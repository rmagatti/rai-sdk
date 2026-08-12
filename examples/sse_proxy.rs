//! The full server-side proxy loop: axum → rai-sdk → SSE → client reassembly.
//!
//! # The pattern
//!
//! A desktop or browser client must never hold provider API keys, so the model
//! call is made by a server the client trusts:
//!
//! ```text
//!   client ──POST /generate──▶  axum handler
//!                                   │  RequestBuilder::stream_wire_events()
//!                                   ▼
//!                               provider (OpenAI, Anthropic, OpenRouter)
//!                                   │  WireStreamEvent
//!                                   ▼
//!   client ◀────SSE data: {…}───  axum handler
//!      │
//!      └── StreamAccumulator ──▶ one complete Response
//! ```
//!
//! The server is stateless. It holds the credentials, meters usage from the
//! `usage` event, and forwards everything else verbatim; the client rebuilds
//! the same stream semantics on the far side using the *same Rust types*.
//!
//! Three properties make this work, and each is exercised below:
//!
//! 1. **Every event serializes.** [`WireStreamEvent`] is a tagged serde enum, so
//!    an SSE `data:` payload is one `serde_json::to_string` call and the client
//!    parses it straight back into the same type.
//! 2. **Failures are events, not silence.** A provider that refuses mid-stream
//!    arrives as `{"type":"error",…}`. A client that receives no terminal event
//!    knows its *connection* died instead — [`StreamAccumulator::finish`] tells
//!    the two apart.
//! 3. **Hang-ups propagate.** When the client disconnects, axum drops the SSE
//!    response, which drops the rai-sdk stream, which aborts the provider
//!    request. No orphaned generation keeps burning tokens.
//!
//! This example runs both halves in one process: it starts the server on an
//! ephemeral loopback port, then acts as its own client.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! cargo run --example sse_proxy
//! ```

use std::convert::Infallible;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use futures::{Stream, StreamExt};
use rai_sdk::client::ModelReady;
use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};
use rai_sdk::{Client, ClientBuilder, Model, ToolDefinition};

// ── Server ─────────────────────────────────────────────────────────────────

/// Stream a generation to the caller as server-sent events.
///
/// Note what this handler does *not* do: buffer, translate, or invent an
/// envelope. The SDK's own event type is the protocol.
async fn generate(
    State(client): State<Arc<Client<ModelReady>>>,
    prompt: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let events = match client
        .request()
        // The browser owns this tool. The proxy advertises its schema so tool
        // calls stream back over WireStreamEvent for the browser to execute.
        .tool_definition(ToolDefinition {
            name: "display_notification".to_string(),
            description: Some("Display a notification in the browser".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "message": { "type": "string" } },
                "required": ["message"],
                "additionalProperties": false
            }),
        })
        .prompt(prompt)
        .stream_wire_events()
        .await
    {
        Ok(events) => events.left_stream(),
        // The request never even opened. Report it in the same envelope so the
        // client has one error path instead of two.
        Err(error) => {
            futures::stream::once(async move { WireStreamEvent::error(&error) }).right_stream()
        }
    };

    // Metering hook: `usage` carries the token counts a server bills against.
    // A real deployment would write them to its accounting store here, and must
    // tolerate their absence — a client that hangs up mid-generation aborts the
    // upstream request, so no usage event is ever produced for it.
    let events = events.inspect(|event| {
        if let WireStreamEvent::Usage { usage } = event {
            eprintln!("[server] usage: {usage:?}");
        }
    });

    let sse = events.map(|event| {
        // `tag()` is the same string as the payload's `"type"` field, so a
        // client can dispatch on the SSE event name or on the JSON — either
        // works, and they cannot disagree.
        Ok(Event::default()
            .event(event.tag())
            .json_data(&event)
            .expect("a wire event always serializes"))
    });

    Sse::new(sse).keep_alive(KeepAlive::default())
}

// ── Client ─────────────────────────────────────────────────────────────────

/// Consume the SSE response and rebuild one `Response` from it.
///
/// The mirror image of the handler: parse each `data:` line back into a
/// [`WireStreamEvent`] and feed it to a [`StreamAccumulator`].
async fn consume(url: &str, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::Client::new()
        .post(url)
        .body(prompt.to_string())
        .send()
        .await?
        .error_for_status()?;

    let mut accumulator = StreamAccumulator::new();
    let mut body = response.bytes_stream();
    let mut buffer = String::new();

    'outer: while let Some(chunk) = body.next().await {
        buffer.push_str(&String::from_utf8_lossy(&chunk?));

        // Minimal SSE framing: events are separated by a blank line, and only
        // the `data:` field carries the payload.
        while let Some(end) = buffer.find("\n\n") {
            let block: String = buffer.drain(..end + 2).collect();

            for line in block.lines() {
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };

                // An event type this build does not know about fails to parse.
                // Skipping it is the right move: a future rai-sdk may add
                // variants, and an old client should degrade rather than die.
                let Ok(event) = serde_json::from_str::<WireStreamEvent>(payload.trim()) else {
                    eprintln!("[client] skipping unrecognized event: {}", payload.trim());
                    continue;
                };

                if let WireStreamEvent::TextDelta { text } = &event {
                    print!("{text}");
                }

                // `push` returns `Err` for an `error` event, which is how the
                // client learns the *provider* refused rather than that the
                // network dropped.
                if let Err(error) = accumulator.push(event) {
                    eprintln!("\n[client] provider error ({}): {error}", error.kind);
                    break 'outer;
                }
            }
        }
    }
    println!();

    // `finish` fails if no terminal event ever arrived, which is exactly the
    // "the connection died" case — distinct from the provider error above.
    let response = accumulator.finish()?;

    println!("[client] model: {}", response.model);
    println!("[client] provider: {}", response.provider);
    println!(
        "[client] finish reason: {}",
        response.finish_reason.as_deref().unwrap_or("-")
    );
    if let Some(usage) = &response.usage {
        println!("[client] usage: {usage:?}");
    }
    println!("[client] reassembled {} characters", response.text().len());

    Ok(())
}

// ── Both halves, one process ───────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(
        ClientBuilder::new()
            .from_env()
            .model(Model::gpt4o_mini())
            .build()?,
    );

    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(client);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}/generate", listener.local_addr()?);
    println!("[server] listening on {url}");

    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    consume(&url, "Explain server-sent events in three sentences.").await?;

    server.abort();
    Ok(())
}
