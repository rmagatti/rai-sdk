# Streaming

Streaming exists for two different reasons, and the SDK offers a method for each:

- You want to **show tokens as they arrive** — use `stream()` and handle events.
- You want a **complete response but delivered over the streaming transport** — use `stream_accumulated()`, which streams internally and hands you a finished `Response`. This is useful for long generations that would otherwise sit near a request timeout.

## Accumulated streaming

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let response = client
        .request()
        .prompt("Write a short launch announcement.")
        .stream_accumulated()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

The result is the same shape as `generate()` — only the transport differs.

## Raw stream events

For incremental output, iterate the stream. This needs `futures::StreamExt` in scope.

```rust,no_run
use futures::StreamExt;
use rai_sdk::{ClientBuilder, Model, provider::ProviderStreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let mut stream = client
        .request()
        .prompt("Count from one to five.")
        .stream()
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            ProviderStreamEvent::Text(text) => print!("{text}"),
            ProviderStreamEvent::Done { .. } => println!(),
            _ => {}
        }
    }

    Ok(())
}
```

Each item is a `Result`, because a stream can fail partway through. Do not use `while let Some(Ok(event))`: that silently swallows mid-stream errors and looks like a clean early finish.

## Event kinds

[`ProviderStreamEvent`](https://docs.rs/rai-sdk/latest/rai_sdk/provider/enum.ProviderStreamEvent.html) normalizes each provider's SSE format:

| Event | Meaning |
| --- | --- |
| `Text(String)` | An incremental text delta. Concatenate them in order. |
| `ToolCallStart { id, name }` | The model began requesting a tool call. |
| `ToolCallChunk { id, arguments }` | A fragment of that call's JSON arguments. Accumulate by `id`. |
| `Done { finish_reason, usage }` | The stream ended; usage is reported here when the provider supplies it. |

Tool-call arguments arrive as fragments that are not individually valid JSON. Buffer all chunks for an `id` and only parse once `Done` arrives.

Match non-exhaustively (`_ => {}`) so new event kinds do not break your code.

## Streaming and tools

`stream()`, `generate_stream_events()`, and `stream_accumulated()` **reject** requests that carry tools, rather than silently ignoring them, with `Error::InvalidRequest`. Executing a tool loop requires sending follow-up requests, which is incompatible with handing you a single continuous stream.

The check uses the request's *effective* tool set, so one client can do both. Opt out per request to stream:

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

# async fn run(tool: rai_sdk::Tool) -> Result<(), Box<dyn std::error::Error>> {
let client = ClientBuilder::new()
    .from_env()
    .model(Model::gpt4o_mini())
    .tool(tool)
    .build()?;

// Tools run here.
let answer = client.request().prompt("Use a tool if needed.").generate().await?;

// And this request streams, because it opts out of tools.
let stream = client
    .request()
    .no_tools()
    .prompt("Just write prose.")
    .stream()
    .await?;
# let _ = (answer, stream);
# Ok(())
# }
```

The rule applies in both directions: adding a tool with `.tool(..)` on a request makes it non-streamable even if the client has no tools, instead of quietly dropping it.

`stream_wire_events()` is the intentional exception. It is the server-side proxy API: it advertises the effective tool definitions to the provider, forwards tool-call start/delta/end events to the remote client, and never runs handlers on the server. The remote client executes the call and sends its result in a later request. This lets a credential-holding proxy support tool-capable clients without owning their tool implementations.

When the proxy has no local handler, register the provider-facing schema directly with `tool_definition()`:

```rust,no_run
use rai_sdk::ToolDefinition;

# async fn run(client: &rai_sdk::Client<rai_sdk::client::ModelReady>) -> Result<(), Box<dyn std::error::Error>> {
let events = client
    .request()
    .tool_definition(ToolDefinition {
        name: "get_weather".into(),
        description: Some("Get the weather for a city".into()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    })
    .prompt("What is the weather in Paris?")
    .stream_wire_events()
    .await?;
# let _ = events;
# Ok(())
# }
```

Executable tools registered with `tool()` are also advertised by `stream_wire_events()`, but their handlers are deliberately not invoked on this path.

If you need incremental output *and* tool execution in one exchange, drive the loop yourself with `generate_once()`, executing calls between turns.

## Cancellation

**Dropping a stream aborts the upstream provider request.** Every streaming method is driven entirely by its consumer: the provider's HTTP response body is polled from inside the returned stream, never from a detached background task. Dropping the stream drops that body, closes the connection, and the provider stops generating.

That holds for the whole family — `stream()`, `generate_stream_events()`, `stream_wire_events()`, and `stream_accumulated()` — and it holds when the surrounding task is cancelled rather than the stream explicitly dropped, which is what a `tokio::time::timeout` or a web-framework client disconnect looks like. Nothing keeps running in the background.

Two consequences to plan for:

- A cancelled generation produces **no terminal event**, so no usage is reported. Providers still bill for what they generated before the abort, so metering cannot rely on the final usage event alone.
- Conversely, there is nothing to clean up. You do not need a cancellation token or an abort handle; letting the stream go out of scope is the whole mechanism.

## Proxying a stream to your own clients

If your server holds the provider credentials and streams results on to a desktop or browser client, the events have to cross a wire. The [`wire`](https://docs.rs/rai-sdk/latest/rai_sdk/wire/index.html) module covers that case:

```text
  client ──POST──▶ your server ──rai-sdk──▶ provider
         ◀──SSE─── WireStreamEvent ◀────────┘
```

`stream_wire_events()` yields `WireStreamEvent`s, which serialize to a tagged JSON object — one SSE `data:` payload each:

```rust,no_run
use futures::StreamExt;
use rai_sdk::{ClientBuilder, Model};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;

let mut events = client
    .request()
    .prompt("Explain SSE in one sentence.")
    .stream_wire_events()
    .await?;

while let Some(event) = events.next().await {
    // event: text_delta
    // data: {"type":"text_delta","text":"Server-sent"}
    println!("event: {}\ndata: {}\n", event.tag(), serde_json::to_string(&event)?);
}
# Ok(())
# }
```

On the receiving side, `StreamAccumulator` is the client-side counterpart of `stream_accumulated()`: feed it the parsed events and it hands back one `Response`, tool calls included.

```rust
use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let payloads = [
    r#"{"type":"message_start","protocol_version":1,"model":"gpt-4o-mini","provider":"openai"}"#,
    r#"{"type":"text_delta","text":"Hello, "}"#,
    r#"{"type":"text_delta","text":"world."}"#,
    r#"{"type":"usage","usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#,
    r#"{"type":"message_stop","finish_reason":"stop"}"#,
];

let mut accumulator = StreamAccumulator::new();
for payload in payloads {
    accumulator.push(serde_json::from_str::<WireStreamEvent>(payload)?)?;
}

let response = accumulator.finish()?;
assert_eq!(response.text(), "Hello, world.");
assert_eq!(response.usage.unwrap().total_tokens, Some(12));
# Ok(())
# }
```

`examples/sse_proxy.rs` runs the whole loop — an axum handler, SSE re-emission, and client-side reassembly — in one process.

### Wire events

Unlike the other streaming methods, `stream_wire_events()` items are not `Result`s. Once the stream is open, every outcome is an event:

| `"type"` | Meaning |
| --- | --- |
| `message_start` | First event of every stream; names the protocol version, model, and provider. |
| `text_delta` | Append this text to the output so far. |
| `tool_call_start` / `tool_call_delta` / `tool_call_end` | A tool call, first incrementally and then assembled. |
| `tool_result` | The output of executing a tool call. Only a proxy that runs tools itself emits this. |
| `usage` | Token counts, emitted once just before the terminal event. |
| `message_stop` | Terminal event of a successful stream. |
| `turn_complete` | An assembled `ConversationTurn`, for history. |
| `error` | Terminal event of a failed stream. |

A mid-stream provider failure arrives as `error` rather than as a truncated response, which is the point: a client that receives no terminal event at all knows its *connection* died instead. `StreamAccumulator::finish()` enforces the distinction — it returns the carried error for the first case and a `stream`-kind error naming the truncation for the second.

### Versioning

The `"type"` strings and each event's field names are a compatibility surface: a server and a client can be built from different `rai-sdk` versions. Renaming or removing one is a breaking change and will be called out in the changelog. Adding a variant is not, so match with a catch-all arm — `WireStreamEvent` and `WireErrorKind` are both `#[non_exhaustive]`, and an unrecognized error kind deserializes into `WireErrorKind::Other` instead of failing.

`WIRE_PROTOCOL_VERSION` names the current revision of the framing and rides on every `message_start`. It is bumped only when a client must react to a framing change, never for additive variants.

## Timeouts

Streaming does not exempt a request from the configured timeout. A long generation can still exceed `AI_TIMEOUT_SECONDS`; raise it for workloads that legitimately run long. See [Configuration](./configuration.md#timeout).
