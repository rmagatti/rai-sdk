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

The streaming methods **reject** requests when tools are registered, rather than silently ignoring them. Executing a tool loop requires sending follow-up requests, which is incompatible with handing you a single continuous stream. You get `Error::InvalidRequest`.

One sharp edge: the check inspects the **client's** tools, not the resolved per-request tool set. So `.no_tools()` on the request does **not** make streaming work on a client that has tools registered — the request is still rejected. Treat streaming as a property of the client.

Your options:

- Use `generate()` and accept non-incremental output.
- Build a separate tool-free client for streaming, and keep the tool-enabled client for `generate()`.
- Drive the loop yourself with `generate_once()`, executing calls between turns.

## Timeouts

Streaming does not exempt a request from the configured timeout. A long generation can still exceed `AI_TIMEOUT_SECONDS`; raise it for workloads that legitimately run long. See [Configuration](./configuration.md#timeout).
