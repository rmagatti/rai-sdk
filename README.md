# rai-sdk

[![crates.io](https://img.shields.io/crates/v/rai-sdk.svg)](https://crates.io/crates/rai-sdk)
[![docs.rs](https://img.shields.io/docsrs/rai-sdk)](https://docs.rs/rai-sdk)
[![CI](https://github.com/rmagatti/rai-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/rmagatti/rai-sdk/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/rai-sdk.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.86-blue.svg)](https://blog.rust-lang.org/2025/04/03/Rust-1.86.0.html)

`rai-sdk` is a Rust SDK for building backend AI workflows across OpenAI, Anthropic, and OpenRouter. It provides typed model selection, typestate request builders, structured output validation, streaming, retry/backoff, multimodal prompts, and automatic tool execution loops.

- **API reference**: <https://docs.rs/rai-sdk>
- **Guide**: <https://rmagatti.github.io/rai-sdk/>

> **Project status:** early and pre-1.0. The crate is usable today, but the public API may change in breaking ways before `1.0`. Pin an exact version if you need stability.

## Features

- **Typed providers and models**: use `Model::gpt4o_mini()`, `Model::claude_sonnet_46()`, `Model::openrouter_auto()`, or custom provider model IDs.
- **Typestate request builders**: `.generate()` is only available after a prompt and model are available at compile time.
- **Structured output**: derive `JsonSchema` and call `.generate_structured::<T>()` or `.generate_structured_once::<T>()`.
- **Tool calling**: register typed async tools; `generate()` executes tool calls and feeds results back to the model until a final answer is produced.
- **Streaming**: consume provider stream events directly, high-level stream events, or use `stream_accumulated()` to stream internally and return a full response. Dropping a stream aborts the upstream provider request.
- **Proxyable streams**: `stream_wire_events()` yields serializable events so a server can re-emit a generation to its own clients over SSE, and `StreamAccumulator` reassembles them on the far side.
- **Retry/backoff**: transient `RateLimit`, `Timeout`, and HTTP errors are retried with configurable exponential backoff and jitter.
- **Multimodal prompts**: send text, image, audio, video, and file content blocks. Provider support varies.

## Installation

```sh
cargo add rai-sdk
```

Or add it to your `Cargo.toml` directly, along with the crates the examples below use:

```toml
[dependencies]
rai-sdk = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures = "0.3"
```

The minimum supported Rust version is **1.86**.

### Feature flags

Providers (all enabled by default):

- `openai` — OpenAI Chat Completions
- `anthropic` — Anthropic Messages
- `openrouter` — OpenRouter (aggregates many vendors)

TLS backend (at least one required when a provider is enabled):

- `rustls-tls` (default) — no system OpenSSL needed, but builds `aws-lc-rs`, which requires **cmake and a C compiler**
- `native-tls` — uses the platform TLS stack and avoids building `aws-lc-rs`/cmake (Linux needs OpenSSL development files)

Since the TLS backend is part of the default feature set, turning defaults off means naming one explicitly:

```toml
[dependencies]
rai-sdk = { version = "0.1", default-features = false, features = ["anthropic", "rustls-tls"] }
```

Building in a minimal container without cmake? Use `native-tls` instead:

```toml
[dependencies]
rai-sdk = { version = "0.1", default-features = false, features = ["anthropic", "native-tls"] }
```

Omitting both while enabling a provider fails the build with an explanatory
message. A providerless `--no-default-features` build remains valid. If Cargo
feature unification enables both TLS features, rai-sdk uses rustls; use the
`default-features = false` form above to avoid compiling it.

## Configuration

Use environment variables:

```sh
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
```

Optional provider settings:

```sh
export OPENAI_BASE_URL="https://api.openai.com/v1"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_HTTP_REFERER="https://your-app.example"
export OPENROUTER_TITLE="Your App"
export OPENROUTER_CATEGORIES="productivity,agents"
export AI_TIMEOUT_SECONDS="120"
```

Optional retry settings:

```sh
export AI_MAX_RETRIES="3"
export AI_RETRY_INITIAL_DELAY_MS="1000"
export AI_RETRY_MAX_DELAY_MS="60000"
export AI_RETRY_BACKOFF_MULTIPLIER="2.0"
export AI_RETRY_JITTER="true"
```

You can also configure everything in code with `ClientBuilder` and `RetryConfig`.

## Basic Chat

```rust
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let response = client
        .request()
        .prompt("Explain Rust ownership in two sentences.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

## OpenRouter

```rust
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .openrouter_http_referer("https://your-app.example")
        .openrouter_title("Your App")
        .model(Model::openrouter_auto())
        .build()?;

    let response = client
        .request()
        .prompt("Pick the best model available and summarize OpenRouter in one paragraph.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

Use curated OpenRouter constructors like `Model::openrouter_gpt5()`, `Model::openrouter_deepseek_r1()`, and `Model::openrouter_qwen3_coder()`, or pass any provider model ID with `Model::openrouter_custom("vendor/model")`.

## Structured Output

`generate_structured()` validates the model response against a generated JSON Schema and deserializes it into your Rust type.

```rust
use rai_sdk::{ClientBuilder, GenerationConfig, JsonSchema, Model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    steps: Vec<String>,
    prep_time_minutes: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let structured = client
        .request()
        .config(GenerationConfig::new().with_temperature(0.2))
        .prompt("Return a simple chocolate cake recipe as JSON.")
        .generate_structured::<Recipe>()
        .await?;

    println!("{:#?}", structured.output);
    Ok(())
}
```

Use `generate_structured_once()` when configured tools should be ignored and you want a single provider response.

## Tool Calling

Tools are typed handlers. `generate()` automatically runs tool calls, appends tool results, and asks the model to continue until it returns a final response.

```rust
use rai_sdk::{ClientBuilder, JsonSchema, Model, Result, Tool, ToolContext};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct WeatherArgs {
    city: String,
    #[serde(default = "default_unit")]
    unit: String,
}

fn default_unit() -> String {
    "celsius".to_string()
}

async fn get_weather(args: WeatherArgs, _ctx: ToolContext) -> Result<serde_json::Value> {
    Ok(json!({
        "city": args.city,
        "temperature": 22,
        "unit": args.unit,
        "condition": "Sunny"
    }))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let weather_tool = Tool::new("get_current_weather")
        .description("Get the current weather in a city.")
        .handler(get_weather)?;

    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .tool(weather_tool)
        .build()?;

    let response = client
        .request()
        .prompt("What is the weather in Paris right now?")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

Use `.generate_once()` if you want the raw provider response with tool calls but do not want the SDK to execute registered tools.

## Streaming

For a complete response assembled from the streaming transport:

```rust
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

For raw stream events:

```rust
use futures::StreamExt;
use rai_sdk::{provider::ProviderStreamEvent, ClientBuilder, Model};

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

## Proxying a Stream Over SSE

When your server holds the provider credentials and streams results on to a desktop or browser client, the events have to cross a wire. `stream_wire_events()` yields `WireStreamEvent`s, which serialize to a tagged JSON object — one SSE `data:` payload each.

```text
  client ──POST──▶ your server ──rai-sdk──▶ provider
         ◀──SSE─── WireStreamEvent ◀────────┘
```

```rust
use futures::StreamExt;
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

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

    Ok(())
}
```

On the receiving side, `StreamAccumulator` is the client-side counterpart of `stream_accumulated()`:

```rust
use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};

fn reassemble(payloads: &[String]) -> Result<rai_sdk::Response, Box<dyn std::error::Error>> {
    let mut accumulator = StreamAccumulator::new();
    for payload in payloads {
        accumulator.push(serde_json::from_str::<WireStreamEvent>(payload)?)?;
    }
    Ok(accumulator.finish()?)
}
```

`cargo run --example sse_proxy` runs the whole loop — axum handler, SSE re-emission, client-side reassembly — in one process.

### Wire format

Unlike the other streaming methods, `stream_wire_events()` items are not `Result`s. Once the stream is open every outcome is an event, tagged with a `"type"` discriminant:

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

A mid-stream provider failure arrives as `error` rather than as a truncated response, so a client can tell "the provider refused" from "the network died" — the latter being a stream that ends with no terminal event at all. `StreamAccumulator::finish()` enforces that distinction.

**The `"type"` strings and each event's field names are a compatibility surface.** A server and a client can be built from different `rai-sdk` versions, so renaming or removing one is a breaking change and will be called out in the changelog; adding a variant is not. `WireStreamEvent` and `WireErrorKind` are both `#[non_exhaustive]`, and an unrecognized error kind deserializes into `WireErrorKind::Other`, so match with a catch-all arm. `WIRE_PROTOCOL_VERSION` names the current revision of the framing and rides on every `message_start`.

## Cancelling a Stream

Dropping a stream aborts the upstream provider request. Every streaming method is driven entirely by its consumer — the provider's response body is polled from inside the returned stream, never from a detached background task — so dropping the stream closes the connection and the provider stops generating. No orphaned generation keeps burning tokens.

That holds when the surrounding task is cancelled rather than the stream explicitly dropped, which is what a `tokio::time::timeout` or a web-framework client disconnect looks like. Note that a cancelled generation reports no usage, so metering cannot rely on the final usage event alone.

## Multimodal Prompt

```rust
use rai_sdk::{ClientBuilder, ContentBlock, Message, Model, Prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let prompt = Prompt::single(Message::user_multimodal(vec![
        ContentBlock::text("Describe this image in one sentence."),
        ContentBlock::image_url("https://example.com/image.png"),
    ]));

    let response = client.request().prompt(prompt).generate().await?;
    println!("{}", response.text());
    Ok(())
}
```

OpenAI and OpenRouter currently serialize image content. Other block types are represented in the common prompt model, but provider-specific support may be incomplete.

## Retry Configuration

```rust
use std::time::Duration;

use rai_sdk::{ClientBuilder, Model, RetryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let retry = RetryConfig::new()
        .with_max_retries(5)
        .with_initial_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(30))
        .with_jitter(true);

    let client = ClientBuilder::new()
        .from_env()
        .model(Model::claude_sonnet_46())
        .retry_config(retry)
        .build()?;

    let response = client
        .request()
        .prompt("Give me three practical Rust error-handling tips.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

Disable retries globally with `ClientBuilder::new().no_retry()` or per request with `.request().no_retry()`.

## Examples

Run bundled examples from this repository:

```sh
cargo run --example basic_chat
cargo run --example structured_output
cargo run --example tool_calling
cargo run --example sse_proxy
```

## Notes

- `generate()` auto-executes registered tools. `generate_once()` does not.
- `generate_structured()` may use tools before producing typed output. `generate_structured_once()` ignores configured tools.
- Streaming with registered tools is intentionally rejected by the raw streaming API.
- Dropping a stream aborts the upstream provider request, so a cancelled generation reports no usage.
- Provider availability is based on enabled Cargo features and configured credentials.

## Documentation

- [API reference on docs.rs](https://docs.rs/rai-sdk) — every public type and method.
- [Guide](https://rmagatti.github.io/rai-sdk/) — task-oriented chapters on configuration, providers, structured output, tool calling, streaming, and retries.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, the commands CI runs, and the testing policy — the test suite is fully offline and must never require API credentials.

Please also read our [Code of Conduct](CODE_OF_CONDUCT.md). To report a security issue, follow [SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
