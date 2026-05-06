# rai-sdk

`rai-sdk` is a Rust SDK for building backend AI workflows across OpenAI, Anthropic, and OpenRouter. It provides typed model selection, typestate request builders, structured output validation, streaming, retry/backoff, multimodal prompts, and automatic tool execution loops.

## Features

- **Typed providers and models**: use `Model::gpt4o_mini()`, `Model::claude_sonnet_46()`, `Model::openrouter_auto()`, or custom provider model IDs.
- **Typestate request builders**: `.generate()` is only available after a prompt and model are available at compile time.
- **Structured output**: derive `JsonSchema` and call `.generate_structured::<T>()` or `.generate_structured_once::<T>()`.
- **Tool calling**: register typed async tools; `generate()` executes tool calls and feeds results back to the model until a final answer is produced.
- **Streaming**: consume provider stream events directly, high-level stream events, or use `stream_accumulated()` to stream internally and return a full response.
- **Retry/backoff**: transient `RateLimit`, `Timeout`, and HTTP errors are retried with configurable exponential backoff and jitter.
- **Multimodal prompts**: send text, image, audio, video, and file content blocks. Provider support varies.

## Installation

Add the SDK to your `Cargo.toml`:

```toml
[dependencies]
rai-sdk = { path = "path/to/rai-sdk" }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures = "0.3"
```

Provider features are enabled by default:

```toml
[features]
default = ["openai", "anthropic", "openrouter"]
```

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
```

## Notes

- `generate()` auto-executes registered tools. `generate_once()` does not.
- `generate_structured()` may use tools before producing typed output. `generate_structured_once()` ignores configured tools.
- Streaming with registered tools is intentionally rejected by the raw streaming API.
- Provider availability is based on enabled Cargo features and configured credentials.
