# rai-sdk

A highly idiomatic, pure-Rust implementation of an SDK for working with AI providers. Designed for robust backends, it features enum-based static dispatch, stateful typed API wrappers, strict typestate builders, fallible tool schemas, and automatic tool execution loops.

## Key Features

- **Enum Dispatch over Trait Objects**: `Client` stores explicit enum variants (`Model::OpenAI` or `Model::Anthropic`), guaranteeing fast, statically dispatched matching.
- **Strongly Typed Providers**: API payloads are strictly mapped to `serde` structs (e.g. `AnthropicRequest`, `OpenAIRequest`).
- **Fallible Tool Schemas**: `Tool::new("...").handler(...)` uses `schemars` and `serde_json` to validate and generate JSON schemas for your tool arguments at compile/build time.
- **Graceful Tool Error Handling**: Instead of crashing on bad tool schemas or parsing errors, the framework wraps execution issues in `ToolArgumentIssue` structs and returns them to the LLM for self-correction.
- **Typestate API Builders**: You cannot call `.generate()` unless you have provided both a model and a prompt, verified purely by the Rust compiler.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rai-sdk = { path = "path/to/rai-sdk" }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Setup

Set your API keys as environment variables:
```sh
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Examples

You can run the examples from the repository using `cargo run --example <name>`.

### 1. Basic Chat
```rust
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .model(Model::gpt4o_mini())
        .build()?;

    let request = client
        .request()
        .prompt("Explain 'Ownership' in Rust in 2 sentences.");

    let response = request.generate().await?;
    println!("Response:\n{}", response.text());
    Ok(())
}
```

### 2. Structured Output
Ask the model to return data conforming strictly to a specific Rust structure.

```rust
use rai_sdk::{ClientBuilder, GenerationConfig, Model};
use serde::{Deserialize, Serialize};
use rai_sdk::schemars::JsonSchema;

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
        .model(Model::gpt4o_mini())
        .build()?;

    let config = GenerationConfig::default()
        .with_temperature(0.2)
        .with_json_schema_for::<Recipe>()?;

    let response = client
        .request()
        .prompt("Recipe for a chocolate cake")
        .config(config)
        .generate().await?;

    let recipe: Recipe = serde_json::from_str(&response.text())?;
    println!("Parsed Recipe: {:?}", recipe);
    Ok(())
}
```

### 3. Tool Calling
Create a tool, give it a typed schema, and `rai-sdk` will automatically call the function when requested by the model and feed the result back into the prompt.

```rust
use rai_sdk::{ClientBuilder, Model, Result, Tool, ToolContext, schemars::JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

async fn get_weather(args: WeatherArgs, _ctx: ToolContext) -> Result<serde_json::Value> {
    println!("Fetching weather for {}...", args.city);
    Ok(json!({ "city": args.city, "temperature": 22, "condition": "Sunny" }))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let weather_tool = Tool::new("get_current_weather")
        .description("Get the current weather in a given city.")
        .handler(get_weather)?;

    let client = ClientBuilder::new()
        .model(Model::claude_sonnet_45()) // Works with Anthropic seamlessly!
        .tools(vec![weather_tool])
        .build()?;

    let response = client
        .request()
        .prompt("What is the weather like in Paris?")
        .generate().await?;

    println!("Final Response:\n{}", response.text());
    Ok(())
}
```
