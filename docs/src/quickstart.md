# Quickstart

## Set a key

```sh
export OPENAI_API_KEY="sk-..."
```

## Make a request

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
        .prompt("Explain Rust ownership in two sentences.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

## What each step does

**`ClientBuilder::new().from_env()`** reads credentials and settings from the environment. Anything you set explicitly on the builder afterwards takes precedence.

**`.model(Model::gpt4o_mini())`** sets the default model. This also changes the builder's type: only after a model is present does `build()` produce a client whose `request()` starts in a model-ready state. That is why the next step does not need to repeat the model.

**`.build()?`** constructs the HTTP client and validates configuration. It fails if the selected provider has no usable credentials.

**`.request().prompt(...)`** starts a request. `prompt()` accepts a `&str`, a `String`, a [`Message`](./multimodal-prompts.md), or a full [`Prompt`](./multimodal-prompts.md).

**`.generate().await?`** sends the request and, if tools are registered, runs the tool loop until the model produces a final answer. Use `generate_once()` for exactly one provider call with no tool execution.

**`response.text()`** concatenates the text content of the response.

## Overriding the model per request

The client's model is a default, not a constraint. Override it on a single request:

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    // Uses Anthropic for this one request; the client default is unchanged.
    let response = client
        .request()
        .model(Model::claude_sonnet_46())
        .prompt("Summarize the Rust borrow checker.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

This requires credentials for whichever provider you name, so the example above needs `ANTHROPIC_API_KEY` in addition to `OPENAI_API_KEY`.

## Tuning generation

Pass a [`GenerationConfig`](https://docs.rs/rai-sdk/latest/rai_sdk/generation/struct.GenerationConfig.html) to control sampling and limits:

```rust,no_run
use rai_sdk::{ClientBuilder, GenerationConfig, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let response = client
        .request()
        .config(
            GenerationConfig::new()
                .with_temperature(0.2)
                .with_max_tokens(512),
        )
        .prompt("List three Rust testing tips.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

Note that `temperature` and `top_p` are ignored for OpenAI reasoning (o-series) models, which do not accept them.

For repeated Anthropic requests that share a long system prompt or tool set,
opt in to Anthropic's ephemeral prompt caching:

```rust
use rai_sdk::GenerationConfig;

let config = GenerationConfig::new().with_prompt_caching(true);
```

This setting adds cache breakpoints to Anthropic requests only. Other providers
silently ignore it, and caching is off by default.

## Next steps

- [Configuration](./configuration.md) — every environment variable and its builder equivalent.
- [Structured output](./structured-output.md) — get a typed value instead of text.
- [Tool calling](./tool-calling.md) — let the model call your code.
- [Streaming](./streaming.md) — render tokens as they arrive.
