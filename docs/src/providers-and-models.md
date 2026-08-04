# Providers and models

A [`Model`](https://docs.rs/rai-sdk/latest/rai_sdk/model/enum.Model.html) value carries both the provider and the wire model ID. Choosing a model therefore chooses a provider — there is no separate provider setting to keep in sync.

```rust
use rai_sdk::{Model, ProviderKind};

let model = Model::gpt4o_mini();
assert_eq!(model.provider(), ProviderKind::OpenAI);
```

## OpenAI

Constructors cover the current GPT and reasoning families, for example:

```rust
use rai_sdk::Model;

let _ = Model::gpt4o_mini();
let _ = Model::gpt4o();
let _ = Model::gpt4_1();
let _ = Model::gpt5();
let _ = Model::gpt5_mini();
let _ = Model::o3();
let _ = Model::o4_mini();
```

Reasoning (o-series) models are detected by the SDK, which omits sampling parameters they reject such as `temperature` and `top_p`. You do not need to special-case that yourself.

## Anthropic

```rust
use rai_sdk::Model;

let _ = Model::claude_sonnet_46();
let _ = Model::claude_opus_47();
let _ = Model::claude_haiku_45();
let _ = Model::claude_35_sonnet();
```

Anthropic model IDs are not vendor-prefixed, unlike OpenRouter's.

## OpenRouter

OpenRouter proxies many vendors behind one API, which makes it a good default when you want breadth without managing several accounts.

```rust
use rai_sdk::Model;

// Let OpenRouter pick.
let _ = Model::openrouter_auto();

// Curated constructors.
let _ = Model::openrouter_gpt5();
let _ = Model::openrouter_claude_sonnet_4_5();
let _ = Model::openrouter_gemini_25_flash();
let _ = Model::openrouter_deepseek_r1();
let _ = Model::openrouter_qwen3_coder();
```

OpenRouter IDs are vendor-prefixed (`vendor/model`).

### Any OpenRouter model

The curated list will always lag the catalog, so pass an ID directly for anything not covered:

```rust
use rai_sdk::Model;

let model = Model::openrouter_custom("mistralai/mistral-large-2512");
```

The ID is passed through verbatim, so a typo surfaces as a provider error rather than a compile error.

### Attribution

OpenRouter identifies calling apps through attribution headers. Set them once on the client:

```rust,no_run
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
        .prompt("Summarize OpenRouter in one paragraph.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

See [Configuration](./configuration.md#openrouter-attribution) for the environment-variable equivalents.

## Choosing a provider

- **OpenAI** — strongest structured-output support via native JSON Schema mode.
- **Anthropic** — long-context work and tool use.
- **OpenRouter** — breadth, fallback, and access to models you do not have direct accounts for. Note that per-vendor quirks leak through: Gemini models reached via OpenRouter reject schemas containing `$schema`, `$defs`, or `$ref`, which is why the SDK normalizes and inlines generated schemas. See [Structured output](./structured-output.md).

## Mixing providers in one process

One client has one default model, but each request can override it, and a client only needs credentials for the providers it actually uses. Check availability at runtime:

```rust,no_run
use rai_sdk::{ClientBuilder, Model, ProviderKind};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    if client.is_provider_available(ProviderKind::Anthropic) {
        let response = client
            .request()
            .model(Model::claude_sonnet_46())
            .prompt("Hello from Anthropic.")
            .generate()
            .await?;
        println!("{}", response.text());
    }

    Ok(())
}
```
