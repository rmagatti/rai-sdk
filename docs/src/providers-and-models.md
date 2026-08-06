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

## OpenAI-compatible endpoints

Ollama, vLLM, LM Studio, llama.cpp's server, and most inference gateways serve `POST {base_url}/chat/completions` in OpenAI's format. The SDK treats that format as a provider in its own right, so a local model is not a special case anywhere else in the API.

Unlike the other providers this one names no service, so the endpoint is set per client rather than from the environment — a process routinely talks to several at once.

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .ollama() // shorthand for http://localhost:11434/v1
        .model(Model::openai_compatible("llama3.1:8b"))
        .build()?;

    let response = client
        .request()
        .prompt("Explain the borrow checker in two sentences.")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

Model identifiers are free-form: there is no catalog to pick from, because they are whatever the operator loaded.

```rust
use rai_sdk::{Model, ProviderKind};

let ollama = Model::openai_compatible("qwen2.5-coder:14b");
let vllm = Model::openai_compatible("Qwen/Qwen2.5-7B-Instruct");

assert_eq!(ollama.provider(), ProviderKind::OpenAICompatible);
assert_eq!(vllm.as_str(), "Qwen/Qwen2.5-7B-Instruct");
```

An API key is optional. With none configured no `Authorization` header is sent at all, which is what a local runtime expects and better than inventing a placeholder token.

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let local = ClientBuilder::new()
    .openai_compatible_base_url("http://localhost:8000/v1")
    .model(Model::openai_compatible("Qwen/Qwen2.5-7B-Instruct"))
    .build()?;

let gateway = ClientBuilder::new()
    .openai_compatible_base_url("https://gateway.internal.example/v1")
    .openai_compatible_key("shared-secret")
    .model(Model::openai_compatible("mixtral-8x7b"))
    .build()?;
# let _ = (local, gateway);
# Ok(())
# }
```

### When the endpoint cannot do something

"OpenAI-compatible" describes a wire format, not a feature set. A small local model may not call tools; a runtime may not honor `response_format`. Both surface as `Error::CapabilityUnsupported`, a variant distinct from the generic HTTP and request errors, so falling back is a match arm rather than a search through an error string.

```rust,no_run
use rai_sdk::{Capability, ClientBuilder, EndpointCapabilities, Model, Tool};

# async fn run(tool: Tool) -> Result<(), Box<dyn std::error::Error>> {
let client = ClientBuilder::new()
    .ollama()
    .model(Model::openai_compatible("llama3.1:8b"))
    .build()?;

match client.request().tool(tool).prompt("What is the weather?").generate().await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.unsupported_capability() == Some(Capability::ToolCalling) => {
        // Retry without tools, switch models, or degrade the feature.
    }
    Err(error) => return Err(error.into()),
}

// Declaring the gap up front turns it into a local failure, with no HTTP call.
let text_only = ClientBuilder::new()
    .ollama()
    .openai_compatible_capabilities(EndpointCapabilities::default().with_tool_calling(false))
    .model(Model::openai_compatible("gemma3:4b"))
    .build()?;
# let _ = text_only;
# Ok(())
# }
```

Capabilities are declared, never probed: auto-detection would cost a round trip on every client build and still be wrong per model.

Two smaller differences are worth knowing. Structured output is requested without OpenAI's `strict` flag, which third-party endpoints implement unevenly — the SDK validates the response against the schema client-side regardless. And the stream parser, shared with the OpenAI provider, tolerates the framing self-hosted servers vary on: `data:` with no space after the colon, a missing `[DONE]` sentinel, and a stream that reports no token usage.

## Choosing a provider

- **OpenAI** — strongest structured-output support via native JSON Schema mode.
- **Anthropic** — long-context work and tool use.
- **OpenRouter** — breadth, fallback, and access to models you do not have direct accounts for. Note that per-vendor quirks leak through: Gemini models reached via OpenRouter reject schemas containing `$schema`, `$defs`, or `$ref`, which is why the SDK normalizes and inlines generated schemas. See [Structured output](./structured-output.md).
- **OpenAI-compatible** — local and self-hosted models, air-gapped deployments, and anything behind an inference gateway. No credential required, and capabilities vary by endpoint and by model.

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
