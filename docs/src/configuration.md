# Configuration

Configuration is client-scoped: credentials, endpoints, timeouts, and retry policy. Per-request settings such as temperature live in [`GenerationConfig`](./quickstart.md#tuning-generation) instead.

## Precedence

`Config` getters fall back to the environment when a field was not set programmatically, so **an explicit value always wins over an environment variable**. This means `from_env()` is a starting point, not a lock-in:

```rust,no_run
use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()                                  // read everything available
        .openai_base_url("http://localhost:8080/v1") // then override one field
        .model(Model::gpt4o_mini())
        .build()?;
    # let _ = client;
    Ok(())
}
```

Order matters within the builder chain: a later setter overwrites an earlier one, including one populated by `from_env()`.

## Credentials

| Variable | Builder method |
| --- | --- |
| `OPENAI_API_KEY` | `.openai_key(..)` |
| `ANTHROPIC_API_KEY` | `.anthropic_key(..)` |
| `OPENROUTER_API_KEY` | `.openrouter_key(..)` |

A missing key is not a construction error. `build()` succeeds and the failure surfaces as `Error::ProviderNotConfigured` when you actually use that provider. This lets one binary support several providers and only require the keys for the ones it uses.

## Endpoints

| Variable | Builder method | Use for |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `.openai_base_url(..)` | Proxies, gateways, Azure OpenAI |
| `ANTHROPIC_BASE_URL` | `.anthropic_base_url(..)` | Proxies, gateways |
| `OPENROUTER_BASE_URL` | `.openrouter_base_url(..)` | Proxies, gateways |

Base URLs are also the seam that makes the SDK testable without network access — point them at a local mock server.

## OpenRouter attribution

OpenRouter uses attribution headers to identify the calling app.

| Variable | Legacy alias | Builder method |
| --- | --- | --- |
| `OPENROUTER_HTTP_REFERER` | `OPENROUTER_APP_URL` | `.openrouter_http_referer(..)` |
| `OPENROUTER_TITLE` | `OPENROUTER_APP_TITLE` | `.openrouter_title(..)` |
| `OPENROUTER_CATEGORIES` | — | `.openrouter_categories(..)` |

The canonical variables win when both are present. `OPENROUTER_CATEGORIES` is comma-separated, and empty entries are trimmed away:

```sh
export OPENROUTER_CATEGORIES="productivity,agents"
```

## Timeout

| Variable | Builder method | Default |
| --- | --- | --- |
| `AI_TIMEOUT_SECONDS` | `.timeout(seconds)` | 120 |

An unparseable value is ignored and the default is kept, rather than failing at startup.

## Retries

| Variable | Default |
| --- | --- |
| `AI_MAX_RETRIES` | 3 |
| `AI_RETRY_INITIAL_DELAY_MS` | 1000 |
| `AI_RETRY_MAX_DELAY_MS` | 60000 |
| `AI_RETRY_BACKOFF_MULTIPLIER` | 2.0 |
| `AI_RETRY_JITTER` | true |

Retry configuration is only populated from the environment when at least one of these variables is recognized; otherwise the built-in defaults apply. In code, use `RetryConfig` — see [Retries and error handling](./retries-and-errors.md).

## Configuring entirely in code

Nothing requires environment variables. Skip `from_env()` and set everything explicitly, which is often preferable in tests and in services that get configuration from a secret manager:

```rust,no_run
use std::time::Duration;

use rai_sdk::{ClientBuilder, Model, RetryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .openai_key(std::env::var("MY_APP_OPENAI_KEY")?)
        .timeout(30)
        .retry_config(
            RetryConfig::new()
                .with_max_retries(5)
                .with_initial_delay(Duration::from_millis(250)),
        )
        .model(Model::gpt4o_mini())
        .build()?;
    # let _ = client;
    Ok(())
}
```

You can also build a [`Config`](https://docs.rs/rai-sdk/latest/rai_sdk/config/struct.Config.html) directly and hand it to `Client::new`.

## Handling secrets

- Never commit API keys. Use environment variables, a secret manager, or a git-ignored `.env` file the process loads itself.
- Do not log a `Config`. Its `Debug` output contains credentials.
- Keys are only read when requested, so a process that never calls a provider never touches its key.
