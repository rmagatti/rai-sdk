# Retries and error handling

Rate limits and timeouts are routine when calling model providers, not exceptional. The SDK retries them by default so you do not have to wrap every call.

## Defaults

| Setting | Default |
| --- | --- |
| Max retries | 3 |
| Initial delay | 1000 ms |
| Max delay | 60000 ms |
| Backoff multiplier | 2.0 |
| Jitter | enabled |

Delays grow exponentially (1s, 2s, 4s, …), are clamped at the maximum, and are randomized by jitter. Jitter matters under load: without it, many clients throttled at the same moment retry in lockstep and re-create the spike that throttled them.

## Configuring retries

```rust,no_run
use std::time::Duration;

use rai_sdk::{ClientBuilder, Model, RetryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let retry = RetryConfig::new()
        .with_max_retries(5)
        .with_initial_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(30))
        .with_backoff_multiplier(2.0)
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

The same values can come from the environment — see [Configuration](./configuration.md#retries).

## Disabling retries

```rust,no_run
# use rai_sdk::{ClientBuilder, Model};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// For every request from this client.
let client = ClientBuilder::new()
    .from_env()
    .no_retry()
    .model(Model::gpt4o_mini())
    .build()?;

// Or for one request only.
let response = client
    .request()
    .no_retry()
    .prompt("One-shot request.")
    .generate()
    .await?;
# println!("{}", response.text());
# Ok(())
# }
```

Disabling retries is the right choice when the caller already has its own retry policy, or when you are inside a request path with a hard latency budget and would rather fail fast. `RetryConfig::none()` is equivalent.

## What is retried

Retried:

- `Error::RateLimit` — the provider throttled you
- `Error::Timeout` — the request exceeded the configured timeout
- Transient HTTP and transport failures

Not retried:

- `Error::Auth` — a bad key will stay bad
- `Error::InvalidRequest` — malformed requests fail deterministically
- `Error::ModelNotAvailable`, `Error::ProviderNotConfigured`, `Error::ProviderNotEnabled`
- `Error::ContentFiltered` — a policy decision, not a transient fault

Check with `error.is_retryable()` rather than enumerating variants.

## Handling errors

`Error` exposes category helpers so you can branch on kind:

```rust,no_run
# use rai_sdk::{ClientBuilder, Model};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
match client.request().prompt("Hello").generate().await {
    Ok(response) => println!("{}", response.text()),
    Err(error) if error.is_auth_error() => eprintln!("check your API key: {error}"),
    Err(error) if error.is_rate_limit() => eprintln!("still throttled after retries: {error}"),
    Err(error) => eprintln!("{} failure: {error}", error.kind_str()),
}
# Ok(())
# }
```

| Helper | Use |
| --- | --- |
| `is_retryable()` | Whether the SDK considers the failure transient |
| `is_auth_error()` | Credential problems |
| `is_rate_limit()` | Throttling |
| `kind_str()` | Stable string category, useful for metrics and logs |
| `provider()` | Which provider failed, when applicable |

`kind_str()` is a better metric label than the full error message, which may contain request-specific detail.

## Notable variants

- `Error::ToolArguments` — a model supplied arguments that failed schema validation. Normally handled internally and returned to the model for self-correction; see [Tool calling](./tool-calling.md#argument-validation).
- `Error::ToolLoopLimitExceeded` — the tool loop hit `max_tool_rounds`.
- `Error::ToolProviderUnsupported` — tool calling is not supported for that provider.
- `Error::ProviderNotEnabled` versus `ProviderNotConfigured` — a missing Cargo feature versus a missing API key. See [Installation](./installation.md#feature-flags).
- `Error::Request` — a provider error with no more specific mapping, including malformed provider responses.

## Interaction with timeouts

The retry budget multiplies the timeout: a 120-second timeout with 3 retries can take well over six minutes worst-case, including backoff. Size the timeout and retry count together against your own deadline instead of tuning them independently.
