# Installation

## Add the crate

```sh
cargo add rai-sdk
```

Or edit `Cargo.toml` directly:

```toml
[dependencies]
rai-sdk = "0.1"
```

## Companion crates

The SDK is async and schema-driven, so most projects also need:

```toml
[dependencies]
rai-sdk = "0.1"
tokio = { version = "1", features = ["full"] }      # async runtime
serde = { version = "1", features = ["derive"] }    # structured output and tool args
serde_json = "1"                                    # tool return values
futures = "0.3"                                     # only for consuming raw streams
```

`serde` and `serde_json` are required for structured output and tool calling. `futures` is only needed if you consume [`stream()`](./streaming.md) directly, since you need `StreamExt` to iterate it.

You do not need to depend on `schemars` separately. The SDK re-exports it as `rai_sdk::schemars` along with the `JsonSchema` derive, which keeps derive-macro versions in lockstep.

## Minimum supported Rust version

**Rust 1.86.** The crate uses edition 2024 (which needs 1.85), and the dependency tree raises the effective floor to 1.86.

The MSRV is declared as `rust-version` in `Cargo.toml` and verified in CI, so it will not drift silently. Treat an MSRV increase as a breaking change.

## Feature flags

| Feature | Default | Enables |
| --- | --- | --- |
| `openai` | yes | OpenAI Chat Completions |
| `anthropic` | yes | Anthropic Messages |
| `openrouter` | yes | OpenRouter |

All three are on by default. To compile only one provider, turn the defaults off:

```toml
[dependencies]
rai-sdk = { version = "0.1", default-features = false, features = ["anthropic"] }
```

Two things are worth knowing about how features interact with configuration:

- A feature controls whether provider support is **compiled in**. Credentials control whether it is **usable at runtime**.
- Requesting a provider whose feature is disabled fails with `Error::ProviderNotEnabled`. Requesting one that is compiled in but has no API key fails with `Error::ProviderNotConfigured`. The two are distinct so you can tell a build-configuration mistake from a deployment mistake.

## Verify the install

```sh
cargo build
```

Then set a key and run a bundled example from a checkout of the repository:

```sh
export OPENAI_API_KEY="sk-..."
cargo run --example basic_chat
```

See [Configuration](./configuration.md) for the full list of environment variables.
