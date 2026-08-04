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
| `rustls-tls` | yes | TLS via rustls |
| `native-tls` | no | TLS via the platform stack |

To compile only one provider, turn the defaults off — but remember that the TLS
backend is part of the default set, so you must name one:

```toml
[dependencies]
rai-sdk = { version = "0.1", default-features = false, features = ["anthropic", "rustls-tls"] }
```

Two things are worth knowing about how provider features interact with
configuration:

- A feature controls whether provider support is **compiled in**. Credentials control whether it is **usable at runtime**.
- Requesting a provider whose feature is disabled fails with `Error::ProviderNotEnabled`. Requesting one that is compiled in but has no API key fails with `Error::ProviderNotConfigured`. The two are distinct so you can tell a build-configuration mistake from a deployment mistake.

## Choosing a TLS backend

At least one TLS backend must be enabled when any provider is enabled. Enabling
a provider with neither is a build error with an explanatory message. A build
with no providers and no TLS backend is valid for consumers that only need the
shared data types.

**`rustls-tls` (default)** needs no system OpenSSL, which makes Linux builds
simpler. The cost is that it builds `aws-lc-rs`, which requires **cmake and a C
compiler**. Most CI images have both; minimal containers often do not.

**`native-tls`** uses the operating system's TLS stack — Security Framework on
macOS, SChannel on Windows, OpenSSL on Linux — and avoids building `aws-lc-rs`
with cmake. Choose it if your build environment lacks cmake, or if you want to
honor the system trust store:

```toml
[dependencies]
rai-sdk = { version = "0.1", default-features = false, features = ["anthropic", "native-tls"] }
```

On Linux, `native-tls` links against the system OpenSSL, so you will need its
development package (`libssl-dev` on Debian and Ubuntu) instead.

Cargo features are additive, so another dependency can cause both TLS backends
to be compiled. rai-sdk explicitly uses rustls when both are available. To keep
`aws-lc-rs` and cmake out of the dependency graph, disable default features and
enable only `native-tls`, as in the example above.

Note that this crate also disables `jsonschema`'s default HTTP schema
resolution. Schemas are generated locally from your Rust types, so a schema
`$ref` can never trigger an outbound request — which keeps the TLS choice
meaningful and removes a class of request-forgery risk. Internal `$ref` and
`$defs` resolution is unaffected.

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
