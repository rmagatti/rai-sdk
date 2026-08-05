# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While `rai-sdk` is pre-1.0, minor version bumps may contain breaking changes to
the public API. Breaking changes are always called out below.

## [Unreleased]

Nothing yet.

## [0.1.1] - 2026-08-05

### Fixed

- Restore hosted API documentation on docs.rs by replacing the removed nightly
  `doc_auto_cfg` feature with `doc_cfg`. CI now reproduces docs.rs's nightly
  `--cfg docsrs` build so this compatibility path stays covered.

## [0.1.0] - 2026-08-05

Initial public release of the SDK. Everything below is new.

### Added

- **Unified client for three providers.** A single `Client` and `ClientBuilder`
  targeting OpenAI, Anthropic, and OpenRouter, so switching provider is a change
  of model rather than a change of code. Each provider sits behind its own Cargo
  feature (`openai`, `anthropic`, `openrouter`), all enabled by default.
- **Selectable TLS backend.** `rustls-tls` (default) or `native-tls`, so
  consumers who cannot build `aws-lc-rs` — which needs cmake and a C compiler —
  can use the platform TLS stack instead. Enabling a provider with neither
  fails the build with an explanatory message; a providerless build needs no
  TLS backend. If feature unification enables both, rai-sdk uses rustls.
  `jsonschema`'s default HTTP schema resolution is disabled so it
  cannot pull a second TLS stack with a hard-coded crypto provider, which also
  means a schema `$ref` can never trigger an outbound request.
- **Typed models.** A `Model` enum with per-provider variants
  (`OpenAIModel`, `AnthropicModel`, `OpenRouterModel`) and convenience
  constructors such as `Model::gpt4o_mini()`, `Model::claude_sonnet_46()`, and
  `Model::openrouter_auto()`, plus escape hatches
  (`Model::openai_custom`, `Model::anthropic_custom`, `Model::openrouter_custom`)
  for model IDs the enum does not know about.
- **Typestate request builders.** `Client::request()` returns a builder that only
  exposes `generate()` and friends once both a prompt and a model are known, so
  incomplete requests are a compile error rather than a runtime one.
- **Structured output.** `generate_structured::<T>()` and
  `generate_structured_once::<T>()` derive a JSON Schema from a `JsonSchema`
  type, ask the provider to conform to it, validate the response against the
  schema, and deserialize into `StructuredOutput<T>`.
- **Tool calling.** Typed async tools via `Tool` and `ToolContext`, with argument
  schemas generated from Rust types and validated before the handler runs.
  `generate()` drives the full loop: it executes requested tool calls, feeds the
  results back, and continues until the model produces a final answer.
  `generate_once()` returns the raw provider response without executing tools.
- **Streaming.** `stream()` yields raw `ProviderStreamEvent`s,
  `generate_stream_events()` yields higher-level `StreamEvent`s including
  assembled tool calls and turn completion, and `stream_accumulated()` uses the
  streaming transport but returns one complete `Response`. Streaming cannot run a
  tool loop, so a request carrying tools is rejected with
  `Error::InvalidRequest`. The check honors per-request overrides:
  `no_tools()` lets a tool-bearing client stream, and a request-level `tool()` is
  rejected rather than silently dropped.
- **Retries with exponential backoff.** `RetryConfig` controls maximum attempts,
  initial and maximum delay, backoff multiplier, and jitter. Transient rate
  limit, timeout, and retryable HTTP failures are retried automatically; retries
  can be disabled globally with `ClientBuilder::no_retry()` or per request with
  `RequestBuilder::no_retry()`.
- **Multimodal prompts.** `ContentBlock` supports text, image, audio, video, and
  file content, composed through `Message::user_multimodal()` and `Prompt`.
  Provider coverage varies; OpenAI and OpenRouter serialize image content today.
- **Configuration.** `Config` and `ClientBuilder` read credentials, base URLs,
  request timeout, OpenRouter attribution headers, and retry settings from
  environment variables via `from_env()`, with everything also settable in code.
- **Conversation history.** `ConversationTurn` and
  `RequestBuilder::generate_with_history()` for multi-turn conversations.
- **Typed errors.** An `Error` enum covering configuration, provider, HTTP,
  serialization, schema validation, and tool errors, along with
  `ToolArgumentIssue` for precise tool argument diagnostics.
- Documentation: API reference on [docs.rs](https://docs.rs/rai-sdk) and a guide
  at <https://rmagatti.github.io/rai-sdk/>.

[Unreleased]: https://github.com/rmagatti/rai-sdk/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/rmagatti/rai-sdk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rmagatti/rai-sdk/releases/tag/v0.1.0
