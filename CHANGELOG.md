# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While `rai-sdk` is pre-1.0, minor version bumps may contain breaking changes to
the public API. Breaking changes are always called out below.

## [Unreleased]

## [0.2.1](https://github.com/rmagatti/rai-sdk/compare/v0.2.0...v0.2.1) - 2026-08-11

### Fixed

- *(anthropic)* report input tokens from streaming usage ([#10](https://github.com/rmagatti/rai-sdk/pull/10))

## [0.2.0](https://github.com/rmagatti/rai-sdk/compare/v0.1.1...v0.2.0) - 2026-08-06

### Breaking changes

- `Model`, `ProviderKind`, and `Error` gain new variants for the
  OpenAI-compatible provider; downstream exhaustive `match`es need new arms.
  (Released as a minor bump per this project's pre-1.0 policy above.)

### Added

- first-class OpenAI-compatible provider (Ollama, vLLM, LM Studio) ([#7](https://github.com/rmagatti/rai-sdk/pull/7))
- *(stream)* add serializable wire events for server-side proxying ([#5](https://github.com/rmagatti/rai-sdk/pull/5))

Everything below is additive at the call site: no existing method, field, or
behavior changed. Three public enums do gain variants — `Model`,
`ProviderKind`, and `Error` — which breaks downstream `match` expressions that
were exhaustive over them.

### Added

- **First-class support for OpenAI-compatible endpoints.** Local and
  self-hosted models — Ollama, vLLM, LM Studio, llama.cpp's server, inference
  gateways — are now a provider of their own rather than a base-URL override on
  the OpenAI one.
  - `ClientBuilder::openai_compatible_base_url()` names the endpoint, with
    `ClientBuilder::ollama()` as shorthand for `http://localhost:11434/v1`.
    `ProviderKind::OpenAICompatible` and
    `Model::openai_compatible("llama3.1:8b")` route to it, and model
    identifiers are free-form because these endpoints publish no catalog.
  - **The endpoint is per client, deliberately not per process.** There is no
    `*_BASE_URL` variable for it: a process routinely talks to several
    compatible endpoints at once, each with its own credentials and
    capabilities, so several clients coexist without interfering.
    `OPENAI_BASE_URL` is unchanged and still redirects only the real OpenAI
    provider.
  - **No API key is required.** With none configured, requests carry no
    `Authorization` header at all rather than a placeholder token;
    `ClientBuilder::openai_compatible_key()` adds a bearer token for gateways
    that want one.
  - **Capability gaps are typed.** "OpenAI-compatible" describes a wire format,
    not a feature set, so a request needing tool calling or structured output
    from an endpoint that lacks it fails with the new
    `Error::CapabilityUnsupported` — distinct from the generic HTTP and request
    errors — carrying a `Capability` a caller can match on through
    `Error::unsupported_capability()`. It is raised either from
    `EndpointCapabilities` declared on the client, before any HTTP call is
    made, or by classifying the endpoint's own rejection. Classification only
    applies when the request actually used the capability, so an unrelated bad
    request stays an ordinary `Error::InvalidRequest`. Capabilities are
    declared, never probed.
  - Chat, streaming, tool calling, and structured output reuse the OpenAI
    provider's request builder and SSE parser rather than forking them.
    Divergences are parameterized instead: structured output is requested
    without OpenAI's `strict` flag, which third-party endpoints implement
    unevenly and which the SDK's client-side schema validation makes redundant.
  - The provider rides the existing `openai` feature, since it is the same wire
    format and shares that module's code. A build that talks only to local
    models enables `openai` and nothing else.
  - `examples/local_model.rs` runs chat, streaming, and a capability fallback
    against a local endpoint with no credentials.
- **Serializable stream events for server-side proxying.** A new `wire` module
  lets a server that holds the provider credentials re-emit a generation to its
  own clients — over SSE or a WebSocket — without losing stream semantics.
  - `RequestBuilder::stream_wire_events()` yields `WireStreamEvent`s: an
    internally tagged serde enum whose `"type"` discriminants
    (`message_start`, `text_delta`, `tool_call_start`, `tool_call_delta`,
    `tool_call_end`, `tool_result`, `usage`, `message_stop`, `turn_complete`,
    `error`) are a documented compatibility surface, pinned by a committed JSON
    fixture per variant. `WIRE_PROTOCOL_VERSION` names the current framing and
    rides on every `message_start`.
  - Items are not `Result`s. A mid-stream provider failure arrives as an
    `error` event carrying a serializable `WireError`, so a client can tell
    "the provider refused" from "the network died" — the latter being a stream
    that ends with no terminal event at all.
  - The terminal `usage` event carries the provider's token counts, so a server
    can meter entitlements and a client can display them from the same payload.
  - `StreamAccumulator` is the receiving half: the client-side counterpart of
    `stream_accumulated()`, reassembling wire events into one `Response`
    including usage and tool calls, and rejecting a truncated stream rather
    than silently returning a short answer.
  - `WireStreamEvent` implements `From<StreamEvent>`, and `StreamEvent`
    implements `TryFrom<WireStreamEvent>`, so existing high-level events can be
    forwarded and recovered without loss.
  - `examples/sse_proxy.rs` runs the whole loop — axum handler, SSE
    re-emission, client-side reassembly — in one process.
- `PartialEq` on the shared data types (`Message`, `ContentBlock`,
  `ImageSource`, `FileSource`, `ToolCall`, `ToolDefinition`,
  `ConversationTurn`, `StreamEvent`, `Prompt`, `Usage`, `Response`,
  `StreamChunk`), so callers can compare and assert on them directly.

### Fixed

- The Chat Completions stream parser now accepts `data:` with no space after
  the colon, which the server-sent events specification allows and self-hosted
  servers emit. Previously such frames were silently dropped. Affects the
  OpenAI and OpenAI-compatible providers.
- `Error::ProviderNotEnabled` names the Cargo feature that actually enables the
  provider rather than assuming it matches the provider's name, via the new
  `ProviderKind::feature_name()`.

### Documentation

- **Cancellation is now specified.** Dropping a stream aborts the upstream
  provider request: every streaming method is driven by its consumer, so
  nothing keeps generating in a detached task and no orphaned generation burns
  tokens. This was already the behavior; it is now documented on `stream()`,
  `generate_stream_events()`, `stream_wire_events()`, and
  `stream_accumulated()`, covered by tests, and explained in the guide — along
  with its consequence that a cancelled generation reports no usage.

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
