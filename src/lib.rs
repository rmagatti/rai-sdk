//! A unified Rust SDK for backend AI workflows across OpenAI, Anthropic, and
//! OpenRouter.
//!
//! `rai-sdk` wraps three provider APIs behind one typed client so switching
//! models does not mean rewriting request-building, streaming, or tool-calling
//! code.
//!
//! # Capabilities
//!
//! - **Typed providers and models** — construct models with [`Model::gpt4o_mini`],
//!   [`Model::claude_sonnet_46`], or [`Model::openrouter_auto`], or pass any
//!   provider model ID directly.
//! - **Typestate request builders** — [`RequestBuilder::generate`] only exists
//!   once a prompt and a model are present, so incomplete requests fail to
//!   compile rather than at runtime.
//! - **Structured output** — derive [`JsonSchema`] and call
//!   [`RequestBuilder::generate_structured`] to validate the response against a
//!   generated schema and deserialize it into your own type.
//! - **Tool calling** — register typed async tools with [`Tool`];
//!   [`RequestBuilder::generate`] runs the tool loop, feeding results back until
//!   the model produces a final answer.
//! - **Streaming** — consume raw provider events, or use
//!   [`RequestBuilder::stream_accumulated`] to stream internally and return a
//!   complete [`Response`].
//! - **Proxyable streams** — [`RequestBuilder::stream_wire_events`] yields
//!   serializable [`WireStreamEvent`]s so a server can re-emit a generation to
//!   its own clients over SSE, and [`StreamAccumulator`] reassembles them on the
//!   far end. See the [`wire`] module.
//! - **Retries** — transient rate-limit, timeout, and HTTP failures are retried
//!   with configurable exponential backoff and jitter via [`RetryConfig`].
//! - **Multimodal prompts** — build prompts from text, image, audio, video, and
//!   file [`ContentBlock`]s. Provider support varies.
//! - **Local and self-hosted models** — point a client at any endpoint speaking
//!   the OpenAI Chat Completions format with
//!   [`ClientBuilder::openai_compatible_base_url`] (or
//!   [`ClientBuilder::ollama`]) and name models with
//!   [`Model::openai_compatible`]. See the `provider::openai_compatible`
//!   module, which the `openai` feature gates.
//!
//! # Quickstart
//!
//! ```no_run
//! use rai_sdk::{ClientBuilder, Model};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ClientBuilder::new()
//!     .from_env()
//!     .model(Model::gpt4o_mini())
//!     .build()?;
//!
//! let response = client
//!     .request()
//!     .prompt("Explain Rust ownership in two sentences.")
//!     .generate()
//!     .await?;
//!
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! [`ClientBuilder::from_env`] reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
//! `OPENROUTER_API_KEY`, along with optional base-URL, timeout, and retry
//! overrides. Everything can also be set explicitly on [`ClientBuilder`] or
//! [`Config`], and explicit values take precedence over the environment.
//!
//! # Cargo features
//!
//! The `openai`, `anthropic`, and `openrouter` features are all enabled by
//! default and gate the corresponding provider support. Disable the defaults to
//! compile against only the providers you use.
//!
//! `openai` additionally gates the OpenAI-compatible provider, which reuses
//! that module's request builder and stream parser rather than duplicating
//! them. A build that talks only to local models therefore enables `openai`
//! and nothing else.
//!
//! Enabling a provider also requires at least one TLS backend:
//!
//! - `rustls-tls` (default) needs no system OpenSSL, but builds `aws-lc-rs`,
//!   which requires cmake and a C compiler.
//! - `native-tls` uses the platform TLS stack instead, avoiding that build
//!   requirement.
//!
//! Because a TLS backend is part of the default feature set, disabling default
//! features means re-enabling one explicitly:
//!
//! ```toml
//! rai-sdk = { version = "0.1", default-features = false, features = ["anthropic", "native-tls"] }
//! ```
//!
//! Cargo features are additive, so dependency feature unification can enable
//! both backends. That configuration is supported and uses rustls; select only
//! `native-tls` as shown above to avoid compiling `aws-lc-rs`.
//!
//! # Further reading
//!
//! The [guide](https://rmagatti.github.io/rai-sdk/) covers each capability in
//! task-oriented chapters. Its examples are compile-checked against this crate,
//! so they stay in sync with the API you see here.
#![cfg_attr(docsrs, feature(doc_cfg))]

// Catch a provider without a TLS backend at compile time. A featureless build
// is valid because it cannot make provider requests and is still useful to
// consumers that only need the crate's shared data types.
#[cfg(all(
    any(feature = "openai", feature = "anthropic", feature = "openrouter"),
    not(any(feature = "rustls-tls", feature = "native-tls"))
))]
compile_error!(
    "rai-sdk has a provider enabled but no TLS backend. \
     This usually means `default-features = false` was set without re-enabling one. \
     Add `rustls-tls` (the default), or `native-tls` if you cannot build aws-lc-rs, \
     which requires cmake and a C compiler."
);

// Compile-check every Rust snippet in the mdBook guide as a doctest, so the
// published guide cannot drift away from the real API. This module only exists
// while rustdoc is collecting doctests, so it adds nothing to the built crate or
// to the rendered documentation.
#[cfg(doctest)]
mod guide {
    macro_rules! chapter {
        ($name:ident, $path:literal) => {
            #[doc = include_str!($path)]
            pub struct $name;
        };
    }

    chapter!(Introduction, "../docs/src/introduction.md");
    chapter!(Installation, "../docs/src/installation.md");
    chapter!(Quickstart, "../docs/src/quickstart.md");
    chapter!(Configuration, "../docs/src/configuration.md");
    chapter!(ProvidersAndModels, "../docs/src/providers-and-models.md");
    chapter!(StructuredOutput, "../docs/src/structured-output.md");
    chapter!(ToolCalling, "../docs/src/tool-calling.md");
    chapter!(Streaming, "../docs/src/streaming.md");
    chapter!(MultimodalPrompts, "../docs/src/multimodal-prompts.md");
    chapter!(RetriesAndErrors, "../docs/src/retries-and-errors.md");
    chapter!(Examples, "../docs/src/examples.md");
    chapter!(Contributing, "../docs/src/contributing.md");
}

pub mod client;
pub mod config;
pub mod error;
pub mod generation;
pub mod message;
pub mod model;
pub mod provider;
pub mod retry;
pub mod tool;
pub mod wire;

pub use client::{Client, ClientBuilder, RequestBuilder};
pub use config::{Config, EndpointCapabilities};
pub use error::{Capability, Error, ProviderKind, Result, ToolArgumentIssue};
pub use generation::GenerationConfig;
pub use message::{
    ContentBlock, ImageSource, Message, Prompt, Response, Role, StreamChunk, StructuredOutput,
    ToolCall, ToolDefinition, Usage,
};
pub use model::{AnthropicModel, Model, OpenAICompatibleModel, OpenAIModel, OpenRouterModel};
pub use retry::RetryConfig;
pub use schemars::{self, JsonSchema};
pub use tool::{Tool, ToolContext};
pub use wire::{
    StreamAccumulator, WIRE_PROTOCOL_VERSION, WireError, WireErrorKind, WireStreamEvent,
};
