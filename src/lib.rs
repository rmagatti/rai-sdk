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
//! - **Retries** — transient rate-limit, timeout, and HTTP failures are retried
//!   with configurable exponential backoff and jitter via [`RetryConfig`].
//! - **Multimodal prompts** — build prompts from text, image, audio, video, and
//!   file [`ContentBlock`]s. Provider support varies.
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
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod client;
pub mod config;
pub mod error;
pub mod generation;
pub mod message;
pub mod model;
pub mod provider;
pub mod retry;
pub mod tool;

pub use client::{Client, ClientBuilder, RequestBuilder};
pub use config::Config;
pub use error::{Error, ProviderKind, Result, ToolArgumentIssue};
pub use generation::GenerationConfig;
pub use message::{
    ContentBlock, ImageSource, Message, Prompt, Response, Role, StreamChunk, StructuredOutput,
    ToolCall, ToolDefinition, Usage,
};
pub use model::{AnthropicModel, Model, OpenAIModel, OpenRouterModel};
pub use retry::RetryConfig;
pub use schemars::{self, JsonSchema};
pub use tool::{Tool, ToolContext};
