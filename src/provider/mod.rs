//! Provider implementations and the low-level streaming event they emit.
//!
//! Each provider module wraps one HTTP API and translates the crate's
//! provider-agnostic [`Prompt`](crate::Prompt)/[`Response`](crate::Response)
//! types to and from that API's wire format. Modules are gated behind the
//! matching Cargo feature (`openai`, `anthropic`, `openrouter`), all of which
//! are enabled by default.
//!
//! Most code should go through [`Client`](crate::Client) instead of using these
//! types directly; they are public so that advanced callers can drive a single
//! provider, and because streaming surfaces [`ProviderStreamEvent`].

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

use crate::message::Usage;

/// A low-level event decoded from a provider's server-sent event stream.
///
/// This is the raw shape yielded by
/// [`RequestBuilder::stream`](crate::RequestBuilder::stream). Providers differ
/// in how they chunk tool calls, so tool arguments arrive as a
/// [`ProviderStreamEvent::ToolCallStart`] followed by any number of
/// [`ProviderStreamEvent::ToolCallChunk`]s that must be concatenated. For a
/// pre-assembled view, use
/// [`RequestBuilder::generate_stream_events`](crate::RequestBuilder::generate_stream_events).
#[derive(Debug, Clone)]
pub enum ProviderStreamEvent {
    /// A fragment of assistant text to append to the output so far.
    Text(String),
    /// A tool call has begun; its arguments follow in later chunks.
    ToolCallStart {
        /// Provider-assigned call identifier.
        id: String,
        /// Name of the tool the model wants to run.
        name: String,
    },
    /// A fragment of the JSON argument string for a tool call.
    ToolCallChunk {
        /// Identifier of the call these arguments belong to.
        id: String,
        /// Partial JSON text to append to previously received fragments.
        arguments: String,
    },
    /// Generation finished.
    ///
    /// Depending on the provider, the finish reason and usage may arrive in
    /// separate `Done` events.
    Done {
        /// Why generation stopped, e.g. `"stop"` or `"tool_use"`.
        finish_reason: Option<String>,
        /// Token usage, when the provider reports it.
        usage: Option<Usage>,
    },
}
