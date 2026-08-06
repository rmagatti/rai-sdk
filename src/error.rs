//! Error types shared by every provider.
//!
//! [`enum@Error`] is the single error type returned across the SDK, and [`Result`]
//! is the matching alias. Errors carry the [`ProviderKind`] they originated
//! from where that is meaningful, and expose classification helpers
//! ([`Error::is_retryable`], [`Error::is_auth_error`], [`Error::kind_str`]) so
//! callers can branch on categories instead of matching every variant.
//!
//! # Examples
//!
//! ```no_run
//! use rai_sdk::{Error, ProviderKind};
//!
//! fn describe(error: &Error) -> String {
//!     if error.is_auth_error() {
//!         return "check your API key".to_string();
//!     }
//!     if error.is_retryable() {
//!         return format!("transient {} failure", error.kind_str());
//!     }
//!     match error.provider() {
//!         Some(ProviderKind::OpenAI) => "OpenAI rejected the request".to_string(),
//!         Some(provider) => format!("{provider} rejected the request"),
//!         None => error.to_string(),
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Identifies which AI provider was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// OpenAI's own API (`https://api.openai.com`).
    OpenAI,
    /// Anthropic's Messages API (`https://api.anthropic.com`).
    Anthropic,
    /// OpenRouter's aggregating API (`https://openrouter.ai`).
    OpenRouter,
    /// A self-hosted or third-party endpoint that speaks the OpenAI Chat
    /// Completions wire format, such as Ollama, vLLM, or LM Studio.
    ///
    /// Unlike the other variants this one names no fixed service: the endpoint
    /// is chosen per client with
    /// [`ClientBuilder::openai_compatible_base_url`](crate::ClientBuilder::openai_compatible_base_url).
    #[serde(rename = "openai-compatible")]
    OpenAICompatible,
}

impl ProviderKind {
    /// The Cargo feature that compiles support for this provider in.
    ///
    /// [`ProviderKind::OpenAICompatible`] rides the `openai` feature, because it
    /// reuses that provider's request builder and stream parser, so this is not
    /// simply the lowercased variant name.
    pub fn feature_name(&self) -> &'static str {
        match self {
            ProviderKind::OpenAI | ProviderKind::OpenAICompatible => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenRouter => "openrouter",
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::OpenAI => write!(f, "openai"),
            ProviderKind::Anthropic => write!(f, "anthropic"),
            ProviderKind::OpenRouter => write!(f, "openrouter"),
            ProviderKind::OpenAICompatible => write!(f, "openai-compatible"),
        }
    }
}

/// An optional part of the Chat Completions API that an endpoint may not
/// implement.
///
/// OpenAI-compatible servers implement the same wire format as OpenAI but not
/// always the same feature set: a small local model may have no tool-calling
/// support, and a runtime may not constrain output to a JSON Schema. Requests
/// that need something the endpoint cannot do fail with
/// [`Error::CapabilityUnsupported`] naming the capability, so a caller can fall
/// back instead of parsing an HTTP error body.
///
/// Declare what an endpoint supports with
/// [`EndpointCapabilities`](crate::EndpointCapabilities).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Advertising tools and receiving tool calls back.
    ToolCalling,
    /// Constraining the response with `response_format` — JSON mode or a JSON
    /// Schema.
    StructuredOutput,
}

impl Capability {
    /// The snake_case name this capability serializes as.
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::ToolCalling => "tool_calling",
            Capability::StructuredOutput => "structured_output",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::ToolCalling => write!(f, "tool calling"),
            Capability::StructuredOutput => write!(f, "structured output"),
        }
    }
}

/// Diagnostic info about a tool argument validation failure.
///
/// One issue is produced per JSON Schema violation found in the arguments a
/// model supplied for a tool call. Issues are serialized back to the model as
/// part of the tool error message so it can correct itself and retry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolArgumentIssue {
    /// JSON pointer to the offending value in the arguments (`$` for the root).
    pub path: String,
    /// JSON pointer to the schema keyword that rejected the value.
    pub schema_path: String,
    /// Human-readable description of the violation.
    pub message: String,
}

/// Errors that can occur when using the AI SDK.
///
/// Every fallible operation in this crate returns this type through
/// [`Result`]. Rather than matching each variant, prefer the classification
/// helpers when you only care about the category of failure.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{Error, ProviderKind};
///
/// let error = Error::RateLimit {
///     provider: ProviderKind::OpenAI,
///     message: "slow down".to_string(),
/// };
///
/// assert!(error.is_rate_limit());
/// assert!(error.is_retryable());
/// assert_eq!(error.kind_str(), "rate_limit");
/// assert_eq!(error.provider(), Some(ProviderKind::OpenAI));
/// ```
#[derive(Error, Debug)]
pub enum Error {
    /// Authentication failed (invalid API key, expired token, etc.)
    #[error("authentication error for {provider}: {message}")]
    Auth {
        /// Provider that rejected the credentials.
        provider: ProviderKind,
        /// Message reported by the provider.
        message: String,
    },

    /// The API request failed.
    ///
    /// Used for provider errors that do not map to a more specific variant,
    /// including malformed provider responses.
    #[error("API request failed for {provider}: {message}")]
    Request {
        /// Provider that produced the failure.
        provider: ProviderKind,
        /// Message reported by the provider, or a description of what was wrong
        /// with its response.
        message: String,
    },

    /// Rate limit exceeded.
    ///
    /// Retryable: see [`Error::is_retryable`].
    #[error("rate limit exceeded for {provider}: {message}")]
    RateLimit {
        /// Provider that throttled the request.
        provider: ProviderKind,
        /// Message reported by the provider.
        message: String,
    },

    /// Invalid request (bad parameters, etc.)
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The requested model is not available or not supported.
    #[error("model not available: {model} for provider {provider}")]
    ModelNotAvailable {
        /// Provider the model was requested from.
        provider: ProviderKind,
        /// Model identifier that was rejected.
        model: String,
    },

    /// Provider not configured (missing API key, etc.)
    #[error("provider {0} is not configured")]
    ProviderNotConfigured(ProviderKind),

    /// Provider feature not enabled.
    #[error(
        "provider {0} feature is not enabled — enable the '{feature}' feature in Cargo.toml",
        feature = .0.feature_name()
    )]
    ProviderNotEnabled(ProviderKind),

    /// The endpoint does not implement a part of the API the request needed.
    ///
    /// Raised for OpenAI-compatible endpoints, which share OpenAI's wire format
    /// without necessarily sharing its feature set. It is deliberately distinct
    /// from [`Error::Request`] and [`Error::InvalidRequest`] so a caller can
    /// degrade gracefully — retry without tools, or parse free-form text
    /// instead of asking for a schema — rather than pattern-matching an HTTP
    /// error body.
    ///
    /// Produced either up front, when
    /// [`EndpointCapabilities`](crate::EndpointCapabilities) says the endpoint
    /// lacks the capability, or from the endpoint's own rejection of a request
    /// that used it.
    #[error("{capability} is not supported by the {provider} endpoint at {base_url}: {message}")]
    CapabilityUnsupported {
        /// Provider that could not serve the request.
        provider: ProviderKind,
        /// Capability the request needed.
        capability: Capability,
        /// Base URL of the endpoint that could not serve it.
        base_url: String,
        /// Why the capability is unavailable: the endpoint's own message, or a
        /// note that it was declared unsupported.
        message: String,
    },

    /// Content was filtered/blocked by the provider.
    #[error("content filtered by {provider}: {reason}")]
    ContentFiltered {
        /// Provider that filtered the content.
        provider: ProviderKind,
        /// Reason the provider gave for filtering.
        reason: String,
    },

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP client error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Stream error.
    #[error("stream error: {0}")]
    Stream(String),

    /// Timeout.
    ///
    /// Retryable: see [`Error::is_retryable`].
    #[error("request timed out for {provider}")]
    Timeout {
        /// Provider whose request timed out.
        provider: ProviderKind,
    },

    /// Tool calling is not supported for the selected provider.
    #[error("tool calling is not supported for provider {provider}")]
    ToolProviderUnsupported {
        /// Provider that does not support tool calling.
        provider: ProviderKind,
    },

    /// Tool arguments failed validation.
    ///
    /// Surfaced to the model as a tool error message rather than aborting the
    /// tool loop, so it can retry with corrected arguments.
    #[error("invalid arguments for tool '{name}': {message}")]
    ToolArguments {
        /// Name of the tool whose arguments were rejected.
        name: String,
        /// Summary of the validation failures.
        message: String,
        /// Per-violation diagnostics, sorted and deduplicated.
        issues: Vec<ToolArgumentIssue>,
    },

    /// A requested tool is not registered.
    #[error("tool not found: {name}")]
    ToolNotFound {
        /// Tool name the model tried to call.
        name: String,
    },

    /// Tool execution exceeded the configured loop limit.
    ///
    /// The limit comes from
    /// [`GenerationConfig::with_max_tool_rounds`](crate::GenerationConfig::with_max_tool_rounds).
    #[error("tool execution exceeded the maximum number of rounds ({max_rounds})")]
    ToolLoopLimitExceeded {
        /// Round limit that was hit.
        max_rounds: usize,
    },

    /// Structured output could not be validated against the requested type.
    #[error("structured output validation failed for {provider} model {model}: {message}")]
    StructuredOutput {
        /// Provider that produced the output.
        provider: ProviderKind,
        /// Model that produced the output.
        model: String,
        /// Why the output was rejected (empty, invalid JSON, schema violation,
        /// or deserialization failure).
        message: String,
    },
}

impl Error {
    /// Returns `true` if this error is likely transient and the request can be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::RateLimit { .. } | Error::Timeout { .. } | Error::Http(_)
        )
    }

    /// Returns `true` if this is an authentication error.
    pub fn is_auth_error(&self) -> bool {
        matches!(self, Error::Auth { .. })
    }

    /// Returns `true` if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, Error::RateLimit { .. })
    }

    /// The capability an endpoint could not provide, if this is a
    /// [`Error::CapabilityUnsupported`].
    ///
    /// This is the hook for falling back to a simpler request shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rai_sdk::{Capability, Error, ProviderKind};
    ///
    /// let error = Error::CapabilityUnsupported {
    ///     provider: ProviderKind::OpenAICompatible,
    ///     capability: Capability::ToolCalling,
    ///     base_url: "http://localhost:11434/v1".to_string(),
    ///     message: "the model does not support tools".to_string(),
    /// };
    ///
    /// assert_eq!(error.unsupported_capability(), Some(Capability::ToolCalling));
    /// assert!(!error.is_retryable());
    /// ```
    pub fn unsupported_capability(&self) -> Option<Capability> {
        match self {
            Error::CapabilityUnsupported { capability, .. } => Some(*capability),
            _ => None,
        }
    }

    /// Short error category string for use as a metrics or logging label.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Error::Auth { .. } => "auth",
            Error::Request { .. } => "request",
            Error::RateLimit { .. } => "rate_limit",
            Error::InvalidRequest(_) => "invalid_request",
            Error::ModelNotAvailable { .. } => "model_not_available",
            Error::ProviderNotConfigured(_) => "provider_not_configured",
            Error::ProviderNotEnabled(_) => "provider_not_enabled",
            Error::CapabilityUnsupported { .. } => "capability_unsupported",
            Error::ContentFiltered { .. } => "content_filtered",
            Error::Config(_) => "config",
            Error::Serialization(_) => "serialization",
            Error::Http(_) => "http",
            Error::Stream(_) => "stream",
            Error::Timeout { .. } => "timeout",
            Error::ToolProviderUnsupported { .. } => "tool_provider_unsupported",
            Error::ToolArguments { .. } => "tool_arguments",
            Error::ToolNotFound { .. } => "tool_not_found",
            Error::ToolLoopLimitExceeded { .. } => "tool_loop_limit_exceeded",
            Error::StructuredOutput { .. } => "structured_output",
        }
    }

    /// Get the provider associated with this error, if any.
    pub fn provider(&self) -> Option<ProviderKind> {
        match self {
            Error::Auth { provider, .. }
            | Error::Request { provider, .. }
            | Error::RateLimit { provider, .. }
            | Error::ModelNotAvailable { provider, .. }
            | Error::ProviderNotConfigured(provider)
            | Error::ProviderNotEnabled(provider)
            | Error::CapabilityUnsupported { provider, .. }
            | Error::ContentFiltered { provider, .. }
            | Error::Timeout { provider }
            | Error::ToolProviderUnsupported { provider }
            | Error::StructuredOutput { provider, .. } => Some(*provider),
            _ => None,
        }
    }
}

/// Result type alias for AI SDK operations.
pub type Result<T> = std::result::Result<T, Error>;
