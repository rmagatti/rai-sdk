//! Serializable stream events for proxying a generation across a network hop.
//!
//! The in-process streaming types ([`ProviderStreamEvent`] and [`StreamEvent`])
//! are shaped for a consumer that lives in the same process as the SDK. A
//! *proxy* deployment does not: a server holds the provider credentials, calls
//! the provider through `rai-sdk`, and re-emits each event to a client over
//! server-sent events (SSE) or a WebSocket. The client then rebuilds the same
//! stream semantics — text deltas, tool-call activity, token usage, stop
//! reason — from what arrived on the wire.
//!
//! ```text
//!   client ──HTTP──▶ your server ──rai-sdk──▶ provider
//!          ◀──SSE─── WireStreamEvent ◀────────┘
//! ```
//!
//! This module is that wire layer:
//!
//! - [`WireStreamEvent`] is a `Serialize`/`Deserialize` event enum with an
//!   explicit, stable JSON representation.
//! - [`WireError`] projects the crate's [`enum@Error`] into something that can
//!   cross a wire, so a mid-stream provider failure arrives as an *event*
//!   rather than as a dropped connection.
//! - [`StreamAccumulator`] is the receiving end of
//!   [`RequestBuilder::stream_accumulated`](crate::RequestBuilder::stream_accumulated):
//!   feed it the events a client parsed off the wire and it hands back one
//!   assembled [`Response`].
//!
//! Produce the events with
//! [`RequestBuilder::stream_wire_events`](crate::RequestBuilder::stream_wire_events).
//!
//! # Wire format
//!
//! [`WireStreamEvent`] is an **internally tagged** serde enum. Every event is a
//! JSON object carrying a `"type"` discriminant alongside that variant's
//! fields:
//!
//! ```json
//! {"type":"message_start","protocol_version":1,"model":"gpt-4o-mini","provider":"openai"}
//! {"type":"text_delta","text":"Hello"}
//! {"type":"usage","usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}
//! {"type":"message_stop","finish_reason":"stop"}
//! ```
//!
//! The full set of `"type"` values is:
//!
//! | `"type"` | Variant | Meaning |
//! | --- | --- | --- |
//! | `message_start` | [`WireStreamEvent::MessageStart`] | First event of every stream; names the protocol version, model, and provider. |
//! | `text_delta` | [`WireStreamEvent::TextDelta`] | Append this text to the output so far. |
//! | `tool_call_start` | [`WireStreamEvent::ToolCallStart`] | A tool call has begun; arguments follow. |
//! | `tool_call_delta` | [`WireStreamEvent::ToolCallDelta`] | A fragment of that call's JSON arguments. |
//! | `tool_call_end` | [`WireStreamEvent::ToolCallEnd`] | The call is complete, with its arguments assembled. |
//! | `tool_result` | [`WireStreamEvent::ToolResult`] | The output of executing a tool call. |
//! | `usage` | [`WireStreamEvent::Usage`] | Token counts for the generation. |
//! | `message_stop` | [`WireStreamEvent::MessageStop`] | Terminal event of a successful stream. |
//! | `turn_complete` | [`WireStreamEvent::TurnComplete`] | An assembled [`ConversationTurn`], for history. |
//! | `error` | [`WireStreamEvent::Error`] | Terminal event of a failed stream. |
//!
//! ## The variant names are a compatibility surface
//!
//! Those `"type"` strings — and the field names inside each event — are part of
//! this crate's public API in exactly the way a function signature is. A server
//! and a client can be built from different `rai-sdk` versions, so renaming a
//! tag silently breaks every deployed client. Treat them as frozen:
//!
//! - **Renaming or removing a `"type"` value, or renaming a field, is a
//!   breaking change** and will only happen in a major (pre-1.0: minor) release,
//!   with a changelog entry.
//! - **Adding a variant, or adding an optional field to an existing variant, is
//!   additive** and can happen in a patch or minor release. Both
//!   [`WireStreamEvent`] and [`WireErrorKind`] are `#[non_exhaustive]`, and
//!   unknown [`WireErrorKind`] values deserialize into
//!   [`WireErrorKind::Other`], so a client compiled against an older version
//!   keeps parsing streams from a newer server. Match with a `_ => {}` arm and
//!   ignore what you do not recognize.
//!
//! [`WIRE_PROTOCOL_VERSION`] names the current revision of this contract and is
//! carried on every [`WireStreamEvent::MessageStart`]. It is bumped only when
//! the *framing* changes in a way a client must react to — not for additive
//! variants. A client that sees a `protocol_version` it does not understand
//! should refuse the stream rather than guess.
//!
//! # Cancellation
//!
//! Dropping a stream aborts the upstream provider request. The "Cancellation"
//! section of
//! [`RequestBuilder::stream_wire_events`](crate::RequestBuilder::stream_wire_events)
//! has the details and what they mean for a proxy.
//!
//! # Examples
//!
//! Server side — turn each event into an SSE `data:` payload:
//!
//! ```no_run
//! use futures::StreamExt;
//! use rai_sdk::{ClientBuilder, Model};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
//! let mut events = client
//!     .request()
//!     .prompt("Explain SSE in one sentence.")
//!     .stream_wire_events()
//!     .await?;
//!
//! while let Some(event) = events.next().await {
//!     let payload = serde_json::to_string(&event)?;
//!     println!("event: {}\ndata: {payload}\n", event.tag());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Client side — rebuild one [`Response`] from the payloads:
//!
//! ```
//! use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let payloads = [
//!     r#"{"type":"message_start","protocol_version":1,"model":"gpt-4o-mini","provider":"openai"}"#,
//!     r#"{"type":"text_delta","text":"Hello, "}"#,
//!     r#"{"type":"text_delta","text":"world."}"#,
//!     r#"{"type":"usage","usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#,
//!     r#"{"type":"message_stop","finish_reason":"stop"}"#,
//! ];
//!
//! let mut accumulator = StreamAccumulator::new();
//! for payload in payloads {
//!     accumulator.push(serde_json::from_str::<WireStreamEvent>(payload)?)?;
//! }
//!
//! let response = accumulator.finish()?;
//! assert_eq!(response.text(), "Hello, world.");
//! assert_eq!(response.usage.unwrap().total_tokens, Some(12));
//! # Ok(())
//! # }
//! ```
//!
//! [`ProviderStreamEvent`]: crate::provider::ProviderStreamEvent

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error::{Error, ProviderKind};
use crate::message::{ConversationTurn, Message, Response, StreamEvent, ToolCall, Usage};

/// The revision of the [`WireStreamEvent`] framing this build speaks.
///
/// Carried on every [`WireStreamEvent::MessageStart`] so a client can check the
/// contract before consuming a stream. See the
/// [module documentation](self#the-variant-names-are-a-compatibility-surface)
/// for when this is bumped.
pub const WIRE_PROTOCOL_VERSION: u32 = 1;

fn default_protocol_version() -> u32 {
    WIRE_PROTOCOL_VERSION
}

/// A stream event in its serializable, over-the-wire form.
///
/// See the [module documentation](self#wire-format) for the JSON shape and the
/// compatibility guarantees attached to the `"type"` tags.
///
/// A well-formed stream starts with exactly one
/// [`MessageStart`](WireStreamEvent::MessageStart) and ends with exactly one
/// terminal event — [`MessageStop`](WireStreamEvent::MessageStop) on success,
/// [`Error`](WireStreamEvent::Error) on failure (see
/// [`is_terminal`](WireStreamEvent::is_terminal)). A stream that simply stops
/// is a *truncated* stream: the connection died. That distinction is the whole
/// point of putting errors on the wire, and [`StreamAccumulator::finish`]
/// enforces it.
///
/// This enum is `#[non_exhaustive]`: new event types are additive, so match
/// with a catch-all arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum WireStreamEvent {
    /// The stream has opened. Always the first event.
    MessageStart {
        /// Revision of the wire framing the sender speaks.
        ///
        /// Defaults to [`WIRE_PROTOCOL_VERSION`] when absent, so payloads
        /// written before this field existed still parse.
        #[serde(default = "default_protocol_version")]
        protocol_version: u32,
        /// Model identifier the request was routed to.
        model: String,
        /// Provider serving the request.
        provider: ProviderKind,
    },

    /// An incremental piece of assistant text to append to the output so far.
    TextDelta {
        /// Text to append.
        text: String,
    },

    /// A tool call has begun; its arguments arrive in later events.
    ToolCallStart {
        /// Provider-assigned call identifier.
        id: String,
        /// Name of the tool the model wants to run.
        name: String,
    },

    /// A fragment of the JSON argument string for a tool call.
    ToolCallDelta {
        /// Identifier of the call these arguments belong to.
        id: String,
        /// Partial JSON text to append to previously received fragments.
        arguments: String,
    },

    /// A tool call has finished streaming, with its arguments assembled.
    ToolCallEnd {
        /// Provider-assigned call identifier.
        id: String,
        /// Name of the tool the model wants to run.
        name: String,
        /// Complete raw JSON argument string.
        arguments: String,
    },

    /// The result of executing a tool call.
    ///
    /// Never emitted by
    /// [`stream_wire_events`](crate::RequestBuilder::stream_wire_events), which
    /// does not run tools. It exists so a proxy that executes tools itself can
    /// report them to its client using the same envelope.
    ToolResult {
        /// Identifier of the call this result answers.
        id: String,
        /// Serialized tool output.
        result: String,
    },

    /// Token usage for the generation.
    ///
    /// Emitted once, just before the terminal event, when the provider reports
    /// usage. This is the event a server bills against and a client displays,
    /// so it deliberately carries the counts rather than a summary.
    Usage {
        /// Token counts as reported by the provider.
        usage: Usage,
    },

    /// Generation finished cleanly. Terminal.
    MessageStop {
        /// Why generation stopped, e.g. `"stop"`, `"length"`, or `"tool_use"`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason: Option<String>,
    },

    /// An assembled conversation turn, ready to be stored as history.
    ///
    /// Never emitted by
    /// [`stream_wire_events`](crate::RequestBuilder::stream_wire_events); it is
    /// the wire form of [`StreamEvent::TurnComplete`], so events from
    /// [`generate_stream_events`](crate::RequestBuilder::generate_stream_events)
    /// can be forwarded without loss.
    TurnComplete {
        /// The complete turn.
        turn: ConversationTurn,
    },

    /// Generation failed. Terminal.
    ///
    /// Receiving this means the provider or the SDK rejected the request
    /// *after* the stream opened. A client that never receives a terminal event
    /// lost its connection instead — a different failure with a different
    /// remedy.
    Error {
        /// What went wrong.
        error: WireError,
    },
}

impl WireStreamEvent {
    /// Build the opening event for a stream, stamped with
    /// [`WIRE_PROTOCOL_VERSION`].
    pub fn message_start(model: impl Into<String>, provider: ProviderKind) -> Self {
        Self::MessageStart {
            protocol_version: WIRE_PROTOCOL_VERSION,
            model: model.into(),
            provider,
        }
    }

    /// Build a terminal error event from a crate [`enum@Error`].
    pub fn error(error: &Error) -> Self {
        Self::Error {
            error: WireError::from(error),
        }
    }

    /// The `"type"` discriminant this event serializes with.
    ///
    /// Useful as the SSE `event:` name, or for logging and metrics labels
    /// without serializing the whole payload.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::TextDelta { .. } => "text_delta",
            Self::ToolCallStart { .. } => "tool_call_start",
            Self::ToolCallDelta { .. } => "tool_call_delta",
            Self::ToolCallEnd { .. } => "tool_call_end",
            Self::ToolResult { .. } => "tool_result",
            Self::Usage { .. } => "usage",
            Self::MessageStop { .. } => "message_stop",
            Self::TurnComplete { .. } => "turn_complete",
            Self::Error { .. } => "error",
        }
    }

    /// Whether this event ends the stream: [`MessageStop`] or [`Error`].
    ///
    /// [`MessageStop`]: WireStreamEvent::MessageStop
    /// [`Error`]: WireStreamEvent::Error
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::MessageStop { .. } | Self::Error { .. })
    }
}

impl From<StreamEvent> for WireStreamEvent {
    /// Forward a high-level [`StreamEvent`] onto the wire without loss.
    ///
    /// Every [`StreamEvent`] has an exact counterpart here, and
    /// [`TryFrom<WireStreamEvent>`](WireStreamEvent) converts it back to the
    /// same value.
    fn from(event: StreamEvent) -> Self {
        match event {
            StreamEvent::TextDelta { text } => Self::TextDelta { text },
            StreamEvent::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCallEnd {
                id,
                name,
                arguments,
            },
            StreamEvent::ToolResult { id, result } => Self::ToolResult { id, result },
            StreamEvent::TurnComplete { turn } => Self::TurnComplete { turn },
        }
    }
}

/// A [`WireStreamEvent`] with no [`StreamEvent`] counterpart.
///
/// The wire enum is a superset: it also models stream framing (`message_start`,
/// `message_stop`), token usage, partial tool-call progress, and errors, none of
/// which [`StreamEvent`] represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrepresentableWireEvent {
    /// The `"type"` tag of the event that could not be converted.
    pub tag: &'static str,
}

impl std::fmt::Display for UnrepresentableWireEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wire event '{}' has no StreamEvent equivalent", self.tag)
    }
}

impl std::error::Error for UnrepresentableWireEvent {}

impl TryFrom<WireStreamEvent> for StreamEvent {
    type Error = UnrepresentableWireEvent;

    fn try_from(event: WireStreamEvent) -> std::result::Result<Self, UnrepresentableWireEvent> {
        let tag = event.tag();
        match event {
            WireStreamEvent::TextDelta { text } => Ok(Self::TextDelta { text }),
            WireStreamEvent::ToolCallEnd {
                id,
                name,
                arguments,
            } => Ok(Self::ToolCall {
                id,
                name,
                arguments,
            }),
            WireStreamEvent::ToolResult { id, result } => Ok(Self::ToolResult { id, result }),
            WireStreamEvent::TurnComplete { turn } => Ok(Self::TurnComplete { turn }),
            _ => Err(UnrepresentableWireEvent { tag }),
        }
    }
}

// ── Errors on the wire ─────────────────────────────────────────────────────

/// A serializable projection of the crate's [`enum@Error`].
///
/// [`enum@Error`] cannot cross a wire: it wraps `reqwest::Error` and
/// `serde_json::Error`, neither of which is `Serialize`. `WireError` keeps the
/// parts a remote client can act on — the category, a human-readable message,
/// the provider, and whether a retry might help — and drops the rest.
///
/// The conversion is deliberately one-way. Rebuilding a `reqwest::Error` from
/// JSON is not possible, so a client matches on [`kind`](WireError::kind)
/// rather than on the original error variant.
///
/// # Examples
///
/// ```
/// use rai_sdk::{Error, ProviderKind};
/// use rai_sdk::wire::{WireError, WireErrorKind};
///
/// let error = Error::RateLimit {
///     provider: ProviderKind::OpenAI,
///     message: "slow down".to_string(),
/// };
///
/// let wire = WireError::from(&error);
/// assert_eq!(wire.kind, WireErrorKind::RateLimit);
/// assert_eq!(wire.provider, Some(ProviderKind::OpenAI));
/// assert!(wire.retryable);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireError {
    /// Error category, mirroring [`Error::kind_str`].
    pub kind: WireErrorKind,

    /// Human-readable description, from the original error's `Display`.
    pub message: String,

    /// Provider the failure came from, when the error names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderKind>,

    /// Whether retrying the request might succeed, from [`Error::is_retryable`].
    pub retryable: bool,
}

impl WireError {
    /// Build an error of `kind` with `message`, with no provider and not
    /// retryable.
    ///
    /// Use this for failures a proxy raises itself — a truncated upstream, a
    /// rejected protocol version — that have no [`enum@Error`] behind them.
    pub fn new(kind: WireErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider: None,
            retryable: false,
        }
    }
}

impl From<&Error> for WireError {
    fn from(error: &Error) -> Self {
        Self {
            kind: WireErrorKind::from(error.kind_str()),
            message: error.to_string(),
            provider: error.provider(),
            retryable: error.is_retryable(),
        }
    }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for WireError {}

/// The category of a [`WireError`], mirroring [`Error::kind_str`].
///
/// Serializes as the same snake_case string [`Error::kind_str`] returns, so the
/// two stay interchangeable in logs and metrics. A value this build does not
/// know deserializes into [`Other`](WireErrorKind::Other) rather than failing,
/// which is what lets an older client keep parsing streams from a newer server.
///
/// This enum is `#[non_exhaustive]`: match with a catch-all arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WireErrorKind {
    /// Authentication failed.
    Auth,
    /// The provider request failed for a reason with no more specific variant.
    Request,
    /// The provider throttled the request.
    RateLimit,
    /// The request itself was invalid.
    InvalidRequest,
    /// The requested model is unavailable.
    ModelNotAvailable,
    /// The provider has no credentials configured.
    ProviderNotConfigured,
    /// The provider's Cargo feature is disabled in the server build.
    ProviderNotEnabled,
    /// The provider filtered the content.
    ContentFiltered,
    /// The SDK was misconfigured.
    Config,
    /// A value failed to serialize or deserialize.
    Serialization,
    /// The HTTP transport failed.
    Http,
    /// The stream itself failed partway through.
    Stream,
    /// The request timed out.
    Timeout,
    /// The provider does not support tool calling.
    ToolProviderUnsupported,
    /// Tool arguments failed schema validation.
    ToolArguments,
    /// A requested tool is not registered.
    ToolNotFound,
    /// The tool loop exceeded its configured round limit.
    ToolLoopLimitExceeded,
    /// Structured output failed validation.
    StructuredOutput,
    /// A category this build does not know about.
    ///
    /// Produced when deserializing a stream from a newer `rai-sdk`. Serializes
    /// back to the original string, so a proxy can forward it untouched.
    #[serde(untagged)]
    Other(String),
}

impl WireErrorKind {
    /// The snake_case name this kind serializes as.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Auth => "auth",
            Self::Request => "request",
            Self::RateLimit => "rate_limit",
            Self::InvalidRequest => "invalid_request",
            Self::ModelNotAvailable => "model_not_available",
            Self::ProviderNotConfigured => "provider_not_configured",
            Self::ProviderNotEnabled => "provider_not_enabled",
            Self::ContentFiltered => "content_filtered",
            Self::Config => "config",
            Self::Serialization => "serialization",
            Self::Http => "http",
            Self::Stream => "stream",
            Self::Timeout => "timeout",
            Self::ToolProviderUnsupported => "tool_provider_unsupported",
            Self::ToolArguments => "tool_arguments",
            Self::ToolNotFound => "tool_not_found",
            Self::ToolLoopLimitExceeded => "tool_loop_limit_exceeded",
            Self::StructuredOutput => "structured_output",
            Self::Other(kind) => kind,
        }
    }
}

impl From<&str> for WireErrorKind {
    fn from(kind: &str) -> Self {
        match kind {
            "auth" => Self::Auth,
            "request" => Self::Request,
            "rate_limit" => Self::RateLimit,
            "invalid_request" => Self::InvalidRequest,
            "model_not_available" => Self::ModelNotAvailable,
            "provider_not_configured" => Self::ProviderNotConfigured,
            "provider_not_enabled" => Self::ProviderNotEnabled,
            "content_filtered" => Self::ContentFiltered,
            "config" => Self::Config,
            "serialization" => Self::Serialization,
            "http" => Self::Http,
            "stream" => Self::Stream,
            "timeout" => Self::Timeout,
            "tool_provider_unsupported" => Self::ToolProviderUnsupported,
            "tool_arguments" => Self::ToolArguments,
            "tool_not_found" => Self::ToolNotFound,
            "tool_loop_limit_exceeded" => Self::ToolLoopLimitExceeded,
            "structured_output" => Self::StructuredOutput,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for WireErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Client-side reassembly ─────────────────────────────────────────────────

/// A tool call whose arguments are still arriving.
#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PendingToolCall {
    fn finish(self) -> ToolCall {
        ToolCall {
            id: self.id,
            name: self.name,
            arguments: serde_json::from_str(&self.arguments).unwrap_or(serde_json::Value::Null),
        }
    }
}

/// Rebuilds one [`Response`] from a stream of [`WireStreamEvent`]s.
///
/// This is the receiving half of the proxy pattern: the client-side equivalent
/// of [`RequestBuilder::stream_accumulated`](crate::RequestBuilder::stream_accumulated),
/// operating on events that arrived over a network rather than on a live
/// provider connection.
///
/// Unlike `stream_accumulated`, this *does* reassemble tool calls: fragments
/// from [`ToolCallStart`](WireStreamEvent::ToolCallStart) /
/// [`ToolCallDelta`](WireStreamEvent::ToolCallDelta) /
/// [`ToolCallEnd`](WireStreamEvent::ToolCallEnd) are concatenated and attached
/// to the assistant message.
///
/// # Truncation is an error
///
/// [`finish`](StreamAccumulator::finish) fails unless the stream was
/// well-formed: it must have opened with
/// [`MessageStart`](WireStreamEvent::MessageStart) and ended with a terminal
/// event. A stream that just stops — the client's socket died mid-generation —
/// yields [`WireErrorKind::Stream`], which is what distinguishes "the network
/// died" from "the provider refused" (the latter arrives as
/// [`WireStreamEvent::Error`] and is returned as its own [`WireError`]).
///
/// # Examples
///
/// ```
/// use rai_sdk::ProviderKind;
/// use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut accumulator = StreamAccumulator::new();
/// accumulator.push(WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI))?;
/// accumulator.push(WireStreamEvent::TextDelta { text: "done".into() })?;
/// accumulator.push(WireStreamEvent::MessageStop { finish_reason: Some("stop".into()) })?;
///
/// let response = accumulator.finish()?;
/// assert_eq!(response.text(), "done");
/// assert_eq!(response.finish_reason.as_deref(), Some("stop"));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct StreamAccumulator {
    protocol_version: Option<u32>,
    model: Option<String>,
    provider: Option<ProviderKind>,
    text: String,
    tool_calls: Vec<ToolCall>,
    pending_tool_call: Option<PendingToolCall>,
    tool_results: Vec<Message>,
    usage: Option<Usage>,
    finish_reason: Option<String>,
    error: Option<WireError>,
    stopped: bool,
}

impl StreamAccumulator {
    /// Start with an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// The protocol version the sender advertised, once
    /// [`MessageStart`](WireStreamEvent::MessageStart) has been seen.
    pub fn protocol_version(&self) -> Option<u32> {
        self.protocol_version
    }

    /// The assistant text accumulated so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Token usage, once a [`Usage`](WireStreamEvent::Usage) event has arrived.
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Whether a terminal event has been seen.
    ///
    /// A stream whose events are exhausted while this is `false` was truncated.
    pub fn is_complete(&self) -> bool {
        self.stopped || self.error.is_some()
    }

    /// Absorb one event.
    ///
    /// # Errors
    ///
    /// Returns the carried [`WireError`] when `event` is
    /// [`WireStreamEvent::Error`], so a caller driving this in a loop can stop
    /// with `?`. The error is remembered, so a later
    /// [`finish`](StreamAccumulator::finish) reports it too.
    pub fn push(&mut self, event: WireStreamEvent) -> std::result::Result<(), WireError> {
        match event {
            WireStreamEvent::MessageStart {
                protocol_version,
                model,
                provider,
            } => {
                self.protocol_version = Some(protocol_version);
                self.model = Some(model);
                self.provider = Some(provider);
            }

            WireStreamEvent::TextDelta { text } => self.text.push_str(&text),

            WireStreamEvent::ToolCallStart { id, name } => {
                self.flush_pending_tool_call();
                self.pending_tool_call = Some(PendingToolCall {
                    id,
                    name,
                    arguments: String::new(),
                });
            }

            WireStreamEvent::ToolCallDelta { id, arguments } => match &mut self.pending_tool_call {
                Some(pending) if pending.id == id => pending.arguments.push_str(&arguments),
                _ => {
                    self.flush_pending_tool_call();
                    self.pending_tool_call = Some(PendingToolCall {
                        id,
                        name: String::new(),
                        arguments,
                    });
                }
            },

            WireStreamEvent::ToolCallEnd {
                id,
                name,
                arguments,
            } => {
                // The assembled form supersedes whatever was buffered for the
                // same call, so drop the partial rather than emitting both.
                if self
                    .pending_tool_call
                    .as_ref()
                    .is_some_and(|pending| pending.id == id)
                {
                    self.pending_tool_call = None;
                } else {
                    self.flush_pending_tool_call();
                }
                self.tool_calls.push(
                    PendingToolCall {
                        id,
                        name,
                        arguments,
                    }
                    .finish(),
                );
            }

            WireStreamEvent::ToolResult { id, result } => {
                self.tool_results.push(Message::tool(result, id));
            }

            WireStreamEvent::Usage { usage } => self.usage = Some(usage),

            WireStreamEvent::MessageStop { finish_reason } => {
                self.flush_pending_tool_call();
                if finish_reason.is_some() {
                    self.finish_reason = finish_reason;
                }
                self.stopped = true;
            }

            WireStreamEvent::TurnComplete { turn } => {
                self.flush_pending_tool_call();
                if self.text.is_empty() {
                    self.text = turn.assistant_message.text_content();
                }
                if self.tool_calls.is_empty() {
                    self.tool_calls = turn.assistant_message.tool_calls.clone();
                }
                self.tool_results.extend(turn.tool_results);
            }

            WireStreamEvent::Error { error } => {
                self.error = Some(error.clone());
                return Err(error);
            }

            // `WireStreamEvent` is `#[non_exhaustive]` in spirit for consumers,
            // but this match is inside the defining crate, so every arm above
            // is checked exhaustively and a new variant is a compile error here.
            #[allow(unreachable_patterns)]
            _ => {}
        }

        Ok(())
    }

    fn flush_pending_tool_call(&mut self) {
        if let Some(pending) = self.pending_tool_call.take() {
            self.tool_calls.push(pending.finish());
        }
    }

    /// Assemble everything pushed so far into one [`Response`].
    ///
    /// # Errors
    ///
    /// - The [`WireError`] from a [`WireStreamEvent::Error`], if one arrived.
    /// - [`WireErrorKind::Stream`] if no
    ///   [`MessageStart`](WireStreamEvent::MessageStart) was seen, since the
    ///   model and provider a [`Response`] requires are unknown.
    /// - [`WireErrorKind::Stream`] if no terminal event was seen, meaning the
    ///   stream was cut short rather than finished.
    pub fn finish(mut self) -> std::result::Result<Response, WireError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        self.flush_pending_tool_call();

        let (Some(model), Some(provider)) = (self.model, self.provider) else {
            return Err(WireError::new(
                WireErrorKind::Stream,
                "stream ended without a message_start event, so the model and provider are unknown",
            ));
        };

        if !self.stopped {
            return Err(WireError::new(
                WireErrorKind::Stream,
                "stream ended before its terminal event (message_stop or error); \
                 the connection was most likely dropped mid-generation",
            ));
        }

        let mut assistant = Message::assistant(self.text);
        assistant.tool_calls = self.tool_calls;

        let mut messages = vec![assistant];
        messages.extend(self.tool_results);

        Ok(Response {
            messages,
            usage: self.usage,
            model,
            provider,
            finish_reason: self.finish_reason,
        })
    }

    /// Drain a stream of wire events into one [`Response`].
    ///
    /// The convenience form of `new()` + `push()` in a loop + `finish()`, for
    /// the common client-side case where the whole stream is consumed at once.
    ///
    /// # Errors
    ///
    /// Same as [`push`](StreamAccumulator::push) and
    /// [`finish`](StreamAccumulator::finish).
    ///
    /// # Examples
    ///
    /// ```
    /// use rai_sdk::ProviderKind;
    /// use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let events = futures::stream::iter(vec![
    ///     WireStreamEvent::message_start("claude-sonnet-4-6", ProviderKind::Anthropic),
    ///     WireStreamEvent::TextDelta { text: "hi".into() },
    ///     WireStreamEvent::MessageStop { finish_reason: Some("stop".into()) },
    /// ]);
    ///
    /// let response = StreamAccumulator::accumulate(events).await?;
    /// assert_eq!(response.text(), "hi");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn accumulate<S>(stream: S) -> std::result::Result<Response, WireError>
    where
        S: Stream<Item = WireStreamEvent>,
    {
        let mut stream = std::pin::pin!(stream);
        let mut accumulator = Self::new();

        while let Some(event) = stream.next().await {
            accumulator.push(event)?;
        }

        accumulator.finish()
    }
}

#[cfg(test)]
mod tests {
    //! Wire-format tests deliberately live *inside* the crate.
    //!
    //! [`WireStreamEvent`] is `#[non_exhaustive]`, so an integration test in
    //! `tests/` — a separate crate — would be forced to write a `_ => {}` arm
    //! and adding a variant would not break it. In here the match is checked
    //! exhaustively, so a new variant fails to compile until it has both a
    //! round-trip case and a committed JSON fixture.

    use super::*;
    use crate::message::{ContentBlock, Message};

    /// Directory holding one committed JSON fixture per variant.
    const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/wire");

    /// Set this to rewrite the fixtures instead of asserting against them.
    const UPDATE_ENV: &str = "RAI_SDK_UPDATE_WIRE_FIXTURES";

    fn sample_usage() -> Usage {
        Usage {
            prompt_tokens: Some(1_024),
            completion_tokens: Some(256),
            total_tokens: Some(1_280),
        }
    }

    fn sample_turn() -> ConversationTurn {
        let mut assistant = Message::assistant("Sunny.");
        assistant.tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "city": "Paris" }),
        }];

        ConversationTurn {
            user_message: Message::user("Weather in Paris?"),
            assistant_message: assistant,
            tool_results: vec![Message::tool("{\"c\":21}", "call_1")],
        }
    }

    /// One representative value per variant.
    ///
    /// The exhaustive `match` in [`fixture_name`] is what forces this list to
    /// grow with the enum.
    fn every_variant() -> Vec<WireStreamEvent> {
        vec![
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::TextDelta {
                text: "Hello, world.".to_string(),
            },
            WireStreamEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            },
            WireStreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                arguments: "{\"city\":".to_string(),
            },
            WireStreamEvent::ToolCallEnd {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
            WireStreamEvent::ToolResult {
                id: "call_1".to_string(),
                result: "{\"celsius\":21}".to_string(),
            },
            WireStreamEvent::Usage {
                usage: sample_usage(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("stop".to_string()),
            },
            WireStreamEvent::TurnComplete {
                turn: sample_turn(),
            },
            WireStreamEvent::error(&Error::RateLimit {
                provider: ProviderKind::Anthropic,
                message: "too many requests".to_string(),
            }),
        ]
    }

    /// Fixture file stem for an event.
    ///
    /// Exhaustive on purpose: adding a variant to [`WireStreamEvent`] breaks
    /// this match, which is the signal to add it to [`every_variant`] and
    /// commit a fixture for it.
    fn fixture_name(event: &WireStreamEvent) -> &'static str {
        match event {
            WireStreamEvent::MessageStart { .. } => "message_start",
            WireStreamEvent::TextDelta { .. } => "text_delta",
            WireStreamEvent::ToolCallStart { .. } => "tool_call_start",
            WireStreamEvent::ToolCallDelta { .. } => "tool_call_delta",
            WireStreamEvent::ToolCallEnd { .. } => "tool_call_end",
            WireStreamEvent::ToolResult { .. } => "tool_result",
            WireStreamEvent::Usage { .. } => "usage",
            WireStreamEvent::MessageStop { .. } => "message_stop",
            WireStreamEvent::TurnComplete { .. } => "turn_complete",
            WireStreamEvent::Error { .. } => "error",
        }
    }

    #[test]
    fn every_variant_is_covered_exactly_once() {
        let mut names: Vec<&str> = every_variant().iter().map(fixture_name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            total,
            "every_variant() must contain each variant exactly once"
        );
    }

    #[test]
    fn round_trip_preserves_every_variant() {
        for event in every_variant() {
            let json = serde_json::to_string(&event).expect("event should serialize");
            let parsed: WireStreamEvent =
                serde_json::from_str(&json).expect("event should deserialize");
            assert_eq!(
                parsed,
                event,
                "round trip changed the '{}' event: {json}",
                fixture_name(&event)
            );
        }
    }

    #[test]
    fn the_tag_matches_the_serialized_discriminant() {
        for event in every_variant() {
            let json = serde_json::to_value(&event).expect("event should serialize");
            assert_eq!(
                json.get("type").and_then(serde_json::Value::as_str),
                Some(event.tag()),
                "tag() disagrees with the serialized discriminant: {json}"
            );
            assert_eq!(event.tag(), fixture_name(&event));
        }
    }

    /// Pins the exact JSON each variant produces.
    ///
    /// The fixtures are the compatibility contract with deployed clients, so an
    /// accidental rename shows up here as a diff rather than in production.
    #[test]
    fn wire_format_matches_the_committed_fixtures() {
        let updating = std::env::var_os(UPDATE_ENV).is_some();

        for event in every_variant() {
            let name = fixture_name(&event);
            let path = std::path::Path::new(FIXTURE_DIR).join(format!("{name}.json"));
            let actual = serde_json::to_value(&event).expect("event should serialize");

            if updating {
                std::fs::create_dir_all(FIXTURE_DIR).expect("create the fixture directory");
                let pretty =
                    serde_json::to_string_pretty(&actual).expect("fixture should serialize");
                std::fs::write(&path, format!("{pretty}\n")).expect("write the fixture");
                continue;
            }

            let raw = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!(
                    "missing wire-format fixture {}: {error}\n\
                     If this variant is new, regenerate the fixtures with:\n    \
                     {UPDATE_ENV}=1 cargo test --all-features wire_format_matches_the_committed_fixtures",
                    path.display()
                )
            });
            let expected: serde_json::Value =
                serde_json::from_str(&raw).expect("fixture should be valid JSON");

            assert_eq!(
                actual,
                expected,
                "the '{name}' event no longer matches {}.\n\
                 Wire-format changes break clients built against an older rai-sdk. \
                 If this change is intentional, note it in CHANGELOG.md and regenerate with:\n    \
                 {UPDATE_ENV}=1 cargo test --all-features wire_format_matches_the_committed_fixtures",
                path.display()
            );

            // The fixture must also parse back, so a hand-edited file cannot
            // drift into a shape the crate can no longer read.
            let parsed: WireStreamEvent =
                serde_json::from_value(expected).expect("fixture should deserialize");
            assert_eq!(parsed, event);
        }
    }

    #[test]
    fn stream_event_conversion_round_trips() {
        let events = vec![
            StreamEvent::TextDelta {
                text: "hi".to_string(),
            },
            StreamEvent::ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
            StreamEvent::ToolResult {
                id: "call_1".to_string(),
                result: "{\"celsius\":21}".to_string(),
            },
            StreamEvent::TurnComplete {
                turn: sample_turn(),
            },
        ];

        for event in events {
            let wire = WireStreamEvent::from(event.clone());
            let back = StreamEvent::try_from(wire).expect("wire event should convert back");
            assert_eq!(back, event);
        }
    }

    #[test]
    fn framing_events_have_no_stream_event_equivalent() {
        let wire = WireStreamEvent::MessageStop {
            finish_reason: None,
        };
        let error = StreamEvent::try_from(wire).expect_err("message_stop is wire-only");
        assert_eq!(error.tag, "message_stop");
        assert!(error.to_string().contains("message_stop"));
    }

    #[test]
    fn message_start_defaults_the_protocol_version_when_absent() {
        let parsed: WireStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","model":"gpt-4o-mini","provider":"openai"}"#,
        )
        .expect("legacy payload should still parse");

        assert_eq!(
            parsed,
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI)
        );
    }

    #[test]
    fn is_terminal_marks_exactly_the_stream_enders() {
        for event in every_variant() {
            let expected = matches!(fixture_name(&event), "message_stop" | "error");
            assert_eq!(event.is_terminal(), expected, "{}", event.tag());
        }
    }

    /// Every [`enum@Error`] variant must map onto a named [`WireErrorKind`].
    ///
    /// Exhaustive so a new `Error` variant cannot silently start serializing as
    /// [`WireErrorKind::Other`].
    #[test]
    fn every_error_variant_maps_to_a_named_kind() {
        let provider = ProviderKind::OpenAI;
        let errors = vec![
            Error::Auth {
                provider,
                message: "bad key".into(),
            },
            Error::Request {
                provider,
                message: "boom".into(),
            },
            Error::RateLimit {
                provider,
                message: "slow down".into(),
            },
            Error::InvalidRequest("nope".into()),
            Error::ModelNotAvailable {
                provider,
                model: "gpt-9".into(),
            },
            Error::ProviderNotConfigured(provider),
            Error::ProviderNotEnabled(provider),
            Error::ContentFiltered {
                provider,
                reason: "policy".into(),
            },
            Error::Config("missing".into()),
            Error::Serialization(
                serde_json::from_str::<serde_json::Value>("{").expect_err("invalid JSON"),
            ),
            Error::Stream("truncated".into()),
            Error::Timeout { provider },
            Error::ToolProviderUnsupported { provider },
            Error::ToolArguments {
                name: "echo".into(),
                message: "bad".into(),
                issues: Vec::new(),
            },
            Error::ToolNotFound {
                name: "echo".into(),
            },
            Error::ToolLoopLimitExceeded { max_rounds: 8 },
            Error::StructuredOutput {
                provider,
                model: "gpt-4o-mini".into(),
                message: "invalid".into(),
            },
        ];

        for error in &errors {
            let wire = WireError::from(error);
            assert!(
                !matches!(wire.kind, WireErrorKind::Other(_)),
                "{} has no named WireErrorKind",
                error.kind_str()
            );
            assert_eq!(wire.kind.as_str(), error.kind_str());
            assert_eq!(wire.message, error.to_string());
            assert_eq!(wire.provider, error.provider());
            assert_eq!(wire.retryable, error.is_retryable());
        }

        // `Error::Http` is the one variant that cannot be constructed outside
        // `reqwest`, so its mapping is asserted on the kind string directly.
        assert_eq!(WireErrorKind::from("http"), WireErrorKind::Http);
    }

    #[test]
    fn unknown_error_kinds_survive_a_round_trip() {
        let json = r#"{"kind":"quantum_flux","message":"from the future","retryable":true}"#;
        let parsed: WireError = serde_json::from_str(json).expect("unknown kind should parse");

        assert_eq!(parsed.kind, WireErrorKind::Other("quantum_flux".into()));
        assert_eq!(parsed.kind.as_str(), "quantum_flux");
        assert_eq!(parsed.provider, None);

        let reserialized = serde_json::to_value(&parsed).expect("should serialize");
        assert_eq!(reserialized["kind"], "quantum_flux");
    }

    #[test]
    fn accumulator_rebuilds_text_usage_and_finish_reason() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::TextDelta {
                text: "Hello, ".into(),
            },
            WireStreamEvent::TextDelta {
                text: "world.".into(),
            },
            WireStreamEvent::Usage {
                usage: sample_usage(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("stop".into()),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }

        assert_eq!(accumulator.protocol_version(), Some(WIRE_PROTOCOL_VERSION));
        assert!(accumulator.is_complete());

        let response = accumulator.finish().expect("stream was well formed");
        assert_eq!(response.text(), "Hello, world.");
        assert_eq!(response.model, "gpt-4o-mini");
        assert_eq!(response.provider, ProviderKind::OpenAI);
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
        assert_eq!(response.usage, Some(sample_usage()));
    }

    #[test]
    fn accumulator_assembles_tool_calls_from_fragments() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::ToolCallStart {
                id: "call_1".into(),
                name: "get_weather".into(),
            },
            WireStreamEvent::ToolCallDelta {
                id: "call_1".into(),
                arguments: "{\"city\":".into(),
            },
            WireStreamEvent::ToolCallDelta {
                id: "call_1".into(),
                arguments: "\"Paris\"}".into(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("tool_use".into()),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }

        let response = accumulator.finish().expect("stream was well formed");
        let tool_calls = &response.messages[0].tool_calls;
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].arguments["city"], "Paris");
    }

    #[test]
    fn a_tool_call_end_supersedes_its_own_fragments() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::ToolCallStart {
                id: "call_1".into(),
                name: "get_weather".into(),
            },
            WireStreamEvent::ToolCallDelta {
                id: "call_1".into(),
                arguments: "{\"city\":".into(),
            },
            WireStreamEvent::ToolCallEnd {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("tool_use".into()),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }

        let response = accumulator.finish().expect("stream was well formed");
        assert_eq!(response.messages[0].tool_calls.len(), 1);
    }

    #[test]
    fn accumulator_keeps_tool_results_as_messages() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::ToolResult {
                id: "call_1".into(),
                result: "{\"celsius\":21}".into(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("stop".into()),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }

        let response = accumulator.finish().expect("stream was well formed");
        assert_eq!(response.messages.len(), 2);
        assert_eq!(response.messages[1].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn an_error_event_stops_the_accumulator() {
        let mut accumulator = StreamAccumulator::new();
        accumulator
            .push(WireStreamEvent::message_start(
                "gpt-4o-mini",
                ProviderKind::OpenAI,
            ))
            .expect("event should be absorbed");

        let error = accumulator
            .push(WireStreamEvent::error(&Error::ContentFiltered {
                provider: ProviderKind::OpenAI,
                reason: "policy".into(),
            }))
            .expect_err("an error event should surface as an error");

        assert_eq!(error.kind, WireErrorKind::ContentFiltered);
        assert!(accumulator.is_complete());
        assert_eq!(
            accumulator
                .finish()
                .expect_err("finish reports it too")
                .kind,
            WireErrorKind::ContentFiltered
        );
    }

    #[test]
    fn a_truncated_stream_is_distinguishable_from_a_provider_error() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::TextDelta {
                text: "half a sen".into(),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }
        assert!(!accumulator.is_complete());

        let error = accumulator
            .finish()
            .expect_err("a truncated stream is an error");
        assert_eq!(error.kind, WireErrorKind::Stream);
        assert!(
            error.message.contains("terminal event"),
            "unhelpful truncation message: {}",
            error.message
        );
    }

    #[test]
    fn a_stream_without_message_start_is_rejected() {
        let mut accumulator = StreamAccumulator::new();
        accumulator
            .push(WireStreamEvent::MessageStop {
                finish_reason: Some("stop".into()),
            })
            .expect("event should be absorbed");

        let error = accumulator
            .finish()
            .expect_err("model and provider are unknown");
        assert_eq!(error.kind, WireErrorKind::Stream);
        assert!(error.message.contains("message_start"));
    }

    #[test]
    fn turn_complete_backfills_an_otherwise_empty_response() {
        let mut accumulator = StreamAccumulator::new();
        for event in [
            WireStreamEvent::message_start("claude-sonnet-4-6", ProviderKind::Anthropic),
            WireStreamEvent::TurnComplete {
                turn: sample_turn(),
            },
            WireStreamEvent::MessageStop {
                finish_reason: Some("tool_use".into()),
            },
        ] {
            accumulator.push(event).expect("event should be absorbed");
        }

        let response = accumulator.finish().expect("stream was well formed");
        assert_eq!(response.text(), "Sunny.");
        assert_eq!(response.messages[0].tool_calls.len(), 1);
    }

    #[tokio::test]
    async fn accumulate_drains_a_whole_stream() {
        let response = StreamAccumulator::accumulate(futures::stream::iter(vec![
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
            WireStreamEvent::TextDelta { text: "ok".into() },
            WireStreamEvent::MessageStop {
                finish_reason: Some("stop".into()),
            },
        ]))
        .await
        .expect("stream was well formed");

        assert_eq!(response.text(), "ok");
    }

    #[test]
    fn multimodal_turns_survive_the_wire() {
        let turn = ConversationTurn {
            user_message: Message::user_multimodal(vec![
                ContentBlock::text("What is this?"),
                ContentBlock::image_url("https://example.com/cat.png"),
            ]),
            assistant_message: Message::assistant("A cat."),
            tool_results: Vec::new(),
        };

        let event = WireStreamEvent::TurnComplete { turn };
        let json = serde_json::to_string(&event).expect("event should serialize");
        let parsed: WireStreamEvent = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(parsed, event);
    }
}
