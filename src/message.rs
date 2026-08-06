//! Prompts, messages, and responses.
//!
//! A request is described by a [`Prompt`], which is just an ordered list of
//! [`Message`]s. Messages are text-only by default and become multimodal when
//! they carry [`ContentBlock`]s instead. Providers answer with a [`Response`]
//! (or a stream of [`StreamChunk`]/[`StreamEvent`] values), and structured
//! requests answer with a [`StructuredOutput<T>`](StructuredOutput) that pairs
//! the parsed value with the raw response.
//!
//! These types are provider-agnostic: each provider translates them to and
//! from its own wire format.
//!
//! # Examples
//!
//! ```no_run
//! use rai_sdk::{ContentBlock, Message, Prompt};
//!
//! // Anything that converts into a `Prompt` can be passed to a request.
//! let simple: Prompt = "Summarize this file.".into();
//!
//! // Multi-turn conversations are built explicitly.
//! let conversation = Prompt::new(vec![
//!     Message::system("You are terse."),
//!     Message::user("Who wrote Dune?"),
//!     Message::assistant("Frank Herbert."),
//!     Message::user("When?"),
//! ]);
//! assert_eq!(conversation.system_message(), Some("You are terse."));
//!
//! // Multimodal messages mix text with images.
//! let vision = Prompt::single(Message::user_multimodal(vec![
//!     ContentBlock::text("What is in this picture?"),
//!     ContentBlock::image_url("https://example.com/cat.png"),
//! ]));
//! assert!(vision.is_multimodal());
//! # let _ = simple;
//! ```

use serde::{Deserialize, Serialize};

use crate::error::ProviderKind;

/// Role of a message participant in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions that steer the whole conversation.
    System,
    /// Input from the end user.
    User,
    /// Output produced by the model.
    Assistant,
    /// The result of executing a tool the model asked for.
    Tool,
}

impl Role {
    /// Return the canonical lowercase name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A tool invocation emitted by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call identifier.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments as a JSON value.
    pub arguments: serde_json::Value,
}

/// Content block for multimodal messages.
///
/// Only [`ContentBlock::Text`] and [`ContentBlock::Image`] are currently
/// translated by the bundled providers; audio, video, and file blocks are
/// modelled here but not yet sent on the wire.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::ContentBlock;
///
/// let caption = ContentBlock::text("Describe this diagram.");
/// let remote = ContentBlock::image_url("https://example.com/diagram.png");
/// let inline = ContentBlock::image_base64("image/png", "iVBORw0KGgo=");
/// # let _ = (caption, remote, inline);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },

    /// An image, by URL or inline base64 data.
    #[serde(rename = "image")]
    Image {
        /// Where the image data comes from.
        source: ImageSource,
    },

    /// An audio clip. Not yet supported by the bundled providers.
    #[serde(rename = "audio")]
    Audio {
        /// Where the audio data comes from.
        source: FileSource,
    },

    /// A video clip. Not yet supported by the bundled providers.
    #[serde(rename = "video")]
    Video {
        /// Where the video data comes from.
        source: FileSource,
    },

    /// An arbitrary file. Not yet supported by the bundled providers.
    #[serde(rename = "file")]
    File {
        /// Where the file data comes from.
        source: FileSource,
    },
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image content block from a URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Url { url: url.into() },
        }
    }

    /// Create an image content block from base64 data.
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    /// Create an audio content block from a URL.
    pub fn audio_url(url: impl Into<String>) -> Self {
        Self::Audio {
            source: FileSource::Url { url: url.into() },
        }
    }

    /// Create an audio content block from base64 data.
    pub fn audio_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Audio {
            source: FileSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    /// Create a video content block from a URL.
    pub fn video_url(url: impl Into<String>) -> Self {
        Self::Video {
            source: FileSource::Url { url: url.into() },
        }
    }

    /// Create a video content block from base64 data.
    pub fn video_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Video {
            source: FileSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }

    /// Create a file content block from a URL.
    pub fn file_url(url: impl Into<String>) -> Self {
        Self::File {
            source: FileSource::Url { url: url.into() },
        }
    }

    /// Create a file content block from base64 data.
    pub fn file_base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::File {
            source: FileSource::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            },
        }
    }
}

/// Image source for multimodal content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    /// A publicly reachable image URL the provider will fetch.
    #[serde(rename = "url")]
    Url {
        /// URL of the image.
        url: String,
    },

    /// Image bytes embedded in the request.
    #[serde(rename = "base64")]
    Base64 {
        /// MIME type of the data, e.g. `image/png`.
        media_type: String,
        /// Base64-encoded image bytes, without a data-URL prefix.
        data: String,
    },
}

/// File source for multimodal content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FileSource {
    /// A publicly reachable URL the provider will fetch.
    #[serde(rename = "url")]
    Url {
        /// URL of the file.
        url: String,
    },

    /// File bytes embedded in the request.
    #[serde(rename = "base64")]
    Base64 {
        /// MIME type of the data, e.g. `audio/mpeg`.
        media_type: String,
        /// Base64-encoded file bytes, without a data-URL prefix.
        data: String,
    },
}

/// A single message in a conversation.
///
/// Supports text-only and multimodal (text + images) content.
///
/// Use the constructors ([`Message::system`], [`Message::user`],
/// [`Message::assistant`], [`Message::tool`], [`Message::user_multimodal`], …)
/// rather than building the struct by hand; they keep the role and the
/// tool-related fields consistent.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::Message;
///
/// let system = Message::system("Answer in one sentence.");
/// let user = Message::user("Why is the sky blue?");
/// assert_eq!(user.text_content(), "Why is the sky blue?");
/// assert!(!user.is_multimodal());
///
/// // Tool results reference the call they answer.
/// let result = Message::tool(r#"{"temp_c":21}"#, "call_abc123");
/// assert_eq!(result.tool_call_id.as_deref(), Some("call_abc123"));
/// # let _ = system;
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who produced this message.
    pub role: Role,

    /// Text content (for simple text-only messages).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,

    /// Multimodal content blocks.  When non-empty, `content` is ignored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_blocks: Vec<ContentBlock>,

    /// Tool calls emitted by an assistant message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// The tool call ID this tool result corresponds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Whether a tool result represents an error.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tool_error: bool,
}

impl Message {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_error: false,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_error: false,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_error: false,
        }
    }

    /// Create an assistant message that includes tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls,
            tool_call_id: None,
            tool_error: false,
        }
    }

    /// Create a successful tool result message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            tool_error: false,
        }
    }

    /// Create a tool result message that represents an error.
    pub fn tool_error(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            content_blocks: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            tool_error: true,
        }
    }

    /// Create a user message with multimodal content blocks.
    pub fn user_multimodal(content_blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: String::new(),
            content_blocks,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_error: false,
        }
    }

    /// Check if this message has multimodal content.
    pub fn is_multimodal(&self) -> bool {
        !self.content_blocks.is_empty()
    }

    /// Check whether the assistant message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Get the text content of this message.
    ///
    /// For multimodal messages, this concatenates all text blocks.
    pub fn text_content(&self) -> String {
        if self.is_multimodal() {
            self.content_blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            self.content.clone()
        }
    }
}

/// A single conversation turn involving user, assistant, and potentially tools.
///
/// Turns are a convenient way to keep history around: replaying them with
/// [`Prompt::with_history`] re-expands them into the flat message list a
/// provider expects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// The user message that opened the turn.
    pub user_message: Message,
    /// The assistant reply, including any tool calls it requested.
    pub assistant_message: Message,
    /// Tool result messages produced for this turn, in execution order.
    pub tool_results: Vec<Message>,
}

/// A stream event from an AI generation request.
///
/// Emitted by
/// [`RequestBuilder::generate_stream_events`](crate::RequestBuilder::generate_stream_events),
/// which assembles low-level provider events into this higher-level shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// An incremental piece of assistant text.
    TextDelta {
        /// Text to append to what has been received so far.
        text: String,
    },
    /// A tool call that has finished streaming its arguments.
    ToolCall {
        /// Provider-assigned call identifier.
        id: String,
        /// Name of the tool the model wants to run.
        name: String,
        /// Raw JSON argument string as streamed by the provider.
        arguments: String,
    },
    /// The result of executing a tool call.
    ToolResult {
        /// Identifier of the call this result answers.
        id: String,
        /// Serialized tool output.
        result: String,
    },
    /// The turn is finished; carries the assembled conversation turn.
    TurnComplete {
        /// The complete turn, ready to be stored as history.
        turn: ConversationTurn,
    },
}

/// A collection of messages forming a prompt.
///
/// `Prompt` implements `From<&str>`, `From<String>`, `From<Message>`, and
/// `From<Vec<Message>>`, so most call sites can pass their input directly to
/// [`RequestBuilder::prompt`](crate::RequestBuilder::prompt).
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{Message, Prompt};
///
/// let prompt = Prompt::single(Message::system("Be brief."))
///     .with_message(Message::user("Define entropy."));
///
/// assert_eq!(prompt.system_message(), Some("Be brief."));
/// assert_eq!(prompt.conversation_messages().len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    /// Messages in provider order; system messages usually come first.
    pub messages: Vec<Message>,
}

impl Prompt {
    /// Build a prompt from a full message list.
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Build a prompt from a single message.
    pub fn single(message: Message) -> Self {
        Self {
            messages: vec![message],
        }
    }

    /// Append a message to the prompt.
    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Return a new prompt with an additional message appended.
    pub fn with_message(mut self, message: Message) -> Self {
        self.push_message(message);
        self
    }

    /// Add a history of conversation turns to the prompt.
    pub fn with_history(mut self, history: Vec<ConversationTurn>) -> Self {
        for turn in history {
            self.push_turn(turn);
        }
        self
    }

    /// Append a single conversation turn to the prompt.
    pub fn push_turn(&mut self, turn: ConversationTurn) {
        self.messages.push(turn.user_message);
        self.messages.push(turn.assistant_message);
        self.messages.extend(turn.tool_results);
    }

    /// Extract the system message if present (first system message).
    pub fn system_message(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str())
    }

    /// Get non-system messages.
    pub fn conversation_messages(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect()
    }

    /// Check if this prompt contains any multimodal content.
    pub fn is_multimodal(&self) -> bool {
        self.messages.iter().any(|m| m.is_multimodal())
    }
}

impl From<&Prompt> for Prompt {
    fn from(prompt: &Prompt) -> Self {
        prompt.clone()
    }
}

impl From<Vec<Message>> for Prompt {
    fn from(messages: Vec<Message>) -> Self {
        Self::new(messages)
    }
}

impl From<Message> for Prompt {
    fn from(message: Message) -> Self {
        Self::single(message)
    }
}

impl From<&str> for Prompt {
    fn from(text: &str) -> Self {
        Prompt {
            messages: vec![Message::user(text.to_string())],
        }
    }
}

impl From<String> for Prompt {
    fn from(text: String) -> Self {
        Prompt {
            messages: vec![Message::user(text)],
        }
    }
}

/// Token usage metadata from the AI response.
///
/// Fields are optional because providers do not all report the same counters,
/// and streaming responses only include usage on the final event.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens consumed by the prompt (input tokens).
    pub prompt_tokens: Option<i32>,
    /// Tokens produced by the model (output tokens).
    pub completion_tokens: Option<i32>,
    /// Prompt plus completion tokens.
    pub total_tokens: Option<i32>,
}

/// The response from an AI generation request.
///
/// # Examples
///
/// ```no_run
/// # async fn run() -> rai_sdk::Result<()> {
/// use rai_sdk::{ClientBuilder, Model};
///
/// let client = ClientBuilder::new()
///     .from_env()
///     .model(Model::gpt4o_mini())
///     .build()?;
///
/// let response = client.request().prompt("Say hi.").generate().await?;
///
/// println!("{}", response.text());
/// println!("served by {} using {}", response.provider, response.model);
/// if let Some(usage) = &response.usage {
///     println!("{:?} total tokens", usage.total_tokens);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    /// The generated message(s).
    pub messages: Vec<Message>,

    /// Token usage information.
    pub usage: Option<Usage>,

    /// The model that was used.
    pub model: String,

    /// The provider that was used.
    pub provider: ProviderKind,

    /// Finish reason (e.g., "stop", "length", "tool_use").
    pub finish_reason: Option<String>,
}

impl Response {
    /// Helper to extract the text content from the first message in the response.
    ///
    /// Returns an empty string if the response contains no messages.
    pub fn text(&self) -> String {
        self.messages
            .first()
            .map(|m| m.text_content())
            .unwrap_or_default()
    }
}

/// A chunk of streamed response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The text content in this chunk.
    pub content: String,

    /// Whether this is the final chunk.
    pub done: bool,

    /// Finish reason (only present on final chunk).
    pub finish_reason: Option<String>,

    /// Usage metadata (only present on final chunk for some providers).
    pub usage: Option<Usage>,
}

/// Parsed structured output together with the underlying AI response.
///
/// Returned by
/// [`RequestBuilder::generate_structured`](crate::RequestBuilder::generate_structured);
/// the raw [`Response`] is kept so callers still have access to usage, model,
/// and finish reason.
#[derive(Debug, Clone)]
pub struct StructuredOutput<T> {
    /// The response content deserialized into `T` and schema-validated.
    pub output: T,
    /// The underlying provider response the value was parsed from.
    pub response: Response,
}

/// A provider-agnostic tool definition sent to the model.
///
/// Produced from a [`Tool`](crate::Tool) when a request is built; providers
/// translate it into their own function/tool schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name the model uses to call it.
    pub name: String,
    /// Optional description telling the model when to use the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the accepted arguments.
    pub input_schema: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_message_extraction() {
        let prompt = Prompt::new(vec![Message::system("Be concise."), Message::user("Hello")]);
        assert_eq!(prompt.system_message(), Some("Be concise."));
        assert_eq!(prompt.conversation_messages().len(), 1);
    }

    #[test]
    fn multimodal_text_content_concatenation() {
        let msg = Message::user_multimodal(vec![
            ContentBlock::text("First"),
            ContentBlock::image_url("https://example.com/img.png"),
            ContentBlock::text("Second"),
        ]);
        assert!(msg.is_multimodal());
        assert_eq!(msg.text_content(), "First\nSecond");
    }

    #[test]
    fn prompt_from_conversions() {
        let p1: Prompt = Message::user("hi").into();
        assert_eq!(p1.messages.len(), 1);

        let p2: Prompt = vec![Message::user("a"), Message::user("b")].into();
        assert_eq!(p2.messages.len(), 2);
    }
}
