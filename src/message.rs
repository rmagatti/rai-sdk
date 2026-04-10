use serde::{Deserialize, Serialize};

use crate::error::ProviderKind;

/// Role of a message participant in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call identifier.
    pub id: String,
    /// Name of the tool to invoke.
    pub name: String,
    /// Arguments as a JSON value.
    pub arguments: serde_json::Value,
}

/// Content block for multimodal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "image")]
    Image { source: ImageSource },

    #[serde(rename = "audio")]
    Audio { source: FileSource },

    #[serde(rename = "video")]
    Video { source: FileSource },

    #[serde(rename = "file")]
    File { source: FileSource },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "url")]
    Url { url: String },

    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

/// File source for multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FileSource {
    #[serde(rename = "url")]
    Url { url: String },

    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

/// A single message in a conversation.
///
/// Supports text-only and multimodal (text + images) content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub user_message: Message,
    pub assistant_message: Message,
    pub tool_results: Vec<Message>,
}

/// A stream event from an AI generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        id: String,
        result: String,
    },
    TurnComplete {
        turn: ConversationTurn,
    },
}

/// A collection of messages forming a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
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

/// Token usage metadata from the AI response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub total_tokens: Option<i32>,
}

/// The response from an AI generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// A chunk of streamed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone)]
pub struct StructuredOutput<T> {
    pub output: T,
    pub response: Response,
}

/// A provider-agnostic tool definition sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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

impl Response {
    /// Helper to extract the text content from the first message in the response.
    pub fn text(&self) -> String {
        self.messages
            .first()
            .map(|m| m.text_content())
            .unwrap_or_default()
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
