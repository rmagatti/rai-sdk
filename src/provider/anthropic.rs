//! Anthropic Messages provider.
//!
//! Talks to `POST {base_url}/messages` (default base URL
//! `https://api.anthropic.com/v1`) using API version `2023-06-01`, supporting
//! non-streaming and streaming generation, tool use, and JSON-Schema structured
//! output. Text and image content blocks are translated; audio, video, and file
//! blocks are not yet supported.
//!
//! Anthropic requires `max_tokens`, so a default of 8192 is used when the
//! generation config does not set one. System messages are hoisted out of the
//! message list into the request's top-level `system` field, and tool results
//! are sent as user-role `tool_result` blocks.

use std::pin::Pin;

use bytes::Bytes;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{debug, error, info, instrument};

use crate::{
    config::Config,
    error::{Error, ProviderKind, Result},
    generation::GenerationConfig,
    message::{
        ContentBlock as CommonContentBlock, ImageSource as CommonImageSource, Message, Prompt,
        Response, Role, ToolCall, ToolDefinition, Usage,
    },
    model::AnthropicModel,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: i32 = 8192;

/// Anthropic Claude provider implementation.
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

impl AnthropicProvider {
    /// Create a new Anthropic provider from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ProviderNotConfigured`] if no Anthropic API key is
    /// available, or [`Error::Config`] if the HTTP client cannot be built.
    pub fn new(config: &Config) -> Result<Self> {
        let api_key = config
            .anthropic_key()
            .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::Anthropic))?;

        let base_url = config
            .anthropic_base_url
            .clone()
            .unwrap_or_else(|| ANTHROPIC_API_URL.to_string());

        let client = super::http_client_builder()
            .timeout(std::time::Duration::from_secs(config.timeout()))
            .build()
            .map_err(|e| Error::Config(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            api_key,
            base_url,
        })
    }

    /// Generate a completion (non-streaming).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Http`] if the request cannot be sent, and
    /// [`Error::Auth`]/[`Error::RateLimit`]/[`Error::InvalidRequest`]/[`Error::Request`]
    /// depending on the status code Anthropic replies with.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate(
        &self,
        model: &AnthropicModel,
        prompt: &Prompt,
        config: &GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Response> {
        info!(
            model = model.as_str(),
            "Generating completion with Anthropic"
        );

        let request = self.build_request(model, prompt, config, false, tool_definitions);
        let url = format!("{}/messages", self.base_url);

        debug!(url = %url, "Sending request to Anthropic");
        debug!(request_payload = ?request, "Anthropic request payload");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Anthropic request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let response_body: AnthropicResponse = response.json().await?;
        debug!(response_payload = ?response_body, "Anthropic response payload");
        self.parse_response(model, response_body)
    }

    /// Generate a completion with streaming.
    ///
    /// The returned stream yields
    /// [`ProviderStreamEvent`](crate::provider::ProviderStreamEvent) values
    /// assembled from Anthropic's server-sent events.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`AnthropicProvider::generate`] while opening
    /// the stream. Transport failures encountered mid-stream are yielded as
    /// [`Error::Stream`] items.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_stream(
        &self,
        model: &AnthropicModel,
        prompt: &Prompt,
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, None);
        let url = format!("{}/messages", self.base_url);

        debug!(url = %url, "Sending streaming request to Anthropic");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Anthropic streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(Self::parse_provider_sse_stream(byte_stream)))
    }

    fn build_request(
        &self,
        model: &AnthropicModel,
        prompt: &Prompt,
        config: &GenerationConfig,
        stream: bool,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> AnthropicRequest {
        let prompt_caching = config.prompt_caching == Some(true);
        let system = prompt.system_message().map(|text| {
            if prompt_caching {
                AnthropicSystemPrompt::Blocks(vec![AnthropicSystemBlock {
                    kind: AnthropicSystemBlockType::Text,
                    text: text.to_string(),
                    cache_control: Some(AnthropicCacheControl::Ephemeral),
                }])
            } else {
                AnthropicSystemPrompt::Text(text.to_string())
            }
        });

        let messages: Vec<AnthropicMessage> = prompt
            .conversation_messages()
            .into_iter()
            .map(Self::build_message)
            .collect();

        AnthropicRequest {
            model: model.as_str().to_string(),
            messages,
            system,
            stream: Some(stream),
            max_tokens: config.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            stop_sequences: config.stop_sequences.clone(),
            output_config: config
                .json_schema
                .clone()
                .map(|schema| AnthropicOutputConfig {
                    format: AnthropicOutputFormat::JsonSchema { schema },
                }),
            tools: tool_definitions.map(|tools| {
                let last_index = tools.len().checked_sub(1);
                tools
                    .iter()
                    .enumerate()
                    .map(|(index, tool)| AnthropicToolDefinition {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        input_schema: tool.input_schema.clone(),
                        cache_control: (prompt_caching && Some(index) == last_index)
                            .then_some(AnthropicCacheControl::Ephemeral),
                    })
                    .collect()
            }),
        }
    }

    fn build_message(message: &Message) -> AnthropicMessage {
        let role = match message.role {
            Role::Assistant => "assistant".to_string(),
            Role::User | Role::Tool | Role::System => "user".to_string(),
        };

        if message.role == Role::Tool {
            return AnthropicMessage {
                role,
                content: AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                    tool_use_id: message.tool_call_id.clone().unwrap_or_default(),
                    content: message.content.clone(),
                    is_error: message.tool_error,
                }]),
            };
        }

        let mut content_blocks = Vec::new();

        if message.is_multimodal() {
            content_blocks.extend(message.content_blocks.iter().map(|block| match block {
                CommonContentBlock::Text { text } => {
                    AnthropicContentBlock::Text { text: text.clone() }
                }
                CommonContentBlock::Image { source } => {
                    let api_source = match source {
                        CommonImageSource::Url { url } => {
                            AnthropicImageSource::Url { url: url.clone() }
                        }
                        CommonImageSource::Base64 { media_type, data } => {
                            AnthropicImageSource::Base64 {
                                media_type: media_type.clone(),
                                data: data.clone(),
                            }
                        }
                    };
                    AnthropicContentBlock::Image { source: api_source }
                }
                _ => unimplemented!(
                    "Audio, Video, and File blocks are not yet supported for Anthropic"
                ),
            }));
        } else if !message.content.is_empty() {
            content_blocks.push(AnthropicContentBlock::Text {
                text: message.content.clone(),
            });
        }

        if message.role == Role::Assistant {
            content_blocks.extend(message.tool_calls.iter().map(|tool_call| {
                AnthropicContentBlock::ToolUse {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    input: tool_call.arguments.clone(),
                }
            }));
        }

        if content_blocks.len() == 1 {
            if let AnthropicContentBlock::Text { text } = &content_blocks[0] {
                if message.tool_calls.is_empty() {
                    return AnthropicMessage {
                        role,
                        content: AnthropicMessageContent::Text(text.clone()),
                    };
                }
            }
        }

        AnthropicMessage {
            role,
            content: AnthropicMessageContent::Blocks(content_blocks),
        }
    }

    fn parse_response(
        &self,
        model: &AnthropicModel,
        response: AnthropicResponse,
    ) -> Result<Response> {
        let content = response
            .content
            .iter()
            .filter_map(|block| {
                if block.r#type == "text" {
                    block.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        let tool_calls = response
            .content
            .into_iter()
            .filter_map(|block| {
                if block.r#type == "tool_use" {
                    Some(ToolCall {
                        id: block.id?,
                        name: block.name?,
                        arguments: block.input.unwrap_or_default(),
                    })
                } else {
                    None
                }
            })
            .collect();

        let usage = Some(Usage {
            prompt_tokens: Some(response.usage.input_tokens),
            completion_tokens: Some(response.usage.output_tokens),
            total_tokens: Some(response.usage.input_tokens + response.usage.output_tokens),
        });

        info!(
            model = model.as_str(),
            stop_reason = ?response.stop_reason,
            "Received response from Anthropic"
        );

        Ok(Response {
            messages: vec![Message::assistant_with_tool_calls(content, tool_calls)],
            usage,
            model: model.as_str().to_string(),
            provider: ProviderKind::Anthropic,
            finish_reason: response.stop_reason,
        })
    }

    fn parse_error(&self, status: u16, body: &str) -> Error {
        if let Ok(error_response) = serde_json::from_str::<AnthropicErrorResponse>(body) {
            let message = error_response.error.message;

            return match status {
                401 => Error::Auth {
                    provider: ProviderKind::Anthropic,
                    message,
                },
                429 => Error::RateLimit {
                    provider: ProviderKind::Anthropic,
                    message,
                },
                400 => Error::InvalidRequest(message),
                _ => Error::Request {
                    provider: ProviderKind::Anthropic,
                    message,
                },
            };
        }

        Error::Request {
            provider: ProviderKind::Anthropic,
            message: format!("HTTP {status}: {body}"),
        }
    }

    fn parse_provider_sse_stream<S>(
        byte_stream: S,
    ) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut buffer = String::new();

        async_stream::stream! {
            tokio::pin!(byte_stream);
            let mut current_tool_id: Option<String> = None;
            // Anthropic splits usage across two events: `message_start` reports
            // the input tokens and `message_delta` the final output tokens, so
            // the input count has to be held here until the delta arrives.
            let mut input_tokens: Option<i32> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(event_end) = buffer.find("\n\n") {
                            let event = buffer[..event_end].to_string();
                            buffer = buffer[event_end + 2..].to_string();

                            let mut event_type = String::new();
                            let mut event_data = String::new();

                            for line in event.lines() {
                                if let Some(t) = line.strip_prefix("event: ") {
                                    event_type = t.to_string();
                                } else if let Some(d) = line.strip_prefix("data: ") {
                                    event_data = d.to_string();
                                }
                            }

                            match event_type.as_str() {
                                "content_block_start" => {
                                    if let Ok(start) = serde_json::from_str::<ContentBlockStart>(&event_data) {
                                        if start.content_block.r#type == "tool_use" {
                                            if let (Some(id), Some(name)) = (start.content_block.id, start.content_block.name) {
                                                current_tool_id = Some(id.clone());
                                                yield Ok(crate::provider::ProviderStreamEvent::ToolCallStart {
                                                    id,
                                                    name,
                                                });
                                            }
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(&event_data) {
                                        if delta.delta.r#type == "text_delta" {
                                            if let Some(text) = delta.delta.text {
                                                if !text.is_empty() {
                                                    yield Ok(crate::provider::ProviderStreamEvent::Text(text));
                                                }
                                            }
                                        } else if delta.delta.r#type == "input_json_delta" {
                                            if let Some(partial) = delta.delta.partial_json {
                                                if let Some(id) = &current_tool_id {
                                                    yield Ok(crate::provider::ProviderStreamEvent::ToolCallChunk {
                                                        id: id.clone(),
                                                        arguments: partial,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                "message_delta" => {
                                    if let Ok(delta) = serde_json::from_str::<MessageDelta>(&event_data) {
                                        let usage = delta.usage.map(|u| Usage {
                                            prompt_tokens: input_tokens,
                                            completion_tokens: Some(u.output_tokens),
                                            total_tokens: input_tokens.map(|input| input + u.output_tokens),
                                        });

                                        if delta.delta.stop_reason.is_some() || usage.is_some() {
                                            yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                finish_reason: delta.delta.stop_reason,
                                                usage,
                                            });
                                        }
                                    }
                                }
                                "message_start" => {
                                    if let Ok(start) = serde_json::from_str::<MessageStart>(&event_data) {
                                        // Only the input count is authoritative here: the
                                        // `output_tokens` reported alongside it is a partial
                                        // figure that `message_delta` later supersedes.
                                        input_tokens = Some(start.message.usage.input_tokens);
                                    }
                                }
                                "error" => {
                                    if let Ok(error) = serde_json::from_str::<StreamError>(&event_data) {
                                        yield Err(Error::Request {
                                            provider: ProviderKind::Anthropic,
                                            message: error.error.message,
                                        });
                                        return;
                                    }
                                }
                                _ => {
                                    debug!(event_type = %event_type, "Ignoring SSE event");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(Error::Stream(e.to_string()));
                        return;
                    }
                }
            }
        }
    }
}

// Anthropic API types

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    max_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicToolDefinition>>,
}

#[derive(Debug, Serialize)]
struct AnthropicToolDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Serialize)]
struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    kind: AnthropicSystemBlockType,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum AnthropicSystemBlockType {
    Text,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicCacheControl {
    Ephemeral,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    format: AnthropicOutputFormat,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicOutputFormat {
    JsonSchema { schema: serde_json::Value },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicImageSource {
    #[serde(rename = "url")]
    Url { url: String },
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponseContentBlock {
    r#type: String,
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: i32,
    output_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStart {
    content_block: ContentBlockStartDetail,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartDetail {
    r#type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDelta {
    delta: ContentDelta,
}

#[derive(Debug, Deserialize)]
struct ContentDelta {
    r#type: String,
    text: Option<String>,
    partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    delta: MessageDeltaContent,
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageStart {
    message: MessageStartDetail,
}

#[derive(Debug, Deserialize)]
struct MessageStartDetail {
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaContent {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    output_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    error: AnthropicErrorDetail,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::message::Prompt;

    fn test_provider() -> AnthropicProvider {
        AnthropicProvider {
            client: Client::new(),
            api_key: "test-key".to_string(),
            base_url: "https://example.com".to_string(),
        }
    }

    #[test]
    fn test_build_request_without_schema_has_no_output_config() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let config = GenerationConfig::new();

        let request = provider.build_request(
            &AnthropicModel::ClaudeSonnet45,
            &prompt,
            &config,
            false,
            None,
        );

        assert!(request.output_config.is_none());
    }

    #[test]
    fn test_build_request_with_schema_sets_output_config() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"]
        });
        let config = GenerationConfig::new().with_json_schema(schema.clone());

        let request = provider.build_request(
            &AnthropicModel::ClaudeSonnet45,
            &prompt,
            &config,
            false,
            None,
        );

        match request.output_config {
            Some(AnthropicOutputConfig {
                format: AnthropicOutputFormat::JsonSchema { schema: actual },
            }) => {
                assert_eq!(actual, schema);
            }
            _ => panic!("Expected Anthropic JSON schema output_config"),
        }
    }

    #[test]
    fn prompt_caching_serializes_system_and_last_tool_breakpoints() {
        let provider = test_provider();
        let prompt = Prompt::new(vec![
            Message::system("Cache this prefix."),
            Message::user("Hello"),
        ]);
        let config = GenerationConfig::new().with_prompt_caching(true);
        let tools = vec![
            ToolDefinition {
                name: "first".to_string(),
                description: None,
                input_schema: serde_json::json!({ "type": "object" }),
            },
            ToolDefinition {
                name: "second".to_string(),
                description: Some("Second tool".to_string()),
                input_schema: serde_json::json!({ "type": "object" }),
            },
        ];

        let request = provider.build_request(
            &AnthropicModel::ClaudeSonnet45,
            &prompt,
            &config,
            false,
            Some(&tools),
        );
        let body = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(
            body["system"],
            serde_json::json!([{
                "type": "text",
                "text": "Cache this prefix.",
                "cache_control": { "type": "ephemeral" }
            }])
        );
        assert!(body["tools"][0].get("cache_control").is_none());
        assert_eq!(
            body["tools"][1]["cache_control"],
            serde_json::json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn prompt_caching_off_preserves_the_existing_system_wire_shape() {
        let provider = test_provider();
        let prompt = Prompt::new(vec![Message::system("Be brief."), Message::user("Hello")]);

        let request = provider.build_request(
            &AnthropicModel::ClaudeSonnet45,
            &prompt,
            &GenerationConfig::new(),
            false,
            None,
        );
        let body = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(body["system"], "Be brief.");
        assert!(
            !body.to_string().contains("cache_control"),
            "default requests must not gain cache-control fields"
        );
    }
}
