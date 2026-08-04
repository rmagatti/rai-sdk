//! OpenAI Chat Completions provider.
//!
//! Talks to `POST {base_url}/chat/completions` (default base URL
//! `https://api.openai.com/v1`), supporting non-streaming and streaming
//! generation, tool calling, JSON mode, and strict JSON-Schema structured
//! output. Text and image content blocks are translated; audio, video, and file
//! blocks are not yet supported.
//!
//! Reasoning (o-series) models do not accept `temperature`/`top_p`, so those
//! fields are dropped automatically for them.

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
    message::{
        ContentBlock, ImageSource, Message, Prompt, Response, ToolCall, ToolDefinition, Usage,
    },
    model::OpenAIModel,
};

const OPENAI_API_URL: &str = "https://api.openai.com/v1";

/// OpenAI provider implementation.
///
/// Stores the API key, base URL, and HTTP client at construction time.
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for OpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

impl OpenAIProvider {
    /// Create a new OpenAI provider from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ProviderNotConfigured`] if no OpenAI API key is
    /// available, or [`Error::Config`] if the HTTP client cannot be built.
    pub fn new(config: &Config) -> Result<Self> {
        let api_key = config
            .openai_key()
            .ok_or(Error::ProviderNotConfigured(ProviderKind::OpenAI))?;

        let base_url = config
            .openai_base_url
            .clone()
            .unwrap_or_else(|| OPENAI_API_URL.to_string());

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
    /// Returns [`Error::Http`] if the request cannot be sent,
    /// [`Error::Auth`]/[`Error::RateLimit`]/[`Error::InvalidRequest`]/[`Error::Request`]
    /// depending on the status code OpenAI replies with, and [`Error::Request`]
    /// if the response contains no choices or a tool call with unparseable
    /// arguments.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate(
        &self,
        model: &OpenAIModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Response> {
        info!(model = model.as_str(), "Generating completion with OpenAI");

        let request = self.build_request(model, prompt, config, false, tool_definitions);
        let url = format!("{}/chat/completions", self.base_url);

        debug!(url = %url, "Sending request to OpenAI");
        debug!(request_payload = ?request, "OpenAI request payload");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenAI request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let response_body: OpenAIResponse = response.json().await?;
        debug!(response_payload = ?response_body, "OpenAI response payload");
        self.parse_response(model, response_body)
    }

    /// Generate a completion with streaming.
    ///
    /// The returned stream yields
    /// [`ProviderStreamEvent`](crate::provider::ProviderStreamEvent) values and
    /// ends when OpenAI sends its `[DONE]` sentinel. Unparseable SSE payloads
    /// are logged and skipped rather than terminating the stream.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`OpenAIProvider::generate`] while opening
    /// the stream. Transport failures encountered mid-stream are yielded as
    /// [`Error::Stream`] items.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_stream(
        &self,
        model: &OpenAIModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, None);
        let url = format!("{}/chat/completions", self.base_url);

        debug!(url = %url, "Sending streaming request to OpenAI");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenAI streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(Self::parse_provider_sse_stream(byte_stream)))
    }

    fn build_request(
        &self,
        model: &OpenAIModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        stream: bool,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> OpenAIRequest {
        let messages: Vec<OpenAIMessage> = prompt
            .messages
            .iter()
            .map(|m| {
                let content = if m.is_multimodal() {
                    Some(OpenAIMessageContent::Blocks(
                        m.content_blocks
                            .iter()
                            .map(|block| match block {
                                ContentBlock::Text { text } => {
                                    OpenAIContentBlock::Text { text: text.clone() }
                                }
                                ContentBlock::Image { source } => {
                                    let url = match source {
                                        ImageSource::Url { url } => url.clone(),
                                        ImageSource::Base64 { media_type, data } => {
                                            format!("data:{media_type};base64,{data}")
                                        }
                                    };
                                    OpenAIContentBlock::ImageUrl {
                                        image_url: OpenAIImageUrl { url },
                                    }
                                }
                                _ => unimplemented!(
                                    "Audio, Video, and File blocks are not yet supported for OpenAI"
                                ),
                            })
                            .collect(),
                    ))
                } else if !m.content.is_empty() {
                    Some(OpenAIMessageContent::Text(m.content.clone()))
                } else {
                    None
                };

                OpenAIMessage {
                    role: m.role.as_str().to_string(),
                    content,
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls: if m.tool_calls.is_empty() {
                        None
                    } else {
                        Some(
                            m.tool_calls
                                .iter()
                                .map(|tc| OpenAIRequestToolCall {
                                    id: tc.id.clone(),
                                    r#type: "function".to_string(),
                                    function: OpenAIRequestFunctionCall {
                                        name: tc.name.clone(),
                                        arguments: tc.arguments.to_string(),
                                    },
                                })
                                .collect(),
                        )
                    },
                }
            })
            .collect();

        let mut request = OpenAIRequest {
            model: model.as_str().to_string(),
            messages,
            stream: Some(stream),
            stream_options: stream.then_some(OpenAIStreamOptions {
                include_usage: true,
            }),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            stop: config.stop_sequences.clone(),
            response_format: None,
            tools: tool_definitions.map(|tools| {
                tools
                    .iter()
                    .map(|tool| OpenAIToolDefinition {
                        r#type: "function".to_string(),
                        function: OpenAIFunctionDefinition {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters: tool.input_schema.clone(),
                        },
                    })
                    .collect()
            }),
        };

        if let Some(json_schema) = config.json_schema.clone() {
            request.response_format = Some(OpenAIResponseFormat::JsonSchema {
                json_schema: OpenAIJsonSchema {
                    name: "structured_output".to_string(),
                    schema: json_schema,
                    strict: true,
                },
            });
        } else if config.json_mode == Some(true) {
            request.response_format = Some(OpenAIResponseFormat::JsonObject);
        }

        // Reasoning models (o1, o3) don't support temperature
        if model.is_reasoning_model() {
            request.temperature = None;
            request.top_p = None;
        }

        request
    }

    fn parse_response(&self, model: &OpenAIModel, response: OpenAIResponse) -> Result<Response> {
        let choice = response.choices.first().ok_or_else(|| Error::Request {
            provider: ProviderKind::OpenAI,
            message: "No choices in response".to_string(),
        })?;

        let content = choice.message.content.clone().unwrap_or_default();
        let tool_calls = choice
            .message
            .tool_calls
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                let arguments = serde_json::from_str(&tc.function.arguments).map_err(|error| {
                    Error::Request {
                        provider: ProviderKind::OpenAI,
                        message: format!(
                            "Invalid tool arguments returned for '{}': {error}",
                            tc.function.name
                        ),
                    }
                })?;

                Ok(ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: Some(u.prompt_tokens),
            completion_tokens: Some(u.completion_tokens),
            total_tokens: Some(u.total_tokens),
        });

        info!(
            model = model.as_str(),
            finish_reason = ?choice.finish_reason,
            "Received response from OpenAI"
        );

        Ok(Response {
            messages: vec![Message::assistant_with_tool_calls(content, tool_calls)],
            usage,
            model: model.as_str().to_string(),
            provider: ProviderKind::OpenAI,
            finish_reason: choice.finish_reason.clone(),
        })
    }

    fn parse_error(&self, status: u16, body: &str) -> Error {
        if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(body) {
            let message = error_response.error.message;
            return match status {
                401 => Error::Auth {
                    provider: ProviderKind::OpenAI,
                    message,
                },
                429 => Error::RateLimit {
                    provider: ProviderKind::OpenAI,
                    message,
                },
                400 => Error::InvalidRequest(message),
                _ => Error::Request {
                    provider: ProviderKind::OpenAI,
                    message,
                },
            };
        }

        Error::Request {
            provider: ProviderKind::OpenAI,
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

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(event_end) = buffer.find("\n\n") {
                            let event = buffer[..event_end].to_string();
                            buffer = buffer[event_end + 2..].to_string();

                            for line in event.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        return;
                                    }

                                    match serde_json::from_str::<OpenAIStreamResponse>(data) {
                                        Ok(stream_response) => {
                                            if let Some(usage) = stream_response.usage {
                                                yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                    finish_reason: stream_response.choices.first().and_then(|c| c.finish_reason.clone()).or(Some("stop".to_string())),
                                                    usage: Some(Usage {
                                                        prompt_tokens: Some(usage.prompt_tokens),
                                                        completion_tokens: Some(usage.completion_tokens),
                                                        total_tokens: Some(usage.total_tokens),
                                                    }),
                                                });
                                                continue;
                                            }

                                            if let Some(choice) = stream_response.choices.first() {
                                                if let Some(content) = &choice.delta.content {
                                                    if !content.is_empty() {
                                                        yield Ok(crate::provider::ProviderStreamEvent::Text(content.clone()));
                                                    }
                                                }

                                                if let Some(tool_calls) = &choice.delta.tool_calls {
                                                    for tc in tool_calls {
                                                        if let Some(id) = &tc.id {
                                                            current_tool_id = Some(id.clone());
                                                            let name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();
                                                            yield Ok(crate::provider::ProviderStreamEvent::ToolCallStart {
                                                                id: id.clone(),
                                                                name,
                                                            });
                                                        }
                                                        if let Some(func) = &tc.function {
                                                            if let Some(args) = &func.arguments {
                                                                let id_to_use = current_tool_id.clone().unwrap_or_default();
                                                                yield Ok(crate::provider::ProviderStreamEvent::ToolCallChunk {
                                                                    id: id_to_use,
                                                                    arguments: args.clone(),
                                                                });
                                                            }
                                                        }
                                                    }
                                                }

                                                if let Some(finish_reason) = &choice.finish_reason {
                                                    yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                        finish_reason: Some(finish_reason.clone()),
                                                        usage: None,
                                                    });
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            debug!(error = %e, data = %data, "Failed to parse SSE chunk");
                                        }
                                    }
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

// ── OpenAI API types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAIStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAIToolDefinition>>,
}

#[derive(Debug, Serialize)]
struct OpenAIStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIToolDefinition {
    r#type: String,
    function: OpenAIFunctionDefinition,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIResponseFormat {
    JsonObject,
    JsonSchema { json_schema: OpenAIJsonSchema },
}

#[derive(Debug, Serialize)]
struct OpenAIJsonSchema {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIRequestToolCall>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIMessageContent {
    Text(String),
    Blocks(Vec<OpenAIContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIContentBlock {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
}

#[derive(Debug, Serialize)]
struct OpenAIImageUrl {
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIRequestToolCall {
    id: String,
    r#type: String,
    function: OpenAIRequestFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIRequestFunctionCall {
    name: String,
    arguments: String,
}

// ── Response types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIResponseToolCall {
    id: String,
    function: OpenAIResponseFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    id: Option<String>,
    function: Option<OpenAIStreamFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIErrorDetail,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorDetail {
    message: String,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::GenerationConfig;

    fn test_provider() -> OpenAIProvider {
        OpenAIProvider {
            client: Client::new(),
            api_key: "test-key".to_string(),
            base_url: "https://example.com".to_string(),
        }
    }

    #[test]
    fn build_request_uses_json_object_mode() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let config = GenerationConfig::new().with_json_mode(true);

        let request = provider.build_request(&OpenAIModel::Gpt4o, &prompt, &config, false, None);

        assert!(matches!(
            request.response_format,
            Some(OpenAIResponseFormat::JsonObject)
        ));
    }

    #[test]
    fn build_request_uses_json_schema_mode() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        });
        let config = GenerationConfig::new().with_json_schema(schema.clone());

        let request = provider.build_request(&OpenAIModel::Gpt4o, &prompt, &config, false, None);

        match request.response_format {
            Some(OpenAIResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "structured_output");
                assert!(json_schema.strict);
                assert_eq!(json_schema.schema, schema);
            }
            _ => panic!("Expected json_schema response format"),
        }
    }

    #[test]
    fn json_schema_takes_precedence_over_json_mode() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let config = GenerationConfig::new()
            .with_json_mode(true)
            .with_json_schema(serde_json::json!({ "type": "object" }));

        let request = provider.build_request(&OpenAIModel::Gpt4o, &prompt, &config, false, None);

        assert!(matches!(
            request.response_format,
            Some(OpenAIResponseFormat::JsonSchema { .. })
        ));
    }
}
