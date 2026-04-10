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
        ContentBlock, ImageSource, Message, Prompt, Response, ToolCall,
        ToolDefinition, Usage,
    },
    model::OpenRouterModel,
};

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter provider implementation.
///
/// Stores the API key, base URL, and HTTP client at construction time.
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    base_url: String,
    app_url: Option<String>,
    app_title: Option<String>,
}

impl std::fmt::Debug for OpenRouterProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .field("app_url", &self.app_url)
            .field("app_title", &self.app_title)
            .finish()
    }
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider from configuration.
    ///
    /// Returns an error if the OpenRouter API key is not configured.
    pub fn new(config: &Config) -> Result<Self> {
        let api_key = config
            .openrouter_key()
            .ok_or(Error::ProviderNotConfigured(ProviderKind::OpenRouter))?;

        let base_url = OPENROUTER_API_URL.to_string();

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout()))
            .build()
            .map_err(|e| Error::Config(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            api_key,
            base_url,
            app_url: config.openrouter_app_url(),
            app_title: config.openrouter_app_title(),
        })
    }

    /// Generate a completion (non-streaming).
    #[instrument(skip(self, prompt, config))]
    pub async fn generate(
        &self,
        model: &OpenRouterModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Response> {
        info!(model = model.as_str(), "Generating completion with OpenRouter");

        let request = self.build_request(model, prompt, config, false, tool_definitions);
        let url = format!("{}/chat/completions", self.base_url);

        debug!(url = %url, "Sending request to OpenRouter");
        debug!(request_payload = ?request, "OpenRouter request payload");

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(app_url) = &self.app_url {
            req_builder = req_builder.header("HTTP-Referer", app_url);
        }
        if let Some(app_title) = &self.app_title {
            req_builder = req_builder.header("X-Title", app_title);
        }

        let response = req_builder
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenRouter request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let response_body: OpenRouterResponse = response.json().await?;
        debug!(response_payload = ?response_body, "OpenRouter response payload");
        self.parse_response(model, response_body)
    }

    /// Generate a completion with streaming.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_stream(
        &self,
        model: &OpenRouterModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, None);
        let url = format!("{}/chat/completions", self.base_url);

        debug!(url = %url, "Sending streaming request to OpenRouter");

        let mut req_builder = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        if let Some(app_url) = &self.app_url {
            req_builder = req_builder.header("HTTP-Referer", app_url);
        }
        if let Some(app_title) = &self.app_title {
            req_builder = req_builder.header("X-Title", app_title);
        }

        let response = req_builder
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenRouter streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(Self::parse_provider_sse_stream(byte_stream)))
    }

    fn build_request(
        &self,
        model: &OpenRouterModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        stream: bool,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> OpenRouterRequest {
        let messages: Vec<OpenRouterMessage> = prompt
            .messages
            .iter()
            .map(|m| {
                let content = if m.is_multimodal() {
                    Some(OpenRouterMessageContent::Blocks(
                        m.content_blocks
                            .iter()
                            .map(|block| match block {
                                ContentBlock::Text { text } => {
                                    OpenRouterContentBlock::Text { text: text.clone() }
                                }
                                ContentBlock::Image { source } => {
                                    let url = match source {
                                        ImageSource::Url { url } => url.clone(),
                                        ImageSource::Base64 { media_type, data } => {
                                            format!("data:{media_type};base64,{data}")
                                        }
                                    };
                                    OpenRouterContentBlock::ImageUrl {
                                        image_url: OpenRouterImageUrl { url },
                                    }
                                }
                                _ => unimplemented!(
                                    "Audio, Video, and File blocks are not yet supported for OpenRouter"
                                ),
                            })
                            .collect(),
                    ))
                } else if !m.content.is_empty() {
                    Some(OpenRouterMessageContent::Text(m.content.clone()))
                } else {
                    None
                };

                OpenRouterMessage {
                    role: m.role.as_str().to_string(),
                    content,
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls: if m.tool_calls.is_empty() {
                        None
                    } else {
                        Some(
                            m.tool_calls
                                .iter()
                                .map(|tc| OpenRouterRequestToolCall {
                                    id: tc.id.clone(),
                                    r#type: "function".to_string(),
                                    function: OpenRouterRequestFunctionCall {
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

        let mut request = OpenRouterRequest {
            model: model.as_str().to_string(),
            messages,
            stream: Some(stream),
            stream_options: stream.then(|| OpenRouterStreamOptions { include_usage: true }),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            stop: config.stop_sequences.clone(),
            response_format: None,
            tools: tool_definitions.map(|tools| {
                tools
                    .iter()
                    .map(|tool| OpenRouterToolDefinition {
                        r#type: "function".to_string(),
                        function: OpenRouterFunctionDefinition {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters: tool.input_schema.clone(),
                        },
                    })
                    .collect()
            }),
        };

        if let Some(json_schema) = config.json_schema.clone() {
            request.response_format = Some(OpenRouterResponseFormat::JsonSchema {
                json_schema: OpenRouterJsonSchema {
                    name: "structured_output".to_string(),
                    schema: json_schema,
                    strict: true,
                },
            });
        } else if config.json_mode == Some(true) {
            request.response_format = Some(OpenRouterResponseFormat::JsonObject);
        }

        request
    }

    fn parse_response(&self, model: &OpenRouterModel, response: OpenRouterResponse) -> Result<Response> {
        let choice = response.choices.first().ok_or_else(|| Error::Request {
            provider: ProviderKind::OpenRouter,
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
                        provider: ProviderKind::OpenRouter,
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
            "Received response from OpenRouter"
        );

        Ok(Response {
            messages: vec![Message::assistant_with_tool_calls(content, tool_calls)],
            usage,
            model: model.as_str().to_string(),
            provider: ProviderKind::OpenRouter,
            finish_reason: choice.finish_reason.clone(),
        })
    }

    fn parse_error(&self, status: u16, body: &str) -> Error {
        if let Ok(error_response) = serde_json::from_str::<OpenRouterErrorResponse>(body) {
            let message = error_response.error.message;
            return match status {
                401 => Error::Auth {
                    provider: ProviderKind::OpenRouter,
                    message,
                },
                429 => Error::RateLimit {
                    provider: ProviderKind::OpenRouter,
                    message,
                },
                400 => Error::InvalidRequest(message),
                _ => Error::Request {
                    provider: ProviderKind::OpenRouter,
                    message,
                },
            };
        }

        Error::Request {
            provider: ProviderKind::OpenRouter,
            message: format!("HTTP {status}: {body}"),
        }
    }

    fn parse_provider_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
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

                                    match serde_json::from_str::<OpenRouterStreamResponse>(data) {
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

// ── OpenRouter API types ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenRouterStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenRouterResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterToolDefinition>>,
}

#[derive(Debug, Serialize)]
struct OpenRouterStreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenRouterToolDefinition {
    r#type: String,
    function: OpenRouterFunctionDefinition,
}

#[derive(Debug, Serialize)]
struct OpenRouterFunctionDefinition {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenRouterResponseFormat {
    JsonObject,
    JsonSchema { json_schema: OpenRouterJsonSchema },
}

#[derive(Debug, Serialize)]
struct OpenRouterJsonSchema {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenRouterMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterRequestToolCall>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenRouterMessageContent {
    Text(String),
    Blocks(Vec<OpenRouterContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenRouterContentBlock {
    Text { text: String },
    ImageUrl { image_url: OpenRouterImageUrl },
}

#[derive(Debug, Serialize)]
struct OpenRouterImageUrl {
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenRouterRequestToolCall {
    id: String,
    r#type: String,
    function: OpenRouterRequestFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenRouterRequestFunctionCall {
    name: String,
    arguments: String,
}

// ── Response types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterMessageResponse {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterResponseToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenRouterResponseToolCall {
    id: String,
    function: OpenRouterResponseFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenRouterResponseFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamResponse {
    choices: Vec<OpenRouterStreamChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamChoice {
    delta: OpenRouterDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamToolCall {
    id: Option<String>,
    function: Option<OpenRouterStreamFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorResponse {
    error: OpenRouterErrorDetail,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorDetail {
    message: String,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::GenerationConfig;

    fn test_provider() -> OpenRouterProvider {
        OpenRouterProvider {
            client: Client::new(),
            api_key: "test-key".to_string(),
            base_url: "https://example.com".to_string(),
            app_url: None,
            app_title: None,
        }
    }

    #[test]
    fn build_request_uses_json_object_mode() {
        let provider = test_provider();
        let prompt = Prompt::single(Message::user("hello"));
        let config = GenerationConfig::new().with_json_mode(true);

        let request = provider.build_request(&OpenRouterModel::Custom("gpt-4o".into()), &prompt, &config, false, None);

        assert!(matches!(
            request.response_format,
            Some(OpenRouterResponseFormat::JsonObject)
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

        let request = provider.build_request(&OpenRouterModel::Custom("gpt-4o".into()), &prompt, &config, false, None);

        match request.response_format {
            Some(OpenRouterResponseFormat::JsonSchema { json_schema }) => {
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

        let request = provider.build_request(&OpenRouterModel::Custom("gpt-4o".into()), &prompt, &config, false, None);

        assert!(matches!(
            request.response_format,
            Some(OpenRouterResponseFormat::JsonSchema { .. })
        ));
    }
}
