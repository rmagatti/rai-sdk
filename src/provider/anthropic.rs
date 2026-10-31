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
//!
//! With [`GenerationConfig::with_prompt_caching`] enabled, ephemeral
//! `cache_control` breakpoints go on the system prompt, on the last tool
//! definition, and on the last content block of the final conversation message:
//! at most three of the four Anthropic allows. The message breakpoint moves with
//! the conversation, so every tool round and every new turn reads what came
//! before it from the cache. Text content is sent as a block array in every
//! position, never as a bare string, so a message is rendered the same way in
//! the request where it carries the breakpoint and in every later request that
//! repeats it without one.
//!
//! [`GenerationConfig::with_reasoning_effort`] goes out as `output_config.effort`,
//! beside the structured-output `format` when both are set. Anthropic documents
//! that changing the effort between requests invalidates the cached prefix, so
//! a conversation that relies on the cache keeps one level.

use super::usage::CostDetails;

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
    generation::{GenerationConfig, ReasoningEffort},
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
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, tool_definitions);
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

        let mut messages: Vec<AnthropicMessage> = prompt
            .conversation_messages()
            .into_iter()
            .map(|message| Self::build_message(message, prompt_caching))
            .collect();
        if prompt_caching {
            // Requests are rebuilt from the whole prompt each round, so this
            // breakpoint always sits on the newest message: the next round or
            // turn reads everything up to here from the cache, and only what it
            // appends is new input.
            if let Some(last) = messages.last_mut() {
                last.mark_cache_breakpoint();
            }
        }

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
            output_config: AnthropicOutputConfig::from_config(config),
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

    fn build_message(message: &Message, prompt_caching: bool) -> AnthropicMessage {
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
                    cache_control: None,
                }]),
            };
        }

        let mut content_blocks = Vec::new();

        if message.is_multimodal() {
            content_blocks.extend(message.content_blocks.iter().map(|block| match block {
                CommonContentBlock::Text { text } => AnthropicContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                },
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
                    AnthropicContentBlock::Image {
                        source: api_source,
                        cache_control: None,
                    }
                }
                _ => unimplemented!(
                    "Audio, Video, and File blocks are not yet supported for Anthropic"
                ),
            }));
        } else if !message.content.is_empty() {
            content_blocks.push(AnthropicContentBlock::Text {
                text: message.content.clone(),
                cache_control: None,
            });
        }

        if message.role == Role::Assistant {
            content_blocks.extend(message.tool_calls.iter().map(|tool_call| {
                AnthropicContentBlock::ToolUse {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    input: tool_call.arguments.clone(),
                    cache_control: None,
                }
            }));
        }

        // A lone text block goes out as a bare string, as it always has, unless
        // caching is on. Then it stays a one-block array wherever the message
        // sits, so the request that marks it and every later request that
        // repeats it differ by the `cache_control` key alone. Anthropic does not
        // reliably match a bare string against the entry written for the marked
        // block, and the cached prefix is then written again instead of read.
        if !prompt_caching && content_blocks.len() == 1 {
            if let AnthropicContentBlock::Text { text, .. } = &content_blocks[0] {
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

        let usage = Some(Usage::from(response.usage));

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
            // Both start and delta counts are cumulative snapshots. Preserve
            // cache/cost fields when a later snapshot only reports output.
            let mut accumulated_usage: Option<Usage> = None;
            let mut usage_emitted = false;

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
                                        let usage = delta.usage.map(|u| {
                                            let accumulated = accumulated_usage.get_or_insert_with(Usage::default);
                                            accumulated.merge(u.into());
                                            accumulated.total_tokens = accumulated.prompt_tokens.zip(accumulated.completion_tokens)
                                                .and_then(|(input, output)| input.checked_add(output));
                                            usage_emitted = true;
                                            accumulated.clone()
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
                                        accumulated_usage.get_or_insert_with(Usage::default)
                                            .merge(start.message.usage.into());
                                    }
                                }
                                "message_stop" => {
                                    if !usage_emitted {
                                        if let Some(usage) = accumulated_usage.take() {
                                            yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                finish_reason: None,
                                                usage: Some(usage),
                                            });
                                        }
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicCacheControl {
    Ephemeral,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<AnthropicOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<ReasoningEffort>,
}

impl AnthropicOutputConfig {
    /// The structured-output format and the reasoning effort, or `None` when
    /// the config sets neither, so a plain request carries no `output_config`.
    fn from_config(config: &GenerationConfig) -> Option<Self> {
        let format = config
            .json_schema
            .clone()
            .map(|schema| AnthropicOutputFormat::JsonSchema { schema });
        let effort = config.reasoning_effort;
        (format.is_some() || effort.is_some()).then_some(Self { format, effort })
    }
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

impl AnthropicMessage {
    /// Puts an ephemeral cache breakpoint on the last content block that can
    /// carry one.
    ///
    /// Anthropic caches the prefix up to and including a marked block (tools,
    /// then system, then messages), and a later request whose prefix matches
    /// reads it back. `build_message` sends every text message as blocks while
    /// caching is on, so marking one only sets a field, and the message is
    /// rendered the same way once it stops being final.
    ///
    /// Leaves the message unchanged when no block can carry a breakpoint (no
    /// blocks at all, or only empty text).
    fn mark_cache_breakpoint(&mut self) {
        match &mut self.content {
            // Never produced while caching is on, the only time this runs. Do
            // not convert a string here: the message would then render
            // differently from the request that repeats it unmarked.
            AnthropicMessageContent::Text(_) => {}
            AnthropicMessageContent::Blocks(blocks) => {
                if let Some(slot) = blocks
                    .iter_mut()
                    .rev()
                    .find_map(AnthropicContentBlock::cache_control_slot)
                {
                    *slot = Some(AnthropicCacheControl::Ephemeral);
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "image")]
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

impl AnthropicContentBlock {
    /// This block's `cache_control` slot, or `None` when Anthropic cannot cache
    /// the block (empty text blocks cannot be cached).
    ///
    /// Every variant is matched by name, so a block type added later (thinking
    /// blocks, for one, cannot carry `cache_control`) has to give its own
    /// answer here.
    fn cache_control_slot(&mut self) -> Option<&mut Option<AnthropicCacheControl>> {
        match self {
            Self::Text {
                text,
                cache_control,
            } => (!text.is_empty()).then_some(cache_control),
            Self::Image { cache_control, .. }
            | Self::ToolUse { cache_control, .. }
            | Self::ToolResult { cache_control, .. } => Some(cache_control),
        }
    }
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
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
    cache_read_input_tokens: Option<i32>,
    cache_creation_input_tokens: Option<i32>,
    // OpenRouter Messages response extensions; native Anthropic omits these.
    cost: Option<f64>,
    cost_details: Option<CostDetails>,
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Self {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            total_tokens: usage
                .input_tokens
                .zip(usage.output_tokens)
                .and_then(|(input, output)| input.checked_add(output)),
            cache_read_tokens: usage.cache_read_input_tokens,
            cache_write_tokens: usage.cache_creation_input_tokens,
            cost_usd: usage.cost,
            upstream_inference_cost_usd: usage.cost_details.and_then(|d| d.upstream_inference_cost),
        }
    }
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
    usage: Option<AnthropicUsage>,
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
struct StreamError {
    error: AnthropicErrorDetail,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentBlock;
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
                format: Some(AnthropicOutputFormat::JsonSchema { schema: actual }),
                effort: None,
            }) => {
                assert_eq!(actual, schema);
            }
            _ => panic!("Expected Anthropic JSON schema output_config"),
        }
    }

    fn effort_request_body(config: &GenerationConfig, prompt: &Prompt) -> serde_json::Value {
        let request = test_provider().build_request(
            &AnthropicModel::ClaudeOpus48,
            prompt,
            config,
            false,
            None,
        );
        serde_json::to_value(request).expect("request should serialize")
    }

    #[test]
    fn reasoning_effort_is_sent_as_output_config_effort() {
        let prompt = Prompt::single(Message::user("hello"));
        for (effort, wire) in [
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::XHigh, "xhigh"),
            (ReasoningEffort::Max, "max"),
        ] {
            let body = effort_request_body(
                &GenerationConfig::new().with_reasoning_effort(effort),
                &prompt,
            );

            assert_eq!(body["output_config"], serde_json::json!({ "effort": wire }));
            // Thinking stays at the model's default; effort alone steers it.
            assert!(body.get("thinking").is_none());
            assert!(body.get("effort").is_none());
        }
    }

    #[test]
    fn reasoning_effort_sits_beside_the_structured_output_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } }
        });
        let config = GenerationConfig::new()
            .with_json_schema(schema.clone())
            .with_reasoning_effort(ReasoningEffort::Medium);

        let body = effort_request_body(&config, &Prompt::single(Message::user("hello")));

        assert_eq!(
            body["output_config"],
            serde_json::json!({
                "format": { "type": "json_schema", "schema": schema },
                "effort": "medium"
            })
        );
    }

    #[test]
    fn reasoning_effort_leaves_the_cached_prefix_unchanged() {
        // The tools, system prompt and messages are what the cache keys on; an
        // effort must add its own field and leave every byte of them alone.
        let tools = three_tools();
        let caching = GenerationConfig::new().with_prompt_caching(true);
        for len in 1..=conversation().len() {
            let prompt = prompt_of(len);
            let build = |config: &GenerationConfig| {
                let request = test_provider().build_request(
                    &AnthropicModel::ClaudeOpus48,
                    &prompt,
                    config,
                    false,
                    Some(&tools),
                );
                serde_json::to_value(request).expect("request should serialize")
            };
            let without = build(&caching);
            let mut with = build(&caching.clone().with_reasoning_effort(ReasoningEffort::Low));

            let output_config = with
                .as_object_mut()
                .expect("request is an object")
                .remove("output_config");
            assert_eq!(output_config, Some(serde_json::json!({ "effort": "low" })));
            assert_eq!(with, without, "{len} messages");
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

    /// One message of every shape `build_message` produces: string shorthand,
    /// an assistant turn with text and two tool calls, a tool result, an error
    /// tool result, an assistant string, and a multimodal user turn.
    fn conversation() -> Vec<Message> {
        let call = |id: &str, desk: &str| ToolCall {
            id: id.to_string(),
            name: "lookup".to_string(),
            arguments: serde_json::json!({ "desk": desk }),
        };
        vec![
            Message::user("Look up the ledger."),
            Message::assistant_with_tool_calls(
                "Looking it up.",
                vec![call("call_1", "HF-007"), call("call_2", "HF-008")],
            ),
            Message::tool("ledger rows", "call_1"),
            Message::tool_error("timed out", "call_2"),
            Message::assistant("Entry 40 is the largest."),
            Message::user_multimodal(vec![
                ContentBlock::text("Compare it with this chart."),
                ContentBlock::image_url("https://example.com/chart.png"),
            ]),
        ]
    }

    /// The system prompt followed by the first `len` messages of
    /// [`conversation`].
    fn prompt_of(len: usize) -> Prompt {
        let mut messages = vec![Message::system("Cache this prefix.")];
        messages.extend(conversation().into_iter().take(len));
        Prompt::new(messages)
    }

    fn three_tools() -> Vec<ToolDefinition> {
        ["first", "second", "third"]
            .map(|name| ToolDefinition {
                name: name.to_string(),
                description: None,
                input_schema: serde_json::json!({ "type": "object" }),
            })
            .to_vec()
    }

    fn request_body(
        prompt: &Prompt,
        prompt_caching: bool,
        tools: Option<&[ToolDefinition]>,
    ) -> serde_json::Value {
        let config = if prompt_caching {
            GenerationConfig::new().with_prompt_caching(true)
        } else {
            GenerationConfig::new()
        };
        let request = test_provider().build_request(
            &AnthropicModel::ClaudeSonnet45,
            prompt,
            &config,
            false,
            tools,
        );
        serde_json::to_value(request).expect("request should serialize")
    }

    /// Every `cache_control` marker anywhere in a serialized request.
    fn breakpoints(value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Object(fields) => fields
                .iter()
                .map(|(key, value)| usize::from(key == "cache_control") + breakpoints(value))
                .sum(),
            serde_json::Value::Array(items) => items.iter().map(breakpoints).sum(),
            _ => 0,
        }
    }

    /// `value` with every `cache_control` marker removed.
    fn without_cache_control(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(fields) => serde_json::Value::Object(
                fields
                    .iter()
                    .filter(|(key, _)| key.as_str() != "cache_control")
                    .map(|(key, value)| (key.clone(), without_cache_control(value)))
                    .collect(),
            ),
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(without_cache_control).collect())
            }
            other => other.clone(),
        }
    }

    /// A serialized message with string content rewritten as the one text block
    /// it stands for.
    fn string_as_text_block(message: &serde_json::Value) -> serde_json::Value {
        let mut message = message.clone();
        if let Some(text) = message["content"].as_str().map(str::to_owned) {
            message["content"] = serde_json::json!([{ "type": "text", "text": text }]);
        }
        message
    }

    fn ephemeral() -> serde_json::Value {
        serde_json::json!({ "type": "ephemeral" })
    }

    #[test]
    fn prompt_caching_marks_only_the_final_message() {
        // The next turn: the conversation so far, then a new plain user turn.
        let mut messages = vec![Message::system("Cache this prefix.")];
        messages.extend(conversation().into_iter().take(5));
        messages.push(Message::user("And the smallest entry?"));
        let prompt = Prompt::new(messages);
        let tools = three_tools();

        let cached = request_body(&prompt, true, Some(&tools));
        let uncached = request_body(&prompt, false, Some(&tools));
        let cached_messages = cached["messages"].as_array().expect("messages array");
        let uncached_messages = uncached["messages"].as_array().expect("messages array");
        let (last, earlier) = cached_messages.split_last().expect("messages present");

        // The final turn is one text block carrying the marker.
        assert_eq!(
            last,
            &serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "And the smallest entry?",
                    "cache_control": { "type": "ephemeral" }
                }]
            })
        );
        // Earlier messages carry no marker. They differ from the uncached
        // request only in sending each lone text turn as a one-block array, the
        // form it keeps from the request that marked it.
        assert_eq!(earlier.iter().map(breakpoints).sum::<usize>(), 0);
        assert_eq!(
            earlier,
            uncached_messages[..uncached_messages.len() - 1]
                .iter()
                .map(string_as_text_block)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn prompt_caching_sends_a_message_the_same_way_once_it_stops_being_final() {
        // Round one ends with two user text turns, as an assistant sends a
        // question and then its turn context. The second carries the marker.
        let round_one = vec![
            Message::system("Cache this prefix."),
            Message::user("Look up the ledger."),
            Message::user("Context: the ledger widget is open."),
        ];
        // Round two repeats them and appends one more message.
        let mut round_two = round_one.clone();
        round_two.push(Message::assistant("Entry 40 is the largest."));

        let first = request_body(&Prompt::new(round_one), true, None);
        let second = request_body(&Prompt::new(round_two), true, None);
        let marked = &first["messages"][1];
        let repeated = &second["messages"][1];

        // The repeated message is the marked one without its marker...
        assert_eq!(&without_cache_control(marked), repeated);
        // ...and both are text blocks, not strings.
        assert_eq!(
            marked,
            &serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Context: the ledger widget is open.",
                    "cache_control": { "type": "ephemeral" }
                }]
            })
        );
        assert_eq!(
            repeated,
            &serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Context: the ledger widget is open."
                }]
            })
        );

        // The question before it is unmarked, and the same, in both rounds.
        let question = serde_json::json!({
            "role": "user",
            "content": [{ "type": "text", "text": "Look up the ledger." }]
        });
        assert_eq!(first["messages"][0], question);
        assert_eq!(second["messages"][0], question);
    }

    #[test]
    fn prompt_caching_renders_every_message_shape_the_same_marked_or_not() {
        let mut messages = conversation();
        messages.push(Message::user("And the smallest entry?"));
        let prefix = |len: usize| {
            let mut prompt = vec![Message::system("Cache this prefix.")];
            prompt.extend(messages[..len].iter().cloned());
            Prompt::new(prompt)
        };

        for len in 1..messages.len() {
            // Message `len - 1` is final, and marked, in the first request and
            // repeated unmarked in the second, which appends one more message.
            let ending_here = request_body(&prefix(len), true, None);
            let one_more = request_body(&prefix(len + 1), true, None);
            let marked = &ending_here["messages"][len - 1];
            let repeated = &one_more["messages"][len - 1];
            assert_eq!(
                &without_cache_control(marked),
                repeated,
                "message {len} must differ only by its marker"
            );
        }

        // With caching on no message is a bare string, which cannot be marked.
        let body = request_body(&prefix(messages.len()), true, None);
        let rendered = body["messages"].as_array().expect("messages array");
        for (index, message) in rendered.iter().enumerate() {
            assert!(
                message["content"].is_array(),
                "message {index} should be a block array: {message}"
            );
        }
    }

    #[test]
    fn prompt_caching_marks_the_last_tool_result_of_a_round() {
        // An agent-loop round: two tool calls, each result in its own
        // user-role message, and nothing after them.
        let body = request_body(&prompt_of(4), true, None);
        let messages = body["messages"].as_array().expect("messages array");

        assert_eq!(
            messages[3],
            serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "timed out",
                    "is_error": true,
                    "cache_control": { "type": "ephemeral" }
                }]
            })
        );
        assert_eq!(messages[..3].iter().map(breakpoints).sum::<usize>(), 0);
    }

    #[test]
    fn prompt_caching_marks_a_final_assistant_turn() {
        let body = request_body(&prompt_of(5), true, None);
        let messages = body["messages"].as_array().expect("messages array");

        assert_eq!(
            messages[4],
            serde_json::json!({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "Entry 40 is the largest.",
                    "cache_control": { "type": "ephemeral" }
                }]
            })
        );
        assert_eq!(messages[..4].iter().map(breakpoints).sum::<usize>(), 0);
    }

    #[test]
    fn prompt_caching_marks_the_last_block_of_a_multi_block_turn() {
        let body = request_body(&prompt_of(6), true, None);
        let blocks = body["messages"][5]["content"]
            .as_array()
            .expect("multimodal turn is a block array");

        assert!(blocks[0].get("cache_control").is_none());
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["cache_control"], ephemeral());
    }

    #[test]
    fn prompt_caching_skips_empty_text_blocks() {
        // Anthropic cannot cache an empty text block, so the marker goes on the
        // nearest earlier block of the final message that can carry it...
        let trailing_empty = Prompt::new(vec![Message::user_multimodal(vec![
            ContentBlock::image_url("https://example.com/chart.png"),
            ContentBlock::text(""),
        ])]);
        let body = request_body(&trailing_empty, true, None);
        let blocks = body["messages"][0]["content"]
            .as_array()
            .expect("multimodal turn is a block array");
        assert_eq!(blocks[0]["cache_control"], ephemeral());
        assert!(blocks[1].get("cache_control").is_none());

        // ...and a final message with nothing cacheable carries no marker.
        let empty_text = Prompt::new(vec![Message::user_multimodal(vec![ContentBlock::text("")])]);
        assert_eq!(
            breakpoints(&request_body(&empty_text, true, None)["messages"]),
            0
        );
    }

    #[test]
    fn prompt_caching_never_exceeds_four_breakpoints() {
        // Anthropic accepts at most four cache_control markers per request.
        // The system prompt, the last tool and the final message use three,
        // whatever the conversation ends with.
        let tools = three_tools();
        for len in 1..=conversation().len() {
            let with_prefix = breakpoints(&request_body(&prompt_of(len), true, Some(&tools)));
            assert!(
                with_prefix <= 4,
                "{with_prefix} breakpoints, {len} messages"
            );
            assert_eq!(
                with_prefix, 3,
                "system, last tool, final message ({len} messages)"
            );

            let bare = Prompt::new(conversation().into_iter().take(len).collect());
            assert_eq!(breakpoints(&request_body(&bare, true, None)), 1);
        }
    }

    #[test]
    fn prompt_caching_off_keeps_the_conversation_wire_bytes() {
        let request = test_provider().build_request(
            &AnthropicModel::ClaudeSonnet45,
            &prompt_of(6),
            &GenerationConfig::new(),
            false,
            None,
        );

        // Today's bytes, captured before message breakpoints existed: with
        // caching off, no block gains a field and no string becomes an array.
        let expected = concat!(
            r#"[{"role":"user","content":"Look up the ledger."},"#,
            r#"{"role":"assistant","content":[{"type":"text","text":"Looking it up."},"#,
            r#"{"type":"tool_use","id":"call_1","name":"lookup","input":{"desk":"HF-007"}},"#,
            r#"{"type":"tool_use","id":"call_2","name":"lookup","input":{"desk":"HF-008"}}]},"#,
            r#"{"role":"user","content":["#,
            r#"{"type":"tool_result","tool_use_id":"call_1","content":"ledger rows"}]},"#,
            r#"{"role":"user","content":["#,
            r#"{"type":"tool_result","tool_use_id":"call_2","content":"timed out","is_error":true}]},"#,
            r#"{"role":"assistant","content":"Entry 40 is the largest."},"#,
            r#"{"role":"user","content":[{"type":"text","text":"Compare it with this chart."},"#,
            r#"{"type":"image","source":{"type":"url","url":"https://example.com/chart.png"}}]}]"#,
        );
        assert_eq!(
            serde_json::to_string(&request.messages).expect("messages should serialize"),
            expected
        );
    }
}
