//! Provider for endpoints that speak the OpenAI Chat Completions wire format.
//!
//! Ollama, vLLM, LM Studio, llama.cpp's server, LocalAI, Text Generation
//! Inference and most inference gateways all expose
//! `POST {base_url}/chat/completions` with OpenAI's request and response
//! shapes. This provider targets that format rather than any one service:
//! the endpoint is named per client, so a process can hold several clients
//! pointed at different servers at once.
//!
//! ```no_run
//! use rai_sdk::{ClientBuilder, Model};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ClientBuilder::new()
//!     .ollama()
//!     .model(Model::openai_compatible("llama3.1:8b"))
//!     .build()?;
//!
//! let response = client.request().prompt("Hello").generate().await?;
//! # println!("{}", response.text());
//! # Ok(())
//! # }
//! ```
//!
//! # How it differs from the OpenAI provider
//!
//! The request builder and the SSE parser are shared with
//! [`openai`](super::openai) — the wire format is the same, and forking it
//! would mean two parsers drifting apart. What differs is everything around it:
//!
//! - **Authentication is optional.** Local runtimes usually accept no
//!   credential. When no key is configured no `Authorization` header is sent at
//!   all, rather than a placeholder key being invented.
//! - **The endpoint is required.** There is no default base URL and no
//!   environment variable; a client without one simply has no compatible
//!   provider.
//! - **Model IDs are free-form.** [`OpenAICompatibleModel`] wraps whatever the
//!   server calls the model, with no catalog to match against.
//! - **`strict` JSON schemas are not requested.** OpenAI's strict mode is a
//!   hard guarantee worth asking for; third-party endpoints implement it
//!   unevenly, some rejecting the flag outright. Structured output is validated
//!   against the schema client-side either way, so the flag is sent as `false`.
//! - **Capability gaps are typed.** See below.
//!
//! The shared stream parser already tolerates the framing divergences that show
//! up across self-hosted servers: `data:` with no space after the colon (the
//! SSE specification makes it optional), a missing `[DONE]` sentinel, absent
//! usage when `stream_options.include_usage` is ignored, and tool-call
//! continuation deltas that omit the call id. Payloads it cannot parse are
//! logged and skipped, so one unrecognized frame cannot truncate a response.
//!
//! # Capability degradation
//!
//! "OpenAI-compatible" describes a wire format, not a feature set. A 3B model
//! behind Ollama may not call tools; a runtime may not constrain output to a
//! JSON Schema. Both cases surface as
//! [`Error::CapabilityUnsupported`] — a
//! variant distinct from the generic HTTP and request errors — so a caller can
//! branch on [`Error::unsupported_capability`](crate::Error::unsupported_capability)
//! and fall back instead of parsing an error string.
//!
//! That error arrives by either of two routes:
//!
//! 1. **Declared.** [`EndpointCapabilities`] says the endpoint lacks the
//!    capability, and the request is refused locally before any HTTP call.
//! 2. **Observed.** The endpoint itself rejects a request that used the
//!    capability, and its own message is classified. Classification only
//!    applies when the request actually carried tools or a `response_format`,
//!    so an unrelated bad request stays an ordinary
//!    [`Error::InvalidRequest`].
//!
//! Nothing is probed: capabilities are declared by the caller, never
//! auto-detected.

use std::pin::Pin;

use futures::Stream;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use tracing::{debug, error, info, instrument};

use super::openai::{ChatRequestOptions, build_chat_request, parse_chat_completions_sse};
use crate::{
    config::{Config, EndpointCapabilities},
    error::{Capability, Error, ProviderKind, Result},
    message::{Message, Prompt, Response, ToolCall, ToolDefinition, Usage},
    model::OpenAICompatibleModel,
};

/// Provider for a single OpenAI-compatible endpoint.
///
/// Stores the endpoint, its optional bearer token, the capabilities it was
/// declared to have, and the HTTP client, all fixed at construction time. One
/// instance serves one endpoint; point a second [`Client`](crate::Client) at a
/// second URL to use both.
pub struct OpenAICompatibleProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    capabilities: EndpointCapabilities,
}

impl std::fmt::Debug for OpenAICompatibleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleProvider")
            .field("base_url", &self.base_url)
            .field(
                "api_key",
                match &self.api_key {
                    Some(_) => &"[REDACTED]",
                    None => &"[UNSET]",
                },
            )
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl OpenAICompatibleProvider {
    /// Create a provider for the endpoint named in `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ProviderNotConfigured`] if no base URL is configured —
    /// unlike the other providers, a missing API key is not an error — or
    /// [`Error::Config`] if the HTTP client cannot be built.
    pub fn new(config: &Config) -> Result<Self> {
        let base_url = config
            .openai_compatible_base_url()
            .ok_or(Error::ProviderNotConfigured(ProviderKind::OpenAICompatible))?;

        let client = super::http_client_builder()
            .timeout(std::time::Duration::from_secs(config.timeout()))
            .build()
            .map_err(|e| Error::Config(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url,
            api_key: config.openai_compatible_key(),
            capabilities: config.openai_compatible_capabilities(),
        })
    }

    /// The endpoint this provider talks to.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// What this endpoint was declared to support.
    pub fn capabilities(&self) -> EndpointCapabilities {
        self.capabilities
    }

    /// Generate a completion (non-streaming).
    ///
    /// # Errors
    ///
    /// Returns [`Error::CapabilityUnsupported`] when the request needs tool
    /// calling or structured output and the endpoint cannot provide it,
    /// [`Error::Http`] if the request cannot be sent,
    /// [`Error::Auth`]/[`Error::RateLimit`]/[`Error::InvalidRequest`]/[`Error::Request`]
    /// depending on the status code the endpoint replies with, and
    /// [`Error::Request`] if the response contains no choices or a tool call
    /// with unparseable arguments.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate(
        &self,
        model: &OpenAICompatibleModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Response> {
        info!(
            model = model.as_str(),
            base_url = %self.base_url,
            "Generating completion with an OpenAI-compatible endpoint"
        );

        let requested = RequestedCapabilities::of(config, tool_definitions);
        self.check_declared_capabilities(requested)?;

        let request = self.build_request(model, prompt, config, false, tool_definitions);
        let url = self.chat_completions_url();

        debug!(url = %url, "Sending request to the OpenAI-compatible endpoint");
        debug!(request_payload = ?request, "OpenAI-compatible request payload");

        let response = self.authorized(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenAI-compatible request failed");
            return Err(self.parse_error(status.as_u16(), &error_body, requested));
        }

        let response_body: ChatCompletionsResponse = response.json().await?;
        debug!(response_payload = ?response_body, "OpenAI-compatible response payload");
        self.parse_response(model, response_body)
    }

    /// Generate a completion with streaming.
    ///
    /// The returned stream yields
    /// [`ProviderStreamEvent`](crate::provider::ProviderStreamEvent) values and
    /// ends on the `[DONE]` sentinel or when the endpoint closes the body,
    /// whichever comes first — self-hosted servers do not all send the
    /// sentinel.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`OpenAICompatibleProvider::generate`] while
    /// opening the stream. Transport failures encountered mid-stream are
    /// yielded as [`Error::Stream`] items.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_stream(
        &self,
        model: &OpenAICompatibleModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        // Streaming never advertises tools — the crate rejects tool-bearing
        // streaming requests earlier — but `response_format` still rides along.
        let requested = RequestedCapabilities::of(config, None);
        self.check_declared_capabilities(requested)?;

        let request = self.build_request(model, prompt, config, true, None);
        let url = self.chat_completions_url();

        debug!(url = %url, "Sending streaming request to the OpenAI-compatible endpoint");

        let response = self.authorized(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenAI-compatible streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body, requested));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(parse_chat_completions_sse(byte_stream)))
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Start a POST, adding `Authorization` only when a key was configured.
    ///
    /// Local endpoints typically reject nothing and require nothing; sending a
    /// made-up bearer token would be indistinguishable from a real credential
    /// leaking into logs on the far side.
    fn authorized(&self, url: &str) -> RequestBuilder {
        let request = self
            .client
            .post(url)
            .header("Content-Type", "application/json");

        match &self.api_key {
            Some(key) => request.header("Authorization", format!("Bearer {key}")),
            None => request,
        }
    }

    fn build_request(
        &self,
        model: &OpenAICompatibleModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
        stream: bool,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> super::openai::OpenAIRequest {
        build_chat_request(
            prompt,
            config,
            ChatRequestOptions {
                model: model.as_str(),
                stream,
                tool_definitions,
                // There is no reasoning-model catalog to consult here, and no
                // compatible endpoint is known to reject sampling parameters.
                drop_sampling_params: false,
                strict_json_schema: false,
            },
        )
    }

    /// Refuse a request the endpoint was declared unable to serve.
    fn check_declared_capabilities(&self, requested: RequestedCapabilities) -> Result<()> {
        if requested.tool_calling && !self.capabilities.tool_calling {
            return Err(self.capability_unsupported(
                Capability::ToolCalling,
                "the endpoint was configured as not supporting tool calling",
            ));
        }

        if requested.structured_output && !self.capabilities.structured_output {
            return Err(self.capability_unsupported(
                Capability::StructuredOutput,
                "the endpoint was configured as not supporting structured output",
            ));
        }

        Ok(())
    }

    fn capability_unsupported(&self, capability: Capability, message: &str) -> Error {
        Error::CapabilityUnsupported {
            provider: ProviderKind::OpenAICompatible,
            capability,
            base_url: self.base_url.clone(),
            message: message.to_string(),
        }
    }

    fn parse_response(
        &self,
        model: &OpenAICompatibleModel,
        response: ChatCompletionsResponse,
    ) -> Result<Response> {
        let choice = response.choices.first().ok_or_else(|| Error::Request {
            provider: ProviderKind::OpenAICompatible,
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
                        provider: ProviderKind::OpenAICompatible,
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
            "Received response from the OpenAI-compatible endpoint"
        );

        Ok(Response {
            messages: vec![Message::assistant_with_tool_calls(content, tool_calls)],
            usage,
            model: model.as_str().to_string(),
            provider: ProviderKind::OpenAICompatible,
            finish_reason: choice.finish_reason.clone(),
        })
    }

    /// Map an unsuccessful HTTP response onto the crate's error type.
    ///
    /// Before the usual status-code mapping, a rejection that names something
    /// the request actually asked for — tools, or a `response_format` — is
    /// reported as [`Error::CapabilityUnsupported`] rather than as a generic
    /// bad request, since "this endpoint cannot do that" and "you sent
    /// nonsense" call for different handling.
    fn parse_error(&self, status: u16, body: &str, requested: RequestedCapabilities) -> Error {
        let message = super::openai::parse_error_message(body);

        if let Some(message) = &message {
            if CAPABILITY_REJECTION_STATUSES.contains(&status) {
                if requested.tool_calling && mentions_unsupported(message, TOOL_SUBJECTS) {
                    return self.capability_unsupported(Capability::ToolCalling, message);
                }
                if requested.structured_output
                    && mentions_unsupported(message, STRUCTURED_OUTPUT_SUBJECTS)
                {
                    return self.capability_unsupported(Capability::StructuredOutput, message);
                }
            }
        }

        let Some(message) = message else {
            return Error::Request {
                provider: ProviderKind::OpenAICompatible,
                message: format!("HTTP {status}: {body}"),
            };
        };

        match status {
            401 | 403 => Error::Auth {
                provider: ProviderKind::OpenAICompatible,
                message,
            },
            429 => Error::RateLimit {
                provider: ProviderKind::OpenAICompatible,
                message,
            },
            400 => Error::InvalidRequest(message),
            _ => Error::Request {
                provider: ProviderKind::OpenAICompatible,
                message,
            },
        }
    }
}

/// Which optional capabilities a single request depends on.
///
/// Capability errors are only raised for what the request actually used, so a
/// text-only call against a text-only endpoint never trips them.
#[derive(Debug, Clone, Copy)]
struct RequestedCapabilities {
    tool_calling: bool,
    structured_output: bool,
}

impl RequestedCapabilities {
    fn of(
        config: &crate::generation::GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Self {
        Self {
            tool_calling: tool_definitions.is_some_and(|tools| !tools.is_empty()),
            structured_output: config.json_schema.is_some() || config.json_mode == Some(true),
        }
    }
}

/// Statuses a server plausibly uses to say "I do not implement that".
///
/// Deliberately excludes 5xx other than 501: a crashing endpoint is not an
/// endpoint that lacks a feature, and misreporting one as the other would send
/// callers down a permanent fallback path over a transient fault.
const CAPABILITY_REJECTION_STATUSES: &[u16] = &[400, 404, 422, 501];

const TOOL_SUBJECTS: &[&str] = &["tool", "function call", "function_call"];

const STRUCTURED_OUTPUT_SUBJECTS: &[&str] = &[
    "response_format",
    "response format",
    "json_schema",
    "json schema",
    "structured output",
    "guided decoding",
    "grammar",
];

/// Whether `message` says one of `subjects` is unavailable.
///
/// Both halves must match: naming the subject alone would classify "tool
/// arguments were invalid" as a missing capability.
fn mentions_unsupported(message: &str, subjects: &[&str]) -> bool {
    const UNAVAILABLE: &[&str] = &[
        "not support",
        "unsupported",
        "not implemented",
        "unrecognized",
        "unknown",
        "no support",
        "does not accept",
    ];

    let message = message.to_ascii_lowercase();

    subjects.iter().any(|subject| message.contains(subject))
        && UNAVAILABLE.iter().any(|marker| message.contains(marker))
}

// ── Response types ─────────────────────────────────────────────────────────
//
// Structurally identical to OpenAI's, which is the point of the format. They
// are named separately here so this provider's parsing does not silently
// inherit an OpenAI-specific change to those types.

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatCompletionsChoice>,
    usage: Option<ChatCompletionsUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsChoice {
    message: ChatCompletionsMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionsToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsToolCall {
    id: String,
    function: ChatCompletionsFunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionsFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::GenerationConfig;

    fn test_provider(capabilities: EndpointCapabilities) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider {
            client: Client::new(),
            base_url: "http://localhost:11434/v1".to_string(),
            api_key: None,
            capabilities,
        }
    }

    fn tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: None,
            input_schema: serde_json::json!({ "type": "object" }),
        }
    }

    #[test]
    fn requested_capabilities_track_what_the_request_uses() {
        let plain = RequestedCapabilities::of(&GenerationConfig::new(), None);
        assert!(!plain.tool_calling);
        assert!(!plain.structured_output);

        let empty_tools = RequestedCapabilities::of(&GenerationConfig::new(), Some(&[]));
        assert!(!empty_tools.tool_calling);

        let with_tools = RequestedCapabilities::of(&GenerationConfig::new(), Some(&[tool()]));
        assert!(with_tools.tool_calling);

        let json_mode =
            RequestedCapabilities::of(&GenerationConfig::new().with_json_mode(true), None);
        assert!(json_mode.structured_output);

        let schema = RequestedCapabilities::of(
            &GenerationConfig::new().with_json_schema(serde_json::json!({ "type": "object" })),
            None,
        );
        assert!(schema.structured_output);
    }

    #[test]
    fn declared_gaps_are_refused_before_any_request() {
        let provider = test_provider(EndpointCapabilities::text_only());

        let error = provider
            .check_declared_capabilities(RequestedCapabilities::of(
                &GenerationConfig::new(),
                Some(&[tool()]),
            ))
            .expect_err("tool calling should be refused");
        assert_eq!(
            error.unsupported_capability(),
            Some(Capability::ToolCalling)
        );

        let error = provider
            .check_declared_capabilities(RequestedCapabilities::of(
                &GenerationConfig::new().with_json_mode(true),
                None,
            ))
            .expect_err("structured output should be refused");
        assert_eq!(
            error.unsupported_capability(),
            Some(Capability::StructuredOutput)
        );
    }

    #[test]
    fn a_fully_capable_endpoint_refuses_nothing() {
        let provider = test_provider(EndpointCapabilities::default());

        provider
            .check_declared_capabilities(RequestedCapabilities::of(
                &GenerationConfig::new().with_json_mode(true),
                Some(&[tool()]),
            ))
            .expect("a fully capable endpoint should accept everything");
    }

    #[test]
    fn unsupported_markers_need_both_a_subject_and_a_denial() {
        assert!(mentions_unsupported(
            "registry/llama3.2:1b does not support tools",
            TOOL_SUBJECTS
        ));
        assert!(mentions_unsupported(
            "unsupported parameter: response_format",
            STRUCTURED_OUTPUT_SUBJECTS
        ));

        // Naming the subject is not enough on its own.
        assert!(!mentions_unsupported(
            "invalid arguments for tool 'get_weather'",
            TOOL_SUBJECTS
        ));
        // Neither is a denial about something else.
        assert!(!mentions_unsupported(
            "unknown model 'llama3.1:8b'",
            TOOL_SUBJECTS
        ));
    }

    #[test]
    fn capability_classification_only_applies_to_what_was_sent() {
        let provider = test_provider(EndpointCapabilities::default());
        let body = serde_json::json!({
            "error": { "message": "this model does not support tools" }
        })
        .to_string();

        let with_tools = provider.parse_error(
            400,
            &body,
            RequestedCapabilities {
                tool_calling: true,
                structured_output: false,
            },
        );
        assert_eq!(
            with_tools.unsupported_capability(),
            Some(Capability::ToolCalling)
        );

        let without_tools = provider.parse_error(
            400,
            &body,
            RequestedCapabilities {
                tool_calling: false,
                structured_output: false,
            },
        );
        assert!(without_tools.unsupported_capability().is_none());
        assert!(matches!(without_tools, Error::InvalidRequest(_)));
    }

    #[test]
    fn transport_failures_are_not_mistaken_for_capability_gaps() {
        let provider = test_provider(EndpointCapabilities::default());
        let body = serde_json::json!({
            "error": { "message": "tools are not supported right now" }
        })
        .to_string();

        let error = provider.parse_error(
            503,
            &body,
            RequestedCapabilities {
                tool_calling: true,
                structured_output: false,
            },
        );

        assert!(error.unsupported_capability().is_none());
        assert!(matches!(error, Error::Request { .. }));
    }

    #[test]
    fn non_json_error_bodies_keep_their_status_and_text() {
        let provider = test_provider(EndpointCapabilities::default());

        let error = provider.parse_error(
            502,
            "<html>bad gateway</html>",
            RequestedCapabilities {
                tool_calling: false,
                structured_output: false,
            },
        );

        assert!(error.to_string().contains("HTTP 502"));
    }

    #[test]
    fn structured_output_does_not_request_strict_schemas() {
        let provider = test_provider(EndpointCapabilities::default());
        let prompt = Prompt::single(Message::user("hello"));
        let config =
            GenerationConfig::new().with_json_schema(serde_json::json!({ "type": "object" }));

        let request = provider.build_request(
            &OpenAICompatibleModel::new("llama3.1:8b"),
            &prompt,
            &config,
            false,
            None,
        );

        let body = serde_json::to_value(&request).expect("request should serialize");
        assert_eq!(body["response_format"]["json_schema"]["strict"], false);
        assert_eq!(body["model"], "llama3.1:8b");
    }
}
