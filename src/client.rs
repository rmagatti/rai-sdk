use std::{any::type_name, marker::PhantomData, pin::Pin};

use futures::{Stream, StreamExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tracing::{debug, error, info, instrument};

use crate::{
    config::Config,
    error::{Error, ProviderKind, Result},
    generation::GenerationConfig,
    message::{Message, Prompt, Response, StructuredOutput, ToolDefinition},
    model::Model,
    retry::RetryConfig,
    tool::{Tool, ToolContext, ToolRegistry},
};

#[cfg(feature = "openai")]
use crate::provider::OpenAIProvider;

#[cfg(feature = "anthropic")]
use crate::provider::AnthropicProvider;

#[cfg(feature = "openrouter")]
use crate::provider::OpenRouterProvider;

#[derive(Clone, Copy)]
enum ToolAvailability {
    Enabled,
    IgnoredForStructuredOnce,
}

#[doc(hidden)]
pub struct ModelMissing;

#[doc(hidden)]
pub struct ModelReady;

/// Unified AI client supporting multiple providers.
pub struct Client<ModelState = ModelMissing> {
    config: Config,
    default_model: Option<Model>,
    default_config: GenerationConfig,
    default_retry_config: RetryConfig,
    tool_registry: ToolRegistry,
    state: PhantomData<ModelState>,

    #[cfg(feature = "openai")]
    openai: Option<OpenAIProvider>,

    #[cfg(feature = "anthropic")]
    anthropic: Option<AnthropicProvider>,

    #[cfg(feature = "openrouter")]
    openrouter: Option<OpenRouterProvider>,
}

impl<ModelState> std::fmt::Debug for Client<ModelState> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Client");
        s.field("default_model", &self.default_model);
        s.field("default_config", &self.default_config);

        #[cfg(feature = "openai")]
        s.field("openai", &self.openai);

        #[cfg(feature = "anthropic")]
        s.field("anthropic", &self.anthropic);

        #[cfg(feature = "openrouter")]
        s.field("openrouter", &self.openrouter);

        s.finish()
    }
}

impl Client<ModelMissing> {
    /// Create a new AI client with the given configuration.
    pub fn new(config: Config) -> Result<Self> {
        let default_retry_config = config.retry_config();
        Self::new_with_defaults(
            config,
            None,
            GenerationConfig::default(),
            default_retry_config,
            ToolRegistry::new(),
        )
    }

    /// Create a builder for configuring a client with defaults.
    pub fn builder() -> ClientBuilder<ModelMissing> {
        ClientBuilder::new()
    }

    /// Create a request builder for a single call.
    pub fn request(&self) -> RequestBuilder<'_, PromptMissing, ModelMissing, ModelMissing> {
        self.request_builder()
    }
}

impl<ModelState> Client<ModelState> {
    fn request_builder(&self) -> RequestBuilder<'_, PromptMissing, ModelState, ModelState> {
        RequestBuilder::new(self)
    }

    fn new_with_defaults(
        config: Config,
        default_model: Option<Model>,
        default_config: GenerationConfig,
        default_retry_config: RetryConfig,
        tool_registry: ToolRegistry,
    ) -> Result<Self> {
        info!("Initializing AI client");

        #[cfg(feature = "openai")]
        let openai = if config.openai_key().is_some() {
            match OpenAIProvider::new(&config) {
                Ok(provider) => {
                    info!("OpenAI provider initialized");
                    Some(provider)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize OpenAI provider");
                    None
                }
            }
        } else {
            tracing::debug!("OpenAI API key not configured, provider disabled");
            None
        };

        #[cfg(feature = "anthropic")]
        let anthropic = if config.anthropic_key().is_some() {
            match AnthropicProvider::new(&config) {
                Ok(provider) => {
                    info!("Anthropic provider initialized");
                    Some(provider)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize Anthropic provider");
                    None
                }
            }
        } else {
            tracing::debug!("Anthropic API key not configured, provider disabled");
            None
        };

        #[cfg(feature = "openrouter")]
        let openrouter = if config.openrouter_key().is_some() {
            match OpenRouterProvider::new(&config) {
                Ok(provider) => {
                    info!("OpenRouter provider initialized");
                    Some(provider)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to initialize OpenRouter provider");
                    None
                }
            }
        } else {
            tracing::debug!("OpenRouter API key not configured, provider disabled");
            None
        };

        info!("AI client initialized successfully");

        Ok(Self {
            config,
            default_model,
            default_config,
            default_retry_config,
            tool_registry,
            state: PhantomData,
            #[cfg(feature = "openai")]
            openai,
            #[cfg(feature = "anthropic")]
            anthropic,
            #[cfg(feature = "openrouter")]
            openrouter,
        })
    }

    async fn generate_with_tools(
        &self,
        model: Model,
        prompt: &Prompt,
        config: &GenerationConfig,
        retry_config: &RetryConfig,
        tool_registry: &ToolRegistry,
    ) -> Result<Response> {
        let Some(tool_definitions) =
            (!tool_registry.is_empty()).then(|| tool_registry.definitions())
        else {
            return crate::retry::with_retry(retry_config, "generate", || {
                self.generate_once_internal(model.clone(), prompt, config, None)
            })
            .await;
        };

        let mut prompt_with_tools = prompt.clone();
        let max_rounds = config.tool_round_limit();

        for round in 0..max_rounds {
            let response = crate::retry::with_retry(retry_config, "generate", || {
                self.generate_once_internal(
                    model.clone(),
                    &prompt_with_tools,
                    config,
                    Some(&tool_definitions),
                )
            })
            .await?;

            let tool_calls: Vec<_> = response
                .messages
                .iter()
                .flat_map(|message| message.tool_calls.iter().cloned())
                .collect();

            if tool_calls.is_empty() {
                return Ok(response);
            }

            prompt_with_tools.messages.extend(response.messages.clone());

            for tool_call in tool_calls {
                let tool_message = tool_registry
                    .execute(
                        &tool_call,
                        ToolContext {
                            provider: model.provider(),
                            model: model.as_str().to_string(),
                            round,
                            tool_name: tool_call.name.clone(),
                            tool_call_id: tool_call.id.clone(),
                        },
                    )
                    .await?;
                prompt_with_tools.messages.push(tool_message);
            }
        }

        Err(Error::ToolLoopLimitExceeded { max_rounds })
    }

    async fn generate_once_internal(
        &self,
        model: Model,
        prompt: &Prompt,
        config: &GenerationConfig,
        tool_definitions: Option<&[ToolDefinition]>,
    ) -> Result<Response> {
        match model {
            #[cfg(feature = "openai")]
            Model::OpenAI(ref openai_model) => {
                let provider = self
                    .openai
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::OpenAI))?;
                provider
                    .generate(openai_model, prompt, config, tool_definitions)
                    .await
            }

            #[cfg(feature = "anthropic")]
            Model::Anthropic(ref anthropic_model) => {
                let provider = self
                    .anthropic
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::Anthropic))?;
                provider
                    .generate(anthropic_model, prompt, config, tool_definitions)
                    .await
            }

            #[cfg(feature = "openrouter")]
            Model::OpenRouter(ref openrouter_model) => {
                let provider = self
                    .openrouter
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::OpenRouter))?;
                provider
                    .generate(openrouter_model, prompt, config, tool_definitions)
                    .await
            }

            #[allow(unreachable_patterns)]
            _ => Err(Error::ProviderNotEnabled(model.provider())),
        }
    }

    /// Generate a completion with streaming.
    #[instrument(skip(self, prompt, config))]
    pub async fn generate_stream(
        &self,
        model: Model,
        prompt: &Prompt,
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        if !self.tool_registry.is_empty() {
            return Err(Error::InvalidRequest(
                "Streaming with tools is not supported".into(),
            ));
        }

        match model {
            #[cfg(feature = "openai")]
            Model::OpenAI(ref openai_model) => {
                let provider = self
                    .openai
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::OpenAI))?;
                provider.generate_stream(openai_model, prompt, config).await
            }

            #[cfg(feature = "anthropic")]
            Model::Anthropic(ref anthropic_model) => {
                let provider = self
                    .anthropic
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::Anthropic))?;
                provider
                    .generate_stream(anthropic_model, prompt, config)
                    .await
            }

            #[cfg(feature = "openrouter")]
            Model::OpenRouter(ref openrouter_model) => {
                let provider = self
                    .openrouter
                    .as_ref()
                    .ok_or_else(|| Error::ProviderNotConfigured(ProviderKind::OpenRouter))?;
                provider
                    .generate_stream(openrouter_model, prompt, config)
                    .await
            }

            #[allow(unreachable_patterns)]
            _ => Err(Error::ProviderNotEnabled(model.provider())),
        }
    }

    /// Check if a provider is available.
    pub fn is_provider_available(&self, provider: ProviderKind) -> bool {
        match provider {
            #[cfg(feature = "openai")]
            ProviderKind::OpenAI => self.openai.is_some(),

            #[cfg(feature = "anthropic")]
            ProviderKind::Anthropic => self.anthropic.is_some(),

            #[cfg(feature = "openrouter")]
            ProviderKind::OpenRouter => self.openrouter.is_some(),

            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl Client<ModelReady> {
    /// Create a request builder that inherits this client's default model.
    pub fn request(&self) -> RequestBuilder<'_, PromptMissing, ModelReady, ModelReady> {
        self.request_builder()
    }
}

enum ToolOverride {
    Inherit,
    Replace(Vec<Tool>),
    Append(Vec<Tool>),
    None,
}

struct ResolvedRequest {
    model: Model,
    config: GenerationConfig,
    retry_config: RetryConfig,
    tool_registry: ToolRegistry,
}

#[doc(hidden)]
pub struct PromptMissing;

#[doc(hidden)]
pub struct PromptReady;

/// Builder for a single AI generation request.
pub struct RequestBuilder<
    'a,
    PromptState = PromptMissing,
    RequestModelState = ModelMissing,
    ClientModelState = ModelMissing,
> {
    client: &'a Client<ClientModelState>,
    model: Option<Model>,
    config: Option<GenerationConfig>,
    retry_config: Option<RetryConfig>,
    prompt: Option<Prompt>,
    tool_override: ToolOverride,
    prompt_state: PhantomData<PromptState>,
    model_state: PhantomData<RequestModelState>,
}

impl<'a, ClientModelState> RequestBuilder<'a, PromptMissing, ClientModelState, ClientModelState> {
    fn new(client: &'a Client<ClientModelState>) -> Self {
        Self {
            client,
            model: None,
            config: None,
            retry_config: None,
            prompt: None,
            tool_override: ToolOverride::Inherit,
            prompt_state: PhantomData,
            model_state: PhantomData,
        }
    }
}

impl<'a, PromptState, RequestModelState, ClientModelState>
    RequestBuilder<'a, PromptState, RequestModelState, ClientModelState>
{
    fn with_prompt_state<NextPromptState>(
        self,
    ) -> RequestBuilder<'a, NextPromptState, RequestModelState, ClientModelState> {
        RequestBuilder {
            client: self.client,
            model: self.model,
            config: self.config,
            retry_config: self.retry_config,
            prompt: self.prompt,
            tool_override: self.tool_override,
            prompt_state: PhantomData,
            model_state: PhantomData,
        }
    }

    fn with_model_state<NextRequestModelState>(
        self,
    ) -> RequestBuilder<'a, PromptState, NextRequestModelState, ClientModelState> {
        RequestBuilder {
            client: self.client,
            model: self.model,
            config: self.config,
            retry_config: self.retry_config,
            prompt: self.prompt,
            tool_override: self.tool_override,
            prompt_state: PhantomData,
            model_state: PhantomData,
        }
    }

    /// Override the model for this request.
    pub fn model(
        mut self,
        model: Model,
    ) -> RequestBuilder<'a, PromptState, ModelReady, ClientModelState> {
        self.model = Some(model);
        self.with_model_state()
    }

    /// Override generation settings for this request.
    pub fn config(mut self, config: GenerationConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Override the retry configuration for this request.
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    /// Disable retries for this request.
    pub fn no_retry(mut self) -> Self {
        self.retry_config = Some(RetryConfig::none());
        self
    }

    /// Set the prompt or conversation history for this request.
    pub fn prompt<P>(
        mut self,
        prompt: P,
    ) -> RequestBuilder<'a, PromptReady, RequestModelState, ClientModelState>
    where
        P: Into<Prompt>,
    {
        self.prompt = Some(prompt.into());
        self.with_prompt_state()
    }

    /// Replace inherited tools with a single request-specific tool.
    pub fn tool(mut self, tool: Tool) -> Self {
        match &mut self.tool_override {
            ToolOverride::Replace(tools) => tools.push(tool),
            _ => self.tool_override = ToolOverride::Replace(vec![tool]),
        }
        self
    }

    /// Replace inherited tools with a custom set for this request.
    pub fn tools<T>(mut self, tools: T) -> Self
    where
        T: IntoIterator<Item = Tool>,
    {
        let mut collected: Vec<_> = tools.into_iter().collect();

        match &mut self.tool_override {
            ToolOverride::Replace(existing) => existing.append(&mut collected),
            _ => self.tool_override = ToolOverride::Replace(collected),
        }

        self
    }

    /// Add one more tool while still keeping client-level tools.
    pub fn additional_tool(mut self, tool: Tool) -> Self {
        match &mut self.tool_override {
            ToolOverride::Replace(tools) => tools.push(tool),
            ToolOverride::Append(tools) => tools.push(tool),
            ToolOverride::Inherit => self.tool_override = ToolOverride::Append(vec![tool]),
            ToolOverride::None => self.tool_override = ToolOverride::Replace(vec![tool]),
        }
        self
    }

    /// Add several request-only tools while still keeping client-level tools.
    pub fn additional_tools<T>(mut self, tools: T) -> Self
    where
        T: IntoIterator<Item = Tool>,
    {
        let mut collected: Vec<_> = tools.into_iter().collect();

        match &mut self.tool_override {
            ToolOverride::Replace(existing) => existing.append(&mut collected),
            ToolOverride::Append(existing) => existing.append(&mut collected),
            ToolOverride::Inherit => self.tool_override = ToolOverride::Append(collected),
            ToolOverride::None => self.tool_override = ToolOverride::Replace(collected),
        }

        self
    }

    /// Disable all tools for this request, including client defaults.
    pub fn no_tools(mut self) -> Self {
        self.tool_override = ToolOverride::None;
        self
    }
}

impl<'a, PromptState, ClientModelState>
    RequestBuilder<'a, PromptState, ModelReady, ClientModelState>
{
    fn resolve(&self) -> Result<ResolvedRequest> {
        let model = self
            .model
            .clone()
            .or_else(|| self.client.default_model.clone())
            .expect("model-ready request builder must contain or inherit a model");

        let config = self
            .config
            .clone()
            .unwrap_or_else(|| self.client.default_config.clone());

        let tool_registry = match &self.tool_override {
            ToolOverride::Inherit => self.client.tool_registry.clone(),
            ToolOverride::Replace(tools) => {
                let mut registry = ToolRegistry::new();
                registry.extend(tools.clone())?;
                registry
            }
            ToolOverride::Append(tools) => {
                let mut registry = self.client.tool_registry.clone();
                registry.extend(tools.clone())?;
                registry
            }
            ToolOverride::None => ToolRegistry::new(),
        };

        let retry_config = self
            .retry_config
            .clone()
            .unwrap_or_else(|| self.client.default_retry_config.clone());

        Ok(ResolvedRequest {
            model,
            config,
            retry_config,
            tool_registry,
        })
    }
}

impl<'a, ClientModelState> RequestBuilder<'a, PromptReady, ModelReady, ClientModelState> {
    /// Generate a response and automatically execute tool calls.
    pub async fn generate(self) -> Result<Response> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");
        self.client
            .generate_with_tools(
                resolved.model,
                prompt,
                &resolved.config,
                &resolved.retry_config,
                &resolved.tool_registry,
            )
            .await
    }

    /// Generate a single provider response without auto-running tools.
    pub async fn generate_once(self) -> Result<Response> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");
        let tool_definitions =
            request_tool_definitions(&resolved.tool_registry, ToolAvailability::Enabled);

        crate::retry::with_retry(&resolved.retry_config, "generate_once", || {
            self.client.generate_once_internal(
                resolved.model.clone(),
                prompt,
                &resolved.config,
                tool_definitions.as_deref(),
            )
        })
        .await
    }

    /// Generate a response that must match the Rust type `T`.
    pub async fn generate_structured<T>(self) -> Result<StructuredOutput<T>>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");
        let config = structured_config_for::<T>(&resolved.config)?;
        let response = self
            .client
            .generate_with_tools(
                resolved.model,
                prompt,
                &config,
                &resolved.retry_config,
                &resolved.tool_registry,
            )
            .await?;

        parse_structured_output(response)
    }

    /// Generate a single structured provider response without auto-running tools.
    pub async fn generate_structured_once<T>(self) -> Result<StructuredOutput<T>>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");
        let config = structured_config_for::<T>(&resolved.config)?;
        let tool_definitions = request_tool_definitions(
            &resolved.tool_registry,
            ToolAvailability::IgnoredForStructuredOnce,
        );

        let response =
            crate::retry::with_retry(&resolved.retry_config, "generate_structured_once", || {
                self.client.generate_once_internal(
                    resolved.model.clone(),
                    prompt,
                    &config,
                    tool_definitions.as_deref(),
                )
            })
            .await?;

        parse_structured_output(response)
    }

    /// Generate a response incorporating a history of conversation turns.
    pub async fn generate_with_history(
        self,
        history: &[crate::message::ConversationTurn],
    ) -> Result<Response> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt")
            .clone()
            .with_history(history.to_vec());

        self.client
            .generate_with_tools(
                resolved.model,
                &prompt,
                &resolved.config,
                &resolved.retry_config,
                &resolved.tool_registry,
            )
            .await
    }

    /// Stream the generation response as high-level stream events.
    pub async fn generate_stream_events(
        self,
    ) -> Result<impl Stream<Item = Result<crate::message::StreamEvent>> + Send> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        let mut stream = crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream(resolved.model.clone(), prompt, &resolved.config)
        })
        .await?;

        let user_message = prompt
            .messages
            .last()
            .cloned()
            .unwrap_or_else(|| crate::message::Message::user(""));

        let stream_events = async_stream::stream! {
            let mut accumulated_content = String::new();
            let mut current_tool_id: Option<String> = None;
            let mut current_tool_name: Option<String> = None;
            let mut current_tool_args = String::new();
            let mut tool_calls = Vec::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        match chunk {
                            crate::provider::ProviderStreamEvent::Text(text) => {
                                accumulated_content.push_str(&text);
                                yield Ok(crate::message::StreamEvent::TextDelta { text });
                            }
                            crate::provider::ProviderStreamEvent::ToolCallStart { id, name } => {
                                if let (Some(tid), Some(tname)) = (current_tool_id.take(), current_tool_name.take()) {
                                    let args_json = serde_json::from_str(&current_tool_args).unwrap_or(serde_json::Value::Null);
                                    tool_calls.push(crate::message::ToolCall {
                                        id: tid.clone(),
                                        name: tname.clone(),
                                        arguments: args_json,
                                    });
                                    yield Ok(crate::message::StreamEvent::ToolCall {
                                        id: tid,
                                        name: tname,
                                        arguments: current_tool_args.clone(),
                                    });
                                    current_tool_args.clear();
                                }
                                current_tool_id = Some(id);
                                current_tool_name = Some(name);
                            }
                            crate::provider::ProviderStreamEvent::ToolCallChunk { id: _, arguments } => {
                                current_tool_args.push_str(&arguments);
                            }
                            crate::provider::ProviderStreamEvent::Done { finish_reason: _, usage: _ } => {
                                if let (Some(tid), Some(tname)) = (current_tool_id.take(), current_tool_name.take()) {
                                    let args_json = serde_json::from_str(&current_tool_args).unwrap_or(serde_json::Value::Null);
                                    tool_calls.push(crate::message::ToolCall {
                                        id: tid.clone(),
                                        name: tname.clone(),
                                        arguments: args_json,
                                    });
                                    yield Ok(crate::message::StreamEvent::ToolCall {
                                        id: tid,
                                        name: tname,
                                        arguments: current_tool_args.clone(),
                                    });
                                }

                                let mut assistant_message = crate::message::Message::assistant(accumulated_content.clone());
                                assistant_message.tool_calls = tool_calls.clone();

                                let turn = crate::message::ConversationTurn {
                                    user_message: user_message.clone(),
                                    assistant_message,
                                    tool_results: Vec::new(),
                                };
                                yield Ok(crate::message::StreamEvent::TurnComplete { turn });
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }
        };

        Ok(stream_events)
    }

    /// Stream the generation response.
    pub async fn stream(
        self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream(resolved.model.clone(), prompt, &resolved.config)
        })
        .await
    }

    /// Stream the generation response and accumulate into a complete [`Response`].
    ///
    /// Uses streaming transport (lower time-to-first-byte) but consumes all
    /// chunks internally and returns a complete response rather than individual
    /// chunks. Useful for progress logging or when you want streaming latency
    /// benefits without manual chunk assembly.
    pub async fn stream_accumulated(self) -> Result<Response> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        let model_str = resolved.model.as_str().to_string();
        let provider = resolved.model.provider();

        let mut stream = crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream(resolved.model.clone(), prompt, &resolved.config)
        })
        .await?;

        let mut accumulated_content = String::new();
        let mut finish_reason = None;
        let mut usage = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            match chunk {
                crate::provider::ProviderStreamEvent::Text(text) => {
                    accumulated_content.push_str(&text);
                }
                crate::provider::ProviderStreamEvent::Done {
                    finish_reason: fr,
                    usage: u,
                } => {
                    if fr.is_some() {
                        finish_reason = fr;
                    }
                    if u.is_some() {
                        usage = u;
                    }
                }
                _ => {}
            }
        }

        Ok(Response {
            messages: vec![Message::assistant(accumulated_content)],
            usage,
            model: model_str,
            provider,
            finish_reason,
        })
    }
}

fn request_tool_definitions(
    tool_registry: &ToolRegistry,
    availability: ToolAvailability,
) -> Option<Vec<ToolDefinition>> {
    if tool_registry.is_empty() {
        return None;
    }

    match availability {
        ToolAvailability::Enabled => Some(tool_registry.definitions()),
        ToolAvailability::IgnoredForStructuredOnce => {
            info!(
                tool_count = tool_registry.definitions().len(),
                "Ignoring configured tools for generate_structured_once; use generate_structured for tool loops"
            );
            None
        }
    }
}

fn structured_config_for<T>(config: &GenerationConfig) -> Result<GenerationConfig>
where
    T: JsonSchema,
{
    let mut config = config.clone();
    config.json_schema = Some(structured_schema_for::<T>()?);
    Ok(config)
}

fn parse_structured_output<T>(response: Response) -> Result<StructuredOutput<T>>
where
    T: DeserializeOwned + JsonSchema,
{
    let provider = response.provider;
    let model = response.model.clone();
    let content = response
        .messages
        .first()
        .map(|message| message.content.trim())
        .unwrap_or_default();

    if content.is_empty() {
        error!(provider = %provider, model = %model, "Structured output was empty");
        return Err(Error::StructuredOutput {
            provider,
            model,
            message: "response content was empty".to_string(),
        });
    }

    let instance = serde_json::from_str::<serde_json::Value>(content).map_err(|parse_error| {
        error!(
            provider = %provider,
            model = %model,
            error = %parse_error,
            response_content = %content,
            "Structured output was not valid JSON"
        );
        Error::StructuredOutput {
            provider,
            model: model.clone(),
            message: parse_error.to_string(),
        }
    })?;

    let schema = structured_schema_for::<T>()?;

    if let Err(validation_error) = jsonschema::validate(&schema, &instance) {
        error!(
            provider = %provider,
            model = %model,
            error = %validation_error,
            response_content = %content,
            response_schema = ?schema,
            "Structured output failed JSON schema validation"
        );
        return Err(Error::StructuredOutput {
            provider,
            model,
            message: validation_error.to_string(),
        });
    }

    match serde_json::from_str::<T>(content) {
        Ok(output) => {
            debug!(
                provider = %provider,
                model = %model,
                output_type = %type_name::<T>(),
                "Structured output validated successfully"
            );
            Ok(StructuredOutput { output, response })
        }
        Err(parse_error) => {
            error!(
                provider = %provider,
                model = %model,
                error = %parse_error,
                response_content = %content,
                "Structured output validation failed"
            );
            Err(Error::StructuredOutput {
                provider,
                model,
                message: parse_error.to_string(),
            })
        }
    }
}

fn structured_schema_for<T>() -> Result<serde_json::Value>
where
    T: JsonSchema,
{
    GenerationConfig::new()
        .with_json_schema_for::<T>()
        .map(|config| {
            config
                .json_schema
                .expect("structured schema should be present")
        })
}

/// Builder for creating an AI client with specific configuration.
pub struct ClientBuilder<ModelState = ModelMissing> {
    config: Config,
    default_model: Option<Model>,
    default_config: GenerationConfig,
    default_retry_config: RetryConfig,
    tools: Vec<Tool>,
    state: PhantomData<ModelState>,
}

impl<ModelState> std::fmt::Debug for ClientBuilder<ModelState> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientBuilder")
            .field("default_model", &self.default_model)
            .field("default_config", &self.default_config)
            .field("tools_count", &self.tools.len())
            .finish()
    }
}

impl ClientBuilder<ModelMissing> {
    /// Start a client builder.
    pub fn new() -> Self {
        Self {
            config: Config::new(),
            default_model: None,
            default_config: GenerationConfig::default(),
            default_retry_config: RetryConfig::default(),
            tools: Vec::new(),
            state: PhantomData,
        }
    }
}

impl<ModelState> ClientBuilder<ModelState> {
    fn with_state<NextModelState>(self) -> ClientBuilder<NextModelState> {
        ClientBuilder {
            config: self.config,
            default_model: self.default_model,
            default_config: self.default_config,
            default_retry_config: self.default_retry_config,
            tools: self.tools,
            state: PhantomData,
        }
    }

    /// Use configuration from environment variables.
    pub fn from_env(mut self) -> Self {
        self.config = Config::from_env();
        self.default_retry_config = self.config.retry_config();
        self
    }

    /// Set the OpenAI API key.
    pub fn openai_key(mut self, key: impl Into<String>) -> Self {
        self.config.openai_api_key = Some(key.into());
        self
    }

    /// Set the OpenAI base URL.
    pub fn openai_base_url(mut self, url: impl Into<String>) -> Self {
        self.config.openai_base_url = Some(url.into());
        self
    }

    /// Set the Anthropic API key.
    pub fn anthropic_key(mut self, key: impl Into<String>) -> Self {
        self.config.anthropic_api_key = Some(key.into());
        self
    }

    /// Set the Anthropic base URL.
    pub fn anthropic_base_url(mut self, url: impl Into<String>) -> Self {
        self.config.anthropic_base_url = Some(url.into());
        self
    }

    /// Set the OpenRouter API key.
    pub fn openrouter_key(mut self, key: impl Into<String>) -> Self {
        self.config.openrouter_api_key = Some(key.into());
        self
    }

    /// Set the OpenRouter base URL.
    pub fn openrouter_base_url(mut self, url: impl Into<String>) -> Self {
        self.config.openrouter_base_url = Some(url.into());
        self
    }

    /// Set the OpenRouter HTTP referer attribution header.
    pub fn openrouter_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.config.openrouter_http_referer = Some(referer.into());
        self
    }

    /// Set the OpenRouter title attribution header.
    pub fn openrouter_title(mut self, title: impl Into<String>) -> Self {
        self.config.openrouter_title = Some(title.into());
        self
    }

    /// Set OpenRouter app categories attribution header.
    pub fn openrouter_categories(mut self, categories: Vec<String>) -> Self {
        self.config.openrouter_categories = Some(categories);
        self
    }

    /// Set the OpenRouter App URL.
    pub fn openrouter_app_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.config.openrouter_app_url = Some(url.clone());
        self.config.openrouter_http_referer = Some(url);
        self
    }

    /// Set the OpenRouter App Title.
    pub fn openrouter_app_title(mut self, title: impl Into<String>) -> Self {
        let title = title.into();
        self.config.openrouter_app_title = Some(title.clone());
        self.config.openrouter_title = Some(title);
        self
    }

    /// Set the request timeout.
    pub fn timeout(mut self, seconds: u64) -> Self {
        self.config.timeout_seconds = Some(seconds);
        self
    }

    /// Set the default model used by request builders.
    pub fn model(mut self, model: Model) -> ClientBuilder<ModelReady> {
        self.default_model = Some(model);
        self.with_state()
    }

    /// Set the default generation config used by request builders.
    pub fn config(mut self, config: GenerationConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Set the default retry configuration for all requests.
    pub fn retry_config(mut self, config: RetryConfig) -> Self {
        self.default_retry_config = config;
        self
    }

    /// Disable retries by default for all requests.
    pub fn no_retry(mut self) -> Self {
        self.default_retry_config = RetryConfig::none();
        self
    }

    /// Register a single tool to be auto-executed by `generate()`.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Register multiple tools to be auto-executed by `generate()`.
    pub fn tools<T>(mut self, tools: T) -> Self
    where
        T: IntoIterator<Item = Tool>,
    {
        self.tools.extend(tools);
        self
    }

    /// Build the AI client.
    pub fn build(self) -> Result<Client<ModelState>> {
        let mut tool_registry = ToolRegistry::new();
        tool_registry.extend(self.tools)?;
        Client::new_with_defaults(
            self.config,
            self.default_model,
            self.default_config,
            self.default_retry_config,
            tool_registry,
        )
    }
}

impl Default for ClientBuilder<ModelMissing> {
    fn default() -> Self {
        Self::new()
    }
}
