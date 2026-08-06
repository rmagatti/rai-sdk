//! The client and request builders that drive generation.
//!
//! [`ClientBuilder`] assembles configuration, a default model, and any shared
//! tools into a [`Client`]. Each call to [`Client::request`] returns a
//! [`RequestBuilder`], a typestate builder whose terminal methods only become
//! available once the request has both a prompt and a model.
//!
//! The builder exposes four families of terminal operations: `generate` and
//! `generate_once` for text, `generate_structured` and
//! `generate_structured_once` for typed output, `stream` and
//! `stream_accumulated` for streaming, and per-request overrides such as
//! configuration, tools, and retry policy. The `_once` variants perform a single
//! provider call and do not execute registered tools.

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

/// Unified AI client for OpenAI, Anthropic, and OpenRouter.
///
/// A client owns provider credentials and HTTP clients, an optional default
/// model, default generation and retry settings, and any tools shared by every
/// request. Build one once and reuse it: individual requests are cheap, but
/// constructing a client initializes a client per configured provider.
///
/// Use [`ClientBuilder`] for the common path, or [`Client::new`] when you
/// already have a [`Config`].
///
/// # Typestate
///
/// The `ModelState` parameter records whether a default model is present.
/// [`ClientBuilder::model`] moves the builder into the model-ready state, and
/// only a model-ready client hands out request builders that can call
/// [`RequestBuilder::generate`] without naming a model. A client built without a
/// default model is still fully usable — every request just has to call
/// [`RequestBuilder::model`] first. Either way, a request missing a model is a
/// compile error rather than a runtime one.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{ClientBuilder, Model};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ClientBuilder::new()
///     .from_env()
///     .model(Model::gpt4o_mini())
///     .build()?;
///
/// // Reuse the same client for many requests.
/// for prompt in ["Define a trait.", "Define a lifetime."] {
///     let response = client.request().prompt(prompt).generate().await?;
///     println!("{}", response.text());
/// }
/// # Ok(())
/// # }
/// ```
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
    /// Create a client from an explicit [`Config`], with no default model.
    ///
    /// Every request from this client must select a model with
    /// [`RequestBuilder::model`]. Use [`ClientBuilder`] instead if you want a
    /// default model or client-level tools.
    ///
    /// A provider whose API key is missing is simply left uninitialized rather
    /// than failing here; using it later returns
    /// [`Error::ProviderNotConfigured`].
    ///
    /// # Errors
    ///
    /// Returns an error if a configured provider's HTTP client cannot be
    /// constructed, for example because the request timeout is invalid.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{Client, Config, Model};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(Config::from_env())?;
    ///
    /// let response = client
    ///     .request()
    ///     .model(Model::gpt4o_mini())
    ///     .prompt("Hello")
    ///     .generate()
    ///     .await?;
    /// # println!("{}", response.text());
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// Equivalent to [`ClientBuilder::new`].
    pub fn builder() -> ClientBuilder<ModelMissing> {
        ClientBuilder::new()
    }

    /// Start a request.
    ///
    /// This client has no default model, so the returned builder requires
    /// [`RequestBuilder::model`] before it will expose `generate` and friends.
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
        #[cfg(not(any(feature = "openai", feature = "anthropic", feature = "openrouter")))]
        let _ = (prompt, config, tool_definitions);

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

    /// Stream a completion for an explicit model and prompt.
    ///
    /// Prefer [`RequestBuilder::stream`], which applies the client's defaults,
    /// retry policy, and per-request tool overrides. This lower-level entry
    /// point is useful when you are driving the model and prompt yourself.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidRequest`] if any tool is registered on this client,
    ///   since streaming cannot run a tool loop. Because this method takes no
    ///   request context, it can only consider the client's tools; use
    ///   [`RequestBuilder::stream`] with [`RequestBuilder::no_tools`] to stream
    ///   from a client that has tools registered.
    /// - [`Error::ProviderNotConfigured`] if the model's provider has no API key.
    /// - [`Error::ProviderNotEnabled`] if its Cargo feature is disabled.
    /// - A transport or provider error if the request itself fails.
    pub async fn generate_stream(
        &self,
        model: Model,
        prompt: &Prompt,
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        ensure_streamable(&self.tool_registry)?;
        self.generate_stream_inner(model, prompt, config).await
    }

    /// Open a provider stream without considering tools.
    ///
    /// Callers are responsible for having already rejected tool-bearing
    /// requests via [`ensure_streamable`].
    #[instrument(skip(self, prompt, config))]
    async fn generate_stream_inner(
        &self,
        model: Model,
        prompt: &Prompt,
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        #[cfg(not(any(feature = "openai", feature = "anthropic", feature = "openrouter")))]
        let _ = (prompt, config);

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

    /// Whether a provider is usable: its feature is enabled and it has
    /// credentials.
    ///
    /// Use this to branch at runtime instead of discovering a missing key
    /// through a failed request.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{ClientBuilder, Model, ProviderKind};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = ClientBuilder::new().from_env().build()?;
    ///
    /// let model = if client.is_provider_available(ProviderKind::Anthropic) {
    ///     Model::claude_sonnet_46()
    /// } else {
    ///     Model::gpt4o_mini()
    /// };
    /// # let _ = model;
    /// # Ok(())
    /// # }
    /// ```
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

    /// The configuration this client was built with.
    ///
    /// Note that the returned [`Config`] contains API keys; do not log it.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl Client<ModelReady> {
    /// Start a request that inherits this client's default model.
    ///
    /// Because the model is already known, the returned builder only needs a
    /// prompt before you can call [`RequestBuilder::generate`]. Override the
    /// model per request with [`RequestBuilder::model`].
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
///
/// Created by [`Client::request`]. Chain overrides, supply a prompt, then call
/// one terminal method. Anything you do not override is inherited from the
/// client.
///
/// # Terminal methods
///
/// | Method | Returns | Runs registered tools |
/// | --- | --- | --- |
/// | [`generate`](Self::generate) | [`Response`] | yes, until a final answer |
/// | [`generate_once`](Self::generate_once) | [`Response`] | no, one provider call |
/// | [`generate_structured`](Self::generate_structured) | [`StructuredOutput<T>`] | yes |
/// | [`generate_structured_once`](Self::generate_structured_once) | [`StructuredOutput<T>`] | no |
/// | [`generate_with_history`](Self::generate_with_history) | [`Response`] | yes |
/// | [`stream`](Self::stream) | stream of provider events | not supported |
/// | [`generate_stream_events`](Self::generate_stream_events) | stream of high-level events | not supported |
/// | [`stream_accumulated`](Self::stream_accumulated) | [`Response`] | not supported |
///
/// Methods ending in `_once` make exactly one provider call and never execute
/// tools.
///
/// # Typestate
///
/// The terminal methods only exist once the builder has both a prompt and a
/// model, so an incomplete request cannot be sent. A model comes either from
/// the client's default or from [`model`](Self::model); the prompt comes from
/// [`prompt`](Self::prompt). If `generate` appears to be missing, one of those
/// two is absent.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{ClientBuilder, GenerationConfig, Model};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ClientBuilder::new()
///     .from_env()
///     .model(Model::gpt4o_mini())
///     .build()?;
///
/// let response = client
///     .request()
///     .model(Model::claude_sonnet_46())                       // override the model
///     .config(GenerationConfig::new().with_temperature(0.2))   // override sampling
///     .no_tools()                                             // ignore client tools
///     .prompt("Summarize the borrow checker.")
///     .generate()
///     .await?;
/// # println!("{}", response.text());
/// # Ok(())
/// # }
/// ```
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

    /// Override the model, and therefore the provider, for this request.
    ///
    /// Takes precedence over the client's default model. Calling this makes the
    /// builder model-ready even if the client has no default.
    pub fn model(
        mut self,
        model: Model,
    ) -> RequestBuilder<'a, PromptState, ModelReady, ClientModelState> {
        self.model = Some(model);
        self.with_model_state()
    }

    /// Override generation settings for this request.
    ///
    /// Replaces the client's default [`GenerationConfig`] wholesale rather than
    /// merging with it, so include every setting you want.
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
    ///
    /// Accepts anything convertible into a [`Prompt`]: a `&str`, a `String`, a
    /// single [`Message`], a `Vec<Message>`, or a full `Prompt` with multi-turn
    /// history and multimodal content.
    ///
    /// # Examples
    ///
    /// ```
    /// use rai_sdk::{Message, Prompt};
    ///
    /// // Each of these is accepted by `prompt()`.
    /// let _: Prompt = "a plain string".into();
    /// let _: Prompt = Message::user("a single message").into();
    /// let _: Prompt = vec![
    ///     Message::system("You are terse."),
    ///     Message::user("Explain lifetimes."),
    /// ]
    /// .into();
    ///
    /// // Or build one up explicitly.
    /// let _ = Prompt::single(Message::system("You are terse."))
    ///     .with_message(Message::user("Explain lifetimes."));
    /// ```
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
    ///
    /// Also the way to stream from a client that has tools registered, since the
    /// streaming methods reject any request carrying tools.
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
    /// Generate a response, automatically executing any tool calls the model
    /// requests.
    ///
    /// This is the method you usually want. If tools are registered, it runs the
    /// loop — send, execute requested tools, append results, send again — until
    /// the model answers without asking for more tools. With no tools
    /// registered, it is a single call.
    ///
    /// Transient failures are retried according to the effective
    /// [`RetryConfig`].
    ///
    /// # Errors
    ///
    /// - [`Error::ProviderNotConfigured`] if the provider has no API key, or
    ///   [`Error::ProviderNotEnabled`] if its Cargo feature is off.
    /// - [`Error::ToolLoopLimitExceeded`] if the model keeps requesting tools
    ///   past [`GenerationConfig::with_max_tool_rounds`] (default 8).
    /// - [`Error::ToolNotFound`] if the model requests a tool that is not
    ///   registered.
    /// - [`Error::RateLimit`], [`Error::Timeout`], or [`Error::Http`] if the
    ///   request still fails after retries.
    /// - [`Error::Auth`], [`Error::InvalidRequest`], [`Error::ContentFiltered`],
    ///   or [`Error::Request`] for provider-side rejections.
    ///
    /// Note that a tool handler returning an error does *not* fail this call:
    /// the error is passed back to the model as tool content so it can react.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{ClientBuilder, Model};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = ClientBuilder::new()
    ///     .from_env()
    ///     .model(Model::gpt4o_mini())
    ///     .build()?;
    ///
    /// let response = client
    ///     .request()
    ///     .prompt("Name one Rust testing crate.")
    ///     .generate()
    ///     .await?;
    ///
    /// println!("{}", response.text());
    /// if let Some(usage) = &response.usage {
    ///     println!("tokens: {:?}", usage.total_tokens);
    /// }
    /// # Ok(())
    /// # }
    /// ```
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

    /// Make exactly one provider call, without executing tools.
    ///
    /// Tool *definitions* are still advertised to the model, so the response may
    /// contain tool calls — they are returned to you on the response messages
    /// instead of being executed. Use this when you want to inspect, gate, or
    /// approve tool calls, or drive the loop yourself.
    ///
    /// # Errors
    ///
    /// Same as [`generate`](Self::generate), except it cannot return
    /// [`Error::ToolLoopLimitExceeded`] or [`Error::ToolNotFound`], since no
    /// tool is executed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{ClientBuilder, Model};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
    /// let response = client
    ///     .request()
    ///     .prompt("What is the weather in Paris?")
    ///     .generate_once()
    ///     .await?;
    ///
    /// for message in &response.messages {
    ///     for call in &message.tool_calls {
    ///         println!("requested {} with {}", call.name, call.arguments);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// A JSON Schema is generated from `T` and sent to the provider, the
    /// response is validated against that schema, and only then deserialized.
    /// Tools still run as in [`generate`](Self::generate).
    ///
    /// `T` must be non-recursive: recursive types force `$ref`/`$defs`, which
    /// strict providers reject. See
    /// [`GenerationConfig::with_json_schema_for`].
    ///
    /// # Errors
    ///
    /// Everything [`generate`](Self::generate) can return, plus
    /// [`Error::StructuredOutput`] if the response is empty, is not valid JSON,
    /// fails schema validation, or does not deserialize into `T`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{ClientBuilder, JsonSchema, Model};
    /// use serde::Deserialize;
    ///
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct Summary {
    ///     title: String,
    ///     bullet_points: Vec<String>,
    /// }
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
    /// let structured = client
    ///     .request()
    ///     .prompt("Summarize the Rust ownership model.")
    ///     .generate_structured::<Summary>()
    ///     .await?;
    ///
    /// println!("{}", structured.output.title);
    /// for point in &structured.output.bullet_points {
    ///     println!("- {point}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
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

    /// Make exactly one provider call and parse the result as `T`.
    ///
    /// Unlike [`generate_once`](Self::generate_once), configured tools are not
    /// even advertised to the model: they are ignored entirely (and a log line
    /// records that). Use this for a pure transformation on a client that
    /// happens to have tools registered.
    ///
    /// # Errors
    ///
    /// Same as [`generate_structured`](Self::generate_structured), minus the
    /// tool-loop errors.
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

    /// Generate a response with prior conversation turns prepended.
    ///
    /// A convenience over assembling the history into the [`Prompt`] yourself:
    /// each [`ConversationTurn`](crate::message::ConversationTurn) contributes
    /// its user message, assistant message, and any tool results, followed by
    /// this request's prompt. Tools run as in [`generate`](Self::generate).
    ///
    /// # Errors
    ///
    /// Same as [`generate`](Self::generate).
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

    /// Stream the response as high-level [`StreamEvent`](crate::message::StreamEvent)s.
    ///
    /// Higher level than [`stream`](Self::stream): text deltas are passed
    /// through, tool-call argument fragments are buffered and emitted as whole
    /// calls, and a final `TurnComplete` event carries the assembled
    /// [`ConversationTurn`](crate::message::ConversationTurn) — convenient for
    /// feeding conversation history back into a later request.
    ///
    /// Registered tools are *not* executed; this only reports what the model
    /// asked for.
    ///
    /// To forward these events to a remote client instead of consuming them in
    /// process, see [`stream_wire_events`](Self::stream_wire_events);
    /// [`WireStreamEvent`](crate::wire::WireStreamEvent) also implements
    /// `From<StreamEvent>` if you would rather convert these.
    ///
    /// # Cancellation
    ///
    /// Dropping the returned stream aborts the upstream provider request. See
    /// the "Cancellation" section of [`stream`](Self::stream).
    ///
    /// # Errors
    ///
    /// Same as [`stream`](Self::stream), including [`Error::InvalidRequest`]
    /// when the request's effective tool set is non-empty. Once the stream is
    /// open, individual items may also be errors.
    pub async fn generate_stream_events(
        self,
    ) -> Result<impl Stream<Item = Result<crate::message::StreamEvent>> + Send> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        ensure_streamable(&resolved.tool_registry)?;

        let mut stream = crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream_inner(resolved.model.clone(), prompt, &resolved.config)
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

    /// Stream the response as serializable
    /// [`WireStreamEvent`](crate::wire::WireStreamEvent)s, ready to forward to a
    /// remote client.
    ///
    /// This is the SDK half of the proxy pattern: your server holds the
    /// provider credentials, calls this, and re-emits each event as an SSE
    /// `data:` payload; the client parses them back into `WireStreamEvent`s and
    /// rebuilds the response with
    /// [`StreamAccumulator`](crate::wire::StreamAccumulator). See the
    /// [`wire`](crate::wire) module for the format and its compatibility
    /// guarantees, and `examples/sse_proxy.rs` for the whole loop.
    ///
    /// # Stream shape
    ///
    /// Unlike the other streaming methods, items are **not** `Result`s. Once the
    /// stream is open every outcome is an event, so a mid-stream provider
    /// failure reaches the client as
    /// [`WireStreamEvent::Error`](crate::wire::WireStreamEvent::Error) instead
    /// of as a silently truncated response. The sequence is:
    ///
    /// 1. exactly one
    ///    [`MessageStart`](crate::wire::WireStreamEvent::MessageStart);
    /// 2. any number of text and tool-call events;
    /// 3. one [`Usage`](crate::wire::WireStreamEvent::Usage), when the provider
    ///    reported token counts;
    /// 4. exactly one terminal event —
    ///    [`MessageStop`](crate::wire::WireStreamEvent::MessageStop) on success,
    ///    [`Error`](crate::wire::WireStreamEvent::Error) on failure.
    ///
    /// Tool-call arguments are reported twice over: incrementally as
    /// [`ToolCallStart`](crate::wire::WireStreamEvent::ToolCallStart) plus
    /// [`ToolCallDelta`](crate::wire::WireStreamEvent::ToolCallDelta) so a UI can
    /// render progress, then once assembled as
    /// [`ToolCallEnd`](crate::wire::WireStreamEvent::ToolCallEnd). A client that
    /// only wants finished calls can ignore the first two.
    ///
    /// Registered tools are *not* executed, exactly as with
    /// [`generate_stream_events`](Self::generate_stream_events).
    ///
    /// # Cancellation
    ///
    /// Dropping the returned stream aborts the upstream provider request. See
    /// the "Cancellation" section of [`stream`](Self::stream) — it matters more
    /// here than anywhere else, because for a proxy the consumer being dropped
    /// *is* the end client hanging up.
    ///
    /// # Errors
    ///
    /// The returned `Result` covers only failures that happen before the stream
    /// opens: the same causes as [`stream`](Self::stream), including
    /// [`Error::InvalidRequest`] when the request's effective tool set is
    /// non-empty. A server that wants its client to see those too can forward
    /// them with
    /// [`WireStreamEvent::error`](crate::wire::WireStreamEvent::error).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use futures::StreamExt;
    /// use rai_sdk::{ClientBuilder, Model};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
    /// let mut events = client
    ///     .request()
    ///     .prompt("Summarize the news.")
    ///     .stream_wire_events()
    ///     .await?;
    ///
    /// while let Some(event) = events.next().await {
    ///     // `data: {"type":"text_delta","text":"..."}`
    ///     println!("data: {}\n", serde_json::to_string(&event)?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream_wire_events(
        self,
    ) -> Result<Pin<Box<dyn Stream<Item = crate::wire::WireStreamEvent> + Send>>> {
        use crate::wire::WireStreamEvent;

        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        ensure_streamable(&resolved.tool_registry)?;

        let model_str = resolved.model.as_str().to_string();
        let provider = resolved.model.provider();

        let mut stream = crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream_inner(resolved.model.clone(), prompt, &resolved.config)
        })
        .await?;

        let wire_events = async_stream::stream! {
            yield WireStreamEvent::message_start(model_str, provider);

            // The assembled `ToolCallEnd` for the call currently streaming.
            // Providers do not delimit tool calls explicitly, so a call is
            // closed by the next `ToolCallStart` or by the end of the stream.
            let mut pending_tool: Option<(String, String, String)> = None;
            let mut finish_reason: Option<String> = None;
            let mut usage: Option<crate::message::Usage> = None;
            let mut failed = false;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        // Terminal: a provider failure is an event, not a
                        // dropped connection.
                        yield WireStreamEvent::error(&error);
                        failed = true;
                        break;
                    }
                };

                match chunk {
                    crate::provider::ProviderStreamEvent::Text(text) => {
                        yield WireStreamEvent::TextDelta { text };
                    }

                    crate::provider::ProviderStreamEvent::ToolCallStart { id, name } => {
                        if let Some((prev_id, prev_name, prev_args)) = pending_tool.take() {
                            yield WireStreamEvent::ToolCallEnd {
                                id: prev_id,
                                name: prev_name,
                                arguments: prev_args,
                            };
                        }
                        pending_tool = Some((id.clone(), name.clone(), String::new()));
                        yield WireStreamEvent::ToolCallStart { id, name };
                    }

                    crate::provider::ProviderStreamEvent::ToolCallChunk { id, arguments } => {
                        // Some providers omit the id on continuation chunks;
                        // attribute those to the call already in flight.
                        let id = match (&pending_tool, id.is_empty()) {
                            (Some((pending_id, _, _)), true) => pending_id.clone(),
                            _ => id,
                        };
                        if let Some((pending_id, _, pending_args)) = pending_tool.as_mut() {
                            if *pending_id == id {
                                pending_args.push_str(&arguments);
                            }
                        }
                        yield WireStreamEvent::ToolCallDelta { id, arguments };
                    }

                    crate::provider::ProviderStreamEvent::Done {
                        finish_reason: reason,
                        usage: reported,
                    } => {
                        if let Some((id, name, arguments)) = pending_tool.take() {
                            yield WireStreamEvent::ToolCallEnd { id, name, arguments };
                        }
                        // Providers may split the finish reason and the usage
                        // across separate `Done` events, so keep the last of
                        // each rather than emitting one terminal event per
                        // `Done`.
                        if reason.is_some() {
                            finish_reason = reason;
                        }
                        if reported.is_some() {
                            usage = reported;
                        }
                    }
                }
            }

            if failed {
                return;
            }

            if let Some((id, name, arguments)) = pending_tool.take() {
                yield WireStreamEvent::ToolCallEnd { id, name, arguments };
            }
            if let Some(usage) = usage {
                yield WireStreamEvent::Usage { usage };
            }
            yield WireStreamEvent::MessageStop { finish_reason };
        };

        // Boxed rather than `impl Stream` so the result borrows nothing and is
        // `'static`. A proxy handler builds this from a shared `Client` and
        // hands it straight to its web framework, which needs an owned,
        // lifetime-free stream.
        Ok(Box::pin(wire_events))
    }

    /// Stream raw provider events as they arrive.
    ///
    /// Use this to render output incrementally. Each item is a [`Result`], since
    /// a stream can fail partway through — do not discard the error case, or a
    /// mid-stream failure will look like a clean end of output.
    ///
    /// # Cancellation
    ///
    /// **Dropping the stream aborts the upstream provider request.** Every
    /// streaming method in this crate is driven entirely by the consumer: the
    /// provider's HTTP response body is polled from inside the returned stream,
    /// never from a detached background task. Dropping the stream therefore
    /// drops the response body and closes the underlying connection, and the
    /// provider stops generating. Nothing keeps running in the background and
    /// no tokens are burned on output nobody will read.
    ///
    /// Two consequences worth planning for:
    ///
    /// - A generation cancelled this way produces **no terminal event** — no
    ///   `Done`, no usage. Providers bill for what they generated before the
    ///   abort, so a server that meters usage cannot rely on the final usage
    ///   event alone.
    /// - Cancellation propagates through wrappers. Dropping the future or
    ///   stream returned by [`generate_stream_events`](Self::generate_stream_events),
    ///   [`stream_wire_events`](Self::stream_wire_events), or
    ///   [`stream_accumulated`](Self::stream_accumulated) — including when the
    ///   whole task is cancelled by `tokio::time::timeout` or by an axum client
    ///   disconnect — aborts the provider request just the same.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] if the request would carry any tool,
    /// because streaming cannot run a tool loop. This considers the request's
    /// effective tool set, so [`no_tools`](Self::no_tools) lets you stream from
    /// a client that has tools registered, and [`tool`](Self::tool) on the
    /// request is rejected even when the client itself has none.
    ///
    /// Otherwise the same causes as [`Client::generate_stream`]:
    /// [`Error::ProviderNotConfigured`], [`Error::ProviderNotEnabled`], or a
    /// transport or provider failure. Once the stream is open, individual items
    /// may also be errors.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use futures::StreamExt;
    /// use rai_sdk::{ClientBuilder, Model, provider::ProviderStreamEvent};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
    /// let mut stream = client
    ///     .request()
    ///     .prompt("Count from one to five.")
    ///     .stream()
    ///     .await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         ProviderStreamEvent::Text(text) => print!("{text}"),
    ///         ProviderStreamEvent::Done { .. } => println!(),
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream(
        self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        ensure_streamable(&resolved.tool_registry)?;

        crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream_inner(resolved.model.clone(), prompt, &resolved.config)
        })
        .await
    }

    /// Stream internally and return one complete [`Response`].
    ///
    /// Uses the streaming transport (lower time-to-first-byte, and less likely
    /// to sit near a timeout on long generations) but consumes every chunk for
    /// you, so the result is shaped exactly like [`generate`](Self::generate).
    /// Reach for this when you want streaming's latency behavior without
    /// handling events.
    ///
    /// Only text and the terminating event are accumulated, so tool calls are
    /// not represented in the returned response.
    ///
    /// # Cancellation
    ///
    /// Dropping the returned future aborts the upstream provider request. See
    /// the "Cancellation" section of [`stream`](Self::stream).
    ///
    /// # Errors
    ///
    /// Same as [`stream`](Self::stream), including [`Error::InvalidRequest`]
    /// when the request's effective tool set is non-empty, plus any error
    /// encountered while consuming the stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rai_sdk::{ClientBuilder, Model};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
    /// let response = client
    ///     .request()
    ///     .prompt("Write a short launch announcement.")
    ///     .stream_accumulated()
    ///     .await?;
    ///
    /// println!("{}", response.text());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream_accumulated(self) -> Result<Response> {
        let resolved = self.resolve()?;
        let prompt = self
            .prompt
            .as_ref()
            .expect("prompt-ready request builder must contain a prompt");

        ensure_streamable(&resolved.tool_registry)?;

        let model_str = resolved.model.as_str().to_string();
        let provider = resolved.model.provider();

        let mut stream = crate::retry::with_retry(&resolved.retry_config, "stream", || {
            self.client
                .generate_stream_inner(resolved.model.clone(), prompt, &resolved.config)
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

/// Reject a streaming request when tools are in play.
///
/// Streaming has no way to execute a tool loop, since that requires issuing
/// follow-up requests. Failing loudly is better than quietly dropping tools the
/// caller registered.
fn ensure_streamable(tool_registry: &ToolRegistry) -> Result<()> {
    if tool_registry.is_empty() {
        return Ok(());
    }

    Err(Error::InvalidRequest(
        "Streaming with tools is not supported. Use generate() to run tools, \
         or no_tools() on the request to stream without them."
            .into(),
    ))
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

/// Builder for creating a [`Client`].
///
/// Set credentials (usually with [`from_env`](Self::from_env)), then optionally
/// a default model, generation config, retry policy, and shared tools, and
/// finish with [`build`](Self::build).
///
/// Explicit setters win over the environment regardless of chain order relative
/// to `from_env()`, because `from_env()` replaces the accumulated config — so
/// call it first.
///
/// # Examples
///
/// ```no_run
/// use std::time::Duration;
///
/// use rai_sdk::{ClientBuilder, GenerationConfig, Model, RetryConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ClientBuilder::new()
///     .from_env()
///     .model(Model::gpt4o_mini())
///     .config(GenerationConfig::new().with_max_tokens(1024))
///     .retry_config(RetryConfig::new().with_initial_delay(Duration::from_millis(250)))
///     .timeout(60)
///     .build()?;
/// # let _ = client;
/// # Ok(())
/// # }
/// ```
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

    /// Load configuration from environment variables.
    ///
    /// Reads the API keys, base URLs, timeout, and retry variables documented in
    /// [`config`](crate::config). This **replaces** any configuration already
    /// accumulated on the builder, so call it first and then override
    /// individual values.
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
    ///
    /// This also moves the builder into the model-ready state, so the resulting
    /// client can start requests that need only a prompt. Individual requests
    /// can still override it with [`RequestBuilder::model`].
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

    /// Register a tool that [`RequestBuilder::generate`] may auto-execute.
    ///
    /// Client-level tools are available to every request. Requests that stream
    /// must opt out with [`RequestBuilder::no_tools`], since streaming cannot
    /// run a tool loop; see [`RequestBuilder::stream`].
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

    /// Build the client.
    ///
    /// Providers with credentials are initialized; providers without them are
    /// left unavailable rather than causing a failure, so this succeeds even if
    /// only one key is present.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidRequest`] if two registered tools share a name, or a
    ///   tool's input schema is invalid.
    /// - An error if a provider's HTTP client cannot be constructed.
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
