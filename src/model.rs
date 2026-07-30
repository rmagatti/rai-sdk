use serde::{Deserialize, Serialize};

use crate::error::ProviderKind;

/// Unified AI model selection across providers.
///
/// Each variant wraps a provider-specific model enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", content = "model")]
pub enum Model {
    OpenAI(OpenAIModel),
    Anthropic(AnthropicModel),
    OpenRouter(OpenRouterModel),
}

impl Model {
    /// Get the model string identifier.
    pub fn as_str(&self) -> &str {
        match self {
            Model::OpenAI(m) => m.as_str(),
            Model::Anthropic(m) => m.as_str(),
            Model::OpenRouter(m) => m.as_str(),
        }
    }

    /// Get the provider for this model.
    pub fn provider(&self) -> ProviderKind {
        match self {
            Model::OpenAI(_) => ProviderKind::OpenAI,
            Model::Anthropic(_) => ProviderKind::Anthropic,
            Model::OpenRouter(_) => ProviderKind::OpenRouter,
        }
    }

    // ── OpenAI convenience constructors ──

    pub fn gpt4o() -> Self {
        Model::OpenAI(OpenAIModel::Gpt4o)
    }

    pub fn gpt4o_mini() -> Self {
        Model::OpenAI(OpenAIModel::Gpt4oMini)
    }

    pub fn gpt4_1() -> Self {
        Model::OpenAI(OpenAIModel::Gpt4_1)
    }

    pub fn o3_mini() -> Self {
        Model::OpenAI(OpenAIModel::O3Mini)
    }

    pub fn o3() -> Self {
        Model::OpenAI(OpenAIModel::O3)
    }

    pub fn o4_mini() -> Self {
        Model::OpenAI(OpenAIModel::O4Mini)
    }

    pub fn gpt5() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5)
    }

    pub fn gpt5_mini() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5Mini)
    }

    pub fn gpt5_nano() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5Nano)
    }

    pub fn gpt5_codex() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5Codex)
    }

    pub fn gpt_5_1() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_1)
    }

    pub fn gpt_5_2() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_2)
    }

    pub fn gpt_5_2_pro() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_2Pro)
    }

    pub fn gpt_5_3_chat() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_3Chat)
    }

    pub fn gpt_5_3_instant() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_3Instant)
    }

    pub fn gpt_5_4() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_4)
    }

    pub fn gpt_5_4_mini() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_4Mini)
    }

    pub fn gpt_5_4_nano() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_4Nano)
    }

    pub fn gpt_5_5() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_5)
    }

    // ── Anthropic convenience constructors ──

    pub fn claude_fable_5() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeFable5)
    }

    pub fn claude_opus_48() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus48)
    }

    pub fn claude_opus_47() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus47)
    }

    pub fn claude_sonnet_46() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeSonnet46)
    }

    pub fn claude_opus_46() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus46)
    }

    pub fn claude_sonnet_4() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeSonnet4)
    }

    pub fn claude_opus_4() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus4)
    }

    pub fn claude_opus_41() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus41)
    }

    pub fn claude_sonnet_45() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeSonnet45)
    }

    pub fn claude_opus_45() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeOpus45)
    }

    pub fn claude_haiku_45() -> Self {
        Model::Anthropic(AnthropicModel::ClaudeHaiku45)
    }

    pub fn claude_35_sonnet() -> Self {
        Model::Anthropic(AnthropicModel::Claude35Sonnet)
    }

    pub fn claude_35_haiku() -> Self {
        Model::Anthropic(AnthropicModel::Claude35Haiku)
    }

    /// Create a model from a custom OpenAI model string.
    pub fn openai_custom(name: impl Into<String>) -> Self {
        Model::OpenAI(OpenAIModel::Custom(name.into()))
    }

    /// Create a model from a custom Anthropic model string.
    pub fn anthropic_custom(name: impl Into<String>) -> Self {
        Model::Anthropic(AnthropicModel::Custom(name.into()))
    }

    /// Create a model from a custom OpenRouter model string.
    pub fn openrouter_custom(name: impl Into<String>) -> Self {
        Model::OpenRouter(OpenRouterModel::Custom(name.into()))
    }

    pub fn openrouter_auto() -> Self {
        Model::OpenRouter(OpenRouterModel::Auto)
    }

    pub fn openrouter_gpt5() -> Self {
        Model::OpenRouter(OpenRouterModel::Gpt5)
    }

    pub fn openrouter_claude_sonnet_4_5() -> Self {
        Model::OpenRouter(OpenRouterModel::ClaudeSonnet4_5)
    }

    pub fn openrouter_gemini_25_flash() -> Self {
        Model::OpenRouter(OpenRouterModel::Gemini25Flash)
    }

    pub fn openrouter_deepseek_r1() -> Self {
        Model::OpenRouter(OpenRouterModel::DeepseekR1)
    }

    pub fn openrouter_qwen3_coder() -> Self {
        Model::OpenRouter(OpenRouterModel::Qwen3Coder)
    }
}

/// OpenAI model variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpenAIModel {
    /// GPT-5.5 — latest flagship model for complex reasoning and coding
    Gpt5_5,
    /// GPT-5.4 — more affordable frontier model for coding and professional work
    Gpt5_4,
    /// GPT-5.4 Mini — stronger mini model for coding, computer use, and subagents
    Gpt5_4Mini,
    /// GPT-5.4 Nano — lowest-latency GPT-5.4 variant
    Gpt5_4Nano,
    /// GPT-5.3 Instant — real-time high-accuracy model
    Gpt5_3Instant,
    /// GPT-5.3 Chat — conversational tuning of GPT-5.3
    Gpt5_3Chat,
    /// GPT-5.2 Pro — highest-capability GPT-5.2 tier
    Gpt5_2Pro,
    /// GPT-5.2
    Gpt5_2,
    /// GPT-5.1
    Gpt5_1,
    /// GPT-5 — flagship GPT-5 model
    Gpt5,
    /// GPT-5 Mini — faster, cheaper GPT-5
    Gpt5Mini,
    /// GPT-5 Nano — smallest, fastest GPT-5
    Gpt5Nano,
    /// GPT-5 Codex — agentic coding-optimized GPT-5
    Gpt5Codex,
    /// GPT-4.1 — high-capability long-context model
    Gpt4_1,
    /// GPT-4o — multimodal model (previous generation)
    Gpt4o,
    /// GPT-4o Mini
    Gpt4oMini,
    /// GPT-4 Turbo
    Gpt4Turbo,
    /// O1 Preview — reasoning model
    O1Preview,
    /// O1 Mini
    O1Mini,
    /// O3 Mini
    O3Mini,
    /// O3
    O3,
    /// O4 Mini
    O4Mini,
    /// Custom model name
    Custom(String),
}

impl OpenAIModel {
    pub fn as_str(&self) -> &str {
        match self {
            // GPT-5 family
            Self::Gpt5_5 => "gpt-5.5",
            Self::Gpt5_4 => "gpt-5.4",
            Self::Gpt5_4Mini => "gpt-5.4-mini",
            Self::Gpt5_4Nano => "gpt-5.4-nano",
            Self::Gpt5_3Instant => "gpt-5.3-instant",
            Self::Gpt5_3Chat => "gpt-5.3-chat",
            Self::Gpt5_2Pro => "gpt-5.2-pro",
            Self::Gpt5_2 => "gpt-5.2",
            Self::Gpt5_1 => "gpt-5.1",
            Self::Gpt5 => "gpt-5",
            Self::Gpt5Mini => "gpt-5-mini",
            Self::Gpt5Nano => "gpt-5-nano",
            Self::Gpt5Codex => "gpt-5-codex",
            // GPT-4 family
            Self::Gpt4_1 => "gpt-4.1",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4Turbo => "gpt-4-turbo",
            // Reasoning (o-series)
            Self::O1Preview => "o1-preview",
            Self::O1Mini => "o1-mini",
            Self::O3Mini => "o3-mini",
            Self::O3 => "o3",
            Self::O4Mini => "o4-mini",
            Self::Custom(s) => s,
        }
    }

    /// Whether this model is a reasoning model (o-series) that doesn't support temperature.
    pub fn is_reasoning_model(&self) -> bool {
        matches!(
            self,
            Self::O1Preview | Self::O1Mini | Self::O3Mini | Self::O3 | Self::O4Mini
        )
    }
}

/// Anthropic Claude model variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnthropicModel {
    /// Claude Fable 5 — most capable widely released Claude model
    ClaudeFable5,
    /// Claude Opus 4.8 — latest Opus-tier model for complex reasoning and coding
    ClaudeOpus48,
    /// Claude Opus 4.7 — legacy Opus-tier model
    ClaudeOpus47,
    /// Claude Opus 4.6 — legacy Opus-tier model with 1M context
    ClaudeOpus46,
    /// Claude Sonnet 4.6 — current best balance of intelligence, speed and cost
    ClaudeSonnet46,
    /// Claude Sonnet 4.5
    ClaudeSonnet45,
    /// Claude Sonnet 4
    ClaudeSonnet4,
    /// Claude Opus 4.5
    ClaudeOpus45,
    /// Claude Opus 4.1 — enhanced Claude 4 Opus
    ClaudeOpus41,
    /// Claude Opus 4 — legacy Claude 4 model
    ClaudeOpus4,
    /// Claude Haiku 4.5
    ClaudeHaiku45,
    /// Claude 3.5 Sonnet
    Claude35Sonnet,
    /// Claude 3.5 Haiku
    Claude35Haiku,
    /// Claude 3 Opus
    Claude3Opus,
    /// Claude 3 Sonnet
    Claude3Sonnet,
    /// Claude 3 Haiku
    Claude3Haiku,
    /// Custom model name
    Custom(String),
}

impl AnthropicModel {
    pub fn as_str(&self) -> &str {
        match self {
            // Claude 5 models
            Self::ClaudeFable5 => "claude-fable-5",
            // Claude 4.8 models
            Self::ClaudeOpus48 => "claude-opus-4-8",
            // Claude 4.7 models
            Self::ClaudeOpus47 => "claude-opus-4-7",
            // Claude 4.6 models
            Self::ClaudeOpus46 => "claude-opus-4-6",
            Self::ClaudeSonnet46 => "claude-sonnet-4-6",
            // Claude 4.5 models
            Self::ClaudeSonnet45 => "claude-sonnet-4-5",
            Self::ClaudeOpus45 => "claude-opus-4-5",
            Self::ClaudeHaiku45 => "claude-haiku-4-5",
            // Claude 4 models
            Self::ClaudeOpus41 => "claude-opus-4-1",
            Self::ClaudeSonnet4 => "claude-sonnet-4-0",
            Self::ClaudeOpus4 => "claude-opus-4-0",
            // Claude 3.5 models
            Self::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            Self::Claude35Haiku => "claude-3-5-haiku-20241022",
            // Claude 3 models
            Self::Claude3Opus => "claude-3-opus-20240229",
            Self::Claude3Sonnet => "claude-3-sonnet-20240229",
            Self::Claude3Haiku => "claude-3-haiku-20240307",
            Self::Custom(s) => s,
        }
    }
}

/// OpenRouter model variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpenRouterModel {
    /// OpenRouter auto-router alias.
    Auto,
    /// OpenRouter free-tier alias.
    Free,

    // OpenAI models
    Gpt5,
    Gpt5Mini,
    Gpt5Nano,
    Gpt5Codex,
    Gpt5_1,
    Gpt5_2,
    Gpt5_2Pro,
    Gpt5_3Chat,
    Gpt5_4,
    Gpt5_4Mini,
    Gpt5_4Nano,
    Gpt5_4Pro,
    Gpt5_5,
    Gpt5_5Pro,
    Gpt4_1,
    Gpt4o,
    O3,
    O3Pro,
    O3DeepResearch,
    O4Mini,
    GptOss120b,

    // Anthropic models
    ClaudeFable5,
    ClaudeSonnet4,
    ClaudeSonnet4_5,
    ClaudeOpus4_1,
    ClaudeOpus4_5,
    ClaudeOpus4_6,
    ClaudeOpus4_6Fast,
    ClaudeOpus4_7,
    ClaudeOpus4_7Fast,
    ClaudeOpus4_8,
    ClaudeOpus4_8Fast,
    ClaudeSonnet4_6,
    ClaudeHaiku4_5,
    Claude3_7Sonnet,

    // Google models
    Gemini35Flash,
    Gemini31ProPreview,
    Gemini31ProPreviewCustomTools,
    Gemini31FlashLite,
    Gemini31FlashLitePreview,
    Gemini31FlashImagePreview,
    Gemini3ProImagePreview,
    Gemini3FlashPreview,
    Gemini25Pro,
    Gemini25Flash,
    Gemini25FlashImage,

    // xAI models
    Grok4_3,
    Grok4_20,
    Grok4_20MultiAgent,
    GrokBuild0_1,
    Grok4,
    Grok4Fast,
    Grok4_1Fast,
    GrokCodeFast1,

    // Meta / Llama models
    Llama4Maverick,
    Llama4Scout,
    Llama3_3_70bInstruct,
    Llama3_2_11bVisionInstruct,

    // Qwen models
    Qwen3Max,
    Qwen3MaxThinking,
    Qwen3Coder,
    Qwen3CoderPlus,
    Qwen3_235bA22b,
    Qwen3Vl235bA22bInstruct,
    Qwen3Vl235bA22bThinking,
    Qwen3_7Max,
    Qwen3_7Plus,
    Qwen3_6MaxPreview,
    Qwen3_6Plus,
    Qwen3_6Flash,

    // DeepSeek models
    DeepseekChatV3_1,
    DeepseekR1,
    DeepseekV3_2,
    DeepseekV4Flash,
    DeepseekV4Pro,

    // Mistral models
    MistralLarge,
    MistralMedium3_1,
    Codestral2508,
    DevstralMedium,
    PixtralLarge2411,
    MistralLarge2512,
    MistralMedium3_5,
    Devstral2512,
    Ministral14b2512,

    // Perplexity models
    SonarPro,
    SonarReasoningPro,
    SonarDeepResearch,

    // Cohere models
    CommandA,

    // Moonshot AI models
    KimiK2_5,
    KimiK2Thinking,
    KimiK2_6,
    KimiK2_7Code,

    // Z.ai models
    Glm5,
    Glm5_1,
    Glm5Turbo,
    Glm4_7,

    // Xiaomi models
    MimoV2Omni,
    MimoV2Flash,
    MimoV2_5,
    MimoV2_5Pro,

    /// Any other OpenRouter `vendor/model` string.
    Custom(String),
}

impl OpenRouterModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Auto => "openrouter/auto",
            Self::Free => "openrouter/free",
            Self::Gpt5 => "openai/gpt-5",
            Self::Gpt5Mini => "openai/gpt-5-mini",
            Self::Gpt5Nano => "openai/gpt-5-nano",
            Self::Gpt5Codex => "openai/gpt-5-codex",
            Self::Gpt5_1 => "openai/gpt-5.1",
            Self::Gpt5_2 => "openai/gpt-5.2",
            Self::Gpt5_2Pro => "openai/gpt-5.2-pro",
            Self::Gpt5_3Chat => "openai/gpt-5.3-chat",
            Self::Gpt5_4 => "openai/gpt-5.4",
            Self::Gpt5_4Mini => "openai/gpt-5.4-mini",
            Self::Gpt5_4Nano => "openai/gpt-5.4-nano",
            Self::Gpt5_4Pro => "openai/gpt-5.4-pro",
            Self::Gpt5_5 => "openai/gpt-5.5",
            Self::Gpt5_5Pro => "openai/gpt-5.5-pro",
            Self::Gpt4_1 => "openai/gpt-4.1",
            Self::Gpt4o => "openai/gpt-4o",
            Self::O3 => "openai/o3",
            Self::O3Pro => "openai/o3-pro",
            Self::O3DeepResearch => "openai/o3-deep-research",
            Self::O4Mini => "openai/o4-mini",
            Self::GptOss120b => "openai/gpt-oss-120b",
            Self::ClaudeFable5 => "anthropic/claude-fable-5",
            Self::ClaudeSonnet4 => "anthropic/claude-sonnet-4",
            Self::ClaudeSonnet4_5 => "anthropic/claude-sonnet-4.5",
            Self::ClaudeOpus4_1 => "anthropic/claude-opus-4.1",
            Self::ClaudeOpus4_5 => "anthropic/claude-opus-4.5",
            Self::ClaudeOpus4_6 => "anthropic/claude-opus-4.6",
            Self::ClaudeOpus4_6Fast => "anthropic/claude-opus-4.6-fast",
            Self::ClaudeOpus4_7 => "anthropic/claude-opus-4.7",
            Self::ClaudeOpus4_7Fast => "anthropic/claude-opus-4.7-fast",
            Self::ClaudeOpus4_8 => "anthropic/claude-opus-4.8",
            Self::ClaudeOpus4_8Fast => "anthropic/claude-opus-4.8-fast",
            Self::ClaudeSonnet4_6 => "anthropic/claude-sonnet-4.6",
            Self::ClaudeHaiku4_5 => "anthropic/claude-haiku-4.5",
            Self::Claude3_7Sonnet => "anthropic/claude-3.7-sonnet",
            Self::Gemini35Flash => "google/gemini-3.5-flash",
            Self::Gemini31ProPreview => "google/gemini-3.1-pro-preview",
            Self::Gemini31ProPreviewCustomTools => "google/gemini-3.1-pro-preview-customtools",
            Self::Gemini31FlashLite => "google/gemini-3.1-flash-lite",
            Self::Gemini31FlashLitePreview => "google/gemini-3.1-flash-lite-preview",
            Self::Gemini31FlashImagePreview => "google/gemini-3.1-flash-image-preview",
            Self::Gemini3ProImagePreview => "google/gemini-3-pro-image-preview",
            Self::Gemini3FlashPreview => "google/gemini-3-flash-preview",
            Self::Gemini25Pro => "google/gemini-2.5-pro",
            Self::Gemini25Flash => "google/gemini-2.5-flash",
            Self::Gemini25FlashImage => "google/gemini-2.5-flash-image",
            Self::Grok4_3 => "x-ai/grok-4.3",
            Self::Grok4_20 => "x-ai/grok-4.20",
            Self::Grok4_20MultiAgent => "x-ai/grok-4.20-multi-agent",
            Self::GrokBuild0_1 => "x-ai/grok-build-0.1",
            Self::Grok4 => "x-ai/grok-4",
            Self::Grok4Fast => "x-ai/grok-4-fast",
            Self::Grok4_1Fast => "x-ai/grok-4.1-fast",
            Self::GrokCodeFast1 => "x-ai/grok-code-fast-1",
            Self::Llama4Maverick => "meta-llama/llama-4-maverick",
            Self::Llama4Scout => "meta-llama/llama-4-scout",
            Self::Llama3_3_70bInstruct => "meta-llama/llama-3.3-70b-instruct",
            Self::Llama3_2_11bVisionInstruct => "meta-llama/llama-3.2-11b-vision-instruct",
            Self::Qwen3Max => "qwen/qwen3-max",
            Self::Qwen3MaxThinking => "qwen/qwen3-max-thinking",
            Self::Qwen3Coder => "qwen/qwen3-coder",
            Self::Qwen3CoderPlus => "qwen/qwen3-coder-plus",
            Self::Qwen3_235bA22b => "qwen/qwen3-235b-a22b",
            Self::Qwen3Vl235bA22bInstruct => "qwen/qwen3-vl-235b-a22b-instruct",
            Self::Qwen3Vl235bA22bThinking => "qwen/qwen3-vl-235b-a22b-thinking",
            Self::Qwen3_7Max => "qwen/qwen3.7-max",
            Self::Qwen3_7Plus => "qwen/qwen3.7-plus",
            Self::Qwen3_6MaxPreview => "qwen/qwen3.6-max-preview",
            Self::Qwen3_6Plus => "qwen/qwen3.6-plus",
            Self::Qwen3_6Flash => "qwen/qwen3.6-flash",
            Self::DeepseekChatV3_1 => "deepseek/deepseek-chat-v3.1",
            Self::DeepseekR1 => "deepseek/deepseek-r1",
            Self::DeepseekV3_2 => "deepseek/deepseek-v3.2",
            Self::DeepseekV4Flash => "deepseek/deepseek-v4-flash",
            Self::DeepseekV4Pro => "deepseek/deepseek-v4-pro",
            Self::MistralLarge => "mistralai/mistral-large",
            Self::MistralMedium3_1 => "mistralai/mistral-medium-3.1",
            Self::Codestral2508 => "mistralai/codestral-2508",
            Self::DevstralMedium => "mistralai/devstral-medium",
            Self::PixtralLarge2411 => "mistralai/pixtral-large-2411",
            Self::MistralLarge2512 => "mistralai/mistral-large-2512",
            Self::MistralMedium3_5 => "mistralai/mistral-medium-3-5",
            Self::Devstral2512 => "mistralai/devstral-2512",
            Self::Ministral14b2512 => "mistralai/ministral-14b-2512",
            Self::SonarPro => "perplexity/sonar-pro",
            Self::SonarReasoningPro => "perplexity/sonar-reasoning-pro",
            Self::SonarDeepResearch => "perplexity/sonar-deep-research",
            Self::CommandA => "cohere/command-a",
            Self::KimiK2_5 => "moonshotai/kimi-k2.5",
            Self::KimiK2Thinking => "moonshotai/kimi-k2-thinking",
            Self::KimiK2_6 => "moonshotai/kimi-k2.6",
            Self::KimiK2_7Code => "moonshotai/kimi-k2.7-code",
            Self::Glm5 => "z-ai/glm-5",
            Self::Glm5_1 => "z-ai/glm-5.1",
            Self::Glm5Turbo => "z-ai/glm-5-turbo",
            Self::Glm4_7 => "z-ai/glm-4.7",
            Self::MimoV2Omni => "xiaomi/mimo-v2-omni",
            Self::MimoV2Flash => "xiaomi/mimo-v2-flash",
            Self::MimoV2_5 => "xiaomi/mimo-v2.5",
            Self::MimoV2_5Pro => "xiaomi/mimo-v2.5-pro",
            Self::Custom(s) => s,
        }
    }
}

impl std::str::FromStr for OpenRouterModel {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let model = match value {
            "openrouter/auto" => Self::Auto,
            "openrouter/free" => Self::Free,
            "openai/gpt-5" => Self::Gpt5,
            "openai/gpt-5-mini" => Self::Gpt5Mini,
            "openai/gpt-5-nano" => Self::Gpt5Nano,
            "openai/gpt-5-codex" => Self::Gpt5Codex,
            "openai/gpt-5.1" => Self::Gpt5_1,
            "openai/gpt-5.2" => Self::Gpt5_2,
            "openai/gpt-5.2-pro" => Self::Gpt5_2Pro,
            "openai/gpt-5.3-chat" => Self::Gpt5_3Chat,
            "openai/gpt-5.4" => Self::Gpt5_4,
            "openai/gpt-5.4-mini" => Self::Gpt5_4Mini,
            "openai/gpt-5.4-nano" => Self::Gpt5_4Nano,
            "openai/gpt-5.4-pro" => Self::Gpt5_4Pro,
            "openai/gpt-5.5" => Self::Gpt5_5,
            "openai/gpt-5.5-pro" => Self::Gpt5_5Pro,
            "openai/gpt-4.1" => Self::Gpt4_1,
            "openai/gpt-4o" => Self::Gpt4o,
            "openai/o3" => Self::O3,
            "openai/o3-pro" => Self::O3Pro,
            "openai/o3-deep-research" => Self::O3DeepResearch,
            "openai/o4-mini" => Self::O4Mini,
            "openai/gpt-oss-120b" => Self::GptOss120b,
            "anthropic/claude-fable-5" => Self::ClaudeFable5,
            "anthropic/claude-sonnet-4" => Self::ClaudeSonnet4,
            "anthropic/claude-sonnet-4.5" => Self::ClaudeSonnet4_5,
            "anthropic/claude-opus-4.1" => Self::ClaudeOpus4_1,
            "anthropic/claude-opus-4.5" => Self::ClaudeOpus4_5,
            "anthropic/claude-opus-4.6" => Self::ClaudeOpus4_6,
            "anthropic/claude-opus-4.6-fast" => Self::ClaudeOpus4_6Fast,
            "anthropic/claude-opus-4.7" => Self::ClaudeOpus4_7,
            "anthropic/claude-opus-4.7-fast" => Self::ClaudeOpus4_7Fast,
            "anthropic/claude-opus-4.8" => Self::ClaudeOpus4_8,
            "anthropic/claude-opus-4.8-fast" => Self::ClaudeOpus4_8Fast,
            "anthropic/claude-sonnet-4.6" => Self::ClaudeSonnet4_6,
            "anthropic/claude-haiku-4.5" => Self::ClaudeHaiku4_5,
            "anthropic/claude-3.7-sonnet" => Self::Claude3_7Sonnet,
            "google/gemini-3.5-flash" => Self::Gemini35Flash,
            "google/gemini-3.1-pro-preview" => Self::Gemini31ProPreview,
            "google/gemini-3.1-pro-preview-customtools" => Self::Gemini31ProPreviewCustomTools,
            "google/gemini-3.1-flash-lite" => Self::Gemini31FlashLite,
            "google/gemini-3.1-flash-lite-preview" => Self::Gemini31FlashLitePreview,
            "google/gemini-3.1-flash-image-preview" => Self::Gemini31FlashImagePreview,
            "google/gemini-3-pro-image-preview" => Self::Gemini3ProImagePreview,
            "google/gemini-3-flash-preview" => Self::Gemini3FlashPreview,
            "google/gemini-2.5-pro" => Self::Gemini25Pro,
            "google/gemini-2.5-flash" => Self::Gemini25Flash,
            "google/gemini-2.5-flash-image" => Self::Gemini25FlashImage,
            "x-ai/grok-4.3" => Self::Grok4_3,
            "x-ai/grok-4.20" => Self::Grok4_20,
            "x-ai/grok-4.20-multi-agent" => Self::Grok4_20MultiAgent,
            "x-ai/grok-build-0.1" => Self::GrokBuild0_1,
            "x-ai/grok-4" => Self::Grok4,
            "x-ai/grok-4-fast" => Self::Grok4Fast,
            "x-ai/grok-4.1-fast" => Self::Grok4_1Fast,
            "x-ai/grok-code-fast-1" => Self::GrokCodeFast1,
            "meta-llama/llama-4-maverick" => Self::Llama4Maverick,
            "meta-llama/llama-4-scout" => Self::Llama4Scout,
            "meta-llama/llama-3.3-70b-instruct" => Self::Llama3_3_70bInstruct,
            "meta-llama/llama-3.2-11b-vision-instruct" => Self::Llama3_2_11bVisionInstruct,
            "qwen/qwen3-max" => Self::Qwen3Max,
            "qwen/qwen3-max-thinking" => Self::Qwen3MaxThinking,
            "qwen/qwen3-coder" => Self::Qwen3Coder,
            "qwen/qwen3-coder-plus" => Self::Qwen3CoderPlus,
            "qwen/qwen3-235b-a22b" => Self::Qwen3_235bA22b,
            "qwen/qwen3-vl-235b-a22b-instruct" => Self::Qwen3Vl235bA22bInstruct,
            "qwen/qwen3-vl-235b-a22b-thinking" => Self::Qwen3Vl235bA22bThinking,
            "qwen/qwen3.7-max" => Self::Qwen3_7Max,
            "qwen/qwen3.7-plus" => Self::Qwen3_7Plus,
            "qwen/qwen3.6-max-preview" => Self::Qwen3_6MaxPreview,
            "qwen/qwen3.6-plus" => Self::Qwen3_6Plus,
            "qwen/qwen3.6-flash" => Self::Qwen3_6Flash,
            "deepseek/deepseek-chat-v3.1" => Self::DeepseekChatV3_1,
            "deepseek/deepseek-r1" => Self::DeepseekR1,
            "deepseek/deepseek-v3.2" => Self::DeepseekV3_2,
            "deepseek/deepseek-v4-flash" => Self::DeepseekV4Flash,
            "deepseek/deepseek-v4-pro" => Self::DeepseekV4Pro,
            "mistralai/mistral-large" => Self::MistralLarge,
            "mistralai/mistral-medium-3.1" => Self::MistralMedium3_1,
            "mistralai/codestral-2508" => Self::Codestral2508,
            "mistralai/devstral-medium" => Self::DevstralMedium,
            "mistralai/pixtral-large-2411" => Self::PixtralLarge2411,
            "mistralai/mistral-large-2512" => Self::MistralLarge2512,
            "mistralai/mistral-medium-3-5" => Self::MistralMedium3_5,
            "mistralai/devstral-2512" => Self::Devstral2512,
            "mistralai/ministral-14b-2512" => Self::Ministral14b2512,
            "perplexity/sonar-pro" => Self::SonarPro,
            "perplexity/sonar-reasoning-pro" => Self::SonarReasoningPro,
            "perplexity/sonar-deep-research" => Self::SonarDeepResearch,
            "cohere/command-a" => Self::CommandA,
            "moonshotai/kimi-k2.5" => Self::KimiK2_5,
            "moonshotai/kimi-k2-thinking" => Self::KimiK2Thinking,
            "moonshotai/kimi-k2.6" => Self::KimiK2_6,
            "moonshotai/kimi-k2.7-code" => Self::KimiK2_7Code,
            "z-ai/glm-5" => Self::Glm5,
            "z-ai/glm-5.1" => Self::Glm5_1,
            "z-ai/glm-5-turbo" => Self::Glm5Turbo,
            "z-ai/glm-4.7" => Self::Glm4_7,
            "xiaomi/mimo-v2-omni" => Self::MimoV2Omni,
            "xiaomi/mimo-v2-flash" => Self::MimoV2Flash,
            "xiaomi/mimo-v2.5" => Self::MimoV2_5,
            "xiaomi/mimo-v2.5-pro" => Self::MimoV2_5Pro,
            other => Self::Custom(other.to_string()),
        };

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_provider_mapping() {
        assert_eq!(Model::gpt4o_mini().provider(), ProviderKind::OpenAI);
        assert_eq!(
            Model::claude_sonnet_45().provider(),
            ProviderKind::Anthropic
        );
        assert_eq!(Model::gpt_5_3_instant().provider(), ProviderKind::OpenAI);
        assert_eq!(
            Model::claude_sonnet_46().provider(),
            ProviderKind::Anthropic
        );
        assert_eq!(Model::claude_opus_46().provider(), ProviderKind::Anthropic);
        assert_eq!(
            Model::openrouter_auto().provider(),
            ProviderKind::OpenRouter
        );
    }

    #[test]
    fn model_string_identifiers() {
        assert_eq!(Model::gpt4o_mini().as_str(), "gpt-4o-mini");
        assert_eq!(Model::claude_sonnet_45().as_str(), "claude-sonnet-4-5");
        assert_eq!(Model::o3_mini().as_str(), "o3-mini");
        assert_eq!(Model::gpt_5_3_instant().as_str(), "gpt-5.3-instant");
        assert_eq!(Model::claude_sonnet_46().as_str(), "claude-sonnet-4-6");
        assert_eq!(Model::claude_opus_46().as_str(), "claude-opus-4-6");
        assert_eq!(Model::openrouter_auto().as_str(), "openrouter/auto");
        assert_eq!(Model::openrouter_gpt5().as_str(), "openai/gpt-5");
        assert_eq!(Model::openrouter_qwen3_coder().as_str(), "qwen/qwen3-coder");
    }

    #[test]
    fn openrouter_model_from_str_handles_known_and_custom_models() {
        assert_eq!(
            "openrouter/auto".parse::<OpenRouterModel>(),
            Ok(OpenRouterModel::Auto)
        );
        assert_eq!(
            "acme/custom-model".parse::<OpenRouterModel>(),
            Ok(OpenRouterModel::Custom("acme/custom-model".to_string()))
        );
    }

    #[test]
    fn reasoning_model_detection() {
        assert!(OpenAIModel::O3.is_reasoning_model());
        assert!(OpenAIModel::O4Mini.is_reasoning_model());
        assert!(!OpenAIModel::Gpt4oMini.is_reasoning_model());
    }

    #[test]
    fn openai_gpt5_family_string_identifiers() {
        let cases = [
            (OpenAIModel::Gpt5_5, "gpt-5.5"),
            (OpenAIModel::Gpt5_4, "gpt-5.4"),
            (OpenAIModel::Gpt5_4Mini, "gpt-5.4-mini"),
            (OpenAIModel::Gpt5_4Nano, "gpt-5.4-nano"),
            (OpenAIModel::Gpt5, "gpt-5"),
            (OpenAIModel::Gpt5Mini, "gpt-5-mini"),
            (OpenAIModel::Gpt5Nano, "gpt-5-nano"),
            (OpenAIModel::Gpt5Codex, "gpt-5-codex"),
            (OpenAIModel::Gpt5_1, "gpt-5.1"),
            (OpenAIModel::Gpt5_2, "gpt-5.2"),
            (OpenAIModel::Gpt5_2Pro, "gpt-5.2-pro"),
            (OpenAIModel::Gpt5_3Instant, "gpt-5.3-instant"),
            (OpenAIModel::Gpt5_3Chat, "gpt-5.3-chat"),
            (OpenAIModel::Gpt4_1, "gpt-4.1"),
        ];

        for (model, expected) in cases {
            assert_eq!(model.as_str(), expected);
            // GPT-5 / GPT-4.1 are not treated as o-series reasoning models.
            assert!(!model.is_reasoning_model());
        }
    }

    #[test]
    fn model_gpt5_constructors_map_to_provider() {
        assert_eq!(Model::gpt5().as_str(), "gpt-5");
        assert_eq!(Model::gpt5().provider(), ProviderKind::OpenAI);
        assert_eq!(Model::gpt_5_5().as_str(), "gpt-5.5");
        assert_eq!(Model::gpt_5_4().as_str(), "gpt-5.4");
        assert_eq!(Model::gpt_5_4_mini().as_str(), "gpt-5.4-mini");
        assert_eq!(Model::gpt_5_4_nano().as_str(), "gpt-5.4-nano");
        assert_eq!(Model::gpt4_1().as_str(), "gpt-4.1");
    }

    #[test]
    fn anthropic_latest_models_string_identifiers() {
        let cases = [
            (AnthropicModel::ClaudeFable5, "claude-fable-5"),
            (AnthropicModel::ClaudeOpus48, "claude-opus-4-8"),
            (AnthropicModel::ClaudeOpus47, "claude-opus-4-7"),
            (AnthropicModel::ClaudeOpus41, "claude-opus-4-1"),
        ];

        for (model, expected) in cases {
            assert_eq!(model.as_str(), expected);
        }

        assert_eq!(Model::claude_fable_5().as_str(), "claude-fable-5");
        assert_eq!(Model::claude_opus_48().as_str(), "claude-opus-4-8");
        assert_eq!(Model::claude_opus_47().as_str(), "claude-opus-4-7");
        assert_eq!(Model::claude_opus_41().provider(), ProviderKind::Anthropic);
    }

    #[test]
    fn openrouter_latest_catalog_models_round_trip() {
        let cases = [
            (OpenRouterModel::Gpt5_5, "openai/gpt-5.5"),
            (OpenRouterModel::ClaudeFable5, "anthropic/claude-fable-5"),
            (OpenRouterModel::ClaudeOpus4_8, "anthropic/claude-opus-4.8"),
            (OpenRouterModel::Gemini35Flash, "google/gemini-3.5-flash"),
            (OpenRouterModel::Grok4_3, "x-ai/grok-4.3"),
            (OpenRouterModel::Qwen3_7Max, "qwen/qwen3.7-max"),
            (OpenRouterModel::DeepseekV4Pro, "deepseek/deepseek-v4-pro"),
            (OpenRouterModel::KimiK2_7Code, "moonshotai/kimi-k2.7-code"),
        ];

        for (model, expected) in cases {
            assert_eq!(model.as_str(), expected);
            assert_eq!(expected.parse::<OpenRouterModel>(), Ok(model));
        }
    }
}
