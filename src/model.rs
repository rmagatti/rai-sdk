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

    pub fn o3_mini() -> Self {
        Model::OpenAI(OpenAIModel::O3Mini)
    }

    pub fn o3() -> Self {
        Model::OpenAI(OpenAIModel::O3)
    }

    pub fn o4_mini() -> Self {
        Model::OpenAI(OpenAIModel::O4Mini)
    }

    pub fn gpt_5_3_instant() -> Self {
        Model::OpenAI(OpenAIModel::Gpt5_3Instant)
    }

    // ── Anthropic convenience constructors ──

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
    /// GPT-4o
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
    /// GPT-5.3 Instant — latest real-time high-accuracy model
    Gpt5_3Instant,
    /// Custom model name
    Custom(String),
}

impl OpenAIModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4Turbo => "gpt-4-turbo",
            Self::O1Preview => "o1-preview",
            Self::O1Mini => "o1-mini",
            Self::O3Mini => "o3-mini",
            Self::O3 => "o3",
            Self::O4Mini => "o4-mini",
            Self::Gpt5_3Instant => "gpt-5.3-instant",
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
    /// Claude Sonnet 4.6 — best balance of intelligence, speed and cost (latest)
    ClaudeSonnet46,
    /// Claude Opus 4.6 — most intelligent model with 1M context (latest)
    ClaudeOpus46,
    /// Claude Sonnet 4.5
    ClaudeSonnet45,
    /// Claude Opus 4.5
    ClaudeOpus45,
    /// Claude Haiku 4.5
    ClaudeHaiku45,
    /// Claude Sonnet 4
    ClaudeSonnet4,
    /// Claude Opus 4
    ClaudeOpus4,
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
            Self::ClaudeSonnet46 => "claude-sonnet-4-6",
            Self::ClaudeOpus46 => "claude-opus-4-6",
            Self::ClaudeSonnet45 => "claude-sonnet-4-5",
            Self::ClaudeOpus45 => "claude-opus-4-5",
            Self::ClaudeHaiku45 => "claude-haiku-4-5",
            Self::ClaudeSonnet4 => "claude-sonnet-4-0",
            Self::ClaudeOpus4 => "claude-opus-4-0",
            Self::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            Self::Claude35Haiku => "claude-3-5-haiku-20241022",
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
    Gpt4_1,
    Gpt4o,
    O3,
    O3Pro,
    O3DeepResearch,
    O4Mini,
    GptOss120b,

    // Anthropic models
    ClaudeSonnet4,
    ClaudeSonnet4_5,
    ClaudeOpus4_1,
    ClaudeOpus4_5,
    ClaudeHaiku4_5,
    Claude3_7Sonnet,

    // Google models
    Gemini31ProPreview,
    Gemini3FlashPreview,
    Gemini25Pro,
    Gemini25Flash,
    Gemini25FlashImage,

    // xAI models
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

    // DeepSeek models
    DeepseekChatV3_1,
    DeepseekR1,
    DeepseekV3_2,

    // Mistral models
    MistralLarge,
    MistralMedium3_1,
    Codestral2508,
    DevstralMedium,
    PixtralLarge2411,

    // Perplexity models
    SonarPro,
    SonarReasoningPro,
    SonarDeepResearch,

    // Cohere models
    CommandA,

    // Moonshot AI models
    KimiK2_5,
    KimiK2Thinking,

    // Z.ai models
    Glm5,

    // Xiaomi models
    MimoV2Omni,

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
            Self::Gpt4_1 => "openai/gpt-4.1",
            Self::Gpt4o => "openai/gpt-4o",
            Self::O3 => "openai/o3",
            Self::O3Pro => "openai/o3-pro",
            Self::O3DeepResearch => "openai/o3-deep-research",
            Self::O4Mini => "openai/o4-mini",
            Self::GptOss120b => "openai/gpt-oss-120b",
            Self::ClaudeSonnet4 => "anthropic/claude-sonnet-4",
            Self::ClaudeSonnet4_5 => "anthropic/claude-sonnet-4.5",
            Self::ClaudeOpus4_1 => "anthropic/claude-opus-4.1",
            Self::ClaudeOpus4_5 => "anthropic/claude-opus-4.5",
            Self::ClaudeHaiku4_5 => "anthropic/claude-haiku-4.5",
            Self::Claude3_7Sonnet => "anthropic/claude-3.7-sonnet",
            Self::Gemini31ProPreview => "google/gemini-3.1-pro-preview",
            Self::Gemini3FlashPreview => "google/gemini-3-flash-preview",
            Self::Gemini25Pro => "google/gemini-2.5-pro",
            Self::Gemini25Flash => "google/gemini-2.5-flash",
            Self::Gemini25FlashImage => "google/gemini-2.5-flash-image",
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
            Self::DeepseekChatV3_1 => "deepseek/deepseek-chat-v3.1",
            Self::DeepseekR1 => "deepseek/deepseek-r1",
            Self::DeepseekV3_2 => "deepseek/deepseek-v3.2",
            Self::MistralLarge => "mistralai/mistral-large",
            Self::MistralMedium3_1 => "mistralai/mistral-medium-3.1",
            Self::Codestral2508 => "mistralai/codestral-2508",
            Self::DevstralMedium => "mistralai/devstral-medium",
            Self::PixtralLarge2411 => "mistralai/pixtral-large-2411",
            Self::SonarPro => "perplexity/sonar-pro",
            Self::SonarReasoningPro => "perplexity/sonar-reasoning-pro",
            Self::SonarDeepResearch => "perplexity/sonar-deep-research",
            Self::CommandA => "cohere/command-a",
            Self::KimiK2_5 => "moonshotai/kimi-k2.5",
            Self::KimiK2Thinking => "moonshotai/kimi-k2-thinking",
            Self::Glm5 => "z-ai/glm-5",
            Self::MimoV2Omni => "xiaomi/mimo-v2-omni",
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
            "openai/gpt-4.1" => Self::Gpt4_1,
            "openai/gpt-4o" => Self::Gpt4o,
            "openai/o3" => Self::O3,
            "openai/o3-pro" => Self::O3Pro,
            "openai/o3-deep-research" => Self::O3DeepResearch,
            "openai/o4-mini" => Self::O4Mini,
            "openai/gpt-oss-120b" => Self::GptOss120b,
            "anthropic/claude-sonnet-4" => Self::ClaudeSonnet4,
            "anthropic/claude-sonnet-4.5" => Self::ClaudeSonnet4_5,
            "anthropic/claude-opus-4.1" => Self::ClaudeOpus4_1,
            "anthropic/claude-opus-4.5" => Self::ClaudeOpus4_5,
            "anthropic/claude-haiku-4.5" => Self::ClaudeHaiku4_5,
            "anthropic/claude-3.7-sonnet" => Self::Claude3_7Sonnet,
            "google/gemini-3.1-pro-preview" => Self::Gemini31ProPreview,
            "google/gemini-3-flash-preview" => Self::Gemini3FlashPreview,
            "google/gemini-2.5-pro" => Self::Gemini25Pro,
            "google/gemini-2.5-flash" => Self::Gemini25Flash,
            "google/gemini-2.5-flash-image" => Self::Gemini25FlashImage,
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
            "deepseek/deepseek-chat-v3.1" => Self::DeepseekChatV3_1,
            "deepseek/deepseek-r1" => Self::DeepseekR1,
            "deepseek/deepseek-v3.2" => Self::DeepseekV3_2,
            "mistralai/mistral-large" => Self::MistralLarge,
            "mistralai/mistral-medium-3.1" => Self::MistralMedium3_1,
            "mistralai/codestral-2508" => Self::Codestral2508,
            "mistralai/devstral-medium" => Self::DevstralMedium,
            "mistralai/pixtral-large-2411" => Self::PixtralLarge2411,
            "perplexity/sonar-pro" => Self::SonarPro,
            "perplexity/sonar-reasoning-pro" => Self::SonarReasoningPro,
            "perplexity/sonar-deep-research" => Self::SonarDeepResearch,
            "cohere/command-a" => Self::CommandA,
            "moonshotai/kimi-k2.5" => Self::KimiK2_5,
            "moonshotai/kimi-k2-thinking" => Self::KimiK2Thinking,
            "z-ai/glm-5" => Self::Glm5,
            "xiaomi/mimo-v2-omni" => Self::MimoV2Omni,
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
}
