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
    /// Custom model name
    Custom(String),
}

impl OpenRouterModel {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Custom(s) => s,
        }
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
    }

    #[test]
    fn model_string_identifiers() {
        assert_eq!(Model::gpt4o_mini().as_str(), "gpt-4o-mini");
        assert_eq!(Model::claude_sonnet_45().as_str(), "claude-sonnet-4-5");
        assert_eq!(Model::o3_mini().as_str(), "o3-mini");
        assert_eq!(Model::gpt_5_3_instant().as_str(), "gpt-5.3-instant");
        assert_eq!(Model::claude_sonnet_46().as_str(), "claude-sonnet-4-6");
        assert_eq!(Model::claude_opus_46().as_str(), "claude-opus-4-6");
    }

    #[test]
    fn reasoning_model_detection() {
        assert!(OpenAIModel::O3.is_reasoning_model());
        assert!(OpenAIModel::O4Mini.is_reasoning_model());
        assert!(!OpenAIModel::Gpt4oMini.is_reasoning_model());
    }
}
