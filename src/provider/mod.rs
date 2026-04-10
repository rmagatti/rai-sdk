#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

use crate::message::Usage;

#[derive(Debug, Clone)]
pub enum ProviderStreamEvent {
    Text(String),
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallChunk {
        id: String,
        arguments: String,
    },
    Done {
        finish_reason: Option<String>,
        usage: Option<Usage>,
    },
}
