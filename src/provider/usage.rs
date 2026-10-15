//! Shared response usage fields for Chat Completions and compatible extensions.

use serde::Deserialize;

#[cfg(any(feature = "openai", feature = "openrouter"))]
use crate::message::Usage;

#[derive(Debug, Deserialize)]
pub(super) struct CostDetails {
    pub upstream_inference_cost: Option<f64>,
}

#[cfg(any(feature = "openai", feature = "openrouter"))]
#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionsUsage {
    prompt_tokens: Option<i32>,
    completion_tokens: Option<i32>,
    total_tokens: Option<i32>,
    prompt_tokens_details: Option<PromptTokensDetails>,
    cost: Option<f64>,
    cost_details: Option<CostDetails>,
}

#[cfg(any(feature = "openai", feature = "openrouter"))]
#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<i32>,
    cache_write_tokens: Option<i32>,
}

#[cfg(any(feature = "openai", feature = "openrouter"))]
impl From<ChatCompletionsUsage> for Usage {
    fn from(usage: ChatCompletionsUsage) -> Self {
        let details = usage.prompt_tokens_details;
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cache_read_tokens: details.as_ref().and_then(|d| d.cached_tokens),
            cache_write_tokens: details.and_then(|d| d.cache_write_tokens),
            cost_usd: usage.cost,
            upstream_inference_cost_usd: usage.cost_details.and_then(|d| d.upstream_inference_cost),
        }
    }
}
