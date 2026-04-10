use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};

use crate::error;

/// Configuration for text generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Temperature for sampling (0.0 to 2.0 typically).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,

    /// Top-p (nucleus) sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling (not supported by all providers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to request JSON output (provider-dependent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<bool>,

    /// JSON Schema for structured output (provider-dependent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,

    /// Maximum number of tool execution rounds before failing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_rounds: Option<usize>,
}

impl GenerationConfig {
    /// Start with an empty config — add only the overrides you need.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    pub fn with_json_mode(mut self, json_mode: bool) -> Self {
        self.json_mode = Some(json_mode);
        self
    }

    pub fn with_json_schema(mut self, json_schema: serde_json::Value) -> Self {
        self.json_schema = Some(json_schema);
        self
    }

    /// Generate a JSON Schema from a Rust type and normalize object schemas
    /// for strict structured-output providers.
    pub fn with_json_schema_for<T>(mut self) -> error::Result<Self>
    where
        T: JsonSchema,
    {
        let mut schema = serde_json::to_value(schema_for!(T))?;
        normalize_strict_json_schema(&mut schema);
        self.json_schema = Some(schema);
        Ok(self)
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = Some(max_tool_rounds);
        self
    }

    /// The effective maximum number of tool-calling rounds (defaults to 8).
    pub fn tool_round_limit(&self) -> usize {
        self.max_tool_rounds.unwrap_or(8)
    }
}

/// Recursively ensure every JSON object schema has `additionalProperties: false`.
///
/// This satisfies providers (like OpenAI strict mode) that require closed schemas.
pub(crate) fn normalize_strict_json_schema(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(obj) => {
            let is_object_schema = obj.get("type").and_then(serde_json::Value::as_str)
                == Some("object")
                || obj.contains_key("properties");

            if is_object_schema {
                obj.entry("type")
                    .or_insert(serde_json::Value::String("object".to_string()));
                obj.entry("additionalProperties")
                    .or_insert(serde_json::Value::Bool(false));
            }

            for value in obj.values_mut() {
                normalize_strict_json_schema(value);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                normalize_strict_json_schema(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chain() {
        let config = GenerationConfig::new()
            .with_temperature(0.5)
            .with_max_tokens(1024)
            .with_top_p(0.9);

        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn tool_round_limit_default() {
        assert_eq!(GenerationConfig::new().tool_round_limit(), 8);
        assert_eq!(
            GenerationConfig::new()
                .with_max_tool_rounds(3)
                .tool_round_limit(),
            3
        );
    }

    #[test]
    fn normalize_adds_additional_properties() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        normalize_strict_json_schema(&mut schema);
        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn normalize_adds_missing_object_type_when_properties_exist() {
        let mut schema = serde_json::json!({
            "properties": {
                "name": { "type": "string" }
            }
        });
        normalize_strict_json_schema(&mut schema);
        assert_eq!(
            schema["type"],
            serde_json::Value::String("object".to_string())
        );
        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn normalize_preserves_explicit_additional_properties() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "entries": {
                    "type": "object",
                    "additionalProperties": { "type": "string" }
                }
            }
        });
        normalize_strict_json_schema(&mut schema);
        // Root gets `false` added
        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        // Nested explicit value is preserved
        assert_eq!(
            schema["properties"]["entries"]["additionalProperties"],
            serde_json::json!({ "type": "string" })
        );
    }
}
