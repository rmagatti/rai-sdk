//! Per-request generation settings.
//!
//! [`GenerationConfig`] carries the knobs that vary from request to request:
//! sampling parameters, token limits, stop sequences, JSON/structured-output
//! mode, and the tool-loop limit. It is built with chained `with_*` methods and
//! can be attached to a client as a default or to a single request as an
//! override.
//!
//! Not every provider honours every field. `top_k` is ignored by providers that
//! do not support it, and OpenAI reasoning models drop `temperature`/`top_p`.
//!
//! # Examples
//!
//! ```no_run
//! use rai_sdk::GenerationConfig;
//!
//! let config = GenerationConfig::new()
//!     .with_temperature(0.2)
//!     .with_max_tokens(1024)
//!     .with_stop_sequences(vec!["\n\n".to_string()]);
//!
//! assert_eq!(config.tool_round_limit(), 8);
//! ```

use schemars::{JsonSchema, generate::SchemaSettings};
use serde::{Deserialize, Serialize};

use crate::error;

/// Configuration for text generation.
///
/// Every field is optional; unset fields are simply omitted from the provider
/// request, so the provider default applies.
///
/// # Examples
///
/// ```no_run
/// use rai_sdk::{GenerationConfig, JsonSchema};
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// #[schemars(crate = "rai_sdk::schemars")]
/// struct Summary {
///     headline: String,
///     bullets: Vec<String>,
/// }
///
/// // Free-form generation.
/// let creative = GenerationConfig::new().with_temperature(0.9);
///
/// // Schema-constrained generation derived from a Rust type.
/// let strict = GenerationConfig::new()
///     .with_temperature(0.0)
///     .with_json_schema_for::<Summary>()?;
/// # let _ = (creative, strict);
/// # Ok::<(), rai_sdk::Error>(())
/// ```
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

    /// Set the sampling temperature. Higher values are more random.
    ///
    /// Ignored for OpenAI reasoning (o-series) models, which do not accept it.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Cap the number of tokens the model may generate.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set nucleus (top-p) sampling.
    ///
    /// Ignored for OpenAI reasoning (o-series) models.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling. Only sent to providers that support it.
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Stop generating as soon as one of these sequences is produced.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    /// Ask the provider for syntactically valid JSON without constraining its
    /// shape.
    ///
    /// A JSON schema set through [`GenerationConfig::with_json_schema`] or
    /// [`GenerationConfig::with_json_schema_for`] takes precedence over this
    /// flag.
    pub fn with_json_mode(mut self, json_mode: bool) -> Self {
        self.json_mode = Some(json_mode);
        self
    }

    /// Constrain the response with a hand-written JSON Schema.
    ///
    /// Prefer [`GenerationConfig::with_json_schema_for`] when the shape is
    /// already expressed as a Rust type.
    pub fn with_json_schema(mut self, json_schema: serde_json::Value) -> Self {
        self.json_schema = Some(json_schema);
        self
    }

    /// Generate a JSON Schema from a Rust type and normalize object schemas
    /// for strict structured-output providers.
    ///
    /// The schema is generated with `inline_subschemas = true` and no top-level
    /// `"$schema"` key, so non-recursive nested types are inlined directly rather
    /// than emitting `"$defs"`/`"$ref"`. This matters because Gemini's
    /// `generation_config.response_schema` (reachable in this SDK through the
    /// OpenRouter provider) rejects schemas containing `"$schema"`, `"$defs"`, or
    /// `"$ref"` keys with a 400 INVALID_ARGUMENT error. See
    /// [`normalize_strict_json_schema`] for the limits of this approach with
    /// recursive types.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`](crate::Error::Serialization) if the
    /// schema generated for `T` cannot be converted to a JSON value.
    pub fn with_json_schema_for<T>(mut self) -> error::Result<Self>
    where
        T: JsonSchema,
    {
        let generator = SchemaSettings::default()
            .with(|settings| {
                settings.inline_subschemas = true;
                settings.meta_schema = None;
            })
            .into_generator();
        let mut schema = serde_json::to_value(generator.into_root_schema_for::<T>())?;
        normalize_strict_json_schema(&mut schema);
        self.json_schema = Some(schema);
        Ok(self)
    }

    /// Limit how many request/tool-execution rounds a single `generate()` call
    /// may run before failing with
    /// [`Error::ToolLoopLimitExceeded`](crate::Error::ToolLoopLimitExceeded).
    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = Some(max_tool_rounds);
        self
    }

    /// The effective maximum number of tool-calling rounds (defaults to 8).
    pub fn tool_round_limit(&self) -> usize {
        self.max_tool_rounds.unwrap_or(8)
    }
}

/// Normalize a generated JSON Schema for strict structured-output providers
/// (notably Gemini's `generation_config.response_schema`, which rejects
/// unrecognized keywords with a 400 INVALID_ARGUMENT error).
///
/// This recursively:
/// - defaults `"additionalProperties"` to `false` on every object schema, without
///   overriding an explicit value that's already present (existing behavior), and
/// - strips any `"$schema"` key, wherever it appears, since Gemini rejects it.
///
/// Note on recursive types: [`GenerationConfig::with_json_schema_for`] configures
/// the schemars generator with `inline_subschemas = true`, which inlines
/// non-recursive nested types so no `"$defs"`/`"$ref"` keys are produced in the
/// common case. However, schemars must still fall back to emitting
/// `"$defs"`/`"$ref"` for *recursive* types (a type that transitively contains
/// itself), since an infinitely-nested structure can't be inlined. This function
/// deliberately does NOT attempt to resolve or flatten those references — doing
/// so would require a general `$ref`-resolution pass, which is out of scope here.
/// Structured-output types passed to `with_json_schema_for` must stay
/// non-recursive to work with Gemini; recursive types will still be rejected.
pub(crate) fn normalize_strict_json_schema(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(obj) => {
            obj.remove("$schema");

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

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct StructuredAnswer {
        answer: String,
        confidence: f64,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct NestedMetadata {
        tags: Vec<String>,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct StructuredEnvelope {
        answer: StructuredAnswer,
        metadata: NestedMetadata,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct Inner {
        a: u64,
        b: String,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct Outer {
        items: Vec<Inner>,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    enum StructuredChoice {
        First,
        Second,
    }

    #[allow(dead_code)]
    #[derive(JsonSchema)]
    struct StructuredWithOptionalAndEnum {
        required_field: String,
        optional_field: Option<String>,
        choice: StructuredChoice,
    }

    /// Recursively asserts that no object key anywhere in the schema starts
    /// with `$` (e.g. `$schema`, `$defs`, `$ref`), which Gemini rejects.
    fn assert_no_dollar_keys(value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                for (key, nested) in object {
                    assert!(
                        !key.starts_with('$'),
                        "schema should not contain a '{key}' keyword: {value}"
                    );
                    assert_no_dollar_keys(nested);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    assert_no_dollar_keys(item);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn test_generation_config_with_json_schema_for() -> error::Result<()> {
        // schema_for! uses the default schemars settings (top-level "$schema",
        // no inlining), which is not what `with_json_schema_for` produces
        // anymore now that it targets Gemini compatibility. Build the expected
        // value with the same settings `with_json_schema_for` uses instead of
        // relying on the macro directly.
        let generator = SchemaSettings::default()
            .with(|settings| {
                settings.inline_subschemas = true;
                settings.meta_schema = None;
            })
            .into_generator();
        let mut expected_schema =
            serde_json::to_value(generator.into_root_schema_for::<StructuredAnswer>())?;
        normalize_strict_json_schema(&mut expected_schema);
        let config = GenerationConfig::new().with_json_schema_for::<StructuredAnswer>()?;

        assert_eq!(config.json_schema, Some(expected_schema));

        Ok(())
    }

    #[test]
    fn test_generation_config_with_json_schema_for_inlines_nested_objects() -> error::Result<()> {
        // Gemini's generation_config.response_schema rejects "$schema",
        // "$defs", and "$ref" — non-recursive nested types must be inlined
        // instead.
        let config = GenerationConfig::new().with_json_schema_for::<StructuredEnvelope>()?;
        let schema = config.json_schema.expect("schema should be present");

        assert_no_dollar_keys(&schema);
        assert!(schema.get("$defs").is_none());
        assert!(schema.get("definitions").is_none());

        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            schema["properties"]["answer"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            schema["properties"]["metadata"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );

        Ok(())
    }

    #[test]
    fn test_generation_config_with_json_schema_for_inlines_vec_of_nested_struct()
    -> error::Result<()> {
        let config = GenerationConfig::new().with_json_schema_for::<Outer>()?;
        let schema = config.json_schema.expect("schema should be present");

        assert_no_dollar_keys(&schema);

        // `Inner`'s properties should be inlined directly under
        // items.items.properties (Outer.items: Vec<Inner>) instead of being
        // referenced via $defs/$ref.
        let inner_properties = &schema["properties"]["items"]["items"]["properties"];
        assert_eq!(inner_properties["a"]["type"], "integer");
        assert_eq!(inner_properties["b"]["type"], "string");

        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            schema["properties"]["items"]["items"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );

        Ok(())
    }

    #[test]
    fn test_generation_config_with_json_schema_for_optional_and_enum_fields() -> error::Result<()> {
        let config =
            GenerationConfig::new().with_json_schema_for::<StructuredWithOptionalAndEnum>()?;
        let schema = config.json_schema.expect("schema should be present");

        assert_no_dollar_keys(&schema);

        let properties = &schema["properties"];
        assert!(properties.get("required_field").is_some());
        assert!(properties.get("optional_field").is_some());
        assert!(properties.get("choice").is_some());

        let required = schema["required"]
            .as_array()
            .expect("required array should be present");
        let required_names: Vec<&str> = required
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect();
        assert!(required_names.contains(&"required_field"));
        assert!(required_names.contains(&"choice"));

        Ok(())
    }
}
