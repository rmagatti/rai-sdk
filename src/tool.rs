//! Typed tools that the model can call during generation.
//!
//! A [`Tool`] pairs a name and description with an async handler. The handler's
//! argument type supplies the tool's JSON Schema through [`JsonSchema`], so the
//! schema advertised to the provider and the type the handler receives cannot
//! drift apart.
//!
//! Incoming arguments are validated against that schema before the handler runs.
//! Validation failures do not abort generation: they are returned to the model
//! as a structured tool-error message describing each problem, so the model can
//! correct itself and call the tool again.

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use schemars::{JsonSchema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    error::{Error, ProviderKind, Result, ToolArgumentIssue},
    generation::normalize_strict_json_schema,
    message::{Message, ToolCall, ToolDefinition},
};

type ToolFuture = Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>>;
type ToolHandler = dyn Fn(serde_json::Value, ToolContext) -> ToolFuture + Send + Sync;

fn format_instance_path(path: impl ToString) -> String {
    let path = path.to_string();
    if path.is_empty() {
        "$".to_string()
    } else {
        path
    }
}

fn collect_validation_issues(
    validator: &jsonschema::Validator,
    value: &serde_json::Value,
) -> Vec<ToolArgumentIssue> {
    let mut issues: Vec<_> = validator
        .iter_errors(value)
        .map(|error| ToolArgumentIssue {
            path: format_instance_path(error.instance_path()),
            schema_path: error.schema_path().to_string(),
            message: error.to_string(),
        })
        .collect();

    issues.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.schema_path.cmp(&b.schema_path))
            .then_with(|| a.message.cmp(&b.message))
    });
    issues.dedup();
    issues
}

fn summarize_validation_issues(issues: &[ToolArgumentIssue]) -> Option<String> {
    if issues.is_empty() {
        return None;
    }

    let messages: Vec<_> = issues.iter().map(|i| i.message.clone()).collect();
    Some(match messages.as_slice() {
        [single] => single.clone(),
        many => format!("{} validation errors: {}", many.len(), many.join("; ")),
    })
}

fn tool_error_content(error: Error) -> Result<String> {
    match error {
        Error::ToolArguments {
            name,
            message,
            issues,
        } => serde_json::to_string(&serde_json::json!({
            "error": {
                "type": "tool_argument_validation",
                "tool": name,
                "retryable": true,
                "message": "Tool arguments failed validation. Call the tool again with corrected arguments.",
                "summary": message,
                "issues": issues,
            }
        }))
        .map_err(Into::into),
        error => serde_json::to_string(&serde_json::json!({ "error": error.to_string() }))
            .map_err(Into::into),
    }
}

/// Metadata passed to each tool handler invocation.
///
/// # Example
///
/// ```rust
/// use rai_sdk::{JsonSchema, Tool};
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// #[schemars(crate = "rai_sdk::schemars")]
/// struct EchoArgs {
///     value: String,
/// }
///
/// let _tool = Tool::new("echo")
///     .description("Return the input together with model metadata")
///     .handler(|args: EchoArgs, ctx: rai_sdk::ToolContext| async move {
///         Ok(format!("{} via {}", args.value, ctx.model))
///     })?;
/// # Ok::<(), rai_sdk::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Provider that requested the tool call.
    pub provider: ProviderKind,
    /// Model ID that requested the tool call, as sent on the wire.
    pub model: String,
    /// Zero-based index of the tool-calling round within the current request.
    ///
    /// Useful for detecting repeated calls and for bounding work in handlers
    /// that may be invoked several times in one generation.
    pub round: usize,
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// Provider-assigned identifier for this specific tool call.
    pub tool_call_id: String,
}

/// A callable tool that can be registered on a client or request.
///
/// Use [`Tool::new`] to start building, chain `.description()` and `.handler()`,
/// where `.handler()` is fallible and validates the schema at build time.
///
/// # Example
///
/// ```rust
/// use rai_sdk::{JsonSchema, Tool};
/// use serde::Deserialize;
///
/// #[derive(Deserialize, JsonSchema)]
/// #[schemars(crate = "rai_sdk::schemars")]
/// struct GetUserInfo {
///     user_id: String,
/// }
///
/// let _tool = Tool::new("get_user_info")
///     .description("Load a user's profile")
///     .handler(|input: GetUserInfo, _ctx: rai_sdk::ToolContext| async move {
///         Ok(serde_json::json!({ "user_id": input.user_id, "email": "user@example.com" }))
///     })?;
/// # Ok::<(), rai_sdk::Error>(())
/// ```
#[derive(Clone)]
pub struct Tool {
    name: String,
    description: Option<String>,
    input_schema: Option<serde_json::Value>,
    handler: Option<Arc<ToolHandler>>,
}

impl Tool {
    /// Start a new tool definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema: None,
            handler: None,
        }
    }

    /// Describe when the model should use this tool.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Override the inferred input schema with a custom JSON Schema.
    pub fn json_schema(mut self, input_schema: serde_json::Value) -> Self {
        self.input_schema = Some(input_schema);
        self
    }

    /// Attach an async handler and infer the input schema from `Args`.
    ///
    /// This is fallible — the schema is validated at build time and an error
    /// is returned if it is invalid.
    pub fn handler<Args, Output, F, Fut>(mut self, handler: F) -> Result<Self>
    where
        Args: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + Send + 'static,
        F: Fn(Args, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Output>> + Send + 'static,
    {
        let tool_name = self.name.clone();
        let schema = schema_for!(Args);
        let mut generated_schema = serde_json::to_value(schema)?;
        normalize_strict_json_schema(&mut generated_schema);

        let mut input_schema = self.input_schema.take().unwrap_or(generated_schema);
        normalize_strict_json_schema(&mut input_schema);

        let validator = Arc::new(jsonschema::validator_for(&input_schema).map_err(|error| {
            Error::InvalidRequest(format!(
                "Tool '{}' has an invalid input schema: {error}",
                self.name
            ))
        })?);

        self.input_schema = Some(input_schema);

        self.handler = Some(Arc::new(move |value, ctx| {
            let tool_name = tool_name.clone();
            let validator = validator.clone();

            let issues = collect_validation_issues(&validator, &value);

            if let Some(message) = summarize_validation_issues(&issues) {
                return Box::pin(async move {
                    Err(Error::ToolArguments {
                        name: tool_name,
                        message,
                        issues,
                    })
                });
            }

            let future = match serde_json::from_value::<Args>(value) {
                Ok(args) => handler(args, ctx),
                Err(error) => {
                    return Box::pin(async move {
                        Err(Error::ToolArguments {
                            name: tool_name,
                            message: error.to_string(),
                            issues: Vec::new(),
                        })
                    });
                }
            };

            Box::pin(async move {
                let output = future.await?;
                serde_json::to_value(output).map_err(Into::into)
            })
        }));

        Ok(self)
    }

    fn into_registered(self) -> Result<RegisteredTool> {
        if self.name.trim().is_empty() {
            return Err(Error::InvalidRequest(
                "Tool name cannot be empty".to_string(),
            ));
        }

        let input_schema = self.input_schema.ok_or_else(|| {
            Error::InvalidRequest(format!("Tool '{}' is missing an input schema", self.name))
        })?;

        let handler = self.handler.ok_or_else(|| {
            Error::InvalidRequest(format!("Tool '{}' is missing a handler", self.name))
        })?;

        Ok(RegisteredTool {
            definition: ToolDefinition {
                name: self.name,
                description: self.description,
                input_schema,
            },
            handler,
        })
    }
}

#[derive(Clone)]
struct RegisteredTool {
    definition: ToolDefinition,
    handler: Arc<ToolHandler>,
}

/// Registry of tools that can be auto-executed during generation.
#[derive(Clone, Default)]
pub(crate) struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    pub(crate) fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub(crate) fn register(&mut self, tool: Tool) -> Result<()> {
        let tool = tool.into_registered()?;
        let name = tool.definition.name.clone();

        if self.tools.contains_key(&name) {
            return Err(Error::InvalidRequest(format!(
                "Tool '{}' is already registered",
                name
            )));
        }

        self.tools.insert(name, tool);
        Ok(())
    }

    pub(crate) fn extend<T>(&mut self, tools: T) -> Result<()>
    where
        T: IntoIterator<Item = Tool>,
    {
        for tool in tools {
            self.register(tool)?;
        }
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub(crate) fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    /// Execute a tool call and return a [`Message`] with the result.
    ///
    /// On handler error, the error is wrapped as a `Message::tool_error()` with
    /// structured JSON content describing the validation failure — tool errors
    /// are NOT hard failures, they become messages the model can retry from.
    pub(crate) async fn execute(
        &self,
        tool_call: &ToolCall,
        context: ToolContext,
    ) -> Result<Message> {
        let registered = self
            .tools
            .get(&tool_call.name)
            .ok_or_else(|| Error::ToolNotFound {
                name: tool_call.name.clone(),
            })?;

        match (registered.handler)(tool_call.arguments.clone(), context).await {
            Ok(result) => Ok(Message::tool(
                serde_json::to_string(&result)?,
                tool_call.id.clone(),
            )),
            Err(error) => Ok(Message::tool_error(
                tool_error_content(error)?,
                tool_call.id.clone(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::sync::Arc;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct AddArgs {
        a: i32,
        b: i32,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[schemars(deny_unknown_fields)]
    struct SearchArgs {
        #[schemars(
            description = "Customer name or email substring to search for",
            length(min = 1)
        )]
        query: String,
        #[schemars(
            description = "Maximum number of results to return",
            range(min = 1, max = 10)
        )]
        limit: Option<usize>,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct OptionalOnlyArgs {
        // Only the generated schema is under test here; the field is never read.
        #[allow(dead_code)]
        #[serde(default)]
        query: Option<String>,
    }

    #[derive(Debug, serde::Serialize)]
    struct AddResult {
        sum: i32,
    }

    fn test_tool_context(tool_name: &str) -> ToolContext {
        ToolContext {
            provider: ProviderKind::OpenAI,
            model: "gpt-4o-mini".to_string(),
            round: 0,
            tool_name: tool_name.to_string(),
            tool_call_id: "call_123".to_string(),
        }
    }

    fn tool_error_payload(message: &Message) -> serde_json::Value {
        serde_json::from_str(&message.content).expect("tool error content should be valid json")
    }

    #[tokio::test]
    async fn tool_registry_executes_registered_handler() {
        let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry
            .register(
                Tool::new("add")
                    .description("Add two numbers")
                    .handler({
                        let calls = calls.clone();
                        move |args: AddArgs, _ctx| {
                            let calls = calls.clone();
                            async move {
                                calls.lock().await.push((args.a, args.b));
                                Ok(AddResult {
                                    sum: args.a + args.b,
                                })
                            }
                        }
                    })
                    .expect("tool should build"),
            )
            .expect("tool should register");

        let message = registry
            .execute(
                &ToolCall {
                    id: "call_123".to_string(),
                    name: "add".to_string(),
                    arguments: serde_json::json!({ "a": 2, "b": 3 }),
                },
                test_tool_context("add"),
            )
            .await
            .expect("tool execution should succeed");

        assert_eq!(message.content, "{\"sum\":5}");
        assert_eq!(message.tool_call_id.as_deref(), Some("call_123"));
        assert!(!message.tool_error);
        assert_eq!(calls.lock().await.as_slice(), &[(2, 3)]);
    }

    #[tokio::test]
    async fn tool_registry_aggregates_schema_validation_errors_before_handler() {
        let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry
            .register(
                Tool::new("search_customers")
                    .description("Search customers by email or name")
                    .handler({
                        let calls = calls.clone();
                        move |args: SearchArgs, _ctx| {
                            let calls = calls.clone();
                            async move {
                                calls.lock().await.push(args.query);
                                Ok(serde_json::json!({ "limit": args.limit }))
                            }
                        }
                    })
                    .expect("tool should build"),
            )
            .expect("tool should register");

        let message = registry
            .execute(
                &ToolCall {
                    id: "call_123".to_string(),
                    name: "search_customers".to_string(),
                    arguments: serde_json::json!({ "query": "", "limit": 25 }),
                },
                test_tool_context("search_customers"),
            )
            .await
            .expect("tool execution should return a tool message");

        let payload = tool_error_payload(&message);

        assert!(message.tool_error);
        assert_eq!(payload["error"]["type"], "tool_argument_validation");
        assert_eq!(payload["error"]["tool"], "search_customers");
        assert_eq!(payload["error"]["retryable"], true);
        assert_eq!(payload["error"]["issues"].as_array().map(Vec::len), Some(2));
        assert!(
            payload["error"]["summary"]
                .as_str()
                .unwrap_or_default()
                .contains("validation errors")
        );
        assert!(calls.lock().await.is_empty());
    }

    #[tokio::test]
    async fn tool_registry_rejects_unknown_fields_in_arguments() {
        let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::new();
        registry
            .register(
                Tool::new("add")
                    .description("Add two numbers")
                    .handler({
                        let calls = calls.clone();
                        move |args: AddArgs, _ctx| {
                            let calls = calls.clone();
                            async move {
                                calls.lock().await.push((args.a, args.b));
                                Ok(AddResult {
                                    sum: args.a + args.b,
                                })
                            }
                        }
                    })
                    .expect("tool should build"),
            )
            .expect("tool should register");

        let message = registry
            .execute(
                &ToolCall {
                    id: "call_123".to_string(),
                    name: "add".to_string(),
                    arguments: serde_json::json!({ "a": 2, "b": 3, "c": 4 }),
                },
                test_tool_context("add"),
            )
            .await
            .expect("tool execution should return a tool message");

        let payload = tool_error_payload(&message);

        assert!(message.tool_error);
        assert_eq!(payload["error"]["type"], "tool_argument_validation");
        assert_eq!(payload["error"]["tool"], "add");
        assert_eq!(payload["error"]["issues"].as_array().map(Vec::len), Some(1));
        assert!(calls.lock().await.is_empty());
    }

    #[test]
    fn tool_builder_rejects_invalid_custom_input_schema() {
        let result = Tool::new("add")
            .json_schema(serde_json::json!({ "type": "not-a-real-json-schema-type" }))
            .handler(|_args: AddArgs, _ctx| async move { Ok(AddResult { sum: 0 }) });

        let error = match result {
            Ok(_) => panic!("invalid custom schemas should fail during registration"),
            Err(error) => error,
        };

        assert!(matches!(error, Error::InvalidRequest(_)));
        assert!(error.to_string().contains("invalid input schema"));
    }

    #[test]
    fn tool_builder_adds_root_object_type_for_optional_only_args() {
        let mut registry = ToolRegistry::new();
        registry
            .register(
                Tool::new("lookup_workspace")
                    .description("Look up workspace information")
                    .handler(|_args: OptionalOnlyArgs, _ctx| async move {
                        Ok(serde_json::json!({ "ok": true }))
                    })
                    .expect("tool should build"),
            )
            .expect("tool should register");

        let definitions = registry.definitions();
        let schema = &definitions[0].input_schema;

        assert_eq!(
            schema["type"],
            serde_json::Value::String("object".to_string())
        );
        assert_eq!(
            schema["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert!(schema.get("properties").is_some());
    }
}
