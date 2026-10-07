//! Bounded, local validation of MCP structured tool results.

use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};

use jsonschema::Validator;
use rmcp::{ServiceError, model::CallToolResult};
use serde_json::Value;
use tokio::sync::Semaphore;

use super::{McpError, RequestOptions};

struct LocalOnly;

impl jsonschema::Retrieve for LocalOnly {
    fn retrieve(
        &self,
        _uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external output-schema references are disabled".into())
    }
}

/// Compiled output schemas, keyed by the caller's tool names.
///
/// Holds validators without retaining a second tool catalog. External schema
/// resolution is refused by an explicit retriever, including when Cargo
/// feature unification enables jsonschema's network or file retrievers.
#[derive(Default)]
pub struct OutputSchemas(BTreeMap<String, Arc<Validator>>);

/// Compiles and validates output schemas away from the async executor.
///
/// Clone or share one processor across sessions to bound concurrent blocking
/// work. A deadline ends the caller's wait; a task already running retains its
/// permit until it finishes. Input byte limits remain the transport and catalog
/// reader's responsibility.
#[derive(Clone)]
pub struct SchemaProcessor {
    permits: Arc<Semaphore>,
}

impl Default for SchemaProcessor {
    fn default() -> Self {
        Self {
            permits: Arc::new(Semaphore::new(4)),
        }
    }
}

impl SchemaProcessor {
    /// Limits the number of concurrent compilation and validation tasks.
    pub fn new(max_concurrent_tasks: NonZeroUsize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max_concurrent_tasks.get())),
        }
    }

    /// Compile declarations once for a catalog, within the operation's budget.
    pub async fn compile(
        &self,
        schemas: Vec<(String, Value)>,
        options: RequestOptions,
    ) -> Result<OutputSchemas, McpError> {
        if schemas.is_empty() {
            return Ok(OutputSchemas::default());
        }
        self.process(options, move || {
            let mut validators = BTreeMap::new();
            for (tool, schema) in schemas {
                let validator = jsonschema::options()
                    .with_retriever(LocalOnly)
                    .build(&schema)
                    .map_err(|_| McpError::InvalidOutputSchema { tool: tool.clone() })?;
                validators.insert(tool, Arc::new(validator));
            }
            Ok(OutputSchemas(validators))
        })
        .await?
    }

    /// Validate a successful structured result without modifying its contents.
    ///
    /// Tools without an output schema and `isError` results pass through. A tool
    /// with a schema must provide matching `structuredContent`. Diagnostics
    /// name only the failed keyword and a bounded schema path, never result data.
    pub async fn validate(
        &self,
        schemas: &OutputSchemas,
        tool: &str,
        result: &CallToolResult,
        options: RequestOptions,
    ) -> Result<(), McpError> {
        if result.is_error == Some(true) {
            return Ok(());
        }
        let Some(validator) = schemas.0.get(tool) else {
            return Ok(());
        };
        let structured = result
            .structured_content
            .as_ref()
            .ok_or_else(|| McpError::OutputSchemaViolation {
                tool: tool.to_owned(),
                detail: "the tool declares an output schema but returned no structuredContent"
                    .to_owned(),
            })?
            .clone();
        let validator = Arc::clone(validator);
        let violation = self
            .process(options, move || {
                validator.validate(&structured).err().map(|error| {
                    let path: String = error.schema_path().as_str().chars().take(200).collect();
                    format!(
                        "keyword {:?} failed at schema path {path:?}",
                        error.kind().keyword()
                    )
                })
            })
            .await?;
        match violation {
            None => Ok(()),
            Some(detail) => Err(McpError::OutputSchemaViolation {
                tool: tool.to_owned(),
                detail,
            }),
        }
    }

    async fn process<T, F>(&self, options: RequestOptions, work: F) -> Result<T, McpError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let expired = options.timeout.is_some_and(|timeout| timeout.is_zero());
        if expired {
            return Err(McpError::Service(ServiceError::Timeout {
                timeout: std::time::Duration::ZERO,
            }));
        }
        let deadline = options
            .timeout
            .and_then(|timeout| tokio::time::Instant::now().checked_add(timeout));
        let run = async {
            let permit = Arc::clone(&self.permits)
                .acquire_owned()
                .await
                .map_err(|_| McpError::SchemaProcessing)?;
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                work()
            })
            .await
            .map_err(|_| McpError::SchemaProcessing)
        };
        let expired = async {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            biased;
            () = expired => Err(McpError::Service(ServiceError::Timeout { timeout: options.timeout.unwrap_or_default() })),
            value = run => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicBool, Ordering},
        time::Duration,
    };

    use super::*;
    use crate::mcp::McpTool;

    const WAIT: Duration = Duration::from_secs(5);

    fn processor() -> SchemaProcessor {
        SchemaProcessor::new(NonZeroUsize::new(1).unwrap())
    }

    #[tokio::test]
    async fn a_timed_out_blocking_task_keeps_its_permit_until_completion() {
        let processor = processor();
        let (started, wait_started) = tokio::sync::oneshot::channel();
        let (release, wait_release) = std::sync::mpsc::channel();
        let (finished, wait_finished) = tokio::sync::oneshot::channel();
        let first = {
            let processor = processor.clone();
            tokio::spawn(async move {
                processor
                    .process(
                        RequestOptions::with_timeout(Duration::from_millis(200)),
                        move || {
                            let _ = started.send(());
                            wait_release.recv_timeout(WAIT).unwrap();
                            let _ = finished.send(());
                        },
                    )
                    .await
            })
        };
        tokio::time::timeout(WAIT, wait_started)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            first.await.unwrap(),
            Err(McpError::Service(ServiceError::Timeout { .. }))
        ));
        assert_eq!(processor.permits.available_permits(), 0);
        let executed = Arc::new(AtomicBool::new(false));
        let marker = Arc::clone(&executed);
        let queued = processor
            .process(
                RequestOptions::with_timeout(Duration::from_millis(20)),
                move || marker.store(true, Ordering::SeqCst),
            )
            .await;
        assert!(matches!(
            queued,
            Err(McpError::Service(ServiceError::Timeout { .. }))
        ));
        assert!(!executed.load(Ordering::SeqCst));
        release.send(()).unwrap();
        tokio::time::timeout(WAIT, wait_finished)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(WAIT, processor.process(RequestOptions::default(), || 42))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(processor.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn dropping_a_waiter_does_not_release_running_blocking_work() {
        let processor = processor();
        let (started, wait_started) = tokio::sync::oneshot::channel();
        let (release, wait_release) = std::sync::mpsc::channel();
        let task = {
            let processor = processor.clone();
            tokio::spawn(async move {
                processor
                    .process(RequestOptions::default(), move || {
                        let _ = started.send(());
                        wait_release.recv_timeout(WAIT).unwrap();
                    })
                    .await
            })
        };
        tokio::time::timeout(WAIT, wait_started)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(processor.permits.available_permits(), 0);
        release.send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(WAIT, processor.process(RequestOptions::default(), || 42))
                .await
                .unwrap()
                .unwrap(),
            42
        );
        assert_eq!(processor.permits.available_permits(), 1);
    }

    fn tool(schema: Value) -> McpTool {
        let mut remote = rmcp::model::Tool::new("typed", "typed", Arc::new(serde_json::Map::new()));
        remote.output_schema = Some(Arc::new(schema.as_object().unwrap().clone()));
        McpTool {
            name: "typed".into(),
            remote,
            schema: Arc::new(std::sync::OnceLock::new()),
        }
    }

    #[tokio::test]
    async fn compilation_is_shared_after_its_first_waiter_is_dropped() {
        let processor = Arc::new(processor());
        let permit = Arc::clone(&processor.permits)
            .acquire_owned()
            .await
            .unwrap();
        let tool = tool(serde_json::json!({"type":"object"}));
        let mut first = Box::pin(tool.output_schemas(&processor));
        assert!(futures::poll!(first.as_mut()).is_pending());
        let flight = tool.schema.get().unwrap().clone();
        drop(first);
        assert!(flight.peek().is_none());
        drop(permit);
        let compiled = tokio::time::timeout(WAIT, tool.output_schemas(&processor))
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let original = flight
            .await
            .unwrap_or_else(|_| panic!("shared compilation failed"));
        assert!(Arc::ptr_eq(&compiled, &original));
    }

    #[tokio::test]
    async fn an_invalid_schema_failure_is_cached_without_recompiling() {
        let processor = Arc::new(processor());
        let tool = tool(serde_json::json!({"type":12}));
        assert!(matches!(
            tool.output_schemas(&processor).await,
            Err(McpError::InvalidOutputSchema { .. })
        ));
        assert!(matches!(tool.schema.get().unwrap().peek(), Some(Err(_))));
        // A held permit would block a fresh compile. The cached error is immediate.
        let _permit = Arc::clone(&processor.permits)
            .acquire_owned()
            .await
            .unwrap();
        assert!(matches!(
            tokio::time::timeout(WAIT, tool.output_schemas(&processor))
                .await
                .unwrap(),
            Err(McpError::InvalidOutputSchema { .. })
        ));
    }

    #[tokio::test]
    async fn external_refs_are_refused_even_with_retrieval_features_enabled() {
        for reference in [
            "https://example.invalid/schema.json",
            "file:///tmp/schema.json",
        ] {
            let result = processor()
                .compile(
                    vec![("remote".into(), serde_json::json!({"$ref":reference}))],
                    RequestOptions::default(),
                )
                .await;
            assert!(matches!(result, Err(McpError::InvalidOutputSchema { .. })));
        }
    }
}
