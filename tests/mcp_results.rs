//! Raw MCP results, output-schema validation and cancellation, offline.
//!
//! The MCP server runs in-process over an in-memory duplex pipe. Nothing here
//! needs a model provider, so the file is gated on `mcp` alone. Every wait on
//! the server is bounded by [`WAIT`], and "exactly once" is checked by a round
//! trip through the server rather than by sleeping.
#![cfg(feature = "mcp")]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use rai_sdk::ToolCall;
use rai_sdk::mcp::rmcp::{
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceError, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, CancelledNotificationParam,
        ContentBlock, JsonObject, ListToolsResult, MetaObject, PaginatedRequestParams, RequestId,
        Resource, ResourceContents, ServerCapabilities, ServerConfig, Tool as McpTool,
    },
    service::{NotificationContext, Peer, RequestContext, RunningService},
};
use rai_sdk::mcp::{
    McpError, McpTools, OutputSchemas, RequestOptions, SchemaProcessor, call_tool, call_tool_until,
};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

/// Upper bound for any single wait on the server; a hang fails the test.
const WAIT: Duration = Duration::from_secs(5);

/// A value that must never appear in an error's text.
const MARKER: &str = "SECRET-MARKER-9137";

fn object(value: Value) -> Arc<JsonObject> {
    match value {
        Value::Object(map) => Arc::new(map),
        other => panic!("expected a JSON object, got {other}"),
    }
}

fn typed_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "n": { "type": "integer" } },
        "required": ["n"],
        "additionalProperties": false
    })
}

fn catalog(with_bad_schema: bool) -> Vec<McpTool> {
    let plain = |name: &'static str| McpTool::new(name, name, object(json!({"type": "object"})));
    let mut tools = vec![
        plain("mixed"),
        plain("mixedError"),
        plain("hang"),
        plain("quick"),
        plain("protocolError"),
        plain("plain"),
        plain("typed").with_raw_output_schema(object(typed_schema())),
    ];
    if with_bad_schema {
        tools.push(plain("badSchema").with_raw_output_schema(object(json!({"type": 12}))));
    }
    tools
}

/// The result `mixed` returns: every content kind, structured content, `_meta`.
fn mixed_result(is_error: bool) -> CallToolResult {
    let content = vec![
        ContentBlock::text("hello"),
        ContentBlock::image("aGVsbG8=", "image/png"),
        ContentBlock::audio("aGVsbG8=", "audio/wav"),
        ContentBlock::resource(ResourceContents::text("body", "file:///a.txt")),
        ContentBlock::resource_link(Resource::new("file:///b.txt", "b")),
    ];
    let mut result = if is_error {
        CallToolResult::error(content)
    } else {
        CallToolResult::success(content)
    };
    result.structured_content = Some(json!({"anything": [1, 2, {"k": null}]}));
    let mut meta = MetaObject::new();
    meta.0.insert("trace".to_owned(), json!("t-1"));
    result.meta = Some(meta);
    result
}

#[derive(Clone)]
struct TestServer {
    /// Whether the catalog also lists a tool whose output schema cannot compile.
    with_bad_schema: bool,
    /// Names of every `tools/call` that reached the handler, in order.
    seen: Arc<Mutex<Vec<String>>>,
    /// Sent when a `hang` call starts, with its JSON-RPC id.
    started: mpsc::UnboundedSender<RequestId>,
    /// Sent for every `notifications/cancelled` the server receives.
    cancelled: mpsc::UnboundedSender<RequestId>,
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(catalog(
            self.with_bad_schema,
        )))
    }

    async fn on_cancelled(
        &self,
        notification: CancelledNotificationParam,
        _context: NotificationContext<RoleServer>,
    ) {
        if let Some(id) = notification.request_id {
            let _ = self.cancelled.send(id);
        }
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.seen.lock().unwrap().push(request.name.to_string());
        let arguments = Value::Object(request.arguments.clone().unwrap_or_default());
        let result = match request.name.as_ref() {
            "mixed" => mixed_result(false),
            "mixedError" => mixed_result(true),
            "hang" => {
                let _ = self.started.send(context.id.clone());
                context.ct.cancelled().await;
                CallToolResult::success(vec![ContentBlock::text("late")])
            }
            "quick" => CallToolResult::success(vec![ContentBlock::text("ok")]),
            "protocolError" => {
                return Err(ErrorData::invalid_params("bad request", None));
            }
            "plain" => CallToolResult::success(vec![ContentBlock::text("plain")]),
            "typed" => match arguments["mode"].as_str().unwrap_or("valid") {
                "valid" => {
                    let mut result = CallToolResult::success(vec![ContentBlock::text("typed")]);
                    result.structured_content = Some(json!({"n": 7}));
                    result
                }
                "invalid" => {
                    let mut result = CallToolResult::success(vec![ContentBlock::text("typed")]);
                    result.structured_content = Some(json!({"n": "seven"}));
                    result
                }
                "secret" => {
                    let mut result = CallToolResult::success(vec![ContentBlock::text("typed")]);
                    result.structured_content = Some(json!({"n": MARKER, MARKER: MARKER}));
                    result
                }
                "missing" => CallToolResult::success(vec![ContentBlock::text("typed")]),
                "error" => {
                    let mut result = CallToolResult::error(vec![ContentBlock::text("failed")]);
                    result.structured_content = Some(json!({"unrelated": true}));
                    result
                }
                other => {
                    return Err(ErrorData::invalid_params(format!("mode {other}"), None));
                }
            },
            "badSchema" => {
                let mut result = CallToolResult::success(vec![ContentBlock::text("x")]);
                result.structured_content = Some(json!({}));
                result
            }
            other => {
                return Err(ErrorData::invalid_params(format!("unknown {other}"), None));
            }
        };
        Ok(result.into())
    }
}

struct Session {
    client: RunningService<RoleClient, ()>,
    seen: Arc<Mutex<Vec<String>>>,
    started: mpsc::UnboundedReceiver<RequestId>,
    cancelled: mpsc::UnboundedReceiver<RequestId>,
}

impl Session {
    fn peer(&self) -> Peer<RoleClient> {
        self.client.peer().clone()
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }

    async fn tools(&self) -> McpTools {
        McpTools::discover(self.peer()).await.unwrap()
    }

    /// A completed round trip: everything the client sent before it has been
    /// handled by the server, so a notification that was going to arrive has.
    async fn round_trip(&self) {
        let result = tokio::time::timeout(
            WAIT,
            call_tool(
                &self.client.peer().clone(),
                CallToolRequestParams::new("quick"),
                RequestOptions::default(),
            ),
        )
        .await
        .expect("round trip should not hang");
        result.expect("round trip should succeed");
    }

    async fn next_started(&mut self) -> RequestId {
        tokio::time::timeout(WAIT, self.started.recv())
            .await
            .expect("the server should start the call")
            .expect("server should still be running")
    }

    async fn next_cancelled(&mut self) -> RequestId {
        tokio::time::timeout(WAIT, self.cancelled.recv())
            .await
            .expect("the server should observe the cancellation")
            .expect("server should still be running")
    }

    /// Asserts no further cancellation arrived, after a round trip.
    async fn assert_no_more_cancellations(&mut self) {
        self.round_trip().await;
        assert!(
            self.cancelled.try_recv().is_err(),
            "expected no further cancellation"
        );
    }
}

async fn session() -> Session {
    session_with(false).await
}

async fn session_with(with_bad_schema: bool) -> Session {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (started_tx, started) = mpsc::unbounded_channel();
    let (cancelled_tx, cancelled) = mpsc::unbounded_channel();
    let server = TestServer {
        with_bad_schema,
        seen: seen.clone(),
        started: started_tx,
        cancelled: cancelled_tx,
    };
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    let client = ().serve(client_io).await.expect("client should connect");
    Session {
        client,
        seen,
        started,
        cancelled,
    }
}

fn tool_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: format!("call_{name}"),
        name: name.to_owned(),
        arguments,
    }
}

fn params(name: &'static str) -> CallToolRequestParams {
    CallToolRequestParams::new(name)
}

fn blocks(result: &CallToolResult) -> Value {
    serde_json::to_value(&result.content).unwrap()
}

fn assert_mixed_preserved(result: &CallToolResult, is_error: bool) {
    let expected = mixed_result(is_error);
    assert_eq!(blocks(result), blocks(&expected));
    assert_eq!(result.structured_content, expected.structured_content);
    assert_eq!(result.is_error, expected.is_error);
    assert_eq!(result.meta, expected.meta);
    let kinds: Vec<_> = blocks(result)
        .as_array()
        .unwrap()
        .iter()
        .map(|block| block["type"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        kinds,
        ["text", "image", "audio", "resource", "resource_link"]
    );
}

fn violation_detail(error: McpError) -> (String, String) {
    match error {
        McpError::OutputSchemaViolation { tool, detail } => (tool, detail),
        other => panic!("expected OutputSchemaViolation, got {other:?}"),
    }
}

// ── Raw results ────────────────────────────────────────────────────────────

#[tokio::test]
async fn raw_call_tool_preserves_mixed_content_structured_meta_and_is_error() {
    let session = session().await;
    let peer = session.peer();

    let ok = call_tool(&peer, params("mixed"), RequestOptions::default())
        .await
        .unwrap();
    assert_mixed_preserved(&ok, false);

    let failed = call_tool(&peer, params("mixedError"), RequestOptions::default())
        .await
        .unwrap();
    assert_mixed_preserved(&failed, true);
}

#[tokio::test]
async fn call_result_preserves_the_complete_result() {
    let session = session().await;
    let tools = session.tools().await;

    let ok = tools
        .call_result(&tool_call("mixed", json!({})), RequestOptions::default())
        .await
        .unwrap();
    assert_mixed_preserved(&ok, false);

    // `isError` results come back as results, not as errors.
    let failed = tools
        .call_result(
            &tool_call("mixedError", json!({})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_mixed_preserved(&failed, true);
}

#[tokio::test]
async fn call_result_routes_renamed_tools_to_their_server_names() {
    let session = session().await;
    let tools = session
        .tools()
        .await
        .rename(|tool| format!("srv_{}", tool.name))
        .unwrap();

    let result = tools
        .call_result(
            &tool_call("srv_mixed", json!({})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_mixed_preserved(&result, false);
    assert_eq!(session.seen(), ["mixed"]);

    let error = tools
        .call_result(&tool_call("mixed", json!({})), RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::ToolNotFound { ref name } if name == "mixed"));
    assert_eq!(session.seen(), ["mixed"], "an unknown name is not sent");
}

#[tokio::test]
async fn invalid_arguments_never_dispatch_and_keep_the_model_path_unchanged() {
    let session = session().await;
    let tools = session.tools().await;

    let error = tools
        .call_result(&tool_call("plain", json!(5)), RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::InvalidArguments { ref tool } if tool == "plain"));

    // The model path still turns it into a tool error for the model.
    let message = tools.call(&tool_call("plain", json!([1]))).await.unwrap();
    assert!(message.tool_error);
    assert!(session.seen().is_empty());

    // `null` arguments are accepted, as before.
    let message = tools.call(&tool_call("plain", Value::Null)).await.unwrap();
    assert!(!message.tool_error);
    assert_eq!(session.seen(), ["plain"]);
}

#[tokio::test]
async fn model_path_conversion_is_unchanged() {
    let session = session().await;
    let tools = session.tools().await;

    // Structured content is what the model receives, and `isError` is kept.
    let message = tools
        .call(&tool_call("mixedError", json!({})))
        .await
        .unwrap();
    assert!(message.tool_error);
    assert_eq!(
        serde_json::from_str::<Value>(&message.content).unwrap(),
        json!({"anything": [1, 2, {"k": null}]})
    );

    let message = tools.call(&tool_call("quick", json!({}))).await.unwrap();
    assert!(!message.tool_error);
    assert_eq!(message.content, "ok");
}

// ── Output-schema validation through McpTools ─────────────────────────────

#[tokio::test]
async fn valid_structured_results_pass_validation() {
    let session = session().await;
    let tools = session.tools().await;

    let result = tools
        .call_result(
            &tool_call("typed", json!({"mode": "valid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.structured_content, Some(json!({"n": 7})));

    let message = tools
        .call(&tool_call("typed", json!({"mode": "valid"})))
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&message.content).unwrap(),
        json!({"n": 7})
    );
}

#[tokio::test]
async fn invalid_structured_results_are_violations_on_both_paths() {
    let session = session().await;
    let tools = session.tools().await;

    let (tool, detail) = violation_detail(
        tools
            .call_result(
                &tool_call("typed", json!({"mode": "invalid"})),
                RequestOptions::default(),
            )
            .await
            .unwrap_err(),
    );
    assert_eq!(tool, "typed");
    assert!(detail.contains("keyword"), "{detail}");
    assert!(!detail.contains("seven"), "result data leaked: {detail}");

    let error = tools
        .call(&tool_call("typed", json!({"mode": "invalid"})))
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::OutputSchemaViolation { .. }));
}

#[tokio::test]
async fn missing_structured_content_is_a_violation() {
    let session = session().await;
    let tools = session.tools().await;

    let (tool, detail) = violation_detail(
        tools
            .call_result(
                &tool_call("typed", json!({"mode": "missing"})),
                RequestOptions::default(),
            )
            .await
            .unwrap_err(),
    );
    assert_eq!(tool, "typed");
    assert!(detail.contains("structuredContent"), "{detail}");
}

#[tokio::test]
async fn error_results_bypass_output_validation() {
    let session = session().await;
    let tools = session.tools().await;

    let result = tools
        .call_result(
            &tool_call("typed", json!({"mode": "error"})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(true));
    assert_eq!(result.structured_content, Some(json!({"unrelated": true})));

    let message = tools
        .call(&tool_call("typed", json!({"mode": "error"})))
        .await
        .unwrap();
    assert!(message.tool_error);
}

#[tokio::test]
async fn an_uncompilable_output_schema_fails_only_its_own_tool() {
    let session = session_with(true).await;
    let tools = session.tools().await;
    let bad = || tool_call("badSchema", json!({}));

    // The bad tool fails before any dispatch, however often it is called.
    for _ in 0..3 {
        let error = tools
            .call_result(&bad(), RequestOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, McpError::InvalidOutputSchema { ref tool } if tool == "badSchema"),
            "{error:?}"
        );
    }
    assert!(session.seen().is_empty(), "the server must not be called");

    // Other tools in the same catalog are unaffected, with or without a schema.
    let typed = tools
        .call_result(
            &tool_call("typed", json!({"mode": "valid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(typed.structured_content, Some(json!({"n": 7})));
    tools
        .call_result(&tool_call("plain", json!({})), RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(session.seen(), ["typed", "plain"]);

    // A good tool's validation still works after the bad tool failed, and the
    // legacy model path reports the same per-tool failure.
    let error = tools
        .call_result(
            &tool_call("typed", json!({"mode": "invalid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::OutputSchemaViolation { .. }));
    let error = tools.call(&bad()).await.unwrap_err();
    assert!(matches!(error, McpError::InvalidOutputSchema { .. }));
    assert_eq!(
        session.seen(),
        ["typed", "plain", "typed"],
        "the bad tool is still never dispatched"
    );
}

#[tokio::test]
async fn a_good_tool_called_first_does_not_hide_the_bad_tools_failure() {
    let session = session_with(true).await;
    let tools = session.tools().await;

    tools
        .call_result(
            &tool_call("typed", json!({"mode": "valid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    let error = tools
        .call_result(
            &tool_call("badSchema", json!({})),
            RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::InvalidOutputSchema { ref tool } if tool == "badSchema"));
    assert_eq!(session.seen(), ["typed"]);
}

// ── Cancellation and deadlines ─────────────────────────────────────────────

#[tokio::test]
async fn a_zero_timeout_never_dispatches() {
    let session = session().await;
    let peer = session.peer();

    let error = call_tool(
        &peer,
        params("hang"),
        RequestOptions::with_timeout(Duration::ZERO),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, McpError::Service(ServiceError::Timeout { .. })),
        "{error:?}"
    );

    let tools = session.tools().await;
    let error = tools
        .call_result(
            &tool_call("plain", json!({})),
            RequestOptions::with_timeout(Duration::ZERO),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        McpError::Service(ServiceError::Timeout { .. })
    ));

    session.round_trip().await;
    assert_eq!(session.seen(), ["quick"], "nothing else reached the server");
}

#[tokio::test]
async fn an_already_cancelled_caller_never_dispatches() {
    let session = session().await;

    let error = call_tool_until(
        &session.peer(),
        params("hang"),
        RequestOptions::default(),
        std::future::ready(()),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, McpError::Cancelled { ref tool } if tool == "hang"));

    session.round_trip().await;
    assert_eq!(session.seen(), ["quick"]);
}

#[tokio::test]
async fn explicit_cancellation_notifies_the_server_exactly_once() {
    let mut session = session().await;
    let peer = session.peer();
    let (cancel, cancelled) = oneshot::channel::<()>();

    let call = tokio::spawn(async move {
        call_tool_until(
            &peer,
            params("hang"),
            RequestOptions::default(),
            async move {
                let _ = cancelled.await;
            },
        )
        .await
    });
    let id = session.next_started().await;
    cancel.send(()).unwrap();

    let result = tokio::time::timeout(WAIT, call)
        .await
        .expect("the caller must not wait for the server")
        .unwrap();
    assert!(matches!(result, Err(McpError::Cancelled { ref tool }) if tool == "hang"));
    assert_eq!(session.next_cancelled().await, id);
    session.assert_no_more_cancellations().await;
}

#[tokio::test]
async fn dropping_a_raw_call_notifies_the_server_exactly_once() {
    let mut session = session().await;
    let peer = session.peer();

    let mut pending = Box::pin(call_tool(&peer, params("hang"), RequestOptions::default()));
    let id = tokio::select! {
        id = session.next_started() => id,
        _ = &mut pending => panic!("a hanging call cannot complete"),
    };
    drop(pending);

    assert_eq!(session.next_cancelled().await, id);
    session.assert_no_more_cancellations().await;
}

#[tokio::test]
async fn the_deadline_is_a_service_timeout_and_cancels_the_call() {
    let mut session = session().await;
    let peer = session.peer();

    let result = tokio::time::timeout(
        WAIT,
        call_tool(
            &peer,
            params("hang"),
            RequestOptions::with_timeout(Duration::from_millis(200)),
        ),
    )
    .await
    .expect("the deadline must end the call");
    assert!(
        matches!(result, Err(McpError::Service(ServiceError::Timeout { .. }))),
        "{result:?}"
    );

    let id = session.next_started().await;
    assert_eq!(session.next_cancelled().await, id);
    session.assert_no_more_cancellations().await;
}

#[tokio::test]
async fn call_result_deadline_also_cancels_on_the_server() {
    let mut session = session().await;
    let tools = session.tools().await;

    let result = tokio::time::timeout(
        WAIT,
        tools.call_result(
            &tool_call("hang", json!({})),
            RequestOptions::with_timeout(Duration::from_millis(200)),
        ),
    )
    .await
    .expect("the deadline must end the call");
    assert!(matches!(
        result,
        Err(McpError::Service(ServiceError::Timeout { .. }))
    ));
    let id = session.next_started().await;
    assert_eq!(session.next_cancelled().await, id);
}

#[tokio::test]
async fn completed_and_failed_calls_send_no_cancellation() {
    let mut session = session().await;
    let peer = session.peer();

    call_tool(&peer, params("quick"), RequestOptions::default())
        .await
        .unwrap();
    call_tool(&peer, params("quick"), RequestOptions::with_timeout(WAIT))
        .await
        .unwrap();
    let error = call_tool(&peer, params("protocolError"), RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        McpError::Service(ServiceError::McpError(_))
    ));

    session.assert_no_more_cancellations().await;
}

// ── SchemaProcessor on its own ─────────────────────────────────────────────

async fn compiled(processor: &SchemaProcessor) -> OutputSchemas {
    processor
        .compile(
            vec![("typed".to_owned(), typed_schema())],
            RequestOptions::default(),
        )
        .await
        .unwrap()
}

fn success_with(structured: Option<Value>) -> CallToolResult {
    let mut result = CallToolResult::success(vec![ContentBlock::text("x")]);
    result.structured_content = structured;
    result
}

#[tokio::test]
async fn schema_processor_validates_independently_of_any_session() {
    let processor = SchemaProcessor::default();
    let schemas = compiled(&processor).await;
    let check = |tool: &'static str, result: CallToolResult| {
        let processor = processor.clone();
        let schemas = &schemas;
        async move {
            processor
                .validate(schemas, tool, &result, RequestOptions::default())
                .await
        }
    };

    check("typed", success_with(Some(json!({"n": 1}))))
        .await
        .unwrap();
    violation_detail(
        check("typed", success_with(Some(json!({"n": "x"}))))
            .await
            .unwrap_err(),
    );
    let (_, detail) = violation_detail(check("typed", success_with(None)).await.unwrap_err());
    assert!(detail.contains("structuredContent"));

    // Error results and tools without a declared schema pass through.
    let mut failed = success_with(Some(json!({"bad": true})));
    failed.is_error = Some(true);
    check("typed", failed).await.unwrap();
    check("undeclared", success_with(None)).await.unwrap();
}

#[tokio::test]
async fn schema_processor_reports_the_tool_with_an_invalid_schema() {
    let processor = SchemaProcessor::new(std::num::NonZeroUsize::MIN);
    let error = processor
        .compile(
            vec![
                ("fine".to_owned(), typed_schema()),
                ("broken".to_owned(), json!({"type": 12})),
            ],
            RequestOptions::default(),
        )
        .await
        .err()
        .expect("an invalid schema must fail the compilation");
    assert!(matches!(error, McpError::InvalidOutputSchema { ref tool } if tool == "broken"));
}

#[tokio::test]
async fn schema_processor_errors_never_carry_result_data() {
    let processor = SchemaProcessor::default();
    let schemas = compiled(&processor).await;
    // The marker is both a value and a key (an unexpected property name).
    let result = success_with(Some(json!({"n": MARKER, MARKER: MARKER})));

    let error = processor
        .validate(&schemas, "typed", &result, RequestOptions::default())
        .await
        .unwrap_err();
    let (tool, detail) = match &error {
        McpError::OutputSchemaViolation { tool, detail } => (tool.clone(), detail.clone()),
        other => panic!("expected OutputSchemaViolation, got {other:?}"),
    };
    assert_eq!(tool, "typed");
    for text in [error.to_string(), format!("{error:?}"), detail] {
        assert!(!text.contains(MARKER), "result data leaked: {text}");
    }

    // The same holds when the violation arrives through a session.
    let session = session().await;
    let tools = session.tools().await;
    let error = tools
        .call_result(
            &tool_call("typed", json!({"mode": "secret"})),
            RequestOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::OutputSchemaViolation { .. }));
    assert!(!error.to_string().contains(MARKER));
    assert!(!format!("{error:?}").contains(MARKER));
}

#[tokio::test]
async fn schema_processor_honors_an_expired_budget() {
    let processor = SchemaProcessor::default();
    let zero = || RequestOptions::with_timeout(Duration::ZERO);

    let error = processor
        .compile(vec![("typed".to_owned(), typed_schema())], zero())
        .await
        .err()
        .expect("a zero budget must not compile");
    assert!(matches!(
        error,
        McpError::Service(ServiceError::Timeout { .. })
    ));

    let schemas = compiled(&processor).await;
    let error = processor
        .validate(
            &schemas,
            "typed",
            &success_with(Some(json!({"n": 1}))),
            zero(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        McpError::Service(ServiceError::Timeout { .. })
    ));
}
