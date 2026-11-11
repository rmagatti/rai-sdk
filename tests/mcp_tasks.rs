//! Tool calls answered with a task (the MCP Tasks extension), offline.
//!
//! The MCP server runs in-process over in-memory duplex pipes. Most tasks are
//! scripted: each `tasks/get` returns the next state from a fixed list, so the
//! tests decide exactly when a task finishes. One tool runs on rmcp's own
//! `TaskManager` to cover real cooperative cancellation. Polling tests run on
//! tokio's paused clock, so no test sleeps in real time.
#![cfg(feature = "mcp")]

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use rai_sdk::ToolCall;
use rai_sdk::mcp::rmcp::{
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceError, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, CancelTaskParams,
        ClientCapabilities, ClientConfig, ContentBlock, CreateTaskResult, DetailedTask, ErrorCode,
        GetTaskParams, GetTaskResult, Implementation, JsonObject, ListToolsResult, MetaObject,
        PaginatedRequestParams, ServerCapabilities, ServerConfig, TASKS_EXTENSION_ID, Task,
        TaskPayload, TaskStatus, Tool as McpTool,
    },
    service::{Peer, RequestContext, RunningService},
    task_manager::{TaskExit, TaskManager, TaskOptions},
};
use rai_sdk::mcp::{
    McpError, McpTools, RequestOptions, SchemaProcessor, TaskState, ToolCallOutcome, call_tool,
    call_tool_or_task, cancel_task, get_task, wait_for_task,
};
use serde_json::{Value, json};
use tokio::time::Instant;

/// Upper bound for any single wait in real time; a hang fails the test.
const WAIT: Duration = Duration::from_secs(5);

/// The poll interval scripted tasks suggest.
const POLL: Duration = Duration::from_secs(2);

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
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

fn catalog() -> Vec<McpTool> {
    let plain =
        |name: &'static str| McpTool::new(name, name, Arc::new(object(json!({"type": "object"}))));
    vec![
        plain("plain"),
        plain("eventually"),
        plain("never"),
        plain("failing"),
        plain("managed"),
        plain("typed").with_raw_output_schema(Arc::new(object(typed_schema()))),
    ]
}

fn completed(result: CallToolResult) -> TaskPayload {
    TaskPayload::Completed {
        result: object(serde_json::to_value(result).unwrap()),
    }
}

fn structured(value: Value) -> CallToolResult {
    let mut result = CallToolResult::success(vec![ContentBlock::text("typed")]);
    result.structured_content = Some(value);
    result
}

/// A task whose `tasks/get` answers walk through `states`, repeating the last.
struct Scripted {
    task: Task,
    states: VecDeque<TaskPayload>,
}

#[derive(Clone, Default)]
struct Store {
    next_id: Arc<AtomicUsize>,
    scripted: Arc<Mutex<HashMap<String, Scripted>>>,
    /// Every `tasks/get` the server answered: task id and (virtual) time.
    gets: Arc<Mutex<Vec<(String, Instant)>>>,
    /// Task ids `tasks/cancel` was received for.
    cancels: Arc<Mutex<Vec<String>>>,
    manager: TaskManager,
}

#[derive(Clone)]
struct TaskServer {
    /// Whether the server advertises the tasks extension.
    tasks: bool,
    store: Store,
}

impl TaskServer {
    fn script(&self, states: Vec<TaskPayload>) -> CallToolResponse {
        let id = format!("task-{}", self.store.next_id.fetch_add(1, Ordering::SeqCst));
        let task = Task::new(
            id.clone(),
            TaskStatus::Working,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        )
        .with_ttl_ms(600_000)
        .with_poll_interval_ms(POLL.as_millis() as u64)
        .with_status_message("queued");
        self.store.scripted.lock().unwrap().insert(
            id,
            Scripted {
                task: task.clone(),
                states: states.into(),
            },
        );
        let mut meta = MetaObject::new();
        meta.0.insert("trace".to_owned(), json!("t-1"));
        CallToolResponse::Task(CreateTaskResult::new(task).with_meta(meta))
    }
}

impl ServerHandler for TaskServer {
    fn get_info(&self) -> ServerConfig {
        let builder = ServerCapabilities::builder().enable_tools();
        let capabilities = if self.tasks {
            builder.enable_tasks().build()
        } else {
            builder.build()
        };
        ServerConfig::new(capabilities)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(catalog()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arguments = Value::Object(request.arguments.clone().unwrap_or_default());
        Ok(match request.name.as_ref() {
            "plain" => CallToolResult::success(vec![ContentBlock::text("plain")]).into(),
            "eventually" => self.script(vec![
                TaskPayload::Working,
                TaskPayload::Working,
                completed(CallToolResult::success(vec![ContentBlock::text("done")])),
            ]),
            "never" => self.script(vec![TaskPayload::Working]),
            "failing" => self.script(vec![
                TaskPayload::Working,
                TaskPayload::Failed {
                    error: object(json!({"code": -32603, "message": "boom"})),
                },
            ]),
            "typed" => {
                let value = match arguments["mode"].as_str() {
                    Some("invalid") => json!({"n": "seven"}),
                    _ => json!({"n": 7}),
                };
                self.script(vec![TaskPayload::Working, completed(structured(value))])
            }
            "managed" => {
                let task = self.store.manager.spawn(TaskOptions::default(), |context| {
                    Box::pin(async move {
                        context.cancelled().await;
                        Err(TaskExit::Cancelled)
                    })
                });
                CallToolResponse::Task(CreateTaskResult::new(task))
            }
            other => return Err(ErrorData::invalid_params(format!("unknown {other}"), None)),
        })
    }

    async fn get_task(
        &self,
        request: GetTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetTaskResult, ErrorData> {
        let id = request.task_id;
        self.store
            .gets
            .lock()
            .unwrap()
            .push((id.clone(), Instant::now()));
        let mut scripted = self.store.scripted.lock().unwrap();
        let Some(entry) = scripted.get_mut(&id) else {
            drop(scripted);
            return self.store.manager.get_task(&id).map(GetTaskResult::new);
        };
        let state = if entry.states.len() > 1 {
            entry.states.pop_front().unwrap()
        } else {
            entry.states[0].clone()
        };
        Ok(GetTaskResult::new(DetailedTask::new(
            entry.task.clone(),
            state,
        )))
    }

    async fn cancel_task(
        &self,
        request: CancelTaskParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        let id = request.task_id;
        self.store.cancels.lock().unwrap().push(id.clone());
        let mut scripted = self.store.scripted.lock().unwrap();
        match scripted.get_mut(&id) {
            Some(entry) => {
                entry.states = VecDeque::from([TaskPayload::Cancelled]);
                Ok(())
            }
            None => {
                drop(scripted);
                self.store.manager.cancel_task(&id)
            }
        }
    }
}

/// A client configuration that declares the tasks extension.
fn tasks_client() -> ClientConfig {
    let mut capabilities = ClientCapabilities::default();
    capabilities
        .extensions
        .get_or_insert_with(Default::default)
        .insert(TASKS_EXTENSION_ID.to_owned(), JsonObject::new());
    ClientConfig::new(capabilities, Implementation::new("tasks-test", "0.0.0"))
}

/// Connect `client` to `server` over a fresh in-memory pipe.
async fn connect(
    server: TaskServer,
    client: ClientConfig,
) -> RunningService<RoleClient, ClientConfig> {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    client
        .serve(client_io)
        .await
        .expect("client should connect")
}

struct Fixture {
    store: Store,
    client: RunningService<RoleClient, ClientConfig>,
}

impl Fixture {
    async fn new() -> Self {
        Self::with(true, tasks_client()).await
    }

    async fn with(tasks: bool, client: ClientConfig) -> Self {
        let store = Store::default();
        let client = connect(
            TaskServer {
                tasks,
                store: store.clone(),
            },
            client,
        )
        .await;
        Self { store, client }
    }

    fn peer(&self) -> Peer<RoleClient> {
        self.client.peer().clone()
    }

    /// Another session to the same server, sharing its task store.
    async fn second_session(&self) -> RunningService<RoleClient, ClientConfig> {
        connect(
            TaskServer {
                tasks: true,
                store: self.store.clone(),
            },
            tasks_client(),
        )
        .await
    }

    fn get_times(&self) -> Vec<Instant> {
        self.store
            .gets
            .lock()
            .unwrap()
            .iter()
            .map(|(_, at)| *at)
            .collect()
    }

    async fn start(&self, name: &'static str) -> String {
        match call_tool_or_task(
            &self.peer(),
            CallToolRequestParams::new(name),
            RequestOptions::default(),
        )
        .await
        .unwrap()
        {
            ToolCallOutcome::Task(handle) => handle.task_id().to_owned(),
            other => panic!("expected a task, got {other:?}"),
        }
    }
}

fn tool_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: format!("call_{name}"),
        name: name.to_owned(),
        arguments,
    }
}

// ── Opting in ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn call_tool_still_reports_a_task_answer_as_unsupported() {
    let fixture = Fixture::new().await;

    let error = call_tool(
        &fixture.peer(),
        CallToolRequestParams::new("eventually"),
        RequestOptions::default(),
    )
    .await
    .unwrap_err();

    match error {
        McpError::UnsupportedResponse { tool, kind } => {
            assert_eq!(tool, "eventually");
            assert_eq!(kind, "task");
        }
        other => panic!("expected UnsupportedResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn call_tool_or_task_returns_the_servers_task() {
    let fixture = Fixture::new().await;

    let outcome = call_tool_or_task(
        &fixture.peer(),
        CallToolRequestParams::new("eventually"),
        RequestOptions::default(),
    )
    .await
    .unwrap();

    let ToolCallOutcome::Task(handle) = outcome else {
        panic!("expected a task, got {outcome:?}");
    };
    assert_eq!(handle.tool, "eventually");
    assert_eq!(handle.task_id(), "task-0");
    assert_eq!(handle.task.status, TaskStatus::Working);
    assert_eq!(handle.task.status_message.as_deref(), Some("queued"));
    assert_eq!(handle.task.created_at, "2026-01-01T00:00:00Z");
    assert_eq!(handle.task.last_updated_at, "2026-01-01T00:00:00Z");
    assert_eq!(handle.task.ttl_ms, Some(600_000));
    assert_eq!(handle.task.poll_interval_ms, Some(2_000));
    assert_eq!(handle.meta.unwrap().0["trace"], "t-1");
}

#[tokio::test]
async fn an_ordinary_result_through_the_task_entry_point_is_completed() {
    // A server without the extension answers as it always has.
    let fixture = Fixture::with(false, tasks_client()).await;

    let outcome = call_tool_or_task(
        &fixture.peer(),
        CallToolRequestParams::new("plain"),
        RequestOptions::default(),
    )
    .await
    .unwrap();

    match outcome {
        ToolCallOutcome::Completed(result) => {
            assert_eq!(result.content[0].as_text().unwrap().text, "plain");
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn a_client_that_did_not_declare_tasks_never_receives_one() {
    let fixture = Fixture::with(true, ClientConfig::default()).await;

    let error = call_tool_or_task(
        &fixture.peer(),
        CallToolRequestParams::new("eventually"),
        RequestOptions::default(),
    )
    .await
    .unwrap_err();

    match error {
        McpError::Service(ServiceError::McpError(error)) => {
            assert_eq!(error.code, ErrorCode::MISSING_REQUIRED_CLIENT_CAPABILITY);
        }
        other => panic!("expected a missing-capability error, got {other:?}"),
    }
}

// ── Reading and cancelling ─────────────────────────────────────────────────

#[tokio::test]
async fn get_task_reports_working_then_the_completed_result() {
    let fixture = Fixture::new().await;
    let id = fixture.start("eventually").await;
    let peer = fixture.peer();

    for _ in 0..2 {
        let snapshot = get_task(&peer, &id, RequestOptions::default())
            .await
            .unwrap();
        assert_eq!(snapshot.state, TaskState::Working);
        assert!(!snapshot.is_terminal());
        assert_eq!(snapshot.task.task_id, id);
    }

    let snapshot = get_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();
    assert!(snapshot.is_terminal());
    assert_eq!(snapshot.task.status, TaskStatus::Completed);
    match snapshot.state {
        TaskState::Completed(result) => {
            assert_eq!(result.content[0].as_text().unwrap().text, "done");
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn a_failed_task_carries_the_servers_error() {
    let fixture = Fixture::new().await;
    let id = fixture.start("failing").await;
    let peer = fixture.peer();

    get_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();
    let snapshot = get_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();

    match snapshot.state {
        TaskState::Failed(error) => {
            assert_eq!(error.code, ErrorCode::INTERNAL_ERROR);
            assert_eq!(error.message, "boom");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_task_cancels_cooperatively_on_the_server() {
    // `managed` runs on rmcp's TaskManager and stops only when cancelled.
    let fixture = Fixture::new().await;
    let id = fixture.start("managed").await;
    let peer = fixture.peer();

    let snapshot = get_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(snapshot.state, TaskState::Working);

    cancel_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();
    let snapshot = wait_for_task(
        &peer,
        &id,
        RequestOptions::with_timeout(WAIT),
        std::future::pending(),
    )
    .await
    .unwrap();
    assert_eq!(snapshot.state, TaskState::Cancelled);
    assert_eq!(snapshot.task.status, TaskStatus::Cancelled);
}

#[tokio::test]
async fn cancelling_a_scripted_task_is_reported_by_the_next_read() {
    let fixture = Fixture::new().await;
    let id = fixture.start("never").await;
    let peer = fixture.peer();

    cancel_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();

    assert_eq!(
        *fixture.store.cancels.lock().unwrap(),
        std::slice::from_ref(&id)
    );
    let snapshot = get_task(&peer, &id, RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(snapshot.state, TaskState::Cancelled);
}

#[tokio::test]
async fn an_unknown_task_id_is_task_not_found() {
    let fixture = Fixture::new().await;
    let peer = fixture.peer();

    let error = get_task(&peer, "no-such-task", RequestOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, McpError::TaskNotFound { ref task_id } if task_id == "no-such-task"),
        "{error:?}"
    );

    let error = cancel_task(&peer, "no-such-task", RequestOptions::default())
        .await
        .unwrap_err();
    assert!(
        matches!(error, McpError::TaskNotFound { ref task_id } if task_id == "no-such-task"),
        "{error:?}"
    );
}

#[tokio::test]
async fn an_already_cancelled_read_is_never_sent() {
    let fixture = Fixture::new().await;
    let id = fixture.start("never").await;

    let error = rai_sdk::mcp::get_task_until(
        &fixture.peer(),
        &id,
        RequestOptions::default(),
        std::future::ready(()),
    )
    .await
    .unwrap_err();

    assert!(
        matches!(error, McpError::TaskRequestCancelled { ref task_id } if *task_id == id),
        "{error:?}"
    );
    assert!(fixture.get_times().is_empty());
}

#[tokio::test]
async fn a_task_created_in_one_session_is_read_from_another() {
    let fixture = Fixture::new().await;
    let id = fixture.start("eventually").await;
    // The first session goes away; only the id is carried over.
    let first = fixture.peer();
    drop(first);

    let later = fixture.second_session().await;
    let mut states = Vec::new();
    for _ in 0..3 {
        let snapshot = get_task(later.peer(), &id, RequestOptions::default())
            .await
            .unwrap();
        states.push(snapshot.state);
    }
    assert_eq!(states[..2], [TaskState::Working, TaskState::Working]);
    assert!(matches!(states[2], TaskState::Completed(_)));
}

// ── Polling ────────────────────────────────────────────────────────────────

#[tokio::test(start_paused = true)]
async fn waiting_polls_at_the_servers_interval_until_terminal() {
    let fixture = Fixture::new().await;
    let id = fixture.start("eventually").await;
    let started = Instant::now();

    let snapshot = wait_for_task(
        &fixture.peer(),
        &id,
        RequestOptions::with_timeout(Duration::from_secs(60)),
        std::future::pending(),
    )
    .await
    .unwrap();

    assert!(matches!(snapshot.state, TaskState::Completed(_)));
    let times = fixture.get_times();
    assert_eq!(times.len(), 3, "working, working, completed");
    assert!(times[0] - started < POLL, "the first read is immediate");
    for pair in times.windows(2) {
        assert!(pair[1] - pair[0] >= POLL, "polled faster than suggested");
        assert!(pair[1] - pair[0] < POLL * 2, "polled slower than suggested");
    }
}

#[tokio::test(start_paused = true)]
async fn waiting_ends_at_the_deadline_with_a_timeout() {
    let fixture = Fixture::new().await;
    let id = fixture.start("never").await;
    let started = Instant::now();

    let error = wait_for_task(
        &fixture.peer(),
        &id,
        RequestOptions::with_timeout(Duration::from_secs(5)),
        std::future::pending(),
    )
    .await
    .unwrap_err();

    match error {
        McpError::Service(ServiceError::Timeout { timeout }) => {
            assert_eq!(timeout, Duration::from_secs(5));
        }
        other => panic!("expected a timeout, got {other:?}"),
    }
    let elapsed = Instant::now() - started;
    assert!(elapsed >= Duration::from_secs(5) && elapsed < Duration::from_secs(6));
    // Reads at 0 s, 2 s and 4 s; the deadline falls in the next pause.
    assert_eq!(fixture.get_times().len(), 3);
    // Ending the wait does not cancel the task.
    assert!(fixture.store.cancels.lock().unwrap().is_empty());
}

#[tokio::test(start_paused = true)]
async fn waiting_ends_when_the_caller_cancels() {
    let fixture = Fixture::new().await;
    let id = fixture.start("never").await;

    let error = wait_for_task(
        &fixture.peer(),
        &id,
        RequestOptions::default(),
        tokio::time::sleep(Duration::from_secs(3)),
    )
    .await
    .unwrap_err();

    assert!(
        matches!(error, McpError::TaskRequestCancelled { ref task_id } if *task_id == id),
        "{error:?}"
    );
    assert_eq!(fixture.get_times().len(), 2, "reads at 0 s and 2 s");
    assert!(fixture.store.cancels.lock().unwrap().is_empty());
}

// ── Through McpTools ───────────────────────────────────────────────────────

#[tokio::test]
async fn catalog_paths_keep_a_task_answer_an_error_unless_asked() {
    let fixture = Fixture::new().await;
    let tools = McpTools::discover(fixture.peer()).await.unwrap();
    let call = tool_call("eventually", json!({}));

    let error = tools.call(&call).await.unwrap_err();
    assert!(matches!(
        error,
        McpError::UnsupportedResponse { kind: "task", .. }
    ));
    let error = tools
        .call_result(&call, RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        McpError::UnsupportedResponse { kind: "task", .. }
    ));

    let outcome = tools
        .call_result_or_task(&call, RequestOptions::default())
        .await
        .unwrap();
    assert!(matches!(outcome, ToolCallOutcome::Task(_)), "{outcome:?}");
}

#[tokio::test]
async fn the_handle_names_the_advertised_tool() {
    let fixture = Fixture::new().await;
    let tools = McpTools::discover(fixture.peer())
        .await
        .unwrap()
        .rename(|tool| format!("srv__{}", tool.name))
        .unwrap();

    let outcome = tools
        .call_result_or_task(
            &tool_call("srv__typed", json!({"mode": "valid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap();
    let ToolCallOutcome::Task(handle) = outcome else {
        panic!("expected a task, got {outcome:?}");
    };
    assert_eq!(handle.tool, "srv__typed");

    let snapshot = tools
        .wait_for_task(
            &handle.tool,
            handle.task_id(),
            RequestOptions::with_timeout(WAIT),
            std::future::pending(),
        )
        .await
        .unwrap();
    match snapshot.state {
        TaskState::Completed(result) => {
            assert_eq!(result.structured_content, Some(json!({"n": 7})));
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn a_completed_task_result_is_validated_like_a_direct_call() {
    let fixture = Fixture::new().await;
    let tools = McpTools::discover(fixture.peer()).await.unwrap();

    let ToolCallOutcome::Task(handle) = tools
        .call_result_or_task(
            &tool_call("typed", json!({"mode": "invalid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap()
    else {
        panic!("expected a task");
    };

    // Still working: nothing to validate yet.
    let snapshot = tools
        .get_task(&handle.tool, handle.task_id(), RequestOptions::default())
        .await
        .unwrap();
    assert_eq!(snapshot.state, TaskState::Working);

    let error = tools
        .get_task(&handle.tool, handle.task_id(), RequestOptions::default())
        .await
        .unwrap_err();
    match error {
        McpError::OutputSchemaViolation { tool, detail } => {
            assert_eq!(tool, "typed");
            assert!(detail.contains("keyword"), "{detail}");
            assert!(!detail.contains("seven"), "result data leaked: {detail}");
        }
        other => panic!("expected OutputSchemaViolation, got {other:?}"),
    }

    // The raw read returns the result unchanged, and the explicit validator
    // rejects it the same way.
    let snapshot = get_task(&fixture.peer(), handle.task_id(), RequestOptions::default())
        .await
        .unwrap();
    let TaskState::Completed(result) = snapshot.state else {
        panic!("expected Completed");
    };
    let processor = SchemaProcessor::default();
    let schemas = processor
        .compile(
            vec![("typed".to_owned(), typed_schema())],
            RequestOptions::default(),
        )
        .await
        .unwrap();
    let error = processor
        .validate(&schemas, "typed", &result, RequestOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::OutputSchemaViolation { .. }));
}

#[tokio::test]
async fn reading_a_task_for_a_tool_outside_the_catalog_sends_nothing() {
    let fixture = Fixture::new().await;
    let tools = McpTools::discover(fixture.peer())
        .await
        .unwrap()
        .filter(|tool| tool.name != "eventually");
    let id = fixture.start("eventually").await;

    let error = tools
        .get_task("eventually", &id, RequestOptions::default())
        .await
        .unwrap_err();

    assert!(matches!(error, McpError::ToolNotFound { ref name } if name == "eventually"));
    assert!(fixture.get_times().is_empty());
}

#[tokio::test]
async fn a_catalog_on_a_later_session_validates_a_task_from_an_earlier_one() {
    let fixture = Fixture::new().await;
    let first = McpTools::discover(fixture.peer()).await.unwrap();
    let ToolCallOutcome::Task(handle) = first
        .call_result_or_task(
            &tool_call("typed", json!({"mode": "valid"})),
            RequestOptions::default(),
        )
        .await
        .unwrap()
    else {
        panic!("expected a task");
    };
    let (tool, task_id) = (handle.tool.clone(), handle.task_id().to_owned());
    drop((first, handle));

    let later = fixture.second_session().await;
    let tools = McpTools::discover(later.peer().clone()).await.unwrap();
    tools
        .get_task(&tool, &task_id, RequestOptions::default())
        .await
        .unwrap();
    let snapshot = tools
        .get_task(&tool, &task_id, RequestOptions::default())
        .await
        .unwrap();
    assert!(matches!(snapshot.state, TaskState::Completed(_)));
}
