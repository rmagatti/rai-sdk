//! Tool calls answered with a task (the MCP Tasks extension,
//! `io.modelcontextprotocol/tasks`).

use std::{future::Future, time::Duration};

use rmcp::{
    ErrorData, RoleClient, ServiceError,
    model::{
        CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskRequest, ClientRequest,
        DetailedTask, ErrorCode, GetTaskParams, GetTaskRequest, GetTaskResult, InputRequests,
        MetaObject, ServerResult, Task, TaskPayload,
    },
    service::Peer,
};

use super::{Budget, McpError, McpTools, RequestFailure, RequestOptions, budgeted, request_inner};

/// Interval used between polls when the server suggests none.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Shortest interval between polls, whatever the server suggests.
const MIN_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Longest interval between polls, whatever the server suggests.
const MAX_POLL_INTERVAL: Duration = Duration::from_secs(60);

/// How a server answered a tool call made through an entry point that
/// accepts tasks, such as [`call_tool_or_task`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolCallOutcome {
    /// The server answered with the tool's final result.
    Completed(CallToolResult),
    /// The server created a task and will produce the result later. Read it
    /// with [`get_task`] or [`wait_for_task`], or cancel it with
    /// [`cancel_task`].
    Task(TaskHandle),
}

/// A task a server created in answer to a tool call.
///
/// Plain data: it holds no session, so the task can be read or cancelled
/// through any peer the server lets reach it. To resume after a restart or
/// from another session, keep [`task_id`](Self::task_id) and
/// [`tool`](Self::tool).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TaskHandle {
    /// The name the call was made under: the catalog's advertised name for
    /// [`McpTools::call_result_or_task`], the server's tool name for
    /// [`call_tool_or_task`]. [`McpTools::get_task`] takes the former to find
    /// the tool's output schema.
    pub tool: String,
    /// The task as the server created it: id, status, status message,
    /// creation and update timestamps, time to live and suggested poll
    /// interval.
    pub task: Task,
    /// The `_meta` the server attached to its answer.
    pub meta: Option<MetaObject>,
}

impl TaskHandle {
    /// The server-generated task id, used by every later task request.
    pub fn task_id(&self) -> &str {
        &self.task.task_id
    }
}

/// The state of a task as of one `tasks/get` response.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TaskSnapshot {
    /// Task metadata as the server reported it. Its `status` agrees with
    /// [`state`](Self::state); `poll_interval_ms` and `ttl_ms` may change
    /// between reads.
    pub task: Task,
    /// The `_meta` the server attached to the response.
    pub meta: Option<MetaObject>,
    /// The status, with its payload.
    pub state: TaskState,
}

impl TaskSnapshot {
    /// Whether the task has finished: completed, failed or cancelled.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

/// A task's status and its status-specific payload.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TaskState {
    /// The server is still working on the call.
    Working,
    /// The server is waiting for the client to answer the listed requests
    /// with `tasks/update`. This module does not answer them; an application
    /// that does can use [`Peer::update_task`].
    InputRequired(InputRequests),
    /// The call finished, and this is its result. A tool that ran and failed
    /// reports that here with `isError: true`, as on a direct call.
    Completed(CallToolResult),
    /// The call failed with a JSON-RPC error, which is what a direct call
    /// would have returned as an error response. This includes a task the
    /// server expired after its time to live.
    Failed(ErrorData),
    /// The task was cancelled before it finished.
    Cancelled,
}

impl TaskState {
    /// Whether this is a final state: completed, failed or cancelled.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Failed(_) | Self::Cancelled)
    }
}

/// Call a tool and accept either its final result or a task.
///
/// The same call as [`call_tool`](super::call_tool), with the same deadline,
/// cancellation and error behavior, except that a task answer is returned as
/// [`ToolCallOutcome::Task`] instead of
/// [`McpError::UnsupportedResponse`]. A server answers with a task only when
/// the client declared the tasks extension; see the
/// [module documentation](super#tasks). An `input_required` answer is still
/// [`McpError::UnsupportedResponse`].
///
/// No output-schema validation is applied, as with `call_tool`. If the call is
/// cancelled or reaches its deadline after dispatch, a task the server created
/// meanwhile is not reported to the caller.
pub async fn call_tool_or_task(
    peer: &Peer<RoleClient>,
    params: CallToolRequestParams,
    options: RequestOptions,
) -> Result<ToolCallOutcome, McpError> {
    call_tool_or_task_until(peer, params, options, std::future::pending()).await
}

/// [`call_tool_or_task`], abandoned when `cancelled` completes first.
pub async fn call_tool_or_task_until<C>(
    peer: &Peer<RoleClient>,
    params: CallToolRequestParams,
    options: RequestOptions,
    cancelled: C,
) -> Result<ToolCallOutcome, McpError>
where
    C: Future<Output = ()>,
{
    super::call_tool_inner(peer, params, options, cancelled).await
}

/// Read a task's current state with `tasks/get`.
///
/// `peer` need not be the session that created the task; see the
/// [module documentation](super#tasks) for what the protocol guarantees about
/// that. A completed result is returned unchanged, without output-schema
/// validation; [`McpTools::get_task`] validates it, or the application can
/// with [`SchemaProcessor::validate`](super::SchemaProcessor::validate).
///
/// An unknown or expired task id is [`McpError::TaskNotFound`]. A completed
/// result or failure the server encoded incorrectly is
/// [`ServiceError::UnexpectedResponse`].
pub async fn get_task(
    peer: &Peer<RoleClient>,
    task_id: &str,
    options: RequestOptions,
) -> Result<TaskSnapshot, McpError> {
    get_task_until(peer, task_id, options, std::future::pending()).await
}

/// [`get_task`], abandoned when `cancelled` completes first.
///
/// Abandoning the read sends `notifications/cancelled` for the `tasks/get`
/// request. The task itself is not cancelled.
pub async fn get_task_until<C>(
    peer: &Peer<RoleClient>,
    task_id: &str,
    options: RequestOptions,
    cancelled: C,
) -> Result<TaskSnapshot, McpError>
where
    C: Future<Output = ()>,
{
    let request = ClientRequest::GetTaskRequest(GetTaskRequest::new(GetTaskParams::new(task_id)));
    match request_inner(peer, request, options, cancelled)
        .await
        .map_err(|failure| task_request_error(task_id, failure))?
    {
        ServerResult::GetTaskResult(result) => snapshot(result),
        _ => Err(McpError::Service(ServiceError::UnexpectedResponse)),
    }
}

/// Ask the server to cancel a task with `tasks/cancel`.
///
/// Cancellation is cooperative: the acknowledgement means the server received
/// the request, not that the task stopped. The task may still complete or
/// fail; read it to learn its final state. An unknown or expired task id is
/// [`McpError::TaskNotFound`] when the server reports it as the protocol
/// recommends.
pub async fn cancel_task(
    peer: &Peer<RoleClient>,
    task_id: &str,
    options: RequestOptions,
) -> Result<(), McpError> {
    cancel_task_until(peer, task_id, options, std::future::pending()).await
}

/// [`cancel_task`], abandoned when `cancelled` completes first.
///
/// A `tasks/cancel` request abandoned after dispatch may still reach the
/// server and take effect.
pub async fn cancel_task_until<C>(
    peer: &Peer<RoleClient>,
    task_id: &str,
    options: RequestOptions,
    cancelled: C,
) -> Result<(), McpError>
where
    C: Future<Output = ()>,
{
    let request =
        ClientRequest::CancelTaskRequest(CancelTaskRequest::new(CancelTaskParams::new(task_id)));
    match request_inner(peer, request, options, cancelled)
        .await
        .map_err(|failure| task_request_error(task_id, failure))?
    {
        ServerResult::TaskAckResult(_) | ServerResult::EmptyResult(_) => Ok(()),
        _ => Err(McpError::Service(ServiceError::UnexpectedResponse)),
    }
}

/// Poll a task with `tasks/get` until it is completed, failed or cancelled.
///
/// An optional convenience over [`get_task`]: it runs in the caller's future
/// and spawns nothing. The first read is sent at once. Between reads it waits
/// for the interval the server last suggested (`pollIntervalMs`), or one
/// second when none was given, kept between 100 milliseconds and 60 seconds.
///
/// [`RequestOptions::timeout`] bounds the whole wait, reads and pauses
/// together, and ends it with [`ServiceError::Timeout`]. When `cancelled`
/// completes first, the wait ends with [`McpError::TaskRequestCancelled`].
/// Neither cancels the task; use [`cancel_task`] for that. Pass
/// `std::future::pending()` for no cancellation signal.
pub async fn wait_for_task<C>(
    peer: &Peer<RoleClient>,
    task_id: &str,
    options: RequestOptions,
    cancelled: C,
) -> Result<TaskSnapshot, McpError>
where
    C: Future<Output = ()>,
{
    budgeted(
        options,
        cancelled,
        || McpError::TaskRequestCancelled {
            task_id: task_id.to_owned(),
        },
        |budget| poll_until_terminal(peer, task_id, budget),
    )
    .await
}

impl McpTools {
    /// [`call_result`](Self::call_result) for applications that accept a task
    /// answer.
    ///
    /// Names, arguments, output schemas and the budget are handled exactly as
    /// by `call_result`. A completed result is validated before return. A task
    /// is returned as [`ToolCallOutcome::Task`] with the advertised name in
    /// [`TaskHandle::tool`]; read it with [`get_task`](Self::get_task) on this
    /// catalog or any catalog of the same server. `call`, `call_result` and
    /// [`tools`](Self::tools) are unchanged and still report a task answer as
    /// [`McpError::UnsupportedResponse`].
    pub async fn call_result_or_task(
        &self,
        call: &crate::ToolCall,
        options: RequestOptions,
    ) -> Result<ToolCallOutcome, McpError> {
        self.call_result_or_task_until(call, options, std::future::pending())
            .await
    }

    /// [`Self::call_result_or_task`], abandoned when `cancelled` completes
    /// first.
    pub async fn call_result_or_task_until<C>(
        &self,
        call: &crate::ToolCall,
        options: RequestOptions,
        cancelled: C,
    ) -> Result<ToolCallOutcome, McpError>
    where
        C: Future<Output = ()>,
    {
        let name = call.name.as_str();
        budgeted(
            options,
            cancelled,
            || McpError::Cancelled {
                tool: name.to_owned(),
            },
            |budget| async move {
                let (tool, params) = self.prepare(name, call.arguments.clone())?;
                let schemas = tool.output_schemas(&self.schema_processor).await?;
                match super::call_tool_inner(
                    &self.peer,
                    params,
                    budget.remaining(),
                    std::future::pending(),
                )
                .await?
                {
                    ToolCallOutcome::Completed(result) => {
                        self.check_output(schemas.as_deref(), name, &result, budget)
                            .await?;
                        Ok(ToolCallOutcome::Completed(result))
                    }
                    ToolCallOutcome::Task(mut handle) => {
                        handle.tool = name.to_owned();
                        Ok(ToolCallOutcome::Task(handle))
                    }
                }
            },
        )
        .await
    }

    /// Read a task created by a call to `tool`, validating a completed result.
    ///
    /// `tool` is the advertised name, as in [`TaskHandle::tool`]. It must be in
    /// this catalog ([`McpError::ToolNotFound`] otherwise, without contacting
    /// the server) and selects the output schema: a completed, successful
    /// result is checked exactly as a direct call's result is, so a mismatch
    /// is [`McpError::OutputSchemaViolation`]. The task is read through this
    /// catalog's peer, which need not be the session that created it.
    /// Otherwise as [`get_task`].
    pub async fn get_task(
        &self,
        tool: &str,
        task_id: &str,
        options: RequestOptions,
    ) -> Result<TaskSnapshot, McpError> {
        self.get_task_until(tool, task_id, options, std::future::pending())
            .await
    }

    /// [`Self::get_task`], abandoned when `cancelled` completes first.
    pub async fn get_task_until<C>(
        &self,
        tool: &str,
        task_id: &str,
        options: RequestOptions,
        cancelled: C,
    ) -> Result<TaskSnapshot, McpError>
    where
        C: Future<Output = ()>,
    {
        budgeted(
            options,
            cancelled,
            || McpError::TaskRequestCancelled {
                task_id: task_id.to_owned(),
            },
            |budget| async move {
                let schemas = self.task_tool_schemas(tool).await?;
                let snapshot = get_task(&self.peer, task_id, budget.remaining()).await?;
                self.check_snapshot(schemas.as_deref(), tool, &snapshot, budget)
                    .await?;
                Ok(snapshot)
            },
        )
        .await
    }

    /// [`wait_for_task`] through this catalog's peer,
    /// validating the final result as [`get_task`](Self::get_task) does.
    pub async fn wait_for_task<C>(
        &self,
        tool: &str,
        task_id: &str,
        options: RequestOptions,
        cancelled: C,
    ) -> Result<TaskSnapshot, McpError>
    where
        C: Future<Output = ()>,
    {
        budgeted(
            options,
            cancelled,
            || McpError::TaskRequestCancelled {
                task_id: task_id.to_owned(),
            },
            |budget| async move {
                let schemas = self.task_tool_schemas(tool).await?;
                let snapshot = poll_until_terminal(&self.peer, task_id, budget).await?;
                self.check_snapshot(schemas.as_deref(), tool, &snapshot, budget)
                    .await?;
                Ok(snapshot)
            },
        )
        .await
    }

    async fn task_tool_schemas(
        &self,
        tool: &str,
    ) -> Result<Option<std::sync::Arc<super::OutputSchemas>>, McpError> {
        let found = self.find(tool).ok_or_else(|| McpError::ToolNotFound {
            name: tool.to_owned(),
        })?;
        found.output_schemas(&self.schema_processor).await
    }

    async fn check_snapshot(
        &self,
        schemas: Option<&super::OutputSchemas>,
        tool: &str,
        snapshot: &TaskSnapshot,
        budget: Budget,
    ) -> Result<(), McpError> {
        match &snapshot.state {
            TaskState::Completed(result) => self.check_output(schemas, tool, result, budget).await,
            _ => Ok(()),
        }
    }
}

async fn poll_until_terminal(
    peer: &Peer<RoleClient>,
    task_id: &str,
    budget: Budget,
) -> Result<TaskSnapshot, McpError> {
    loop {
        let snapshot = get_task(peer, task_id, budget.remaining()).await?;
        if snapshot.is_terminal() {
            return Ok(snapshot);
        }
        tokio::time::sleep(poll_interval(snapshot.task.poll_interval_ms)).await;
    }
}

fn poll_interval(suggested_ms: Option<u64>) -> Duration {
    suggested_ms
        .map_or(DEFAULT_POLL_INTERVAL, Duration::from_millis)
        .clamp(MIN_POLL_INTERVAL, MAX_POLL_INTERVAL)
}

pub(super) fn handle(tool: String, task: Task, meta: Option<MetaObject>) -> TaskHandle {
    TaskHandle { tool, task, meta }
}

fn task_request_error(task_id: &str, failure: RequestFailure) -> McpError {
    match failure {
        RequestFailure::Cancelled => McpError::TaskRequestCancelled {
            task_id: task_id.to_owned(),
        },
        // The extension reserves Invalid params for an unknown or expired id:
        // `taskId` is the only parameter these requests carry.
        RequestFailure::Service(ServiceError::McpError(error))
            if error.code == ErrorCode::INVALID_PARAMS =>
        {
            McpError::TaskNotFound {
                task_id: task_id.to_owned(),
            }
        }
        RequestFailure::Service(error) => McpError::Service(error),
    }
}

fn snapshot(result: GetTaskResult) -> Result<TaskSnapshot, McpError> {
    let meta = result.meta;
    let DetailedTask { task, payload, .. } = result.task;
    let malformed = || McpError::Service(ServiceError::UnexpectedResponse);
    let state = match payload {
        TaskPayload::Working => TaskState::Working,
        TaskPayload::InputRequired { input_requests } => TaskState::InputRequired(input_requests),
        TaskPayload::Completed { result } => TaskState::Completed(
            serde_json::from_value(serde_json::Value::Object(result)).map_err(|_| malformed())?,
        ),
        TaskPayload::Failed { error } => TaskState::Failed(
            serde_json::from_value(serde_json::Value::Object(error)).map_err(|_| malformed())?,
        ),
        TaskPayload::Cancelled => TaskState::Cancelled,
        _ => return Err(malformed()),
    };
    Ok(TaskSnapshot { task, meta, state })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_interval_follows_the_server_within_bounds() {
        assert_eq!(poll_interval(None), DEFAULT_POLL_INTERVAL);
        assert_eq!(poll_interval(Some(2_500)), Duration::from_millis(2_500));
        assert_eq!(poll_interval(Some(0)), MIN_POLL_INTERVAL);
        assert_eq!(poll_interval(Some(u64::MAX)), MAX_POLL_INTERVAL);
    }
}
