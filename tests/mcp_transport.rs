//! Request deadlines and cancellation at the transport boundary, offline.
//!
//! The client reaches an in-process MCP server through a wrapper transport
//! that records every JSON-RPC message in both directions and can stall the
//! session on purpose:
//!
//! - it can block the session's event loop inside a send, so the peer's
//!   outbound queue is no longer drained and fills up;
//! - it can hold `notifications/cancelled` deliveries open indefinitely.
//!
//! Every wait in a test is bounded by [`WAIT`], and the transport never
//! blocks for longer than [`BLOCK_LIMIT`].
#![cfg(feature = "mcp")]

use std::{
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc as std_mpsc,
    },
    time::Duration,
};

use futures::FutureExt;
use rai_sdk::mcp::rmcp::{
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceError, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ClientJsonRpcMessage,
        ContentBlock, ServerCapabilities, ServerConfig, ServerJsonRpcMessage,
    },
    service::{Peer, RequestContext, RunningService},
    transport::{IntoTransport, Transport, async_rw::TransportAdapterAsyncCombinedRW},
};
use rai_sdk::mcp::{McpError, RequestOptions, call_tool, call_tool_until};
use serde_json::Value;
use tokio::sync::oneshot;

/// Upper bound for any single wait in a test; a hang fails the test.
const WAIT: Duration = Duration::from_secs(5);

/// Longest time the transport blocks the session loop. Dropping the release
/// sender ends the block at once, including when a test fails.
const BLOCK_LIMIT: Duration = Duration::from_secs(10);

/// How long a stalled cancellation delivery stays open: far longer than any
/// wait in a test, so a caller that waited for it would fail on [`WAIT`].
const STALL: Duration = Duration::from_secs(60);

/// Notifications offered to a blocked session. Comfortably more than the
/// peer's outbound queue holds, so the queue is full afterwards.
const QUEUE_FILL_ATTEMPTS: usize = 4096;

const CANCELLED: &str = "notifications/cancelled";
const ROOTS_CHANGED: &str = "notifications/roots/list_changed";

/// Everything the client transport saw, plus the stalls it is told to apply.
#[derive(Default)]
struct Wire {
    /// Client-to-server messages, in the order the session loop sent them.
    sent: Mutex<Vec<Value>>,
    /// Server-to-client messages, in the order the session loop read them.
    received: Mutex<Vec<Value>>,
    /// When set, the next `tools/call` blocks the session loop until the
    /// paired sender is used or dropped.
    block_next_call: Mutex<Option<std_mpsc::Receiver<()>>>,
    /// Set once the session loop is blocked.
    loop_blocked: AtomicBool,
    /// Hold every `notifications/cancelled` delivery open for [`STALL`].
    stall_cancellations: AtomicBool,
    /// Cancellations actually written to the server.
    cancellations_delivered: AtomicUsize,
}

impl Wire {
    fn sent(&self) -> Vec<Value> {
        self.sent.lock().unwrap().clone()
    }

    /// Names of every `tools/call` that reached the transport, in order.
    fn calls(&self) -> Vec<String> {
        self.sent()
            .iter()
            .filter(|message| message["method"] == "tools/call")
            .filter_map(|message| message["params"]["name"].as_str().map(str::to_owned))
            .collect()
    }

    /// JSON-RPC ids of every `tools/call` for `name`, in order.
    fn call_ids(&self, name: &str) -> Vec<Value> {
        self.sent()
            .into_iter()
            .filter(|message| {
                message["method"] == "tools/call" && message["params"]["name"] == name
            })
            .map(|message| message["id"].clone())
            .collect()
    }

    /// Request ids named by every `notifications/cancelled` the client sent.
    fn cancelled_ids(&self) -> Vec<Value> {
        self.sent()
            .iter()
            .filter(|message| message["method"] == CANCELLED)
            .map(|message| message["params"]["requestId"].clone())
            .collect()
    }

    fn count(&self, method: &str) -> usize {
        self.sent()
            .iter()
            .filter(|message| message["method"] == method)
            .count()
    }

    /// Whether the session loop has read the server's answer to `id`.
    fn answered(&self, id: &Value) -> bool {
        self.received.lock().unwrap().iter().any(|message| {
            &message["id"] == id
                && (message.get("result").is_some() || message.get("error").is_some())
        })
    }

    fn delivered(&self) -> usize {
        self.cancellations_delivered.load(Ordering::SeqCst)
    }
}

/// A client transport that records traffic and applies the [`Wire`] stalls.
struct Recorded<T> {
    inner: T,
    wire: Arc<Wire>,
}

impl<T> Transport<RoleClient> for Recorded<T>
where
    T: Transport<RoleClient>,
{
    type Error = T::Error;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let message = serde_json::to_value(&item).expect("client messages serialize");
        let is_call = message["method"] == "tools/call";
        let is_cancellation = message["method"] == CANCELLED;
        self.wire.sent.lock().unwrap().push(message);

        if is_call {
            let gate = self.wire.block_next_call.lock().unwrap().take();
            if let Some(gate) = gate {
                // `send` runs inside the session's event loop, so blocking
                // here stops the loop from draining the peer's outbound
                // queue until the test releases it.
                self.wire.loop_blocked.store(true, Ordering::SeqCst);
                let _ = gate.recv_timeout(BLOCK_LIMIT);
            }
        }

        let stall = is_cancellation && self.wire.stall_cancellations.load(Ordering::SeqCst);
        let send = (!stall).then(|| self.inner.send(item));
        let wire = Arc::clone(&self.wire);
        async move {
            match send {
                Some(send) => send.await?,
                None => tokio::time::sleep(STALL).await,
            }
            if is_cancellation {
                wire.cancellations_delivered.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<ServerJsonRpcMessage>> + Send {
        let wire = Arc::clone(&self.wire);
        let next = self.inner.receive();
        async move {
            let message = next.await;
            if let Some(message) = &message {
                let value = serde_json::to_value(message).expect("server messages serialize");
                wire.received.lock().unwrap().push(value);
            }
            message
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

#[derive(Clone)]
struct TestServer;

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        match request.name.as_ref() {
            "quick" | "unread" => {
                Ok(CallToolResult::success(vec![ContentBlock::text("ok")]).into())
            }
            "hang" => {
                context.ct.cancelled().await;
                Ok(CallToolResult::success(vec![ContentBlock::text("late")]).into())
            }
            other => Err(ErrorData::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}

struct Session {
    client: RunningService<RoleClient, ()>,
    wire: Arc<Wire>,
}

impl Session {
    fn peer(&self) -> Peer<RoleClient> {
        self.client.peer().clone()
    }

    /// A completed round trip: everything the client queued before it has
    /// been written to the transport.
    async fn round_trip(&self) {
        let result = tokio::time::timeout(
            WAIT,
            call_tool(
                &self.peer(),
                params("quick"),
                RequestOptions::with_timeout(WAIT),
            ),
        )
        .await
        .expect("round trip should not hang");
        result.expect("round trip should succeed");
    }

    /// Lets a cancellation spawned by a dropped call reach the session's
    /// queue, then completes a round trip queued behind it. Relies on the
    /// current-thread runtime of `#[tokio::test]`.
    async fn flush(&self) {
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        self.round_trip().await;
    }
}

async fn session() -> Session {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = TestServer.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    let wire = Arc::new(Wire::default());
    let transport = Recorded {
        inner: IntoTransport::<RoleClient, std::io::Error, TransportAdapterAsyncCombinedRW>::into_transport(
            client_io,
        ),
        wire: Arc::clone(&wire),
    };
    let client = tokio::time::timeout(WAIT, ().serve(transport))
        .await
        .expect("the handshake should not hang")
        .expect("client should connect");
    Session { client, wire }
}

fn params(name: &'static str) -> CallToolRequestParams {
    CallToolRequestParams::new(name)
}

/// Waits, bounded by [`WAIT`], until `done` holds.
async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let waited = tokio::time::timeout(WAIT, async {
        while !done() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(waited.is_ok(), "timed out waiting for {what}");
}

fn assert_timeout(result: Result<CallToolResult, McpError>, expected: Duration) {
    match result {
        Err(McpError::Service(ServiceError::Timeout { timeout })) => assert_eq!(timeout, expected),
        other => panic!("expected a service timeout, got {other:?}"),
    }
}

/// The hard budget covers the wait for a place in the session's outbound
/// queue, and neither the deadline nor an explicit cancellation that ends that
/// wait leaves anything queued behind: the abandoned calls never reach the
/// transport and owe no cancellation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deadlines_and_cancellation_cover_a_blocked_dispatch() {
    let session = session().await;
    let peer = session.peer();
    let (release, gate) = std_mpsc::channel::<()>();
    *session.wire.block_next_call.lock().unwrap() = Some(gate);

    let blocker = tokio::spawn({
        let peer = peer.clone();
        async move { call_tool(&peer, params("quick"), RequestOptions::default()).await }
    });
    eventually("the session loop to block", || {
        session.wire.loop_blocked.load(Ordering::SeqCst)
    })
    .await;

    // Nothing drains the outbound queue now, so it fills and stays full.
    for _ in 0..QUEUE_FILL_ATTEMPTS {
        let _ = tokio::task::unconstrained(peer.notify_roots_list_changed()).now_or_never();
    }

    let budget = Duration::from_millis(150);
    let timed_out = tokio::time::timeout(
        WAIT,
        call_tool(&peer, params("late"), RequestOptions::with_timeout(budget)),
    )
    .await
    .expect("the deadline must end a blocked dispatch");
    assert_timeout(timed_out, budget);

    let cancelled = tokio::time::timeout(
        WAIT,
        call_tool_until(
            &peer,
            params("lateCancelled"),
            RequestOptions::default(),
            tokio::time::sleep(Duration::from_millis(100)),
        ),
    )
    .await
    .expect("cancellation must end a blocked dispatch");
    assert!(
        matches!(cancelled, Err(McpError::Cancelled { ref tool }) if tool == "lateCancelled"),
        "{cancelled:?}"
    );

    drop(release);
    tokio::time::timeout(WAIT, blocker)
        .await
        .expect("the blocked call should finish once released")
        .expect("the blocked call should not panic")
        .expect("the blocked call should succeed");
    session.round_trip().await;

    let queued = session.wire.count(ROOTS_CHANGED);
    assert!(
        queued > 0 && queued < QUEUE_FILL_ATTEMPTS,
        "the outbound queue should have filled: it accepted {queued} of {QUEUE_FILL_ATTEMPTS}"
    );
    assert_eq!(
        session.wire.calls(),
        ["quick", "quick"],
        "abandoned calls must never reach the transport"
    );
    assert!(
        session.wire.cancelled_ids().is_empty(),
        "undispatched calls owe no cancellation"
    );
}

/// After dispatch, a deadline or an explicit cancellation returns to the
/// caller while the cancellation notice is still undelivered, and each call
/// still sends exactly one notice for its own request.
#[tokio::test]
async fn deadlines_and_cancellation_do_not_wait_for_cancellation_delivery() {
    let session = session().await;
    let peer = session.peer();
    session
        .wire
        .stall_cancellations
        .store(true, Ordering::SeqCst);

    let budget = Duration::from_millis(200);
    let timed_out = tokio::time::timeout(
        WAIT,
        call_tool(&peer, params("hang"), RequestOptions::with_timeout(budget)),
    )
    .await
    .expect("a stalled cancellation must not hold the caller past its deadline");
    assert_timeout(timed_out, budget);
    assert_eq!(session.wire.delivered(), 0);
    assert_eq!(
        session.wire.call_ids("hang").len(),
        1,
        "the call was dispatched before its deadline"
    );

    let (cancel, cancelled) = oneshot::channel::<()>();
    let call = tokio::spawn({
        let peer = peer.clone();
        async move {
            call_tool_until(
                &peer,
                params("hang"),
                RequestOptions::default(),
                async move {
                    let _ = cancelled.await;
                },
            )
            .await
        }
    });
    eventually("the second call to be dispatched", || {
        session.wire.call_ids("hang").len() == 2
    })
    .await;
    cancel.send(()).expect("the call should still be waiting");
    let result = tokio::time::timeout(WAIT, call)
        .await
        .expect("a stalled cancellation must not hold a cancelled caller")
        .expect("the call should not panic");
    assert!(
        matches!(result, Err(McpError::Cancelled { ref tool }) if tool == "hang"),
        "{result:?}"
    );

    eventually("both cancellations to be sent", || {
        session.wire.cancelled_ids().len() >= 2
    })
    .await;
    session.round_trip().await;
    assert_eq!(
        session.wire.cancelled_ids(),
        session.wire.call_ids("hang"),
        "exactly one cancellation per abandoned call"
    );
    assert_eq!(
        session.wire.delivered(),
        0,
        "both deliveries are still stalled"
    );
}

/// A call dropped after its response reached the client, but before the
/// caller read it, completed on the server and owes no cancellation. The
/// unanswered call dropped first shows the same flush does catch one.
#[tokio::test]
async fn a_delivered_but_unread_response_is_not_cancelled() {
    let session = session().await;
    let peer = session.peer();

    let mut unanswered = Box::pin(call_tool(&peer, params("hang"), RequestOptions::default()));
    assert!(futures::poll!(unanswered.as_mut()).is_pending());
    eventually("the unanswered call to be sent", || {
        !session.wire.call_ids("hang").is_empty()
    })
    .await;
    drop(unanswered);
    session.flush().await;
    let hang = session.wire.call_ids("hang");
    assert_eq!(session.wire.cancelled_ids(), hang);

    let mut unread = Box::pin(call_tool(
        &peer,
        params("unread"),
        RequestOptions::default(),
    ));
    assert!(futures::poll!(unread.as_mut()).is_pending());
    eventually("the unread call to be sent", || {
        !session.wire.call_ids("unread").is_empty()
    })
    .await;
    let id = session.wire.call_ids("unread").remove(0);
    // The session loop hands a response to the caller in the same poll that
    // reads it, so once it has been read, it is waiting for the caller.
    eventually("the response to reach the client", || {
        session.wire.answered(&id)
    })
    .await;
    drop(unread);
    session.flush().await;
    assert_eq!(
        session.wire.cancelled_ids(),
        hang,
        "a delivered response owes no cancellation"
    );
}
