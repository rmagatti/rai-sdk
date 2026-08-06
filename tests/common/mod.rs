//! Shared helpers for the offline integration test suite.
//!
//! Everything here is deterministic and offline. No helper ever contacts a real
//! provider: HTTP traffic either goes to a [`wiremock::MockServer`] or to the
//! raw loopback SSE server in [`split_sse`].
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use rai_sdk::provider::ProviderStreamEvent;
use rai_sdk::{ClientBuilder, Error, Result, RetryConfig};
use wiremock::{MockServer, Request, Respond, ResponseTemplate};

// ── Credentials ────────────────────────────────────────────────────────────
//
// Tests always configure keys explicitly on the builder so that the suite is
// independent of whatever the developer happens to have exported. These values
// are obvious placeholders and are only ever sent to a local mock server.

pub const OPENAI_TEST_KEY: &str = "test-openai-key";
pub const ANTHROPIC_TEST_KEY: &str = "test-anthropic-key";
pub const OPENROUTER_TEST_KEY: &str = "test-openrouter-key";

// ── Client builders pointed at a mock server ───────────────────────────────

/// An OpenAI-backed client builder pointed at `base_url`, with retries off.
///
/// Retries are disabled by default so error-mapping assertions observe the
/// first response instead of a retried one. Tests that exercise retry behavior
/// re-enable it with [`fast_retry`].
pub fn openai_builder(base_url: &str) -> ClientBuilder {
    ClientBuilder::new()
        .openai_key(OPENAI_TEST_KEY)
        .openai_base_url(base_url.to_string())
        .no_retry()
}

/// An Anthropic-backed client builder pointed at `base_url`, with retries off.
pub fn anthropic_builder(base_url: &str) -> ClientBuilder {
    ClientBuilder::new()
        .anthropic_key(ANTHROPIC_TEST_KEY)
        .anthropic_base_url(base_url.to_string())
        .no_retry()
}

/// An OpenRouter-backed client builder pointed at `base_url`, with retries off.
pub fn openrouter_builder(base_url: &str) -> ClientBuilder {
    ClientBuilder::new()
        .openrouter_key(OPENROUTER_TEST_KEY)
        .openrouter_base_url(base_url.to_string())
        .no_retry()
}

/// A retry configuration with millisecond delays and no jitter.
///
/// Keeps retry tests fast (a full 3-retry run sleeps ~7ms) while still
/// exercising the real exponential progression, and makes elapsed-time
/// assertions reproducible by disabling jitter.
pub fn fast_retry(max_retries: u32) -> RetryConfig {
    RetryConfig::new()
        .with_max_retries(max_retries)
        .with_initial_delay(Duration::from_millis(2))
        .with_max_delay(Duration::from_millis(50))
        .with_backoff_multiplier(2.0)
        .with_jitter(false)
}

// ── Scripted mock responses ────────────────────────────────────────────────

/// A single canned HTTP response used by [`Script`].
#[derive(Clone)]
pub struct Step {
    status: u16,
    content_type: &'static str,
    body: String,
}

impl Step {
    /// A JSON response with the given status.
    pub fn json(status: u16, body: serde_json::Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.to_string(),
        }
    }

    /// A `200 OK` JSON response.
    pub fn ok(body: serde_json::Value) -> Self {
        Self::json(200, body)
    }

    /// A response with a raw, possibly non-JSON, body.
    pub fn raw(status: u16, content_type: &'static str, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type,
            body: body.into(),
        }
    }

    /// A `200 OK` `text/event-stream` response.
    pub fn sse(body: impl Into<String>) -> Self {
        Self::raw(200, "text/event-stream", body)
    }

    fn template(&self) -> ResponseTemplate {
        ResponseTemplate::new(self.status)
            .insert_header("content-type", self.content_type)
            .set_body_string(self.body.clone())
    }
}

/// A responder that returns a different response per call, in order.
///
/// Once the script is exhausted the last step repeats, which keeps
/// "fail then succeed forever" retry scenarios simple to express.
pub struct Script {
    steps: Vec<Step>,
    calls: AtomicUsize,
}

impl Script {
    /// Build a script from an ordered list of responses. Must be non-empty.
    pub fn new(steps: Vec<Step>) -> Self {
        assert!(!steps.is_empty(), "a script needs at least one step");
        Self {
            steps,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Respond for Script {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let step = self
            .steps
            .get(index)
            .unwrap_or_else(|| self.steps.last().expect("script is non-empty"));
        step.template()
    }
}

// ── Request inspection ─────────────────────────────────────────────────────

/// Every request body the mock server received, parsed as JSON.
pub async fn received_json_bodies(server: &MockServer) -> Vec<serde_json::Value> {
    server
        .received_requests()
        .await
        .expect("wiremock request recording should be enabled")
        .iter()
        .map(|request| {
            serde_json::from_slice(&request.body).unwrap_or_else(|error| {
                panic!(
                    "recorded request body should be JSON: {error}\nbody: {}",
                    String::from_utf8_lossy(&request.body)
                )
            })
        })
        .collect()
}

/// The value of `header` on the request at `index`, if present.
pub async fn received_header(server: &MockServer, index: usize, header: &str) -> Option<String> {
    let requests = server
        .received_requests()
        .await
        .expect("wiremock request recording should be enabled");
    let request = requests
        .get(index)
        .unwrap_or_else(|| panic!("expected at least {} recorded request(s)", index + 1));
    request
        .headers
        .get(header)
        .map(|value| value.to_str().expect("header should be valid UTF-8").into())
}

/// How many requests the mock server received.
pub async fn request_count(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .expect("wiremock request recording should be enabled")
        .len()
}

// ── SSE helpers ────────────────────────────────────────────────────────────

/// Join pre-rendered SSE event blocks into a single response body.
///
/// Each block is terminated with the blank line that both provider parsers use
/// as their event delimiter.
pub fn sse_body(events: &[&str]) -> String {
    events
        .iter()
        .map(|event| format!("{event}\n\n"))
        .collect::<String>()
}

/// An OpenAI-style `data: {json}` event line.
pub fn data_event(payload: serde_json::Value) -> String {
    format!("data: {payload}")
}

/// An Anthropic-style `event: <type>` / `data: {json}` event block.
pub fn named_event(event: &str, payload: serde_json::Value) -> String {
    format!("event: {event}\ndata: {payload}")
}

/// Drain a provider stream, failing the test on the first stream error.
pub async fn collect_events<S>(stream: S) -> Vec<ProviderStreamEvent>
where
    S: futures::Stream<Item = Result<ProviderStreamEvent>>,
{
    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("stream should not yield an error"));
    }
    events
}

/// Drain a provider stream, keeping successes and errors in arrival order.
pub async fn collect_results<S>(stream: S) -> Vec<Result<ProviderStreamEvent>>
where
    S: futures::Stream<Item = Result<ProviderStreamEvent>>,
{
    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// Render a stream event as a compact, comparable string.
///
/// `ProviderStreamEvent` implements neither `PartialEq` nor `Serialize`, so
/// tests compare these summaries instead of writing nested `matches!` chains.
pub fn describe_event(event: &ProviderStreamEvent) -> String {
    match event {
        ProviderStreamEvent::Text(text) => format!("text:{text}"),
        ProviderStreamEvent::ToolCallStart { id, name } => format!("tool_start:{id}:{name}"),
        ProviderStreamEvent::ToolCallChunk { id, arguments } => {
            format!("tool_chunk:{id}:{arguments}")
        }
        ProviderStreamEvent::Done {
            finish_reason,
            usage,
        } => format!(
            "done:{}:{}",
            finish_reason.as_deref().unwrap_or("-"),
            usage
                .as_ref()
                .map(|usage| format!(
                    "{}/{}/{}",
                    opt(usage.prompt_tokens),
                    opt(usage.completion_tokens),
                    opt(usage.total_tokens)
                ))
                .unwrap_or_else(|| "-".to_string())
        ),
    }
}

fn opt(value: Option<i32>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

/// Concatenate the text of every [`ProviderStreamEvent::Text`] event.
pub fn stream_text(events: &[ProviderStreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ProviderStreamEvent::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

// ── A raw SSE server that splits one event across two TCP writes ───────────

/// A single-connection loopback HTTP server that writes a response body in
/// several separate flushes.
///
/// `wiremock` always hands the whole body to the client at once, which cannot
/// reproduce an SSE event arriving split across two byte chunks — exactly the
/// case where the providers' `String` buffering logic could break. This server
/// writes each part with a small gap so the client observes distinct chunks.
///
/// The response deliberately omits `Content-Length` and uses
/// `Connection: close`, so the body is terminated by EOF and no manual chunked
/// transfer-encoding framing is required.
pub struct SplitSseServer {
    base_url: String,
    handle: tokio::task::JoinHandle<()>,
}

impl SplitSseServer {
    /// Base URL to hand to a provider's `*_base_url` setting.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl Drop for SplitSseServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Serve `parts` on one connection, flushing between each part.
pub async fn split_sse(parts: Vec<String>) -> SplitSseServer {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("read listener address");

    let handle = tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };

        // Read the request head (and whatever body arrived with it). The
        // content is irrelevant — the point is to not leave it unread.
        let mut scratch = [0_u8; 8192];
        let _ = socket.read(&mut scratch).await;

        let head = "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Connection: close\r\n\r\n";
        if socket.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        if socket.flush().await.is_err() {
            return;
        }

        for part in parts {
            if socket.write_all(part.as_bytes()).await.is_err() {
                return;
            }
            if socket.flush().await.is_err() {
                return;
            }
            // Force the parts into separate reads on the client side.
            tokio::time::sleep(Duration::from_millis(40)).await;
        }

        let _ = socket.shutdown().await;
    });

    SplitSseServer {
        base_url: format!("http://{addr}"),
        handle,
    }
}

// ── A raw SSE server that reports when the client hangs up ─────────────────

/// A single-connection loopback SSE server that writes `parts`, then holds the
/// response body open until the client goes away.
///
/// Cancellation cannot be observed from the client side — the point of the
/// behavior is that *nothing* happens afterwards. So the assertion has to come
/// from the server: this one parks on a read after sending its parts, which
/// completes only when the peer closes the connection, and reports that through
/// [`DisconnectSseServer::wait_for_disconnect`].
///
/// The response uses `Connection: close` with no `Content-Length`, so the body
/// is terminated by EOF. Never sending EOF is what keeps the generation
/// "in progress" for as long as the test needs.
pub struct DisconnectSseServer {
    base_url: String,
    streaming: Option<tokio::sync::oneshot::Receiver<()>>,
    disconnected: Option<tokio::sync::oneshot::Receiver<()>>,
    handle: tokio::task::JoinHandle<()>,
}

impl DisconnectSseServer {
    /// Base URL to hand to a provider's `*_base_url` setting.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Wait until the request has arrived and every part has been written.
    ///
    /// Lets a test cancel a consumer at a point where the generation is
    /// genuinely in flight, without guessing at a delay.
    pub async fn wait_until_streaming(&mut self) {
        let receiver = self
            .streaming
            .take()
            .expect("wait_until_streaming should only be called once");

        tokio::time::timeout(Duration::from_secs(10), receiver)
            .await
            .expect("the client never opened the upstream connection")
            .expect("the server task ended before it started streaming");
    }

    /// Wait for the client to close the connection.
    ///
    /// Panics if that does not happen within a generous grace period, which is
    /// exactly the failure being guarded against: an upstream request that
    /// outlives its consumer is an orphaned generation.
    pub async fn wait_for_disconnect(&mut self) {
        let receiver = self
            .disconnected
            .take()
            .expect("wait_for_disconnect should only be called once");

        // The timeout only bounds the failure case. The success path resolves
        // as soon as the socket closes, so this adds no fixed delay.
        tokio::time::timeout(Duration::from_secs(10), receiver)
            .await
            .expect("the client never closed the upstream connection")
            .expect("the server task ended without reporting a disconnect");
    }
}

impl Drop for DisconnectSseServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Serve `parts` on one connection, then wait for the client to disconnect.
pub async fn sse_until_disconnect(parts: Vec<String>) -> DisconnectSseServer {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("read listener address");
    let (notify, disconnected) = tokio::sync::oneshot::channel();
    let (notify_streaming, streaming) = tokio::sync::oneshot::channel();

    let handle = tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };

        let mut scratch = [0_u8; 8192];
        let _ = socket.read(&mut scratch).await;

        let head = "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Connection: close\r\n\r\n";
        if socket.write_all(head.as_bytes()).await.is_err() {
            return;
        }

        for part in parts {
            if socket.write_all(part.as_bytes()).await.is_err() {
                return;
            }
            if socket.flush().await.is_err() {
                return;
            }
        }

        let _ = notify_streaming.send(());

        // Park here. The client sends nothing more on this socket, so this read
        // resolves only when it closes the connection (`Ok(0)`) or resets it.
        loop {
            match socket.read(&mut scratch).await {
                Ok(0) | Err(_) => break,
                Ok(_) => continue,
            }
        }

        let _ = notify.send(());
    });

    DisconnectSseServer {
        base_url: format!("http://{addr}"),
        streaming: Some(streaming),
        disconnected: Some(disconnected),
        handle,
    }
}

/// Serve `parts` as a *chunked* response, then hang up without the terminating
/// zero-length chunk.
///
/// That is a malformed HTTP body, so the client's transport reports a mid-stream
/// failure — the "the upstream died halfway through" case, as opposed to a
/// provider that explicitly refuses. The two must be distinguishable, hence the
/// deliberately broken framing.
pub async fn truncated_chunked_sse(parts: Vec<String>) -> SplitSseServer {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("read listener address");

    let handle = tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };

        let mut scratch = [0_u8; 8192];
        let _ = socket.read(&mut scratch).await;

        let head = "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\n\
             Transfer-Encoding: chunked\r\n\r\n";
        if socket.write_all(head.as_bytes()).await.is_err() {
            return;
        }

        for part in parts {
            let chunk = format!("{:x}\r\n{part}\r\n", part.len());
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
            if socket.flush().await.is_err() {
                return;
            }
        }

        // No `0\r\n\r\n`: close mid-body so the client sees a truncated
        // response rather than a clean end of stream.
        drop(socket);
    });

    SplitSseServer {
        base_url: format!("http://{addr}"),
        handle,
    }
}

// ── Error assertions ───────────────────────────────────────────────────────

/// Unwrap a `Result` that must be an error, reporting the success case.
pub fn expect_error<T: std::fmt::Debug>(result: Result<T>) -> Error {
    match result {
        Ok(value) => panic!("expected an error, got: {value:?}"),
        Err(error) => error,
    }
}

// ── Env-var isolation without `unsafe` ─────────────────────────────────────
//
// `Cargo.toml` sets `[lints.rust] unsafe_code = "forbid"`, and in edition 2024
// `std::env::set_var`/`remove_var` are `unsafe`. `forbid` cannot be relaxed with
// `#[allow]`, so no test in this package may mutate its own process env.
//
// Instead, an env test re-executes the test binary as a child process:
// `std::process::Command::env`/`env_remove` are safe, and the child starts from
// a known-clean slate. That also makes these tests immune to whatever the
// developer has exported, which mutating the current process would not be.

/// Marker that tells a re-executed test binary it is the child.
const ENV_CHILD_MARKER: &str = "RAI_SDK_ENV_TEST_CHILD";

/// Every environment variable the SDK reads.
///
/// The child process has all of these removed before the variables under test
/// are applied, so the developer's shell can never influence an assertion.
pub const SDK_ENV_VARS: &[&str] = &[
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_HTTP_REFERER",
    "OPENROUTER_APP_URL",
    "OPENROUTER_TITLE",
    "OPENROUTER_APP_TITLE",
    "OPENROUTER_CATEGORIES",
    "AI_TIMEOUT_SECONDS",
    "AI_MAX_RETRIES",
    "AI_RETRY_INITIAL_DELAY_MS",
    "AI_RETRY_MAX_DELAY_MS",
    "AI_RETRY_BACKOFF_MULTIPLIER",
    "AI_RETRY_JITTER",
];

/// Whether this process is the re-executed child of an env test.
pub fn in_env_child() -> bool {
    std::env::var(ENV_CHILD_MARKER).is_ok()
}

/// Re-run `test_name` in a child process whose environment contains exactly
/// `vars` out of [`SDK_ENV_VARS`].
///
/// Panics if the child fails, or if it did not run exactly one test — which
/// guards against a `test_name` typo silently turning into a no-op pass.
pub fn run_in_clean_env(test_name: &str, vars: &[(&str, &str)]) {
    let exe = std::env::current_exe().expect("locate the current test binary");

    let mut command = std::process::Command::new(exe);
    command
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(ENV_CHILD_MARKER, "1");

    for name in SDK_ENV_VARS {
        command.env_remove(name);
    }
    for (name, value) in vars {
        command.env(name, value);
    }

    let output = command.output().expect("spawn the child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "child run of `{test_name}` failed\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("1 passed"),
        "child run of `{test_name}` did not execute exactly one test \
         (is the name spelled correctly?)\n--- stdout ---\n{stdout}"
    );
}
