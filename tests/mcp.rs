//! MCP tool catalogs against an in-process MCP server.
//!
//! Every test is offline: the MCP server runs in-process over an in-memory
//! duplex pipe, and provider traffic goes to a local `wiremock` mock. The
//! provider half needs a concrete provider, so the file is gated on `openai`
//! as well as `mcp`.
#![cfg(all(feature = "mcp", feature = "openai"))]

mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use common::{Script, Step, openai_builder, received_json_bodies};
use rai_sdk::mcp::rmcp::{
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, JsonObject,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool as McpTool,
    },
    service::{Peer, RequestContext, RunningService},
};
use rai_sdk::mcp::{McpError, McpTools};
use rai_sdk::{Message, Model, ToolCall};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use wiremock::MockServer;
use wiremock::matchers::{method, path};

/// A schema the SDK's own generator would never produce: no
/// `additionalProperties`, a `$schema` key, and a nested object. Any
/// normalization on the way to the provider would change it.
fn quote_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "symbol": { "type": "string", "minLength": 1 },
            "fields": {
                "type": "object",
                "properties": { "bid": { "type": "boolean" } }
            }
        },
        "required": ["symbol"]
    })
}

fn object(value: Value) -> Arc<JsonObject> {
    match value {
        Value::Object(map) => Arc::new(map),
        other => panic!("expected a JSON object, got {other}"),
    }
}

fn catalog() -> Vec<McpTool> {
    vec![
        McpTool::new(
            "getQuote",
            "Latest quote for a symbol",
            object(quote_schema()),
        ),
        McpTool::new(
            "echoText",
            "Echo text back",
            object(json!({"type": "object"})),
        ),
        McpTool::new(
            "failDomain",
            "Reports a domain failure",
            object(json!({"type": "object"})),
        ),
        McpTool::new(
            "failProtocol",
            "Fails at the protocol level",
            object(json!({"type": "object"})),
        ),
        McpTool::new(
            "chart",
            "Returns an image",
            object(json!({"type": "object"})),
        ),
        McpTool::new(
            "whoami",
            "Reports the session's caller",
            object(json!({"type": "object"})),
        ),
        McpTool::new(
            "waitForCancel",
            "Blocks until cancelled",
            object(json!({"type": "object"})),
        ),
    ]
}

/// A test MCP server. `caller` stands in for the identity a real server would
/// derive from the session's credentials.
#[derive(Clone)]
struct TestServer {
    caller: &'static str,
    calls: Arc<AtomicUsize>,
    cancelled: mpsc::UnboundedSender<()>,
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
        let mut result = ListToolsResult::with_all_items(catalog());
        if self.caller == "cursor-loop" {
            // A misbehaving server that always claims there is another page.
            result.next_cursor = Some("again".to_string());
        }
        Ok(result)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let arguments = Value::Object(request.arguments.clone().unwrap_or_default());
        let result = match request.name.as_ref() {
            "getQuote" => {
                let mut result = CallToolResult::success(vec![ContentBlock::text(
                    "text fallback that should not be used",
                )]);
                result.structured_content = Some(json!({
                    "symbol": arguments["symbol"],
                    "bid": 101.25,
                }));
                result
            }
            "echoText" => CallToolResult::success(vec![
                ContentBlock::text(arguments["text"].as_str().unwrap_or_default()),
                ContentBlock::text("second block"),
            ]),
            "failDomain" => CallToolResult::error(vec![ContentBlock::text("symbol not found")]),
            "failProtocol" => {
                return Err(ErrorData::invalid_params(
                    "bad request at the protocol level",
                    None,
                ));
            }
            "chart" => CallToolResult::success(vec![ContentBlock::image("aGVsbG8=", "image/png")]),
            "whoami" => CallToolResult::success(vec![ContentBlock::text(self.caller)]),
            "waitForCancel" => {
                context.ct.cancelled().await;
                let _ = self.cancelled.send(());
                CallToolResult::success(vec![ContentBlock::text("late")])
            }
            other => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool {other}"),
                    None,
                ));
            }
        };
        Ok(result.into())
    }
}

struct Session {
    client: RunningService<RoleClient, ()>,
    calls: Arc<AtomicUsize>,
    cancelled: mpsc::UnboundedReceiver<()>,
}

impl Session {
    fn peer(&self) -> Peer<RoleClient> {
        self.client.peer().clone()
    }
}

async fn session(caller: &'static str) -> Session {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let calls = Arc::new(AtomicUsize::new(0));
    let (cancelled_tx, cancelled) = mpsc::unbounded_channel();
    let server = TestServer {
        caller,
        calls: calls.clone(),
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
        calls,
        cancelled,
    }
}

fn tool_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: format!("call_{name}"),
        name: name.to_string(),
        arguments,
    }
}

#[tokio::test]
async fn discover_imports_names_descriptions_and_schemas_unchanged() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    assert_eq!(tools.len(), catalog().len());
    let definitions = tools.definitions();
    let names: Vec<_> = definitions.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "getQuote",
            "echoText",
            "failDomain",
            "failProtocol",
            "chart",
            "whoami",
            "waitForCancel"
        ]
    );
    assert_eq!(
        definitions[0].description.as_deref(),
        Some("Latest quote for a symbol")
    );
    assert_eq!(definitions[0].input_schema, quote_schema());
    assert!(tools.contains("getQuote"));
    assert!(!tools.contains("get_quote"));
}

#[tokio::test]
async fn structured_content_is_preferred_over_text() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let message = tools
        .call(&tool_call("getQuote", json!({ "symbol": "AAPL" })))
        .await
        .unwrap();

    assert!(!message.tool_error);
    assert_eq!(message.tool_call_id.as_deref(), Some("call_getQuote"));
    let content: Value = serde_json::from_str(&message.content).unwrap();
    assert_eq!(content, json!({ "symbol": "AAPL", "bid": 101.25 }));
}

#[tokio::test]
async fn text_content_is_sent_verbatim() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let message = tools
        .call(&tool_call("echoText", json!({ "text": "hello \"world\"" })))
        .await
        .unwrap();

    assert!(!message.tool_error);
    assert_eq!(message.content, "hello \"world\"\nsecond block");
}

#[tokio::test]
async fn is_error_results_become_tool_errors() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let message = tools
        .call(&tool_call("failDomain", json!({})))
        .await
        .unwrap();

    assert!(message.tool_error);
    assert_eq!(message.content, "symbol not found");
}

#[tokio::test]
async fn protocol_errors_are_mcp_errors_not_tool_results() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let error = tools
        .call(&tool_call("failProtocol", json!({})))
        .await
        .unwrap_err();

    assert!(matches!(error, McpError::Service(_)), "got {error:?}");
}

#[tokio::test]
async fn unsupported_content_is_reported_not_dropped() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let error = tools
        .call(&tool_call("chart", json!({})))
        .await
        .unwrap_err();

    match error {
        McpError::UnsupportedContent { tool, kind } => {
            assert_eq!(tool, "chart");
            assert_eq!(kind, "image");
        }
        other => panic!("expected UnsupportedContent, got {other:?}"),
    }
}

#[tokio::test]
async fn unknown_tools_are_not_sent_to_the_server() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let error = tools
        .call(&tool_call("dropTables", json!({})))
        .await
        .unwrap_err();

    assert!(matches!(error, McpError::ToolNotFound { ref name } if name == "dropTables"));
    assert_eq!(session.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn non_object_arguments_are_a_correctable_tool_error() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let message = tools
        .call(&tool_call("echoText", json!("hi")))
        .await
        .unwrap();

    assert!(message.tool_error);
    let content: Value = serde_json::from_str(&message.content).unwrap();
    assert_eq!(content["error"]["type"], "tool_argument_validation");
    assert_eq!(content["error"]["retryable"], true);
    assert_eq!(session.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rename_changes_advertised_names_but_calls_the_server_name() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer())
        .await
        .unwrap()
        .filter(|tool| tool.name != "waitForCancel")
        .rename(|tool| format!("quotes__{}", tool.name))
        .unwrap();

    assert!(tools.contains("quotes__echoText"));
    assert!(!tools.contains("echoText"));
    assert!(!tools.contains("quotes__waitForCancel"));
    assert_eq!(
        tools
            .remote_tool("quotes__getQuote")
            .map(|tool| tool.name.as_ref()),
        Some("getQuote")
    );

    let message = tools
        .call(&tool_call("quotes__echoText", json!({ "text": "routed" })))
        .await
        .unwrap();
    assert_eq!(message.content, "routed\nsecond block");
}

#[tokio::test]
async fn colliding_names_are_rejected() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let error = tools.rename(|_| "same".to_string()).unwrap_err();
    assert!(
        matches!(error, McpError::InvalidCatalog(_)),
        "got {error:?}"
    );

    let mut duplicated = catalog();
    duplicated.push(duplicated[0].clone());
    let error = McpTools::from_catalog(session.peer(), duplicated).unwrap_err();
    assert!(
        matches!(error, McpError::InvalidCatalog(_)),
        "got {error:?}"
    );
}

#[tokio::test]
async fn cancellation_notifies_the_server() {
    let mut session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();
    let (cancel, cancelled) = oneshot::channel::<()>();

    let call = tool_call("waitForCancel", json!({}));
    let pending = tools.call_until(&call, async move {
        let _ = cancelled.await;
    });
    let (result, ()) = tokio::join!(pending, async move {
        tokio::task::yield_now().await;
        let _ = cancel.send(());
    });

    match result.unwrap_err() {
        McpError::Cancelled { tool } => assert_eq!(tool, "waitForCancel"),
        other => panic!("expected Cancelled, got {other:?}"),
    }
    tokio::time::timeout(std::time::Duration::from_secs(5), session.cancelled.recv())
        .await
        .expect("server should observe the cancellation")
        .expect("server should still be running");
}

#[tokio::test]
async fn already_cancelled_callers_never_dispatch() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let error = tools
        .call_until(
            &tool_call("echoText", json!({ "text": "never" })),
            std::future::ready(()),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, McpError::Cancelled { ref tool } if tool == "echoText"));

    // A later call on the same session is the first the server ever handles.
    let message = tools.call(&tool_call("whoami", json!({}))).await.unwrap();
    assert_eq!(message.content, "alice");
    assert_eq!(session.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn repeated_pagination_cursors_are_rejected() {
    let session = session("cursor-loop").await;

    let error = McpTools::discover(session.peer()).await.unwrap_err();

    assert!(
        matches!(error, McpError::InvalidCatalog(_)),
        "got {error:?}"
    );
}

#[tokio::test]
async fn catalogs_are_isolated_per_session() {
    let alice = session("alice").await;
    let bob = session("bob").await;
    let alice_tools = McpTools::discover(alice.peer()).await.unwrap();
    let bob_tools = McpTools::discover(bob.peer()).await.unwrap();

    let whoami = tool_call("whoami", json!({}));
    let calls = (0..8).map(|i| {
        let tools = if i % 2 == 0 { &alice_tools } else { &bob_tools };
        let whoami = &whoami;
        async move { (i, tools.call(whoami).await.unwrap()) }
    });
    for (i, message) in futures::future::join_all(calls).await {
        let expected = if i % 2 == 0 { "alice" } else { "bob" };
        assert_eq!(message.content, expected);
    }
    assert_eq!(alice.calls.load(Ordering::SeqCst), 4);
    assert_eq!(bob.calls.load(Ordering::SeqCst), 4);
}

fn chat_completion_tool_call(name: &str, arguments: Value) -> Value {
    json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": name, "arguments": arguments.to_string() }
                }]
            }
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

fn chat_completion_text(text: &str) -> Value {
    json!({
        "id": "chatcmpl-2",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": { "role": "assistant", "content": text }
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

fn advertised_parameters(body: &Value, name: &str) -> Value {
    body["tools"]
        .as_array()
        .expect("request should advertise tools")
        .iter()
        .find(|tool| tool["function"]["name"] == name)
        .unwrap_or_else(|| panic!("tool {name} should be advertised"))["function"]["parameters"]
        .clone()
}

#[tokio::test]
async fn definitions_reach_the_provider_unchanged() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let server = MockServer::start().await;
    wiremock::Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![Step::ok(chat_completion_tool_call(
            "getQuote",
            json!({ "symbol": "MSFT" }),
        ))]))
        .mount(&server)
        .await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .unwrap();

    let response = client
        .request()
        .tool_definitions(tools.definitions())
        .prompt("Quote MSFT")
        .generate_once()
        .await
        .unwrap();

    let bodies = received_json_bodies(&server).await;
    assert_eq!(
        advertised_parameters(&bodies[0], "getQuote"),
        quote_schema()
    );

    // The application owns the loop: execute what the model asked for.
    let calls: Vec<ToolCall> = response
        .messages
        .iter()
        .flat_map(|message| message.tool_calls.clone())
        .collect();
    assert_eq!(calls.len(), 1);
    let result: Message = tools.call(&calls[0]).await.unwrap();
    let content: Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(content["symbol"], "MSFT");
}

#[tokio::test]
async fn executable_tools_run_inside_generate() {
    let session = session("alice").await;
    let tools = McpTools::discover(session.peer()).await.unwrap();

    let server = MockServer::start().await;
    wiremock::Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![
            Step::ok(chat_completion_tool_call("failDomain", json!({}))),
            Step::ok(chat_completion_text("That symbol does not exist.")),
        ]))
        .mount(&server)
        .await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .unwrap();

    let response = client
        .request()
        .tools(tools.tools())
        .prompt("Quote XXXX")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text(), "That symbol does not exist.");
    assert_eq!(session.calls.load(Ordering::SeqCst), 1);

    let bodies = received_json_bodies(&server).await;
    assert_eq!(
        advertised_parameters(&bodies[0], "getQuote"),
        quote_schema()
    );
    let tool_message = bodies[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("second request should carry the tool result")
        .clone();
    assert_eq!(tool_message["tool_call_id"], "call_1");
    assert_eq!(tool_message["content"], "symbol not found");
}
