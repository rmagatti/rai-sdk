# MCP tools

The `mcp` feature exposes the tools of a [Model Context Protocol](https://modelcontextprotocol.io) server to a model. You don't write a handler or a schema per tool. You read the server's catalog once and register all of it.

```toml
[dependencies]
rai-sdk = { version = "0.3", features = ["mcp"] }
```

The feature builds on [`rmcp`](https://docs.rs/rmcp), the official Rust MCP SDK, re-exported as `rai_sdk::mcp::rmcp`. Only rmcp's client role is enabled. The feature adds no transport, TLS or auth stack, so you connect with whatever transport and credentials your application already uses. It requires **Rust 1.88**, rmcp's minimum. The rest of the crate keeps its own, lower minimum.

## Registering a catalog

`McpTools::discover` reads a server's full tool catalog from a connected session. `tools()` turns it into executable tools for `generate()`:

```rust,no_run
use rai_sdk::mcp::McpTools;
use rai_sdk::mcp::rmcp::{RoleClient, service::Peer};
use rai_sdk::{ClientBuilder, Model};

# async fn run(peer: Peer<RoleClient>) -> Result<(), Box<dyn std::error::Error>> {
let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
let catalog = McpTools::discover(peer).await?;

let response = client
    .request()
    .tools(catalog.tools())
    .prompt("Which symbols moved most today?")
    .generate()
    .await?;
println!("{}", response.text());
# Ok(())
# }
```

`discover` follows pagination and rejects a server that repeats a cursor. It has no deadline of its own, so wrap it in your application's timeout. To cap how much of a catalog is read, page it yourself with `rai_sdk::mcp::list_tools` and bind the result with `McpTools::from_catalog`.

Each tool is advertised with the server's own name, description and `inputSchema`. The schema is passed through unchanged. Schemas generated from Rust types are normalized for strict providers, but MCP schemas are not, and arguments are not validated locally because the server validates its own input.

## Running your own tool loop

Streaming methods do not execute tools. An application that streams, or that wants to authorize, log or route each call itself, advertises the catalog as definitions and executes calls with `call`:

```rust,no_run
use rai_sdk::mcp::McpTools;
use rai_sdk::{Client, Message, ToolCall};
use rai_sdk::client::ModelReady;

# async fn turn(client: &Client<ModelReady>, catalog: &McpTools) -> Result<(), Box<dyn std::error::Error>> {
let response = client
    .request()
    .tool_definitions(catalog.definitions())
    .prompt("Quote MSFT")
    .generate_once()
    .await?;

let calls: Vec<ToolCall> = response
    .messages
    .iter()
    .flat_map(|message| message.tool_calls.clone())
    .collect();

let mut results: Vec<Message> = Vec::new();
for call in &calls {
    if catalog.contains(&call.name) {
        results.push(catalog.call(call).await?);
    }
}
// Append the assistant message and `results` to the history and continue.
# Ok(())
# }
```

The same definitions work with `stream_wire_events`, which advertises tools and forwards tool calls without executing them.

## Results and errors

`call` returns the tool-result `Message` to send back to the model:

| Server response | Result |
| --- | --- |
| Result with `structuredContent` | `Message::tool` holding the structured content as JSON |
| Result with text content only | `Message::tool` holding the text verbatim (blocks joined by newlines) |
| Result with `isError: true` | `Message::tool_error` with the same content rules, so the model can react |
| Image, audio or binary resource content | `Err(McpError::UnsupportedContent)`; nothing is silently dropped |
| Arguments that are not a JSON object | `Message::tool_error` the model can correct; the server is not called |
| Transport, protocol or JSON-RPC error | `Err(McpError::Service)` |
| Tool not in the catalog | `Err(McpError::ToolNotFound)` |

A domain failure reported by the tool is a message for the model. A failure to reach or speak to the server is an error for the application, which decides whether to retry, tell the model, or end the turn. Inside `generate()`, both are reported to the model as tool errors, like any other tool handler failure.

## Structured results and output validation

`McpTools::call_result` and `call_result_until` return the original `CallToolResult`, including content blocks, `structuredContent`, `_meta` and `isError`. Use them when a UI needs typed JSON or non-text content alongside the model's message. `call` and `call_until` use the same execution path, then convert the result to a `Message`.

```rust,no_run
# use rai_sdk::{ToolCall, mcp::{McpTools, RequestOptions}};
# async fn run(catalog: &McpTools, call: &ToolCall) -> Result<(), Box<dyn std::error::Error>> {
let result = catalog.call_result(
    call,
    RequestOptions::with_timeout(std::time::Duration::from_secs(10)),
).await?;
// `result.structured_content` remains a JSON value; no message re-parsing is needed.
# Ok(())
# }
```

All catalog call paths validate successful `structuredContent` against a tool's declared `outputSchema`. A missing or non-matching value returns `McpError::OutputSchemaViolation`; `isError` results skip validation. This also applies to the existing `call`, `call_until` and `tools()` paths. Tools with no output schema retain their previous behavior.

Compilation is lazy and cached per tool, including failed compilations. An invalid schema prevents that tool from being dispatched and leaves other tools usable. Compilation continues independently when its first caller cancels or reaches its deadline, so a later caller can reuse the work. Clones and filtered catalogs share that cache; renaming creates new schema keys.

`SchemaProcessor` bounds compilation and validation on blocking workers (four by default). Share a cloned processor across catalogs with `with_schema_processor` to apply one limit across sessions. A deadline ends the caller's wait; blocking work already running keeps its permit until completion. External HTTP and file schema references are always refused. Diagnostics include bounded schema information, never result values.

## Applications with their own catalog

The free `list_tools` and `call_tool` functions accept a connected `Peer` and typed MCP parameters. They retain no catalog, impose no authorization policy and return typed MCP results unchanged. This lets an application enforce its own page, byte and tool limits without keeping a second catalog solely for execution.

Use `SchemaProcessor::compile` once for the retained output schemas and `validate` on returned results. This explicit path uses the same validator as `McpTools`. Batch compilation reports any invalid declaration in the supplied batch; applications can compile individual tools to isolate failures. Transport body limits, catalog size limits, credentials, authorization and cache lifetimes remain application-owned.

`RequestOptions::with_timeout` sets one hard budget covering queue wait and response. For catalog calls it also covers the wait for compilation and result validation. Progress notifications do not extend it. A zero timeout does not dispatch; an unrepresentably large timeout is treated as unbounded. `call_tool_until` additionally accepts a cancellation future. Input-required rounds and task handles remain unsupported.

## Names

`filter` keeps a subset of the catalog and `rename` changes the names advertised to the model. Calls still go to the server under each tool's own name:

```rust,no_run
# use rai_sdk::mcp::{McpError, McpTools};
# fn names(catalog: McpTools) -> Result<McpTools, McpError> {
let catalog = catalog
    .filter(|tool| !tool.annotations.as_ref().is_some_and(|a| a.destructive_hint == Some(true)))
    .rename(|tool| format!("quotes__{}", tool.name))?;
# Ok(catalog)
# }
```

Use `rename` to combine several servers without collisions or to keep names an existing prompt relies on. Empty or repeated names are rejected with `McpError::InvalidCatalog`. As with any tool, a name the provider does not accept is rejected by the provider. `remote_tool` returns the server's full declaration of a tool, including its title, annotations and output schema.

## Credentials and isolation

`McpTools` binds a tool catalog to a client session. Authentication belongs to the transport. When a server authorizes each caller, connect a session with that caller's credentials, build an `McpTools` from it, and attach it to that caller's requests with `.tools(..)` or `.tool_definitions(..)`. Clones share both the session and catalog. The application must keep the original value, its clones, and the generated tools scoped to the same authenticated caller or credential context. The adapter has no global cache and does not enforce caller identity.

Call `discover` each turn to refresh the catalog, or discover once and reuse the value or its clones across turns for the same credential context. `from_catalog` binds an already-fetched catalog to a session without listing tools again. Applications that cache catalogs must key them by the server and applicable caller or credential context, and bind them to a session authenticated for that context. Cache lifetimes, eviction, reconnects and refreshes after tool-list notifications are application-owned.

## Cancellation

`call_until` races a call against any future, such as `CancellationToken::cancelled()`:

```rust,no_run
# use rai_sdk::mcp::{McpError, McpTools};
# use rai_sdk::{Message, ToolCall};
# async fn run(
#     catalog: &McpTools,
#     call: &ToolCall,
#     stop: tokio::sync::oneshot::Receiver<()>,
# ) -> Result<Message, McpError> {
catalog
    .call_until(call, async move {
        let _ = stop.await;
    })
    .await
# }
```

If the future completes first, the call returns `McpError::Cancelled` straight away. Cancellation is checked before the call is dispatched, so a caller that is already cancelled never sends it. A call that was already sent is cancelled on the server with `notifications/cancelled`, delivered from a background task so a stalled session cannot delay the return. MCP cancellation is advisory. The server may already have finished, or may finish anyway, so a cancelled call to a tool with side effects may still have taken effect. Dropping a dispatched call future also sends cancellation asynchronously. Completed responses and protocol errors do not send cancellation.

## Limitations

- Results that need further client interaction (`input_required` rounds or task handles) return `McpError::UnsupportedResponse`.
- The model-facing methods cannot forward image or audio tool results. The result-preserving methods return those blocks unchanged.
- `tools()` and `definitions()` reflect the catalog at the time it was read. Read it again to pick up a server's `tools/list_changed`.
