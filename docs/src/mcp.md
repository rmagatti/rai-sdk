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

`discover` follows pagination and rejects a server that repeats a cursor. It has no deadline of its own, so wrap it in your application's timeout. To cap how much of a catalog is read, page it yourself with `Peer::list_tools` and bind the result with `McpTools::from_catalog`.

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

`McpTools` holds only the session it was read from and that session's catalog. Authentication belongs to the transport. When a server authorizes each caller, connect a session with that caller's credentials, build an `McpTools` from it, and attach it to that caller's requests with `.tools(..)` or `.tool_definitions(..)`. Nothing is global, and catalogs are never shared between values, so one caller's session is never used for another's calls. `from_catalog` binds a catalog you have already fetched (and cached, per server and caller) to a new session without listing tools again.

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

If the future completes first, the call returns `McpError::Cancelled` straight away. Cancellation is checked before the call is dispatched, so a caller that is already cancelled never sends it. A call that was already sent is cancelled on the server with `notifications/cancelled`, delivered from a background task so a stalled session cannot delay the return. MCP cancellation is advisory. The server may already have finished, or may finish anyway, so a cancelled call to a tool with side effects may still have taken effect. Dropping a `call` future stops waiting without notifying the server.

## Limitations

- Results that need further client interaction (`input_required` rounds or task handles) return `McpError::UnsupportedResponse`.
- Image and audio results are not forwarded. Providers do not accept them as tool-result content through this SDK's message model.
- `tools()` and `definitions()` reflect the catalog at the time it was read. Read it again to pick up a server's `tools/list_changed`.
