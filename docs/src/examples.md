# Examples

The repository ships runnable examples. Clone it and run them with `cargo run --example`.

```sh
git clone https://github.com/rmagatti/rai-sdk
cd rai-sdk
export OPENAI_API_KEY="sk-..."
```

These make real API calls and will consume credit.

## basic_chat

```sh
cargo run --example basic_chat
```

The smallest complete request: build a client from the environment, send a prompt, print the text. Start here to confirm your credentials work.

Source: [`examples/basic_chat.rs`](https://github.com/rmagatti/rai-sdk/blob/main/examples/basic_chat.rs)

## structured_output

```sh
cargo run --example structured_output
```

Derives `JsonSchema` on a struct and uses `generate_structured` to get a validated, typed value back instead of text. See [Structured output](./structured-output.md).

Source: [`examples/structured_output.rs`](https://github.com/rmagatti/rai-sdk/blob/main/examples/structured_output.rs)

## tool_calling

```sh
cargo run --example tool_calling
```

Registers a typed tool and lets `generate()` run the loop: the model requests the call, the SDK executes the handler, feeds the result back, and the model produces a final answer. See [Tool calling](./tool-calling.md).

Source: [`examples/tool_calling.rs`](https://github.com/rmagatti/rai-sdk/blob/main/examples/tool_calling.rs)

## sse_proxy

```sh
cargo run --example sse_proxy
```

The full server-side proxy loop in one process: an axum handler streams a generation with `stream_wire_events()`, re-emits each event as an SSE `data:` payload, and a client in the same binary parses them back and reassembles one `Response` with `StreamAccumulator`. Reach for this when your server holds the provider credentials and streams results on to a desktop or browser client. See [Streaming](./streaming.md#proxying-a-stream-to-your-own-clients).

Source: [`examples/sse_proxy.rs`](https://github.com/rmagatti/rai-sdk/blob/main/examples/sse_proxy.rs)

## Using a different provider

All four examples use `Model::gpt4o_mini()`. Change the constructor and set that provider's key to try another:

```rust,no_run
# use rai_sdk::Model;
let _ = Model::claude_sonnet_46();  // needs ANTHROPIC_API_KEY
let _ = Model::openrouter_auto();   // needs OPENROUTER_API_KEY
```

## Seeing what the SDK is doing

The SDK emits `tracing` spans and events. Install a subscriber and set `RUST_LOG` to inspect requests, retries, and tool execution:

```sh
RUST_LOG=rai_sdk=debug cargo run --example tool_calling
```

`tracing-subscriber` is already a dev-dependency, so this works in the examples without adding anything.

## Testing without API calls

The SDK's own test suite is fully offline: base URLs are pointed at a mock HTTP server, so no test needs credentials. You can use the same approach in your project by setting `OPENAI_BASE_URL` (or the equivalent) at a local mock. See [Configuration](./configuration.md#endpoints) and [Contributing](./contributing.md).
