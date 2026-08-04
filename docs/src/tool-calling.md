# Tool calling

Tools let the model call your code. You register typed handlers; when the model asks for one, the SDK validates the arguments, runs the handler, feeds the result back, and asks the model to continue — until it produces a final answer.

## Defining a tool

A tool's argument type supplies its JSON Schema, so the schema advertised to the provider and the type your handler receives cannot drift apart.

```rust,no_run
use rai_sdk::{ClientBuilder, JsonSchema, Model, Result, Tool, ToolContext};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct WeatherArgs {
    city: String,
    #[serde(default = "default_unit")]
    unit: String,
}

fn default_unit() -> String {
    "celsius".to_string()
}

async fn get_weather(args: WeatherArgs, _ctx: ToolContext) -> Result<serde_json::Value> {
    Ok(json!({
        "city": args.city,
        "temperature": 22,
        "unit": args.unit,
        "condition": "Sunny"
    }))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let weather_tool = Tool::new("get_current_weather")
        .description("Get the current weather in a city.")
        .handler(get_weather)?;

    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .tool(weather_tool)
        .build()?;

    let response = client
        .request()
        .prompt("What is the weather in Paris right now?")
        .generate()
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

`.handler()` is fallible on purpose: it generates and validates the schema at build time, so an unrepresentable argument type fails when you construct the tool rather than mid-conversation.

Write a real `description`. It is the only thing telling the model when the tool applies.

## Handler context

Handlers receive a [`ToolContext`](https://docs.rs/rai-sdk/latest/rai_sdk/tool/struct.ToolContext.html) alongside the typed arguments:

| Field | Use |
| --- | --- |
| `provider` | Which provider requested the call |
| `model` | Wire model ID that requested it |
| `round` | Zero-based tool-loop round, useful for bounding repeated work |
| `tool_name` | Name of the tool being invoked |
| `tool_call_id` | Provider-assigned ID for this call |

## The tool loop

`generate()` repeats: send the request, execute any requested tools, append results, send again. It stops when the model returns a final answer without tool calls.

Bound the loop with `max_tool_rounds` (default 8):

```rust,no_run
use rai_sdk::GenerationConfig;

let config = GenerationConfig::new().with_max_tool_rounds(3);
# let _ = config;
```

Exceeding the limit fails with `Error::ToolLoopLimitExceeded`, which prevents a model that keeps calling tools from looping indefinitely.

## Getting tool calls without executing them

`generate_once()` performs a single provider call and returns tool calls without running your handlers. Use it when you want to inspect, gate, or approve calls before they take effect:

```rust,no_run
# use rai_sdk::{ClientBuilder, Model};
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
let response = client
    .request()
    .prompt("What is the weather in Paris?")
    .generate_once()
    .await?;

for message in &response.messages {
    if message.has_tool_calls() {
        for call in &message.tool_calls {
            println!("model wants {} with {}", call.name, call.arguments);
        }
    }
}
# Ok(())
# }
```

Tool calls live on the individual [`Message`](https://docs.rs/rai-sdk/latest/rai_sdk/message/struct.Message.html) values in `response.messages`, alongside `has_tool_calls()` for a quick check.

## Argument validation

Arguments are validated against the tool's schema before the handler runs. A validation failure is **not** a hard error: the SDK returns a structured tool-error message to the model describing each violation, so it can correct itself and call the tool again.

That means your handler only ever sees arguments that already satisfy the schema, and a confused model produces a retry rather than a failed request. Each violation is reported as a [`ToolArgumentIssue`](https://docs.rs/rai-sdk/latest/rai_sdk/error/struct.ToolArgumentIssue.html) with the offending path, the schema keyword that rejected it, and a message.

Tighten schemas to get better self-correction. `schemars` attributes carry through:

```rust
use rai_sdk::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
struct SearchArgs {
    #[schemars(description = "Name or email substring to search for", length(min = 1))]
    query: String,
    #[schemars(description = "Maximum results to return", range(min = 1, max = 10))]
    limit: Option<usize>,
}
```

Errors returned by your handler are also surfaced to the model as tool-error content rather than aborting generation. Return an error when a call genuinely cannot be satisfied and you want the model to react to that fact.

## Per-request tools

Tools can be registered on the client (shared by every request) or per request:

| Method | Effect |
| --- | --- |
| `.tool(..)` / `.tools(..)` on the request | Replaces the client's tools for this request |
| `.additional_tool(..)` / `.additional_tools(..)` | Adds to the client's tools |
| `.no_tools()` | Disables tools for this request |

## Limitations

- Raw streaming rejects requests with registered tools. See [Streaming](./streaming.md).
- A provider that does not support tool calling produces `Error::ToolProviderUnsupported`.
- Tool names must be unique per client; registering a duplicate is an error.
