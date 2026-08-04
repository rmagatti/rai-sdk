# Structured output

Structured output turns a model response into a typed Rust value. The SDK generates a JSON Schema from your type, asks the provider to conform to it, validates the response against the schema, and only then deserializes.

That last part matters: valid JSON is not the same as JSON matching your type. Validation happens before deserialization so failures come with schema-level diagnostics rather than an opaque serde error.

## Basic usage

Derive `Deserialize` and `JsonSchema`, then call `generate_structured::<T>()`:

```rust,no_run
use rai_sdk::{ClientBuilder, GenerationConfig, JsonSchema, Model};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    steps: Vec<String>,
    prep_time_minutes: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let structured = client
        .request()
        .config(GenerationConfig::new().with_temperature(0.2))
        .prompt("Return a simple chocolate cake recipe as JSON.")
        .generate_structured::<Recipe>()
        .await?;

    println!("{:#?}", structured.output);
    println!("raw text: {}", structured.response.text());
    Ok(())
}
```

The result is a [`StructuredOutput<T>`](https://docs.rs/rai-sdk/latest/rai_sdk/message/struct.StructuredOutput.html) with two fields: `output` (your typed value) and `response` (the underlying response, including usage).

Use `JsonSchema` from `rai_sdk` rather than depending on `schemars` yourself — the SDK re-exports both the trait and the derive so versions cannot mismatch.

## With or without tools

| Method | Tools | Provider calls |
| --- | --- | --- |
| `generate_structured::<T>()` | May call registered tools first | One or more |
| `generate_structured_once::<T>()` | Ignores configured tools | Exactly one |

Use the `_once` variant when you want a pure transformation and no tool side effects, even though the client has tools registered.

## Validation failures

If the response does not satisfy the schema, you get an error instead of a partially-populated value. Distinguish that from transport problems by inspecting the error:

```rust,no_run
# use rai_sdk::{ClientBuilder, JsonSchema, Model};
# use serde::Deserialize;
# #[derive(Debug, Deserialize, JsonSchema)]
# struct Recipe { name: String }
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = ClientBuilder::new().from_env().model(Model::gpt4o_mini()).build()?;
match client
    .request()
    .prompt("Return a recipe as JSON.")
    .generate_structured::<Recipe>()
    .await
{
    Ok(structured) => println!("{:?}", structured.output),
    Err(error) if error.is_retryable() => eprintln!("transient: {error}"),
    Err(error) => eprintln!("did not match the schema: {error}"),
}
# Ok(())
# }
```

Lowering the temperature and describing the desired shape in the prompt both reduce validation failures.

## Schema generation details

The generated schema is deliberately conservative, because strict providers reject anything unexpected:

- **Subschemas are inlined.** Generation runs with `inline_subschemas = true` and no top-level `"$schema"`, so nested non-recursive types are emitted inline instead of producing `"$defs"`/`"$ref"`.
- **`additionalProperties` defaults to `false`** on every object schema, without overriding a value you set explicitly.
- **`"$schema"` keys are stripped** wherever they appear.

The reason is Gemini: reached through OpenRouter, its `response_schema` rejects schemas containing `"$schema"`, `"$defs"`, or `"$ref"` with a 400 `INVALID_ARGUMENT`.

### Recursive types are not supported

Inlining cannot represent a type that transitively contains itself, so `schemars` falls back to `"$defs"`/`"$ref"` for recursive types. The SDK does not resolve those references. A recursive structured-output type will therefore be rejected by strict providers. Flatten the shape — for example, by replacing nested self-references with an ID or a bounded depth — if you need Gemini compatibility.

## JSON mode versus schema mode

For cases where you want valid JSON but do not care about its shape:

```rust,no_run
use rai_sdk::GenerationConfig;

let config = GenerationConfig::new().with_json_mode(true);
# let _ = config;
```

A schema always wins over the `json_mode` flag. You can also supply a hand-written schema with `with_json_schema(..)`, or generate one from a type without performing a request using `with_json_schema_for::<T>()`.
