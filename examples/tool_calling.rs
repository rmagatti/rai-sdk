//! Automatic tool execution.
//!
//! Registers a typed tool and lets `generate` run the tool loop: the model
//! requests the call, the SDK executes the handler, feeds the result back, and
//! the model produces a final answer.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! cargo run --example tool_calling
//! ```

use rai_sdk::{ClientBuilder, Model, Result, Tool, ToolContext, schemars::JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// 1. Define typed arguments for your tool.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct WeatherArgs {
    /// The city to get the weather for (e.g., "Paris", "Tokyo").
    city: String,
    /// Whether to return Celsius or Fahrenheit.
    #[serde(default = "default_unit")]
    unit: String,
}

fn default_unit() -> String {
    "celsius".to_string()
}

/// 2. Define the asynchronous handler logic.
async fn get_weather(args: WeatherArgs, _ctx: ToolContext) -> Result<serde_json::Value> {
    println!(
        "  [Tool Execution] Fetching weather for {} in {}...",
        args.city, args.unit
    );

    // In a real app, you would make an API call here.
    // For this example, we return mock data.
    let temp = if args.city.to_lowercase() == "paris" {
        22
    } else {
        18
    };

    Ok(json!({
        "city": args.city,
        "temperature": temp,
        "unit": args.unit,
        "condition": "Sunny",
    }))
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 3. Build the fallible tool using the typed handler.
    let weather_tool = Tool::new("get_current_weather")
        .description("Get the current weather in a given city.")
        .handler(get_weather)?;

    // 4. Initialize the client with the tool registered.
    // By default, `rai-sdk` will automatically execute tool calls and loop back to the model.
    let client = ClientBuilder::new()
        .model(Model::gpt4o_mini())
        .tools(vec![weather_tool])
        .build()?;

    // 5. Ask a question that requires the tool.
    let request = client
        .request()
        .prompt("What is the weather like in Paris right now?");

    println!("Sending request to OpenAI (expecting tool use)...");

    let response = request.generate().await?;

    println!("\nFinal Response:\n{}", response.text());

    Ok(())
}
