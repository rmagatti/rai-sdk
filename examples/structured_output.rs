//! Validated, typed output driven by a generated JSON Schema.
//!
//! Shows how deriving `JsonSchema` lets `generate_structured` validate the
//! model response and deserialize it into a Rust struct.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! cargo run --example structured_output
//! ```

use rai_sdk::{ClientBuilder, GenerationConfig, Model, schemars::JsonSchema};
use serde::{Deserialize, Serialize};

/// Define the struct you want the model to return.
/// It must derive `Deserialize`, `Serialize`, and `JsonSchema`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    steps: Vec<String>,
    prep_time_minutes: u32,
    difficulty: String,
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the client builder with a specific model.
    let client = ClientBuilder::new().model(Model::gpt4o_mini()).build()?;

    // 2. We want structured output. We pass the struct type into `GenerationConfig`.
    let config = GenerationConfig::default()
        .with_temperature(0.2)
        .with_json_schema_for::<Recipe>()?;

    // 3. Create a request with the prompt and the custom configuration.
    let request = client
        .request()
        .prompt("Give me a recipe for a simple and tasty chocolate cake.")
        .config(config);

    println!("Sending request to OpenAI (expecting structured JSON)...");

    // 4. Generate the response. The output will be pure JSON matching our schema.
    let response = request.generate().await?;
    let output_text = response.text();

    println!("\nRaw JSON Response:\n{}", output_text);

    // 5. Parse the returned text directly into our Rust struct.
    let recipe: Recipe = serde_json::from_str(&output_text)?;

    println!("\nParsed Recipe Object:");
    println!("Name: {}", recipe.name);
    println!("Prep Time: {} min", recipe.prep_time_minutes);
    println!("Difficulty: {}", recipe.difficulty);
    println!("Ingredients: {:?}", recipe.ingredients);

    Ok(())
}
