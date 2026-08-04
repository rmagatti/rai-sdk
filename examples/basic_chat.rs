//! Minimal chat completion against OpenAI.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! ```sh
//! cargo run --example basic_chat
//! ```

use rai_sdk::{ClientBuilder, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: ensure the OPENAI_API_KEY environment variable is set.

    // 1. Initialize the client builder with a specific model.
    let client = ClientBuilder::new().model(Model::gpt4o_mini()).build()?;

    // 2. Create a request with a simple prompt.
    let request = client
        .request()
        .prompt("Explain the concept of 'Ownership' in Rust in 2 sentences.");

    println!("Sending request to OpenAI...");

    // 3. Generate the response.
    let response = request.generate().await?;

    println!("\nResponse:\n{}", response.text());

    Ok(())
}
