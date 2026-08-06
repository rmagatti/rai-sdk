//! Chat with a local model over an OpenAI-compatible endpoint.
//!
//! Requires no API key — just a server speaking the OpenAI Chat Completions
//! format. Defaults to Ollama on `http://localhost:11434/v1`; override the
//! endpoint and model with `RAI_EXAMPLE_BASE_URL` and `RAI_EXAMPLE_MODEL` to
//! point at vLLM, LM Studio, or anything else.
//!
//! ```sh
//! ollama serve &
//! ollama pull llama3.1:8b
//! cargo run --example local_model
//!
//! # Or against another runtime:
//! RAI_EXAMPLE_BASE_URL=http://localhost:8000/v1 \
//! RAI_EXAMPLE_MODEL=Qwen/Qwen2.5-7B-Instruct \
//!   cargo run --example local_model
//! ```

use futures::StreamExt;
use rai_sdk::{Capability, ClientBuilder, Model, provider::ProviderStreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("RAI_EXAMPLE_BASE_URL")
        .unwrap_or_else(|_| rai_sdk::config::OLLAMA_BASE_URL.to_string());
    let model = std::env::var("RAI_EXAMPLE_MODEL").unwrap_or_else(|_| "llama3.1:8b".to_string());

    println!("Using {model} at {base_url}\n");

    // 1. The endpoint is a property of the client, not of the process. Build a
    //    second client with a second base URL to talk to both at once.
    let client = ClientBuilder::new()
        .openai_compatible_base_url(&base_url)
        // .openai_compatible_key("...") // only if the endpoint requires one
        .model(Model::openai_compatible(&model))
        .build()?;

    // 2. Generation is identical to any other provider.
    let response = client
        .request()
        .prompt("Explain the borrow checker in two sentences.")
        .generate()
        .await?;

    println!("Response:\n{}\n", response.text());

    // 3. So is streaming.
    print!("Streaming: ");
    let mut stream = client
        .request()
        .prompt("Count from one to five.")
        .stream()
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            ProviderStreamEvent::Text(text) => {
                print!("{text}");
                use std::io::Write;
                std::io::stdout().flush()?;
            }
            ProviderStreamEvent::Done { .. } => println!(),
            _ => {}
        }
    }

    // 4. Not every local model can call tools. When one cannot, the failure is
    //    a typed capability error rather than an opaque HTTP 400, so falling
    //    back is a match arm instead of a string search.
    let weather = rai_sdk::Tool::new("get_weather")
        .description("Look up the weather for a city")
        .handler(|args: WeatherArgs, _ctx| async move {
            Ok(serde_json::json!({ "city": args.city, "forecast": "sunny" }))
        })?;

    println!("\nAsking for a tool call...");
    match client
        .request()
        .tool(weather)
        .prompt("What is the weather in Paris?")
        .generate()
        .await
    {
        Ok(response) => println!("{}", response.text()),
        Err(error) if error.unsupported_capability() == Some(Capability::ToolCalling) => {
            println!("{model} cannot call tools; answering without them instead.");
            let response = client
                .request()
                .no_tools()
                .prompt("What is the weather usually like in Paris in spring?")
                .generate()
                .await?;
            println!("{}", response.text());
        }
        Err(error) => return Err(error.into()),
    }

    Ok(())
}

#[derive(serde::Deserialize, rai_sdk::JsonSchema)]
struct WeatherArgs {
    city: String,
}
