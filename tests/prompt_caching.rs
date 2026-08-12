//! Anthropic prompt-caching request serialization through offline mock servers.
#![cfg(feature = "anthropic")]

mod common;

use common::{Script, Step, anthropic_builder, received_json_bodies};
use rai_sdk::{GenerationConfig, Message, Model, Prompt, Tool};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

fn anthropic_response() -> serde_json::Value {
    json!({
        "content": [{ "type": "text", "text": "done" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 12, "output_tokens": 2 }
    })
}

async fn anthropic_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(Script::new(vec![Step::ok(anthropic_response())]))
        .mount(&server)
        .await;
    server
}

fn test_tool(name: &'static str) -> Tool {
    Tool::new(name)
        .description(format!("Run {name}"))
        .json_schema(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }))
        .handler(|_: serde_json::Value, _| async { Ok(json!({})) })
        .expect("test tool should build")
}

#[tokio::test]
async fn anthropic_prompt_caching_marks_system_and_last_tool() {
    let server = anthropic_server().await;
    let client = anthropic_builder(&server.uri())
        .model(Model::claude_sonnet_46())
        .build()
        .expect("client should build");
    let prompt = Prompt::new(vec![
        Message::system("You are concise."),
        Message::user("Use a tool."),
    ]);

    client
        .request()
        .tools([test_tool("first"), test_tool("second")])
        .config(GenerationConfig::new().with_prompt_caching(true))
        .prompt(prompt)
        .generate_once()
        .await
        .expect("mocked generation should succeed");

    let bodies = received_json_bodies(&server).await;
    let body = &bodies[0];
    assert_eq!(
        body["system"],
        json!([{
            "type": "text",
            "text": "You are concise.",
            "cache_control": { "type": "ephemeral" }
        }])
    );

    let tools = body["tools"].as_array().expect("tools should be an array");
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"], json!({ "type": "ephemeral" }));
}

#[tokio::test]
async fn anthropic_prompt_caching_defaults_to_the_existing_wire_shape() {
    let server = anthropic_server().await;
    let client = anthropic_builder(&server.uri())
        .model(Model::claude_sonnet_46())
        .build()
        .expect("client should build");

    client
        .request()
        .prompt(vec![Message::system("Be brief."), Message::user("Hello")])
        .generate_once()
        .await
        .expect("mocked generation should succeed");

    let bodies = received_json_bodies(&server).await;
    assert_eq!(
        bodies[0],
        json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "Hello" }],
            "system": "Be brief.",
            "stream": false,
            "max_tokens": 8192
        })
    );
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn non_anthropic_provider_ignores_prompt_caching() {
    use common::openai_builder;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![Step::ok(json!({
            "choices": [{
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4 }
        }))]))
        .mount(&server)
        .await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .expect("client should build");

    client
        .request()
        .config(GenerationConfig::new().with_prompt_caching(true))
        .prompt("Hello")
        .generate_once()
        .await
        .expect("non-Anthropic generation should succeed");

    let bodies = received_json_bodies(&server).await;
    assert!(
        !bodies[0].to_string().contains("cache_control"),
        "the Anthropic-only setting must not alter OpenAI JSON"
    );
}
