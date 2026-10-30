//! Reasoning effort on the wire, per provider, through offline mock servers.
#![cfg(any(feature = "anthropic", feature = "openai", feature = "openrouter"))]

mod common;

use common::{Script, Step, received_json_bodies};
use rai_sdk::{GenerationConfig, Model, ReasoningEffort};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

#[cfg(any(feature = "openai", feature = "openrouter"))]
fn chat_completion() -> serde_json::Value {
    json!({
        "choices": [{
            "message": { "role": "assistant", "content": "done" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4 }
    })
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_tool_loop_sends_the_same_effort_on_every_round() {
    use rai_sdk::{Message, Tool};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(Script::new(vec![
            Step::ok(json!({
                "content": [{ "type": "tool_use", "id": "call_1", "name": "lookup", "input": {} }],
                "stop_reason": "tool_use",
                "usage": { "input_tokens": 12, "output_tokens": 2 }
            })),
            Step::ok(json!({
                "content": [{ "type": "text", "text": "done" }],
                "stop_reason": "end_turn",
                "usage": { "input_tokens": 14, "output_tokens": 1 }
            })),
        ]))
        .mount(&server)
        .await;
    let client = common::anthropic_builder(&server.uri())
        .model(Model::anthropic_custom("anthropic/claude-sonnet-5.5"))
        .build()
        .expect("client should build");
    let lookup = Tool::new("lookup")
        .description("Look a value up")
        .json_schema(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }))
        .handler(|_: serde_json::Value, _| async { Ok(json!({ "value": 1 })) })
        .expect("test tool should build");

    client
        .request()
        .tools([lookup])
        .config(
            GenerationConfig::new()
                .with_max_tokens(1024)
                .with_prompt_caching(true)
                .with_reasoning_effort(ReasoningEffort::Low),
        )
        .prompt(vec![
            Message::system("You are concise."),
            Message::user("Look it up."),
        ])
        .generate()
        .await
        .expect("mocked tool loop should succeed");

    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies.len(), 2, "one request per tool round");
    assert_eq!(
        bodies[0],
        json!({
            "model": "anthropic/claude-sonnet-5.5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Look it up.",
                    "cache_control": { "type": "ephemeral" }
                }]
            }],
            "system": [{
                "type": "text",
                "text": "You are concise.",
                "cache_control": { "type": "ephemeral" }
            }],
            "stream": false,
            "max_tokens": 1024,
            "output_config": { "effort": "low" },
            "tools": [{
                "name": "lookup",
                "description": "Look a value up",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                },
                "cache_control": { "type": "ephemeral" }
            }]
        })
    );
    // Anthropic renders the effort into the prompt, so a later round that
    // changed or dropped it would miss the cache the first round wrote.
    assert_eq!(bodies[1]["output_config"], bodies[0]["output_config"]);
    assert!(bodies[1].get("thinking").is_none());
}

#[cfg(feature = "openrouter")]
#[tokio::test]
async fn openrouter_chat_request_carries_effort_as_reasoning_effort() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![Step::ok(chat_completion())]))
        .mount(&server)
        .await;
    let client = common::openrouter_builder(&server.uri())
        .model(Model::openrouter_custom("anthropic/claude-sonnet-5.5"))
        .build()
        .expect("client should build");

    client
        .request()
        .config(GenerationConfig::new().with_reasoning_effort(ReasoningEffort::XHigh))
        .prompt("Hello")
        .generate_once()
        .await
        .expect("mocked generation should succeed");

    let bodies = received_json_bodies(&server).await;
    assert_eq!(
        bodies[0],
        json!({
            "model": "anthropic/claude-sonnet-5.5",
            "messages": [{ "role": "user", "content": "Hello" }],
            "stream": false,
            "reasoning": { "effort": "xhigh" }
        })
    );
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_request_ignores_reasoning_effort() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![
            Step::ok(chat_completion()),
            Step::ok(chat_completion()),
        ]))
        .mount(&server)
        .await;
    let client = common::openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .expect("client should build");

    for config in [
        GenerationConfig::new(),
        GenerationConfig::new().with_reasoning_effort(ReasoningEffort::Max),
    ] {
        client
            .request()
            .config(config)
            .prompt("Hello")
            .generate_once()
            .await
            .expect("mocked generation should succeed");
    }

    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies.len(), 2);
    assert_eq!(
        bodies[1], bodies[0],
        "OpenAI has its own reasoning vocabulary; the Anthropic levels must not reach it"
    );
}
