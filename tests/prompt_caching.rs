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
async fn anthropic_prompt_caching_moves_the_breakpoint_to_each_rounds_final_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(Script::new(vec![
            Step::ok(json!({
                "content": [{ "type": "tool_use", "id": "call_1", "name": "lookup", "input": {} }],
                "stop_reason": "tool_use",
                "usage": { "input_tokens": 12, "output_tokens": 2 }
            })),
            Step::ok(anthropic_response()),
        ]))
        .mount(&server)
        .await;
    let client = anthropic_builder(&server.uri())
        .model(Model::claude_sonnet_46())
        .build()
        .expect("client should build");

    client
        .request()
        .tools([test_tool("lookup")])
        .config(GenerationConfig::new().with_prompt_caching(true))
        .prompt(vec![
            Message::system("You are concise."),
            Message::user("Look it up."),
        ])
        .generate()
        .await
        .expect("mocked tool loop should succeed");

    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies.len(), 2, "one request per tool round");

    // Round 1 ends with the user's turn, so that turn carries the breakpoint.
    assert_eq!(
        bodies[0]["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Look it up.",
                "cache_control": { "type": "ephemeral" }
            }]
        }])
    );

    // Round 2 repeats round 1's message in the same form, minus the marker,
    // and moves the breakpoint to its own final message, the tool result, so
    // round 1's prefix is read from cache.
    let round_two = bodies[1]["messages"]
        .as_array()
        .expect("messages should be an array");
    assert_eq!(round_two.len(), 3);
    let mut round_one_message = bodies[0]["messages"][0].clone();
    round_one_message["content"][0]
        .as_object_mut()
        .expect("round 1's message should be a text block")
        .remove("cache_control");
    assert_eq!(round_two[0], round_one_message);
    assert_eq!(round_two[1]["content"][0]["type"], "tool_use");
    assert!(!round_two[1].to_string().contains("cache_control"));
    assert_eq!(round_two[2]["content"][0]["type"], "tool_result");
    assert_eq!(
        round_two[2]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    // System, last tool and final message: three of Anthropic's four slots.
    for body in &bodies {
        assert_eq!(body.to_string().matches("cache_control").count(), 3);
    }
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
