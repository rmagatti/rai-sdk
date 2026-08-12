//! How the streaming methods interact with tool registration.
//!
//! Streaming cannot execute a tool loop, so the in-process streaming methods
//! reject requests carrying tools. The wire-events proxy path is the exception:
//! it advertises definitions without executing them and forwards tool calls to
//! the remote client. These tests pin both behaviors and the effective tool-set
//! rules around request overrides.
//!
//! Every test is offline: streams are served by a local `wiremock` mock.
//!
//! The behavior under test is provider-independent, but the tests need a
//! concrete provider to stream from, so the whole file is gated on `openai`.
#![cfg(feature = "openai")]

mod common;

use common::{
    Script, Step, collect_events, data_event, expect_error, openai_builder, received_json_bodies,
    sse_body, stream_text,
};
#[cfg(feature = "anthropic")]
use common::{anthropic_builder, named_event};
use futures::StreamExt;
#[cfg(feature = "anthropic")]
use rai_sdk::wire::WireStreamEvent;
use rai_sdk::{Error, JsonSchema, Model, Tool, ToolContext, ToolDefinition};
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    value: String,
}

/// A tool that is never expected to run in these tests.
fn echo_tool() -> Tool {
    Tool::new("echo")
        .description("Echo the input back")
        .handler(
            |args: EchoArgs, _ctx: ToolContext| async move { Ok(json!({ "value": args.value })) },
        )
        .expect("echo tool should build")
}

/// A minimal OpenAI-style text stream.
fn text_stream_body() -> String {
    sse_body(&[
        &data_event(json!({
            "choices": [{ "delta": { "content": "hello " }, "index": 0 }]
        })),
        &data_event(json!({
            "choices": [{ "delta": { "content": "world" }, "index": 0 }]
        })),
        &data_event(json!({
            "choices": [{ "delta": {}, "finish_reason": "stop", "index": 0 }]
        })),
        "data: [DONE]",
    ])
}

async fn streaming_mock() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![Step::sse(text_stream_body())]))
        .mount(&server)
        .await;
    server
}

fn assert_rejected_for_tools(error: Error) {
    match error {
        Error::InvalidRequest(message) => assert!(
            message.contains("Streaming with tools is not supported"),
            "unexpected InvalidRequest message: {message}"
        ),
        other => panic!("expected InvalidRequest for a tool-bearing stream, got: {other:?}"),
    }
}

#[tokio::test]
async fn streaming_from_a_tool_free_client_works() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .expect("client should build");

    let stream = client
        .request()
        .prompt("stream please")
        .stream()
        .await
        .expect("a tool-free client should be allowed to stream");

    assert_eq!(stream_text(&collect_events(stream).await), "hello world");
}

#[tokio::test]
async fn a_client_level_tool_still_blocks_streaming() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .prompt("stream please")
            .stream()
            .await
            .map(|_| "a stream"),
    );
    assert_rejected_for_tools(error);

    assert_eq!(
        server
            .received_requests()
            .await
            .expect("request recording")
            .len(),
        0,
        "a rejected stream must not reach the provider"
    );
}

#[tokio::test]
async fn no_tools_lets_a_tool_bearing_client_stream() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    // The regression this guards: the check used to inspect the client's
    // registry, so `no_tools()` could not enable streaming.
    let stream = client
        .request()
        .no_tools()
        .prompt("stream please")
        .stream()
        .await
        .expect("no_tools() should make the request streamable");

    assert_eq!(stream_text(&collect_events(stream).await), "hello world");
}

#[tokio::test]
async fn a_request_level_tool_blocks_streaming_on_a_tool_free_client() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .expect("client should build");

    // The inverse regression: a per-request tool used to be silently dropped,
    // streaming as though it had never been registered.
    let error = expect_error(
        client
            .request()
            .tool(echo_tool())
            .prompt("stream please")
            .stream()
            .await
            .map(|_| "a stream"),
    );
    assert_rejected_for_tools(error);
}

#[tokio::test]
async fn stream_accumulated_follows_the_same_tool_rules() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .prompt("stream please")
            .stream_accumulated()
            .await,
    );
    assert_rejected_for_tools(error);

    let response = client
        .request()
        .no_tools()
        .prompt("stream please")
        .stream_accumulated()
        .await
        .expect("no_tools() should make the request streamable");
    assert_eq!(response.text(), "hello world");
}

#[tokio::test]
async fn generate_stream_events_follows_the_same_tool_rules() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .prompt("stream please")
            .generate_stream_events()
            .await
            .map(|_| "a stream"),
    );
    assert_rejected_for_tools(error);

    let _stream = client
        .request()
        .no_tools()
        .prompt("stream please")
        .generate_stream_events()
        .await
        .expect("no_tools() should make the request streamable");
}

#[tokio::test]
async fn the_low_level_client_api_still_uses_the_client_registry() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    // `Client::generate_stream` takes no request context, so it can only
    // consider the client's own tools. That behavior is unchanged.
    let error = expect_error(
        client
            .generate_stream(
                Model::gpt4o_mini(),
                &"stream please".into(),
                &Default::default(),
            )
            .await
            .map(|_| "a stream"),
    );
    assert_rejected_for_tools(error);
}

#[tokio::test]
async fn wire_events_accept_and_advertise_registered_tools() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .tool(echo_tool())
        .build()
        .expect("client should build");

    let mut stream = client
        .request()
        .prompt("stream please")
        .stream_wire_events()
        .await
        .expect("wire events should support registered tools");
    while stream.next().await.is_some() {}

    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["tools"][0]["function"]["name"], "echo");
    assert_eq!(
        bodies[0]["tools"][0]["function"]["description"],
        "Echo the input back"
    );
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn wire_events_advertise_definition_only_tools_and_forward_anthropic_tool_use() {
    let server = MockServer::start().await;
    let body = sse_body(&[
        &named_event(
            "message_start",
            json!({
                "type": "message_start",
                "message": { "usage": { "input_tokens": 12, "output_tokens": 0 } }
            }),
        ),
        &named_event(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "echo",
                    "input": {}
                }
            }),
        ),
        &named_event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"value\":" }
            }),
        ),
        &named_event(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "\"hello\"}" }
            }),
        ),
        &named_event(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use", "stop_sequence": null },
                "usage": { "output_tokens": 8 }
            }),
        ),
        &named_event("message_stop", json!({ "type": "message_stop" })),
    ]);

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(Script::new(vec![Step::sse(body)]))
        .mount(&server)
        .await;

    let definition = ToolDefinition {
        name: "echo".to_string(),
        description: Some("Echo the input back".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        }),
    };
    let client = anthropic_builder(&server.uri())
        .model(Model::claude_sonnet_46())
        .build()
        .expect("client should build");

    let mut stream = client
        .request()
        .tool_definition(definition.clone())
        .prompt("use echo")
        .stream_wire_events()
        .await
        .expect("wire streaming should advertise definition-only tools");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    assert!(matches!(
        events.get(1),
        Some(WireStreamEvent::ToolCallStart { id, name })
            if id == "toolu_123" && name == "echo"
    ));
    assert!(matches!(
        events.get(2),
        Some(WireStreamEvent::ToolCallDelta { id, arguments })
            if id == "toolu_123" && arguments == "{\"value\":"
    ));
    assert!(matches!(
        events.get(3),
        Some(WireStreamEvent::ToolCallDelta { id, arguments })
            if id == "toolu_123" && arguments == "\"hello\"}"
    ));
    assert!(matches!(
        events.get(4),
        Some(WireStreamEvent::ToolCallEnd { id, name, arguments })
            if id == "toolu_123" && name == "echo" && arguments == "{\"value\":\"hello\"}"
    ));
    assert!(matches!(
        events.last(),
        Some(WireStreamEvent::MessageStop { finish_reason })
            if finish_reason.as_deref() == Some("tool_use")
    ));

    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies.len(), 1);
    assert_eq!(
        bodies[0]["tools"],
        json!([{
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema
        }])
    );
}

#[tokio::test]
async fn definition_only_tools_remain_rejected_by_non_wire_streams() {
    let server = streaming_mock().await;
    let client = openai_builder(&server.uri())
        .model(Model::gpt4o_mini())
        .build()
        .expect("client should build");
    let definition = ToolDefinition {
        name: "echo".to_string(),
        description: Some("Echo the input back".to_string()),
        input_schema: json!({ "type": "object" }),
    };

    let error = expect_error(
        client
            .request()
            .tool_definition(definition)
            .prompt("stream please")
            .stream()
            .await
            .map(|_| "a stream"),
    );
    assert_rejected_for_tools(error);
}
