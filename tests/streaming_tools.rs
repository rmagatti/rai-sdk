//! How the streaming methods interact with tool registration.
//!
//! Streaming cannot execute a tool loop, so a request carrying tools is
//! rejected. The subtlety these tests pin down is *which* tool set is
//! consulted: the request's effective set, not the client's. That makes
//! `no_tools()` a real escape hatch, and makes a request-level `tool()` an error
//! even on a client that has none.
//!
//! Every test is offline: streams are served by a local `wiremock` mock.

mod common;

use common::{
    Script, Step, collect_events, data_event, expect_error, openai_builder, sse_body, stream_text,
};
use rai_sdk::{Error, JsonSchema, Model, Tool, ToolContext};
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
