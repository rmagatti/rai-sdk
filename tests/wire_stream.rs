//! `stream_wire_events()` against a mock provider: framing, usage, errors, and
//! cancellation.
//!
//! The behavior is provider-independent, but the tests need a concrete provider
//! to stream from. Most use OpenAI; the "provider refuses mid-stream" case uses
//! Anthropic, because its SSE dialect is the one that carries an explicit
//! `error` event. Each test is gated on the feature it needs so the file
//! participates in every CI feature combination.
//!
//! Every test is offline: streams come from `wiremock` or from a raw loopback
//! socket in `common`.
#![cfg(any(feature = "openai", feature = "anthropic"))]

mod common;

use futures::StreamExt;
use rai_sdk::wire::{StreamAccumulator, WireErrorKind, WireStreamEvent};
use rai_sdk::{Model, ProviderKind, WIRE_PROTOCOL_VERSION};

/// Drain a wire stream into a vector.
async fn collect_wire<S>(stream: S) -> Vec<WireStreamEvent>
where
    S: futures::Stream<Item = WireStreamEvent>,
{
    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// The `"type"` tags of a stream, in order.
fn tags(events: &[WireStreamEvent]) -> Vec<&'static str> {
    events.iter().map(WireStreamEvent::tag).collect()
}

/// Re-emit every event as JSON and parse it back, the way a proxy would.
fn through_the_wire(events: &[WireStreamEvent]) -> Vec<WireStreamEvent> {
    events
        .iter()
        .map(|event| {
            let payload = serde_json::to_string(event).expect("event should serialize");
            serde_json::from_str(&payload).unwrap_or_else(|error| {
                panic!("a payload this crate wrote should parse back: {payload}: {error}")
            })
        })
        .collect()
}

#[cfg(feature = "openai")]
mod openai {
    use super::*;

    use common::{
        Script, Step, data_event, openai_builder, sse_body, sse_until_disconnect,
        truncated_chunked_sse,
    };
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer};

    /// An OpenAI-style stream: two text deltas, a finish reason, then usage.
    fn text_stream_body() -> String {
        sse_body(&[
            &data_event(json!({
                "choices": [{ "delta": { "content": "Rust " }, "index": 0 }]
            })),
            &data_event(json!({
                "choices": [{ "delta": { "content": "is fast." }, "index": 0 }]
            })),
            &data_event(json!({
                "choices": [{ "delta": {}, "finish_reason": "stop", "index": 0 }]
            })),
            &data_event(json!({
                "choices": [],
                "usage": { "prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15 }
            })),
            "data: [DONE]",
        ])
    }

    fn tool_stream_body() -> String {
        sse_body(&[
            &data_event(json!({
                "choices": [{
                    "delta": { "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": { "name": "get_weather", "arguments": "" }
                    }] },
                    "index": 0
                }]
            })),
            &data_event(json!({
                "choices": [{
                    "delta": { "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": "{\"city\":" }
                    }] },
                    "index": 0
                }]
            })),
            &data_event(json!({
                "choices": [{
                    "delta": { "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": "\"Paris\"}" }
                    }] },
                    "index": 0
                }]
            })),
            &data_event(json!({
                "choices": [{ "delta": {}, "finish_reason": "tool_calls", "index": 0 }]
            })),
            "data: [DONE]",
        ])
    }

    async fn mock_with(body: String) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(Script::new(vec![Step::sse(body)]))
            .mount(&server)
            .await;
        server
    }

    async fn wire_events_from(body: String) -> Vec<WireStreamEvent> {
        let server = mock_with(body).await;
        let client = openai_builder(&server.uri())
            .model(Model::gpt4o_mini())
            .build()
            .expect("client should build");

        let stream = client
            .request()
            .prompt("stream please")
            .stream_wire_events()
            .await
            .expect("the stream should open");

        collect_wire(stream).await
    }

    #[tokio::test]
    async fn a_generation_is_framed_by_message_start_and_message_stop() {
        let events = wire_events_from(text_stream_body()).await;

        assert_eq!(
            tags(&events),
            vec![
                "message_start",
                "text_delta",
                "text_delta",
                "usage",
                "message_stop"
            ]
        );

        assert_eq!(
            events[0],
            WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI)
        );
        assert!(events.last().expect("non-empty").is_terminal());
        assert_eq!(
            events.iter().filter(|event| event.is_terminal()).count(),
            1,
            "a stream must have exactly one terminal event"
        );
    }

    #[tokio::test]
    async fn the_opening_event_names_the_protocol_version_model_and_provider() {
        let events = wire_events_from(text_stream_body()).await;

        match &events[0] {
            WireStreamEvent::MessageStart {
                protocol_version,
                model,
                provider,
            } => {
                assert_eq!(*protocol_version, WIRE_PROTOCOL_VERSION);
                assert_eq!(model, "gpt-4o-mini");
                assert_eq!(*provider, ProviderKind::OpenAI);
            }
            other => panic!("expected a message_start, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_whole_loop_rebuilds_the_response_on_the_far_side() {
        // Server → JSON → client, which is the arrangement this feature exists
        // for. The assertion is on the reassembled response, not on the events.
        let events = wire_events_from(text_stream_body()).await;
        let received = through_the_wire(&events);

        let response = StreamAccumulator::accumulate(futures::stream::iter(received))
            .await
            .expect("the proxied stream should reassemble");

        assert_eq!(response.text(), "Rust is fast.");
        assert_eq!(response.model, "gpt-4o-mini");
        assert_eq!(response.provider, ProviderKind::OpenAI);
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));

        let usage = response.usage.expect("usage should reach the client");
        assert_eq!(usage.prompt_tokens, Some(11));
        assert_eq!(usage.completion_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(15));
    }

    #[tokio::test]
    async fn usage_is_emitted_once_and_before_the_terminal_event() {
        let events = wire_events_from(text_stream_body()).await;
        let tags = tags(&events);

        let usage_positions: Vec<usize> = tags
            .iter()
            .enumerate()
            .filter(|(_, tag)| **tag == "usage")
            .map(|(index, _)| index)
            .collect();

        assert_eq!(usage_positions, vec![tags.len() - 2]);
    }

    #[tokio::test]
    async fn tool_calls_stream_as_start_deltas_and_an_assembled_end() {
        let events = wire_events_from(tool_stream_body()).await;

        assert_eq!(
            tags(&events),
            vec![
                "message_start",
                "tool_call_start",
                "tool_call_delta",
                "tool_call_delta",
                "tool_call_delta",
                "tool_call_end",
                "message_stop"
            ]
        );

        // The provider emits an empty first argument fragment; the assembled
        // end event is what a client should actually act on.
        assert_eq!(
            events[5],
            WireStreamEvent::ToolCallEnd {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            }
        );

        let response =
            StreamAccumulator::accumulate(futures::stream::iter(through_the_wire(&events)))
                .await
                .expect("the proxied stream should reassemble");

        let tool_calls = &response.messages[0].tool_calls;
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].arguments["city"], "Paris");
    }

    #[tokio::test]
    async fn a_truncated_upstream_becomes_an_error_event() {
        // The upstream dies halfway through. The consumer must still receive a
        // terminal event rather than an inexplicably short stream.
        let server = truncated_chunked_sse(vec![sse_body(&[&data_event(json!({
            "choices": [{ "delta": { "content": "half a sen" }, "index": 0 }]
        }))])])
        .await;

        let client = openai_builder(server.base_url())
            .model(Model::gpt4o_mini())
            .build()
            .expect("client should build");

        let events = collect_wire(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        )
        .await;

        assert_eq!(tags(&events), vec!["message_start", "text_delta", "error"]);

        match events.last().expect("non-empty") {
            WireStreamEvent::Error { error } => {
                assert_eq!(error.kind, WireErrorKind::Stream);
            }
            other => panic!("expected an error event, got {other:?}"),
        }

        // And it must survive the wire as an error, not as a truncation.
        let error = StreamAccumulator::accumulate(futures::stream::iter(through_the_wire(&events)))
            .await
            .expect_err("a failed stream should not reassemble");
        assert_eq!(error.kind, WireErrorKind::Stream);
    }

    /// Dropping the consumer must abort the provider request.
    ///
    /// This is the "no orphaned generations" guarantee: when a proxy's client
    /// hangs up, the server stops paying for tokens nobody will read. The mock
    /// provider holds its response body open and reports the moment the socket
    /// closes.
    #[tokio::test]
    async fn dropping_the_stream_aborts_the_upstream_request() {
        let mut server = sse_until_disconnect(vec![sse_body(&[&data_event(json!({
            "choices": [{ "delta": { "content": "still going" }, "index": 0 }]
        }))])])
        .await;

        let client = openai_builder(server.base_url())
            .model(Model::gpt4o_mini())
            .build()
            .expect("client should build");

        let mut stream = Box::pin(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        );

        assert_eq!(
            stream.next().await.map(|event| event.tag()),
            Some("message_start")
        );
        assert_eq!(
            stream.next().await.map(|event| event.tag()),
            Some("text_delta"),
            "the provider should still be mid-generation"
        );

        // The consumer goes away. Nothing else in the SDK holds the request.
        drop(stream);

        server.wait_for_disconnect().await;
    }

    /// The same guarantee for the other streaming entry points, since they
    /// share the mechanism and would regress together.
    #[tokio::test]
    async fn dropping_a_high_level_stream_also_aborts_the_upstream_request() {
        let mut server = sse_until_disconnect(vec![sse_body(&[&data_event(json!({
            "choices": [{ "delta": { "content": "still going" }, "index": 0 }]
        }))])])
        .await;

        let client = openai_builder(server.base_url())
            .model(Model::gpt4o_mini())
            .build()
            .expect("client should build");

        let mut stream = Box::pin(
            client
                .request()
                .prompt("stream please")
                .generate_stream_events()
                .await
                .expect("the stream should open"),
        );

        assert!(stream.next().await.is_some(), "expected a first event");
        drop(stream);

        server.wait_for_disconnect().await;
    }

    /// Cancelling the surrounding task — which is what an axum client
    /// disconnect looks like — must abort the upstream too.
    #[tokio::test]
    async fn cancelling_the_consuming_task_aborts_the_upstream_request() {
        let mut server = sse_until_disconnect(vec![sse_body(&[&data_event(json!({
            "choices": [{ "delta": { "content": "still going" }, "index": 0 }]
        }))])])
        .await;

        let base_url = server.base_url().to_string();
        let consumer = tokio::spawn(async move {
            let client = openai_builder(&base_url)
                .model(Model::gpt4o_mini())
                .build()
                .expect("client should build");

            // Never completes: the mock never ends its response body.
            let _ = client
                .request()
                .prompt("stream please")
                .stream_accumulated()
                .await;
        });

        // Cancel only once the generation is genuinely in flight, so the
        // assertion is about cancellation rather than about a race to connect.
        server.wait_until_streaming().await;
        consumer.abort();
        let _ = consumer.await;

        server.wait_for_disconnect().await;
    }

    #[tokio::test]
    async fn a_tool_bearing_request_streams_and_advertises_the_tools() {
        use rai_sdk::{JsonSchema, Tool, ToolContext};
        use serde::Deserialize;

        #[derive(Debug, Deserialize, JsonSchema)]
        struct EchoArgs {
            value: String,
        }

        let server = mock_with(text_stream_body()).await;
        let client = openai_builder(&server.uri())
            .model(Model::gpt4o_mini())
            .tool(
                Tool::new("echo")
                    .description("Echo the input back")
                    .handler(|args: EchoArgs, _ctx: ToolContext| async move {
                        Ok(json!({ "value": args.value }))
                    })
                    .expect("echo tool should build"),
            )
            .build()
            .expect("client should build");

        // The wire-events proxy path advertises tools without executing
        // them, so a tool-bearing request streams instead of erroring (the
        // in-process streaming methods keep rejecting — pinned in
        // tests/streaming_tools.rs alongside the advertised-body assertions).
        let events = client
            .request()
            .prompt("stream please")
            .stream_wire_events()
            .await
            .expect("wire events advertise registered tools without executing them");
        let tags: Vec<&'static str> = events.map(|event| event.tag()).collect().await;
        assert_eq!(tags.last(), Some(&"message_stop"));
    }
}

#[cfg(feature = "anthropic")]
mod anthropic {
    use super::*;

    use common::{Script, Step, anthropic_builder, named_event, sse_body};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer};

    /// A stream that starts fine and is then refused by the provider.
    fn refused_stream_body() -> String {
        sse_body(&[
            &named_event(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Sure, I can" }
                }),
            ),
            &named_event(
                "error",
                json!({
                    "type": "error",
                    "error": { "type": "overloaded_error", "message": "Overloaded" }
                }),
            ),
        ])
    }

    /// A provider that refuses mid-stream must be distinguishable from a
    /// connection that died: the former arrives as a typed `error` event.
    #[tokio::test]
    async fn a_mid_stream_provider_error_arrives_as_an_error_event() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(Script::new(vec![Step::sse(refused_stream_body())]))
            .mount(&server)
            .await;

        let client = anthropic_builder(&server.uri())
            .model(Model::claude_sonnet_46())
            .build()
            .expect("client should build");

        let events = collect_wire(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        )
        .await;

        assert_eq!(tags(&events), vec!["message_start", "text_delta", "error"]);

        match events.last().expect("non-empty") {
            WireStreamEvent::Error { error } => {
                assert_eq!(error.kind, WireErrorKind::Request);
                assert_eq!(error.provider, Some(ProviderKind::Anthropic));
                assert!(
                    error.message.contains("Overloaded"),
                    "the provider's own message should reach the client: {}",
                    error.message
                );
            }
            other => panic!("expected an error event, got {other:?}"),
        }

        // The distinction the client actually acts on: "the provider refused"
        // carries a category and a message; "the network died" would not.
        let error = StreamAccumulator::accumulate(futures::stream::iter(through_the_wire(&events)))
            .await
            .expect_err("a refused stream should not reassemble");
        assert_eq!(error.kind, WireErrorKind::Request);
        assert!(error.message.contains("Overloaded"));
    }

    #[tokio::test]
    async fn the_opening_event_names_the_anthropic_provider() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(Script::new(vec![Step::sse(sse_body(&[&named_event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": { "output_tokens": 7 }
                }),
            )]))]))
            .mount(&server)
            .await;

        let client = anthropic_builder(&server.uri())
            .model(Model::claude_sonnet_46())
            .build()
            .expect("client should build");

        let events = collect_wire(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        )
        .await;

        assert_eq!(
            tags(&events),
            vec!["message_start", "usage", "message_stop"]
        );
        match &events[0] {
            WireStreamEvent::MessageStart { provider, .. } => {
                assert_eq!(*provider, ProviderKind::Anthropic);
            }
            other => panic!("expected a message_start, got {other:?}"),
        }

        let response =
            StreamAccumulator::accumulate(futures::stream::iter(through_the_wire(&events)))
                .await
                .expect("the proxied stream should reassemble");
        assert_eq!(
            response
                .usage
                .expect("usage should survive")
                .completion_tokens,
            Some(7)
        );
        assert_eq!(response.finish_reason.as_deref(), Some("end_turn"));
    }

    /// The token counts of a whole generation, in the two events Anthropic
    /// spreads them over. The `output_tokens` on `message_start` is the partial
    /// count at that point, not the final one.
    fn split_usage_stream_body() -> String {
        sse_body(&[
            &named_event(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_01",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": { "input_tokens": 42, "output_tokens": 1 }
                    }
                }),
            ),
            &named_event(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Hello" }
                }),
            ),
            &named_event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": { "output_tokens": 7 }
                }),
            ),
        ])
    }

    /// Anthropic is the one dialect that splits usage across events: input
    /// tokens ride on `message_start` and output tokens on `message_delta`.
    /// Reading either event in isolation loses half the accounting, so the
    /// input count has to survive until the delta arrives.
    #[tokio::test]
    async fn usage_split_across_two_events_is_reported_whole() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(Script::new(vec![Step::sse(split_usage_stream_body())]))
            .mount(&server)
            .await;

        let client = anthropic_builder(&server.uri())
            .model(Model::claude_sonnet_46())
            .build()
            .expect("client should build");

        let events = collect_wire(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        )
        .await;

        assert_eq!(
            tags(&events),
            vec!["message_start", "text_delta", "usage", "message_stop"]
        );

        let response =
            StreamAccumulator::accumulate(futures::stream::iter(through_the_wire(&events)))
                .await
                .expect("the proxied stream should reassemble");
        let usage = response.usage.expect("usage should survive");
        assert_eq!(usage.prompt_tokens, Some(42));
        // The final output count, not the partial one from `message_start`.
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(49));
    }

    /// Without a `message_start` there is no input count to report, and an
    /// unknown count is left unknown rather than guessed at from the half of
    /// the accounting that did arrive.
    #[tokio::test]
    async fn usage_without_a_message_start_leaves_the_prompt_count_unknown() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(Script::new(vec![Step::sse(sse_body(&[&named_event(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "end_turn" },
                    "usage": { "output_tokens": 7 }
                }),
            )]))]))
            .mount(&server)
            .await;

        let client = anthropic_builder(&server.uri())
            .model(Model::claude_sonnet_46())
            .build()
            .expect("client should build");

        let events = collect_wire(
            client
                .request()
                .prompt("stream please")
                .stream_wire_events()
                .await
                .expect("the stream should open"),
        )
        .await;

        let usage = events
            .iter()
            .find_map(|event| match event {
                WireStreamEvent::Usage { usage } => Some(usage),
                _ => None,
            })
            .expect("the stream should still report the output tokens it saw");

        assert_eq!(usage.prompt_tokens, None);
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, None);
    }
}
