//! Offline usage accounting across provider codecs, stream wrappers and wire events.

use rai_sdk::Usage;
use rai_sdk::wire::{StreamAccumulator, WireStreamEvent};
use serde_json::json;

#[cfg(any(feature = "openai", feature = "anthropic", feature = "openrouter"))]
mod common;

#[test]
fn usage_wire_round_trip_preserves_unknown_and_explicit_zero() {
    let unknown: Usage = serde_json::from_value(json!({})).unwrap();
    assert_eq!(unknown, Usage::default());
    let old = json!({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4});
    let parsed: Usage = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), old);

    let payload = json!({
        "type": "usage",
        "usage": {
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "cost_usd": 0.0, "upstream_inference_cost_usd": 0.0
        }
    });
    let event: WireStreamEvent = serde_json::from_value(payload).unwrap();
    let encoded = serde_json::to_value(&event).unwrap();
    assert_eq!(encoded["usage"]["cache_read_tokens"], 0);
    assert_eq!(encoded["usage"]["cost_usd"], 0.0);
    let WireStreamEvent::Usage { usage } = event else {
        panic!("expected usage");
    };
    assert_eq!(usage.prompt_tokens, None);
    assert_eq!(usage.cache_read_tokens, Some(0));
    assert_eq!(usage.cache_write_tokens, Some(0));
    assert_eq!(usage.cost_usd, Some(0.0));
    assert_eq!(usage.upstream_inference_cost_usd, Some(0.0));
}

#[test]
fn wire_accumulator_merges_cumulative_partial_snapshots() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .push(WireStreamEvent::message_start(
            "test-model",
            rai_sdk::ProviderKind::Anthropic,
        ))
        .unwrap();
    for payload in [
        json!({"prompt_tokens":12,"completion_tokens":1,"total_tokens":13,
            "cache_read_tokens":100,"cache_write_tokens":20,"cost_usd":0.02,
            "upstream_inference_cost_usd":0.03}),
        json!({"completion_tokens":3,"total_tokens":15}),
        json!({"completion_tokens":3,"total_tokens":15}),
        json!({"prompt_tokens":null,"cache_read_tokens":0,"cost_usd":0.0,
            "upstream_inference_cost_usd":0.0}),
        json!({}),
    ] {
        accumulator
            .push(WireStreamEvent::Usage {
                usage: serde_json::from_value(payload).unwrap(),
            })
            .unwrap();
    }
    accumulator
        .push(WireStreamEvent::MessageStop {
            finish_reason: Some("end_turn".into()),
        })
        .unwrap();
    assert_eq!(
        accumulator.finish().unwrap().usage.unwrap(),
        Usage {
            prompt_tokens: Some(12),
            completion_tokens: Some(3),
            total_tokens: Some(15),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(20),
            cost_usd: Some(0.0),
            upstream_inference_cost_usd: Some(0.0),
        }
    );
}

#[cfg(any(feature = "openai", feature = "openrouter"))]
mod chat_completions {
    use super::*;
    use common::{Script, Step, data_event, sse_body};
    use futures::StreamExt;
    use rai_sdk::{Client, Model, provider::ProviderStreamEvent};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer};

    fn clients(url: &str) -> Vec<(Client, Model)> {
        let mut clients = Vec::new();
        #[cfg(feature = "openai")]
        {
            clients.push((
                common::openai_builder(url).build().unwrap(),
                Model::gpt4o_mini(),
            ));
            clients.push((
                common::openai_compatible_builder(url).build().unwrap(),
                Model::openai_compatible("test-model"),
            ));
        }
        #[cfg(feature = "openrouter")]
        clients.push((
            common::openrouter_builder(url).build().unwrap(),
            Model::openrouter_auto(),
        ));
        clients
    }

    #[tokio::test]
    async fn sync_maps_cache_and_cost_without_inventing_absent_values() {
        // OpenAI's prompt_tokens_details and OpenRouter's usage-accounting schema.
        // Cost fields on other compatible endpoints are USD extension coverage.
        let cases = [
            (json!(null), None),
            (json!({}), Some(Usage::default())),
            (
                json!({"prompt_tokens_details":null,"cost":null,"cost_details":null}),
                Some(Usage::default()),
            ),
            (
                json!({"prompt_tokens_details":{"cached_tokens":null,"cache_write_tokens":null},
                "cost_details":{"upstream_inference_cost":null}}),
                Some(Usage::default()),
            ),
            (
                json!({"prompt_tokens":100,"completion_tokens":2,"total_tokens":102,
                "prompt_tokens_details":{"cached_tokens":60,"cache_write_tokens":20},
                "cost":0.003,"cost_details":{"upstream_inference_cost":0.002}}),
                Some(Usage {
                    prompt_tokens: Some(100),
                    completion_tokens: Some(2),
                    total_tokens: Some(102),
                    cache_read_tokens: Some(60),
                    cache_write_tokens: Some(20),
                    cost_usd: Some(0.003),
                    upstream_inference_cost_usd: Some(0.002),
                }),
            ),
            (
                json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
                "prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
                "cost":0.0,"cost_details":{"upstream_inference_cost":0.0}}),
                Some(Usage {
                    prompt_tokens: Some(0),
                    completion_tokens: Some(0),
                    total_tokens: Some(0),
                    cache_read_tokens: Some(0),
                    cache_write_tokens: Some(0),
                    cost_usd: Some(0.0),
                    upstream_inference_cost_usd: Some(0.0),
                }),
            ),
            // An upstream inference cost alone is not the charge to the account.
            (
                json!({"cost_details":{"upstream_inference_cost":0.002}}),
                Some(Usage {
                    upstream_inference_cost_usd: Some(0.002),
                    ..Usage::default()
                }),
            ),
            (
                json!({"prompt_tokens":100,"completion_tokens":2,"total_tokens":102}),
                Some(Usage {
                    prompt_tokens: Some(100),
                    completion_tokens: Some(2),
                    total_tokens: Some(102),
                    ..Usage::default()
                }),
            ),
        ];
        for (usage, expected) in cases {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(Script::new(vec![Step::ok(json!({
                    "choices":[{"message":{"content":null,"tool_calls":[{
                        "id":"call_1","function":{"name":"lookup","arguments":"{}"}
                    }]},"finish_reason":"tool_calls"}],"usage":usage
                }))]))
                .mount(&server)
                .await;
            for (client, model) in clients(&server.uri()) {
                let response = client
                    .request()
                    .model(model.clone())
                    .prompt("Hello")
                    .generate_once()
                    .await
                    .unwrap();
                assert_eq!(response.usage, expected);
                assert_eq!(response.messages[0].tool_calls.len(), 1);
            }
        }
    }

    #[tokio::test]
    async fn streams_keep_content_tools_and_partial_cumulative_usage() {
        let body = sse_body(&[
            &data_event(
                json!({"choices":[{"delta":{"content":"hello", "tool_calls":[{
                    "id":"call_1","function":{"name":"lookup","arguments":"{\"city\":"}
                }]}}],"usage":{
                    "prompt_tokens":100,"completion_tokens":1,"total_tokens":101,
                    "prompt_tokens_details":{"cached_tokens":60,"cache_write_tokens":20},
                    "cost":0.003,"cost_details":{"upstream_inference_cost":0.002}
                }}),
            ),
            &data_event(json!({"choices":[{"delta":{"tool_calls":[{
                "function":{"arguments":"\"Paris\"}"}
            }]},"finish_reason":"tool_calls"}]})),
            &data_event(json!({"choices":[],"usage":{
                "completion_tokens":2,"total_tokens":102,
                "prompt_tokens_details":{"cached_tokens":0},"cost":null
            }})),
            &data_event(json!({"choices":[],"usage":{
                "completion_tokens":2,"total_tokens":102,
                "cost":0.0,"cost_details":{"upstream_inference_cost":0.0}
            }})),
            &data_event(json!({"choices":[],"usage":null})),
            "data: [DONE]",
        ]);
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(Script::new(vec![Step::sse(body)]))
            .mount(&server)
            .await;
        let expected = Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(2),
            total_tokens: Some(102),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(20),
            cost_usd: Some(0.0),
            upstream_inference_cost_usd: Some(0.0),
        };
        for (client, model) in clients(&server.uri()) {
            let events = common::collect_events(
                client
                    .request()
                    .model(model.clone())
                    .prompt("Hi")
                    .stream()
                    .await
                    .unwrap(),
            )
            .await;
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, ProviderStreamEvent::Text(t) if t == "hello"))
            );
            assert!(events.iter().any(
                |e| matches!(e, ProviderStreamEvent::ToolCallStart { id, .. } if id == "call_1")
            ));
            let Some(ProviderStreamEvent::Done {
                usage: Some(usage), ..
            }) = events.last()
            else {
                panic!("expected final usage");
            };
            assert_eq!(usage, &expected);
            let response = client
                .request()
                .model(model.clone())
                .prompt("Hi")
                .stream_accumulated()
                .await
                .unwrap();
            assert_eq!(response.text(), "hello");
            assert_eq!(response.usage.as_ref(), Some(&expected));
            assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));

            let mut events = client
                .request()
                .model(model.clone())
                .prompt("Hi")
                .stream_wire_events()
                .await
                .unwrap();
            let mut accumulator = StreamAccumulator::new();
            let mut tool_ends = 0;
            while let Some(event) = events.next().await {
                if matches!(event, WireStreamEvent::ToolCallEnd { .. }) {
                    tool_ends += 1;
                }
                // Proxy serialization must preserve the new fields too.
                accumulator
                    .push(serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap())
                    .unwrap();
            }
            assert_eq!(tool_ends, 1);
            let response = accumulator.finish().unwrap();
            assert_eq!(response.usage.as_ref(), Some(&expected));
            assert_eq!(
                response.messages[0].tool_calls[0].arguments,
                json!({"city":"Paris"})
            );
            assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));

            let events = client
                .request()
                .model(model.clone())
                .prompt("Hi")
                .generate_stream_events()
                .await
                .unwrap();
            let events: Vec<_> = Box::pin(events).collect().await;
            let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, rai_sdk::message::StreamEvent::TurnComplete { .. }))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, rai_sdk::message::StreamEvent::ToolCall { .. }))
                    .count(),
                1
            );
            let Some(rai_sdk::message::StreamEvent::TurnComplete { turn }) = events.last() else {
                panic!("expected one complete turn");
            };
            assert_eq!(
                turn.assistant_message.tool_calls[0].arguments,
                json!({"city":"Paris"})
            );
        }
    }

    #[tokio::test]
    async fn usage_without_finish_reason_defaults_to_stop_only_at_stream_end() {
        for sentinel in [true, false] {
            let mut body = sse_body(&[
                &data_event(json!({"choices":[],"usage":{"prompt_tokens":1,"cost":0.0}})),
                &data_event(json!({"choices":[{"delta":{"content":"after usage"}}]})),
            ]);
            if sentinel {
                body.push_str("data: [DONE]\n\n");
            }
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(Script::new(vec![Step::sse(body)]))
                .mount(&server)
                .await;
            for (client, model) in clients(&server.uri()) {
                let events = common::collect_events(
                    client
                        .request()
                        .model(model)
                        .prompt("Hi")
                        .stream()
                        .await
                        .unwrap(),
                )
                .await;
                assert!(matches!(
                    &events[0],
                    ProviderStreamEvent::Done {
                        finish_reason: None,
                        ..
                    }
                ));
                assert!(matches!(&events[1], ProviderStreamEvent::Text(t) if t=="after usage"));
                assert!(
                    matches!(events.last(), Some(ProviderStreamEvent::Done { finish_reason:Some(reason),.. }) if reason=="stop")
                );
            }
        }
    }
}

#[cfg(feature = "anthropic")]
mod anthropic {
    use super::*;
    use common::{Script, Step, anthropic_builder, named_event, sse_body};
    use futures::StreamExt;
    use rai_sdk::Model;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer};

    #[tokio::test]
    async fn message_stop_preserves_start_usage_without_duplicating_delta_usage() {
        for delta_kind in 0..3 {
            let mut body = sse_body(&[&named_event(
                "message_start",
                json!({"message":{"usage":{
                    "input_tokens":12,"output_tokens":1,
                    "cache_read_input_tokens":100,"cache_creation_input_tokens":20
                }}}),
            )]);
            if delta_kind > 0 {
                body.push_str(&sse_body(&[&named_event(
                    "message_delta",
                    json!({"delta":{"stop_reason":"end_turn"},"usage":
                        if delta_kind == 2 { json!({"output_tokens":3}) } else { json!(null) }
                    }),
                )]));
            }
            body.push_str(&sse_body(&[&named_event("message_stop", json!({}))]));
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/messages"))
                .respond_with(Script::new(vec![Step::sse(body)]))
                .mount(&server)
                .await;
            let client = anthropic_builder(&server.uri())
                .model(Model::claude_sonnet_46())
                .build()
                .unwrap();
            let events =
                common::collect_events(client.request().prompt("Hi").stream().await.unwrap()).await;
            let usages: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    rai_sdk::provider::ProviderStreamEvent::Done {
                        usage: Some(usage), ..
                    } => Some(usage),
                    _ => None,
                })
                .collect();
            assert_eq!(usages.len(), 1);
            assert_eq!(usages[0].cache_read_tokens, Some(100));
            assert_eq!(usages[0].cache_write_tokens, Some(20));
            assert_eq!(
                usages[0].completion_tokens,
                Some(if delta_kind == 2 { 3 } else { 1 })
            );
            assert_eq!(
                usages[0].total_tokens,
                Some(if delta_kind == 2 { 15 } else { 13 })
            );
            let response = client
                .request()
                .prompt("Hi")
                .stream_accumulated()
                .await
                .unwrap();
            assert_eq!(response.usage.as_ref(), Some(usages[0]));
            assert_eq!(
                response.finish_reason.as_deref(),
                (delta_kind > 0).then_some("end_turn")
            );
        }
    }

    #[tokio::test]
    async fn a_midstream_error_does_not_emit_a_complete_turn() {
        let body = sse_body(&[
            &named_event(
                "message_start",
                json!({"message":{"usage":{"input_tokens":12,"output_tokens":1}}}),
            ),
            &named_event(
                "message_delta",
                json!({"delta":{},"usage":{"output_tokens":3}}),
            ),
            &named_event("error", json!({"error":{"message":"overloaded"}})),
        ]);
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(Script::new(vec![Step::sse(body)]))
            .mount(&server)
            .await;
        let client = anthropic_builder(&server.uri())
            .model(Model::claude_sonnet_46())
            .build()
            .unwrap();
        let events = client
            .request()
            .prompt("Hi")
            .generate_stream_events()
            .await
            .unwrap();
        let events: Vec<_> = Box::pin(events).collect().await;
        assert_eq!(events.len(), 1);
        assert!(events[0].is_err());
    }

    #[tokio::test]
    async fn sync_maps_native_cache_counts_and_openrouter_messages_cost_extensions() {
        // OpenRouter /openapi.json: MessagesResult.usage extends AnthropicUsage
        // with cost and cost_details. Native Anthropic omits those fields.
        for (extensions, read, write, cost, upstream) in [
            (json!({}), None, None, None, None),
            (
                json!({"cache_read_input_tokens":null,"cache_creation_input_tokens":null,
                "cost":null,"cost_details":null}),
                None,
                None,
                None,
                None,
            ),
            (
                json!({"cache_read_input_tokens":0,"cache_creation_input_tokens":0,
                "cost":0.0,"cost_details":{"upstream_inference_cost":0.0}}),
                Some(0),
                Some(0),
                Some(0.0),
                Some(0.0),
            ),
            (
                json!({"cache_read_input_tokens":100,"cache_creation_input_tokens":20,
                "cost":0.003,"cost_details":{"upstream_inference_cost":0.002}}),
                Some(100),
                Some(20),
                Some(0.003),
                Some(0.002),
            ),
        ] {
            let mut usage = json!({"input_tokens":12,"output_tokens":3});
            usage
                .as_object_mut()
                .unwrap()
                .extend(extensions.as_object().unwrap().clone());
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/messages"))
                .respond_with(Script::new(vec![Step::ok(json!({
                    "content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{}}],
                    "stop_reason":"tool_use","usage":usage
                }))]))
                .mount(&server)
                .await;
            let client = anthropic_builder(&server.uri())
                .model(Model::claude_sonnet_46())
                .build()
                .unwrap();
            let response = client
                .request()
                .prompt("Hello")
                .generate_once()
                .await
                .unwrap();
            assert_eq!(response.messages[0].tool_calls.len(), 1);
            assert_eq!(
                response.usage.unwrap(),
                Usage {
                    prompt_tokens: Some(12),
                    completion_tokens: Some(3),
                    total_tokens: Some(15),
                    cache_read_tokens: read,
                    cache_write_tokens: write,
                    cost_usd: cost,
                    upstream_inference_cost_usd: upstream,
                }
            );
        }
    }

    #[tokio::test]
    async fn streaming_merges_start_and_partial_deltas_including_compatible_cost_extensions() {
        // Cost on streaming Messages events is compatibility coverage, not a
        // guarantee in OpenRouter's current MessagesStartEvent/MessagesDeltaEvent schema.
        for include_cost in [false, true] {
            let mut start = json!({"input_tokens":12,"output_tokens":1,
                "cache_read_input_tokens":100,"cache_creation_input_tokens":20});
            let mut final_usage =
                json!({"input_tokens":null,"output_tokens":null,"cache_read_input_tokens":0});
            if include_cost {
                start["cost"] = json!(0.003);
                start["cost_details"] = json!({"upstream_inference_cost":0.002});
                final_usage["cost"] = json!(0.0);
                final_usage["cost_details"] = json!({"upstream_inference_cost":0.0});
            }
            let body = sse_body(&[
                &named_event("message_start", json!({"message":{"usage":start}})),
                &named_event(
                    "content_block_delta",
                    json!({"delta":{"type":"text_delta","text":"hi"}}),
                ),
                &named_event(
                    "message_delta",
                    json!({"delta":{},"usage":{"output_tokens":3}}),
                ),
                &named_event(
                    "message_delta",
                    json!({"delta":{},"usage":{"output_tokens":3,"cache_creation_input_tokens":null,"cost":null,"cost_details":null}}),
                ),
                &named_event("message_delta", json!({"delta":{},"usage":final_usage})),
                &named_event(
                    "message_delta",
                    json!({"delta":{"stop_reason":"end_turn"},"usage":null}),
                ),
                &named_event("message_stop", json!({})),
            ]);
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/messages"))
                .respond_with(Script::new(vec![Step::sse(body)]))
                .mount(&server)
                .await;
            let client = anthropic_builder(&server.uri())
                .model(Model::claude_sonnet_46())
                .build()
                .unwrap();
            let expected = Usage {
                prompt_tokens: Some(12),
                completion_tokens: Some(3),
                total_tokens: Some(15),
                cache_read_tokens: Some(0),
                cache_write_tokens: Some(20),
                cost_usd: include_cost.then_some(0.0),
                upstream_inference_cost_usd: include_cost.then_some(0.0),
            };
            let response = client
                .request()
                .prompt("Hi")
                .stream_accumulated()
                .await
                .unwrap();
            assert_eq!(response.text(), "hi");
            assert_eq!(response.usage.as_ref(), Some(&expected));
            let mut events = client
                .request()
                .prompt("Hi")
                .stream_wire_events()
                .await
                .unwrap();
            let mut accumulator = StreamAccumulator::new();
            while let Some(event) = events.next().await {
                accumulator.push(event).unwrap();
            }
            assert_eq!(accumulator.finish().unwrap().usage, Some(expected));
        }
    }
}
