//! The wire contract seen from outside the crate.
//!
//! These tests take the position of a consumer of the published crate: they
//! parse JSON that a *server* wrote and reassemble it, without touching any
//! provider. That is deliberately the same direction of travel as a real proxy
//! client, and it means the whole file runs in every Cargo feature combination,
//! including a providerless build.
//!
//! The exhaustive per-variant round-trip and the committed wire-format
//! fixtures live in `src/wire.rs` instead: `WireStreamEvent` is
//! `#[non_exhaustive]`, so a `match` written out here would be forced to carry
//! a wildcard arm and a newly added variant would slip past it.

use rai_sdk::message::{ConversationTurn, Message, StreamEvent};
use rai_sdk::wire::{StreamAccumulator, WireError, WireErrorKind, WireStreamEvent};
use rai_sdk::{ProviderKind, WIRE_PROTOCOL_VERSION};

/// The `data:` payloads a server would write for a short generation.
///
/// Written out as literal strings rather than serialized from Rust values: the
/// point is to prove the crate can read bytes it did not just produce.
const TEXT_STREAM: &[&str] = &[
    r#"{"type":"message_start","protocol_version":1,"model":"gpt-4o-mini","provider":"openai"}"#,
    r#"{"type":"text_delta","text":"Rust "}"#,
    r#"{"type":"text_delta","text":"is "}"#,
    r#"{"type":"text_delta","text":"fast."}"#,
    r#"{"type":"usage","usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}"#,
    r#"{"type":"message_stop","finish_reason":"stop"}"#,
];

fn parse(payload: &str) -> WireStreamEvent {
    serde_json::from_str(payload).unwrap_or_else(|error| panic!("{payload} should parse: {error}"))
}

fn accumulate(payloads: &[&str]) -> Result<rai_sdk::Response, WireError> {
    let mut accumulator = StreamAccumulator::new();
    for payload in payloads {
        accumulator.push(parse(payload))?;
    }
    accumulator.finish()
}

// ── The full receiving loop ────────────────────────────────────────────────

#[test]
fn sse_payloads_reassemble_into_one_response() {
    let response = accumulate(TEXT_STREAM).expect("a well-formed stream should accumulate");

    assert_eq!(response.text(), "Rust is fast.");
    assert_eq!(response.model, "gpt-4o-mini");
    assert_eq!(response.provider, ProviderKind::OpenAI);
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn token_counts_survive_the_wire() {
    let response = accumulate(TEXT_STREAM).expect("a well-formed stream should accumulate");

    // Entitlement accounting on the server and the token readout in the client
    // both depend on this event arriving intact.
    let usage = response.usage.expect("the usage event should be preserved");
    assert_eq!(usage.prompt_tokens, Some(11));
    assert_eq!(usage.completion_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(15));
}

#[test]
fn the_opening_event_advertises_the_protocol_version() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .push(parse(TEXT_STREAM[0]))
        .expect("message_start should be absorbed");

    assert_eq!(accumulator.protocol_version(), Some(WIRE_PROTOCOL_VERSION));
}

#[test]
fn tool_calls_reassemble_from_their_fragments() {
    let response = accumulate(&[
        r#"{"type":"message_start","protocol_version":1,"model":"gpt-4o-mini","provider":"openai"}"#,
        r#"{"type":"tool_call_start","id":"call_1","name":"get_weather"}"#,
        r#"{"type":"tool_call_delta","id":"call_1","arguments":"{\"city\":"}"#,
        r#"{"type":"tool_call_delta","id":"call_1","arguments":"\"Paris\"}"}"#,
        r#"{"type":"tool_call_end","id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}"#,
        r#"{"type":"message_stop","finish_reason":"tool_use"}"#,
    ])
    .expect("a well-formed stream should accumulate");

    let tool_calls = &response.messages[0].tool_calls;
    assert_eq!(
        tool_calls.len(),
        1,
        "the assembled call should not double up"
    );
    assert_eq!(tool_calls[0].name, "get_weather");
    assert_eq!(tool_calls[0].arguments["city"], "Paris");
}

// ── Errors vs. dropped connections ─────────────────────────────────────────

#[test]
fn a_provider_error_arrives_as_a_typed_event() {
    let error = accumulate(&[
        TEXT_STREAM[0],
        r#"{"type":"text_delta","text":"I can"}"#,
        r#"{"type":"error","error":{"kind":"content_filtered","message":"content filtered by openai: policy","provider":"openai","retryable":false}}"#,
    ])
    .expect_err("an error event should surface as an error");

    assert_eq!(error.kind, WireErrorKind::ContentFiltered);
    assert_eq!(error.provider, Some(ProviderKind::OpenAI));
    assert!(!error.retryable);
    assert!(error.message.contains("policy"));
}

#[test]
fn a_retryable_error_says_so() {
    let error = accumulate(&[
        TEXT_STREAM[0],
        r#"{"type":"error","error":{"kind":"rate_limit","message":"slow down","provider":"anthropic","retryable":true}}"#,
    ])
    .expect_err("an error event should surface as an error");

    assert_eq!(error.kind, WireErrorKind::RateLimit);
    assert!(error.retryable);
}

#[test]
fn a_dropped_connection_is_not_mistaken_for_a_clean_finish() {
    // The whole reason errors travel as events: without a terminal event, the
    // only honest conclusion is that the network died.
    let error = accumulate(&TEXT_STREAM[..3]).expect_err("a truncated stream should not succeed");

    assert_eq!(error.kind, WireErrorKind::Stream);
    assert!(
        error.message.contains("terminal event"),
        "the truncation message should explain itself: {}",
        error.message
    );
}

#[test]
fn an_unknown_error_kind_does_not_break_an_older_client() {
    // Forward compatibility: a server on a newer rai-sdk may report a category
    // this build has no variant for. It must still parse.
    let error = accumulate(&[
        TEXT_STREAM[0],
        r#"{"type":"error","error":{"kind":"budget_exhausted","message":"out of credits","retryable":false}}"#,
    ])
    .expect_err("an error event should surface as an error");

    assert_eq!(error.kind, WireErrorKind::Other("budget_exhausted".into()));
    assert_eq!(error.kind.as_str(), "budget_exhausted");
    assert_eq!(error.to_string(), "budget_exhausted: out of credits");
}

#[test]
fn an_unknown_event_type_is_refused_rather_than_guessed() {
    // Documented behavior, pinned here so it cannot change silently: an event
    // type this build does not know fails to parse. Callers are expected to log
    // and skip such a payload; quietly inventing a meaning for it would be
    // worse than ignoring it.
    let result = serde_json::from_str::<WireStreamEvent>(
        r#"{"type":"reasoning_delta","text":"thinking..."}"#,
    );

    assert!(result.is_err(), "an unknown event type should not parse");
}

// ── Interop with the in-process event type ─────────────────────────────────

#[test]
fn high_level_events_survive_a_trip_through_json() {
    let events = vec![
        StreamEvent::TextDelta {
            text: "hello".to_string(),
        },
        StreamEvent::ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"city":"Paris"}"#.to_string(),
        },
        StreamEvent::ToolResult {
            id: "call_1".to_string(),
            result: r#"{"celsius":21}"#.to_string(),
        },
        StreamEvent::TurnComplete {
            turn: ConversationTurn {
                user_message: Message::user("Weather?"),
                assistant_message: Message::assistant("Sunny."),
                tool_results: Vec::new(),
            },
        },
    ];

    for event in events {
        let json = serde_json::to_string(&WireStreamEvent::from(event.clone()))
            .expect("the wire event should serialize");
        let parsed: WireStreamEvent = serde_json::from_str(&json).expect("it should parse back");
        let restored = StreamEvent::try_from(parsed).expect("it should convert back");

        assert_eq!(restored, event, "lossy for {json}");
    }
}

#[test]
fn framing_events_have_no_high_level_equivalent() {
    let unrepresentable = [
        WireStreamEvent::message_start("gpt-4o-mini", ProviderKind::OpenAI),
        WireStreamEvent::MessageStop {
            finish_reason: None,
        },
        WireStreamEvent::Usage {
            usage: Default::default(),
        },
    ];

    for event in unrepresentable {
        let tag = event.tag();
        let error = StreamEvent::try_from(event).expect_err("{tag} is wire-only");
        assert_eq!(error.tag, tag);
    }
}

// ── SSE naming ─────────────────────────────────────────────────────────────

#[test]
fn the_tag_is_usable_as_an_sse_event_name() {
    // Servers name the SSE `event:` field from `tag()`, so it has to match the
    // discriminant a client dispatches on.
    for payload in TEXT_STREAM {
        let event = parse(payload);
        let json: serde_json::Value = serde_json::from_str(payload).expect("valid JSON");
        assert_eq!(json["type"], event.tag());
    }
}

#[test]
fn only_the_last_event_is_terminal() {
    let (last, rest) = TEXT_STREAM.split_last().expect("the stream is not empty");

    for payload in rest {
        assert!(
            !parse(payload).is_terminal(),
            "{payload} should not end the stream"
        );
    }
    assert!(parse(last).is_terminal(), "{last} should end the stream");
}
