//! Prompts, messages, multimodal content blocks, and response accessors.
//!
//! The serialization assertions matter because `Message` and `ContentBlock` are
//! part of the public API surface: callers persist conversation history as JSON
//! and expect it to deserialize again on the next release.

use rai_sdk::{ContentBlock, ImageSource, Message, Prompt, Response, Role, Usage};

fn to_json<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("value should serialize")
}

// ── Roles ──────────────────────────────────────────────────────────────────

#[test]
fn role_as_str_and_serde_agree_on_lowercase_names() {
    let cases = [
        (Role::System, "system"),
        (Role::User, "user"),
        (Role::Assistant, "assistant"),
        (Role::Tool, "tool"),
    ];

    for (role, expected) in cases {
        assert_eq!(role.as_str(), expected);
        assert_eq!(to_json(&role), serde_json::json!(expected));
        assert_eq!(
            serde_json::from_value::<Role>(serde_json::json!(expected)).expect("deserialize role"),
            role
        );
    }
}

// ── Message constructors ───────────────────────────────────────────────────

#[test]
fn message_constructors_set_the_expected_role_and_content() {
    let system = Message::system("Be concise.");
    assert_eq!(system.role, Role::System);
    assert_eq!(system.content, "Be concise.");
    assert!(!system.is_multimodal());
    assert!(!system.has_tool_calls());
    assert!(system.tool_call_id.is_none());
    assert!(!system.tool_error);

    let user = Message::user("Hello");
    assert_eq!(user.role, Role::User);
    assert_eq!(user.content, "Hello");

    let assistant = Message::assistant("Hi there");
    assert_eq!(assistant.role, Role::Assistant);
    assert_eq!(assistant.content, "Hi there");
}

#[test]
fn tool_result_messages_carry_the_call_id_and_error_flag() {
    let ok = Message::tool("{\"sum\":5}", "call_1");
    assert_eq!(ok.role, Role::Tool);
    assert_eq!(ok.tool_call_id.as_deref(), Some("call_1"));
    assert!(!ok.tool_error);

    let failed = Message::tool_error("{\"error\":\"boom\"}", "call_2");
    assert_eq!(failed.role, Role::Tool);
    assert_eq!(failed.tool_call_id.as_deref(), Some("call_2"));
    assert!(failed.tool_error);
}

#[test]
fn assistant_with_tool_calls_reports_and_preserves_the_calls() {
    let message = Message::assistant_with_tool_calls(
        "Looking that up",
        vec![rai_sdk::ToolCall {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            arguments: serde_json::json!({ "city": "Paris" }),
        }],
    );

    assert!(message.has_tool_calls());
    assert_eq!(message.tool_calls.len(), 1);
    assert_eq!(message.tool_calls[0].name, "get_weather");
    assert_eq!(message.tool_calls[0].arguments["city"], "Paris");
    // Text content and tool calls coexist on one assistant message.
    assert_eq!(message.text_content(), "Looking that up");
}

// ── Multimodal content ─────────────────────────────────────────────────────

#[test]
fn content_block_constructors_serialize_to_internally_tagged_json() {
    assert_eq!(
        to_json(&ContentBlock::text("hello")),
        serde_json::json!({ "type": "text", "text": "hello" })
    );

    assert_eq!(
        to_json(&ContentBlock::image_url("https://example.com/cat.png")),
        serde_json::json!({
            "type": "image",
            "source": { "type": "url", "url": "https://example.com/cat.png" }
        })
    );

    assert_eq!(
        to_json(&ContentBlock::image_base64("image/png", "aGk=")),
        serde_json::json!({
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": "aGk=" }
        })
    );
}

#[test]
fn audio_video_and_file_blocks_serialize_with_their_own_tags() {
    let cases = [
        (
            to_json(&ContentBlock::audio_url("https://example.com/a.mp3")),
            "audio",
            serde_json::json!({ "type": "url", "url": "https://example.com/a.mp3" }),
        ),
        (
            to_json(&ContentBlock::audio_base64("audio/mpeg", "YQ==")),
            "audio",
            serde_json::json!({ "type": "base64", "media_type": "audio/mpeg", "data": "YQ==" }),
        ),
        (
            to_json(&ContentBlock::video_url("https://example.com/v.mp4")),
            "video",
            serde_json::json!({ "type": "url", "url": "https://example.com/v.mp4" }),
        ),
        (
            to_json(&ContentBlock::video_base64("video/mp4", "Yg==")),
            "video",
            serde_json::json!({ "type": "base64", "media_type": "video/mp4", "data": "Yg==" }),
        ),
        (
            to_json(&ContentBlock::file_url("https://example.com/doc.pdf")),
            "file",
            serde_json::json!({ "type": "url", "url": "https://example.com/doc.pdf" }),
        ),
        (
            to_json(&ContentBlock::file_base64("application/pdf", "Yw==")),
            "file",
            serde_json::json!({ "type": "base64", "media_type": "application/pdf", "data": "Yw==" }),
        ),
    ];

    for (json, expected_type, expected_source) in cases {
        assert_eq!(json["type"], expected_type);
        assert_eq!(json["source"], expected_source);
    }
}

#[test]
fn content_blocks_round_trip_through_json() {
    let blocks = vec![
        ContentBlock::text("describe this"),
        ContentBlock::image_url("https://example.com/cat.png"),
        ContentBlock::image_base64("image/jpeg", "ZGF0YQ=="),
        ContentBlock::audio_url("https://example.com/a.mp3"),
        ContentBlock::video_base64("video/mp4", "ZGF0YQ=="),
        ContentBlock::file_url("https://example.com/doc.pdf"),
    ];

    let encoded = to_json(&blocks);
    let decoded: Vec<ContentBlock> =
        serde_json::from_value(encoded.clone()).expect("content blocks should round-trip");

    assert_eq!(to_json(&decoded), encoded);
}

#[test]
fn image_source_variants_are_distinguished_by_their_type_tag() {
    let url = ImageSource::Url {
        url: "https://example.com/cat.png".to_string(),
    };
    let base64 = ImageSource::Base64 {
        media_type: "image/png".to_string(),
        data: "aGk=".to_string(),
    };

    assert_eq!(to_json(&url)["type"], "url");
    assert_eq!(to_json(&base64)["type"], "base64");
    assert_eq!(to_json(&base64)["media_type"], "image/png");
}

#[test]
fn multimodal_text_content_joins_only_the_text_blocks() {
    let message = Message::user_multimodal(vec![
        ContentBlock::text("First"),
        ContentBlock::image_url("https://example.com/cat.png"),
        ContentBlock::audio_url("https://example.com/a.mp3"),
        ContentBlock::text("Second"),
    ]);

    assert!(message.is_multimodal());
    assert_eq!(message.role, Role::User);
    // Non-text blocks are skipped; remaining text is newline-joined.
    assert_eq!(message.text_content(), "First\nSecond");
}

#[test]
fn multimodal_messages_drop_the_plain_content_field() {
    // `content` is populated but `content_blocks` takes precedence, and the
    // empty string is skipped during serialization.
    let message = Message::user_multimodal(vec![ContentBlock::text("only blocks")]);
    let json = to_json(&message);

    assert!(
        json.get("content").is_none(),
        "empty `content` should be omitted, got {json}"
    );
    assert_eq!(json["content_blocks"][0]["text"], "only blocks");
}

#[test]
fn message_serialization_omits_every_empty_optional_field() {
    let json = to_json(&Message::user("hi"));
    let object = json.as_object().expect("message serializes to an object");

    // `serde_json::Map` is ordered, so compare the key set rather than the
    // emission order.
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["content", "role"],
        "a plain text message should serialize only `role` and `content`"
    );
    assert_eq!(object["role"], "user");
    assert_eq!(object["content"], "hi");
}

#[test]
fn tool_error_flag_is_only_serialized_when_true() {
    let ok = to_json(&Message::tool("result", "call_1"));
    assert!(
        ok.get("tool_error").is_none(),
        "a successful tool result should not serialize `tool_error`, got {ok}"
    );

    let failed = to_json(&Message::tool_error("result", "call_1"));
    assert_eq!(failed["tool_error"], true);
}

// ── Prompts ────────────────────────────────────────────────────────────────

#[test]
fn prompt_constructors_and_conversions_produce_equivalent_prompts() {
    assert_eq!(Prompt::single(Message::user("hi")).messages.len(), 1);
    assert_eq!(
        Prompt::new(vec![Message::system("s"), Message::user("u")])
            .messages
            .len(),
        2
    );

    let from_str: Prompt = "hi".into();
    assert_eq!(from_str.messages.len(), 1);
    assert_eq!(from_str.messages[0].role, Role::User);
    assert_eq!(from_str.messages[0].content, "hi");

    let from_string: Prompt = "hi".to_string().into();
    assert_eq!(from_string.messages[0].content, "hi");

    let from_message: Prompt = Message::assistant("a").into();
    assert_eq!(from_message.messages[0].role, Role::Assistant);

    let from_vec: Prompt = vec![Message::user("a"), Message::user("b")].into();
    assert_eq!(from_vec.messages.len(), 2);
}

#[test]
fn prompt_push_and_with_message_append_in_order() {
    let mut prompt = Prompt::single(Message::system("s"));
    prompt.push_message(Message::user("u1"));
    let prompt = prompt.with_message(Message::assistant("a1"));

    let roles: Vec<Role> = prompt.messages.iter().map(|m| m.role).collect();
    assert_eq!(roles, vec![Role::System, Role::User, Role::Assistant]);
}

#[test]
fn prompt_system_message_returns_the_first_system_message_only() {
    let prompt = Prompt::new(vec![
        Message::user("before"),
        Message::system("first"),
        Message::system("second"),
    ]);

    assert_eq!(prompt.system_message(), Some("first"));
    assert_eq!(prompt.conversation_messages().len(), 1);
    assert_eq!(prompt.conversation_messages()[0].content, "before");
}

#[test]
fn prompt_without_a_system_message_reports_none() {
    let prompt = Prompt::single(Message::user("hi"));
    assert_eq!(prompt.system_message(), None);
    assert_eq!(prompt.conversation_messages().len(), 1);
}

#[test]
fn prompt_is_multimodal_when_any_message_has_content_blocks() {
    let text_only = Prompt::new(vec![Message::user("a"), Message::assistant("b")]);
    assert!(!text_only.is_multimodal());

    let mixed = Prompt::new(vec![
        Message::user("a"),
        Message::user_multimodal(vec![ContentBlock::image_url("https://example.com/c.png")]),
    ]);
    assert!(mixed.is_multimodal());
}

#[test]
fn prompt_round_trips_through_json_with_multimodal_and_tool_messages() {
    let prompt = Prompt::new(vec![
        Message::system("Be concise."),
        Message::user_multimodal(vec![
            ContentBlock::text("What is this?"),
            ContentBlock::image_url("https://example.com/cat.png"),
        ]),
        Message::assistant_with_tool_calls(
            String::new(),
            vec![rai_sdk::ToolCall {
                id: "call_1".to_string(),
                name: "classify".to_string(),
                arguments: serde_json::json!({ "url": "https://example.com/cat.png" }),
            }],
        ),
        Message::tool("{\"label\":\"cat\"}", "call_1"),
    ]);

    let encoded = to_json(&prompt);
    let decoded: Prompt =
        serde_json::from_value(encoded.clone()).expect("prompt should round-trip");

    assert_eq!(to_json(&decoded), encoded);
    assert_eq!(decoded.messages.len(), 4);
    assert_eq!(decoded.system_message(), Some("Be concise."));
    assert!(decoded.messages[1].is_multimodal());
    assert!(decoded.messages[2].has_tool_calls());
    assert_eq!(decoded.messages[3].tool_call_id.as_deref(), Some("call_1"));
}

// ── Responses and usage ────────────────────────────────────────────────────

fn response_with(messages: Vec<Message>) -> Response {
    Response {
        messages,
        usage: None,
        model: "gpt-4o-mini".to_string(),
        provider: rai_sdk::ProviderKind::OpenAI,
        finish_reason: Some("stop".to_string()),
    }
}

#[test]
fn response_text_reads_the_first_message() {
    let response = response_with(vec![
        Message::assistant("first"),
        Message::assistant("second"),
    ]);

    // `Response::text()` deliberately reads only the first message rather than
    // concatenating every message in the response.
    assert_eq!(response.text(), "first");
}

#[test]
fn response_text_accumulates_text_blocks_of_a_multimodal_message() {
    let response = response_with(vec![Message::user_multimodal(vec![
        ContentBlock::text("part one"),
        ContentBlock::image_url("https://example.com/cat.png"),
        ContentBlock::text("part two"),
    ])]);

    assert_eq!(response.text(), "part one\npart two");
}

#[test]
fn response_text_is_empty_when_there_are_no_messages() {
    assert_eq!(response_with(Vec::new()).text(), "");
}

#[test]
fn usage_defaults_to_all_unknown_and_round_trips() {
    let default = Usage::default();
    assert_eq!(default.prompt_tokens, None);
    assert_eq!(default.completion_tokens, None);
    assert_eq!(default.total_tokens, None);

    let usage = Usage {
        prompt_tokens: Some(11),
        completion_tokens: Some(7),
        total_tokens: Some(18),
    };
    let decoded: Usage = serde_json::from_value(to_json(&usage)).expect("usage should round-trip");
    assert_eq!(decoded.prompt_tokens, Some(11));
    assert_eq!(decoded.completion_tokens, Some(7));
    assert_eq!(decoded.total_tokens, Some(18));
}

#[test]
fn usage_accepts_partially_reported_token_counts() {
    // Anthropic streaming only reports output tokens, so partial usage must
    // deserialize rather than fail.
    let usage: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": null,
        "completion_tokens": 5,
        "total_tokens": null
    }))
    .expect("partial usage should deserialize");

    assert_eq!(usage.prompt_tokens, None);
    assert_eq!(usage.completion_tokens, Some(5));
    assert_eq!(usage.total_tokens, None);
}

#[test]
fn response_round_trips_through_json() {
    let response = Response {
        messages: vec![Message::assistant("hi")],
        usage: Some(Usage {
            prompt_tokens: Some(3),
            completion_tokens: Some(1),
            total_tokens: Some(4),
        }),
        model: "claude-sonnet-4-6".to_string(),
        provider: rai_sdk::ProviderKind::Anthropic,
        finish_reason: Some("end_turn".to_string()),
    };

    let encoded = to_json(&response);
    assert_eq!(encoded["provider"], "anthropic");

    let decoded: Response =
        serde_json::from_value(encoded).expect("response should deserialize again");
    assert_eq!(decoded.text(), "hi");
    assert_eq!(decoded.model, "claude-sonnet-4-6");
    assert_eq!(decoded.finish_reason.as_deref(), Some("end_turn"));
    assert_eq!(
        decoded.usage.and_then(|usage| usage.total_tokens),
        Some(4),
        "usage should survive the round-trip"
    );
}
