//! The OpenAI-compatible provider: local and self-hosted endpoints.
//!
//! What is worth pinning down here is not that Chat Completions parsing works —
//! that is shared with the OpenAI provider and covered by its own tests — but
//! the things this provider does differently:
//!
//! - the endpoint is per client, so several can coexist in one process;
//! - authentication is optional, and absent means *no header at all*;
//! - model identifiers are free-form strings the endpoint chose;
//! - a capability the endpoint lacks is a distinct, typed error rather than an
//!   HTTP failure a caller would have to string-match.
//!
//! Every test is offline: endpoints are local `wiremock` servers, and no test
//! reads or requires a credential.
//!
//! The provider rides the `openai` feature, which is also what gates the
//! request builder and stream parser it reuses.
#![cfg(feature = "openai")]

mod common;

use common::{
    Script, Step, collect_events, data_event, describe_event, expect_error,
    openai_compatible_builder, received_header, received_json_bodies, request_count, sse_body,
    stream_text,
};
use rai_sdk::{
    Capability, EndpointCapabilities, Error, JsonSchema, Model, ProviderKind, Tool, ToolContext,
    config::OLLAMA_BASE_URL,
};
use serde::Deserialize;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

// ── Fixtures ───────────────────────────────────────────────────────────────

/// A model identifier in the shape a local runtime uses: no vendor prefix, a
/// tag, and characters no provider catalog would enumerate.
const LOCAL_MODEL: &str = "llama3.1:8b";

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    value: String,
}

/// A minimal structured-output target for the `response_format` tests.
#[derive(Debug, Deserialize, JsonSchema)]
struct Summary {
    title: String,
}

fn echo_tool() -> Tool {
    Tool::new("echo")
        .description("Echo the input back")
        .handler(
            |args: EchoArgs, _ctx: ToolContext| async move { Ok(json!({ "value": args.value })) },
        )
        .expect("echo tool should build")
}

fn chat_completion(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": LOCAL_MODEL,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14 }
    })
}

/// A mock endpoint that answers every chat completion with `content`.
async fn endpoint_replying(content: &str) -> MockServer {
    endpoint_with(Step::ok(chat_completion(content))).await
}

/// A mock endpoint that answers every chat completion with one canned step.
async fn endpoint_with(step: Step) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(Script::new(vec![step]))
        .mount(&server)
        .await;
    server
}

// ── Chat ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_local_endpoint_answers_a_chat_request() {
    let server = endpoint_replying("Ownership moves values.").await;
    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let response = client
        .request()
        .prompt("Explain ownership.")
        .generate()
        .await
        .expect("the endpoint should answer");

    assert_eq!(response.text(), "Ownership moves values.");
    assert_eq!(response.provider, ProviderKind::OpenAICompatible);
    assert_eq!(response.model, LOCAL_MODEL);
    assert_eq!(
        response.usage.as_ref().and_then(|usage| usage.total_tokens),
        Some(14)
    );

    // The free-form identifier reaches the endpoint verbatim: no catalog
    // lookup, no normalization, no vendor prefix.
    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies[0]["model"], LOCAL_MODEL);
}

#[tokio::test]
async fn the_endpoint_is_configured_per_client_not_from_the_environment() {
    // `ollama()` is shorthand for the default local endpoint. Asserting on the
    // resolved config keeps this offline — nothing is dialed.
    let client = openai_compatible_builder("http://ignored.invalid/v1")
        .ollama()
        .build()
        .expect("client should build");

    assert_eq!(
        client.config().openai_compatible_base_url(),
        Some(OLLAMA_BASE_URL.to_string())
    );
    assert!(client.is_provider_available(ProviderKind::OpenAICompatible));

    // No endpoint configured means no provider, rather than a default one.
    let bare = rai_sdk::ClientBuilder::new()
        .build()
        .expect("client should build");
    assert!(!bare.is_provider_available(ProviderKind::OpenAICompatible));
}

// ── Authentication ─────────────────────────────────────────────────────────

#[tokio::test]
async fn no_authorization_header_is_sent_when_no_key_is_configured() {
    let server = endpoint_replying("hi").await;
    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    client
        .request()
        .prompt("hello")
        .generate()
        .await
        .expect("an unauthenticated endpoint should answer");

    // Not "Bearer " with an empty or placeholder token — absent entirely.
    assert_eq!(received_header(&server, 0, "authorization").await, None);
}

#[tokio::test]
async fn a_configured_key_is_sent_as_a_bearer_token() {
    let server = endpoint_replying("hi").await;
    let client = openai_compatible_builder(&server.uri())
        .openai_compatible_key("test-gateway-key")
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    client
        .request()
        .prompt("hello")
        .generate()
        .await
        .expect("the endpoint should answer");

    assert_eq!(
        received_header(&server, 0, "authorization").await,
        Some("Bearer test-gateway-key".to_string())
    );
}

// ── Several endpoints in one process ───────────────────────────────────────

#[tokio::test]
async fn clients_with_different_base_urls_coexist() {
    let laptop = endpoint_replying("from the laptop").await;
    let cluster = endpoint_replying("from the cluster").await;

    let laptop_client = openai_compatible_builder(&laptop.uri())
        .model(Model::openai_compatible("llama3.1:8b"))
        .build()
        .expect("client should build");
    let cluster_client = openai_compatible_builder(&cluster.uri())
        .openai_compatible_key("cluster-key")
        .model(Model::openai_compatible("Qwen/Qwen2.5-7B-Instruct"))
        .build()
        .expect("client should build");

    // Interleaved, and with the second client built before the first is used,
    // so a shared or last-writer-wins endpoint would be caught.
    let (from_laptop, from_cluster) = tokio::join!(
        laptop_client.request().prompt("who are you?").generate(),
        cluster_client.request().prompt("who are you?").generate(),
    );

    assert_eq!(
        from_laptop.expect("laptop endpoint should answer").text(),
        "from the laptop"
    );
    assert_eq!(
        from_cluster.expect("cluster endpoint should answer").text(),
        "from the cluster"
    );

    // Each endpoint saw exactly its own request, with its own model and
    // credentials.
    assert_eq!(request_count(&laptop).await, 1);
    assert_eq!(request_count(&cluster).await, 1);
    assert_eq!(
        received_json_bodies(&laptop).await[0]["model"],
        "llama3.1:8b"
    );
    assert_eq!(
        received_json_bodies(&cluster).await[0]["model"],
        "Qwen/Qwen2.5-7B-Instruct"
    );
    assert_eq!(received_header(&laptop, 0, "authorization").await, None);
    assert_eq!(
        received_header(&cluster, 0, "authorization").await,
        Some("Bearer cluster-key".to_string())
    );
}

// ── Streaming ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn streaming_reuses_the_chat_completions_parser() {
    let server = endpoint_with(Step::sse(sse_body(&[
        &data_event(json!({ "choices": [{ "delta": { "content": "tick " }, "index": 0 }] })),
        &data_event(json!({ "choices": [{ "delta": { "content": "tock" }, "index": 0 }] })),
        &data_event(json!({
            "choices": [{ "delta": {}, "finish_reason": "stop", "index": 0 }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6 }
        })),
        "data: [DONE]",
    ])))
    .await;

    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let stream = client
        .request()
        .prompt("count")
        .stream()
        .await
        .expect("the endpoint should stream");
    let events = collect_events(stream).await;

    assert_eq!(stream_text(&events), "tick tock");
    assert_eq!(
        describe_event(events.last().expect("a terminal event")),
        "done:stop:4/2/6"
    );

    // Streaming asks for usage; endpoints that ignore the hint are handled by
    // the test below.
    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies[0]["stream"], true);
    assert_eq!(bodies[0]["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn streaming_tolerates_a_server_that_omits_the_space_the_sentinel_and_the_usage() {
    // Three divergences at once, all seen from self-hosted servers: `data:`
    // with no space after the colon (the SSE spec makes it optional), no
    // `[DONE]` sentinel, and no usage despite `include_usage`.
    let server = endpoint_with(Step::sse(sse_body(&[
        &format!(
            "data:{}",
            json!({ "choices": [{ "delta": { "content": "terse" }, "index": 0 }] })
        ),
        &format!(
            "data:{}",
            json!({ "choices": [{ "delta": {}, "finish_reason": "stop", "index": 0 }] })
        ),
    ])))
    .await;

    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let response = client
        .request()
        .prompt("be terse")
        .stream_accumulated()
        .await
        .expect("a sentinel-less stream should still complete");

    assert_eq!(response.text(), "terse");
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    assert!(
        response.usage.is_none(),
        "an endpoint that reports no usage should not have usage invented for it"
    );
}

// ── Capability degradation ─────────────────────────────────────────────────

#[tokio::test]
async fn a_declared_capability_gap_fails_before_the_request_is_sent() {
    let server = endpoint_replying("unreachable").await;
    let client = openai_compatible_builder(&server.uri())
        .openai_compatible_capabilities(EndpointCapabilities::default().with_tool_calling(false))
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .tool(echo_tool())
            .prompt("use the tool")
            .generate()
            .await,
    );

    assert_capability_error(&error, Capability::ToolCalling);
    assert_eq!(
        request_count(&server).await,
        0,
        "a declared gap should be caught locally, not by the endpoint"
    );
}

#[tokio::test]
async fn an_endpoint_that_refuses_tools_produces_a_typed_capability_error() {
    // The shape Ollama returns for a model with no tool support.
    let server = endpoint_with(Step::json(
        400,
        json!({
            "error": {
                "message": "registry.ollama.ai/library/llama3.1:8b does not support tools",
                "type": "invalid_request_error"
            }
        }),
    ))
    .await;

    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .tool(echo_tool())
            .prompt("use the tool")
            .generate()
            .await,
    );

    assert_capability_error(&error, Capability::ToolCalling);
    assert!(
        error.to_string().contains("does not support tools"),
        "the endpoint's own message should survive: {error}"
    );
}

#[tokio::test]
async fn an_endpoint_that_refuses_response_format_names_structured_output() {
    let server = endpoint_with(Step::json(
        400,
        json!({ "error": { "message": "unsupported parameter: response_format" } }),
    ))
    .await;

    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let error = expect_error(
        client
            .request()
            .prompt("summarize")
            .generate_structured::<Summary>()
            .await,
    );

    assert_capability_error(&error, Capability::StructuredOutput);
}

#[tokio::test]
async fn an_unrelated_rejection_stays_an_ordinary_request_error() {
    // Same status, same vocabulary, but the request never asked for tools —
    // so this must not be mistaken for a capability gap and send the caller
    // down a permanent fallback path.
    let server = endpoint_with(Step::json(
        400,
        json!({ "error": { "message": "unknown model 'llama3.1:8b'" } }),
    ))
    .await;

    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let error = expect_error(client.request().prompt("hello").generate().await);

    assert_eq!(error.unsupported_capability(), None);
    assert!(
        matches!(error, Error::InvalidRequest(_)),
        "expected InvalidRequest, got: {error:?}"
    );
}

#[tokio::test]
async fn structured_output_asks_for_a_non_strict_schema() {
    let server = endpoint_replying(r#"{"title":"ok"}"#).await;
    let client = openai_compatible_builder(&server.uri())
        .model(Model::openai_compatible(LOCAL_MODEL))
        .build()
        .expect("client should build");

    let structured = client
        .request()
        .prompt("summarize")
        .generate_structured::<Summary>()
        .await
        .expect("the endpoint should answer");

    assert_eq!(structured.output.title, "ok");

    // OpenAI's strict mode is a contract third-party endpoints implement
    // unevenly; the schema is enforced client-side instead.
    let bodies = received_json_bodies(&server).await;
    assert_eq!(bodies[0]["response_format"]["type"], "json_schema");
    assert_eq!(bodies[0]["response_format"]["json_schema"]["strict"], false);
}

/// Assert an error is the typed capability failure for `capability`.
///
/// The point of the variant is that a caller can branch on it without matching
/// a status code or grepping a message, so that is what is asserted: the
/// classification, the provider, and that it is not treated as transient.
fn assert_capability_error(error: &Error, capability: Capability) {
    assert_eq!(
        error.unsupported_capability(),
        Some(capability),
        "expected a {capability} capability error, got: {error:?}"
    );
    assert_eq!(error.provider(), Some(ProviderKind::OpenAICompatible));
    assert_eq!(error.kind_str(), "capability_unsupported");
    assert!(
        !error.is_retryable(),
        "a missing capability will not appear on a retry"
    );
}
