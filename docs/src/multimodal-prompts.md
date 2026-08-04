# Multimodal prompts

A plain string prompt is shorthand. Underneath, a request carries a [`Prompt`](https://docs.rs/rai-sdk/latest/rai_sdk/message/struct.Prompt.html): a sequence of [`Message`](https://docs.rs/rai-sdk/latest/rai_sdk/message/struct.Message.html) values, each with a role and either simple text or a list of content blocks.

## Roles and messages

```rust
use rai_sdk::{Message, Prompt};

let prompt = Prompt::single(Message::system("You are a terse Rust expert."))
    .with_message(Message::user("Why does the borrow checker reject this?"));

assert_eq!(prompt.system_message(), Some("You are a terse Rust expert."));
```

`Prompt::single` starts from one message, `Prompt::new` takes a whole `Vec<Message>`, and `with_message` appends. A `Vec<Message>` also converts directly with `.into()`, so you rarely need to name `Prompt` at all.

Roles are `System`, `User`, `Assistant`, and `Tool`. Providers differ in how they handle system prompts — Anthropic takes it as a separate top-level field rather than a message — and the SDK handles that translation, so you can express it as a message consistently.

## Multi-turn conversations

Build history by listing messages in order:

```rust
use rai_sdk::{Message, Prompt};

let prompt = Prompt::new(vec![
    Message::user("What is a lifetime?"),
    Message::assistant("A lifetime names how long a reference is valid."),
    Message::user("Show me a case where elision fails."),
]);

assert_eq!(prompt.messages.len(), 3);
```

If you already have the previous turns as [`ConversationTurn`](https://docs.rs/rai-sdk/latest/rai_sdk/message/struct.ConversationTurn.html) values, `Prompt::with_history` expands each turn into its user message, assistant message, and tool results for you — or use `generate_with_history` on the request builder.

## Images

Use `Message::user_multimodal` with content blocks:

```rust,no_run
use rai_sdk::{ClientBuilder, ContentBlock, Message, Model, Prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .from_env()
        .model(Model::gpt4o_mini())
        .build()?;

    let prompt = Prompt::single(Message::user_multimodal(vec![
        ContentBlock::text("Describe this image in one sentence."),
        ContentBlock::image_url("https://example.com/image.png"),
    ]));

    let response = client.request().prompt(prompt).generate().await?;
    println!("{}", response.text());
    Ok(())
}
```

Order is meaningful: put the instruction text before or after the image deliberately, since models attend to position.

### URL versus inline data

[`ImageSource`](https://docs.rs/rai-sdk/latest/rai_sdk/message/enum.ImageSource.html) supports either a URL or base64 data with a media type. Use a URL when the provider can reach it, and base64 for local or private images that must travel in the request body.

## Content block types

`ContentBlock` covers `Text`, `Image`, `Audio`, `Video`, and `File`.

**Provider support is uneven, and this is the most important caveat in this chapter.** OpenAI and OpenRouter currently serialize image content. The other block types exist in the common prompt model, but provider-specific serialization may be incomplete — a block a provider does not support may simply not reach the model rather than producing a loud error.

Verify behavior for your provider and model before relying on audio, video, or file blocks in production. Choose a model that documents support for the modality you need; multimodal capability is per-model, not per-provider.

## Checking a prompt

```rust
use rai_sdk::{ContentBlock, Message, Prompt};

let prompt = Prompt::single(Message::user_multimodal(vec![
    ContentBlock::text("Hello"),
    ContentBlock::image_url("https://example.com/image.png"),
]));

assert!(prompt.is_multimodal());
```

`Prompt::system_message()` returns the first system message, if any.
