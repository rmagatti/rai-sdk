# Introduction

`rai-sdk` is a Rust SDK for backend AI workflows. It puts OpenAI, Anthropic, and OpenRouter behind one typed client, so changing model or provider does not mean rewriting request construction, tool plumbing, or stream parsing.

## What it solves

Each provider has its own request shape, its own tool-calling protocol, and its own streaming event format. Writing directly against them means three code paths that drift apart. `rai-sdk` normalizes all three behind a single API while keeping provider-specific escape hatches available.

It also handles the parts that are tedious to get right:

- **Tool loops.** When a model asks to call a tool, something has to execute it, append the result, and ask the model to continue — until it stops asking. `generate()` does that for you.
- **Structured output.** Getting valid JSON out of a model is not the same as getting JSON that matches your type. The SDK generates a JSON Schema from your Rust type, validates the response against it, then deserializes.
- **Transient failures.** Rate limits and timeouts are normal, not exceptional. Retries with exponential backoff and jitter are built in.
- **Stream parsing.** Server-sent events arrive split across arbitrary byte boundaries. The SDK buffers and reassembles them correctly.

## Supported providers

| Provider | Feature flag | Notes |
| --- | --- | --- |
| OpenAI | `openai` | Chat Completions API |
| Anthropic | `anthropic` | Messages API |
| OpenRouter | `openrouter` | Aggregates many vendors behind an OpenAI-compatible API |

All three are enabled by default. See [Installation](./installation.md) to compile only what you need.

## Design notes

- **Typestate builders.** `generate()` does not exist as a method until the request has both a prompt and a model. Incomplete requests are a compile error, not a runtime one.
- **Explicit `_once` variants.** Methods ending in `_once` make exactly one provider call and never execute registered tools. The plain variants run the full loop. The distinction is in the name rather than a boolean argument.
- **Errors carry provenance.** [`Error`](https://docs.rs/rai-sdk/latest/rai_sdk/enum.Error.html) records which provider failed and exposes category helpers such as `is_retryable()` and `is_auth_error()`, so callers can branch on kind instead of matching every variant.

## Project status

Early and pre-1.0. The crate works, but the public API may change in breaking ways before `1.0`. Pin an exact version if you need stability.

## Where to go next

- [Installation](./installation.md) — add the crate and pick feature flags.
- [Quickstart](./quickstart.md) — a working request in a few lines.
- [API reference on docs.rs](https://docs.rs/rai-sdk) — every public item.
