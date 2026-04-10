import re

with open("src/provider/anthropic.rs", "r") as f:
    content = f.read()

# 1. Update `generate_stream` to call the new stream parser
impl_stream = """    pub async fn generate_stream(
        &self,
        model: &AnthropicModel,
        prompt: &Prompt,
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, None);
        let url = format!("{}/messages", self.base_url);

        debug!(url = %url, "Sending streaming request to Anthropic");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "Anthropic streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(Self::parse_provider_sse_stream(byte_stream)))
    }"""

content = re.sub(r'    pub async fn generate_stream\([\s\S]*?unimplemented!\(\)\n    }', impl_stream, content)

# 2. Add `parse_provider_sse_stream`
parse_func = """    fn parse_provider_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut buffer = String::new();

        async_stream::stream! {
            tokio::pin!(byte_stream);
            let mut current_tool_id: Option<String> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(event_end) = buffer.find("\\n\\n") {
                            let event = buffer[..event_end].to_string();
                            buffer = buffer[event_end + 2..].to_string();

                            let mut event_type = String::new();
                            let mut event_data = String::new();

                            for line in event.lines() {
                                if let Some(t) = line.strip_prefix("event: ") {
                                    event_type = t.to_string();
                                } else if let Some(d) = line.strip_prefix("data: ") {
                                    event_data = d.to_string();
                                }
                            }

                            match event_type.as_str() {
                                "content_block_start" => {
                                    if let Ok(start) = serde_json::from_str::<ContentBlockStart>(&event_data) {
                                        if start.content_block.r#type == "tool_use" {
                                            if let (Some(id), Some(name)) = (start.content_block.id, start.content_block.name) {
                                                current_tool_id = Some(id.clone());
                                                yield Ok(crate::provider::ProviderStreamEvent::ToolCallStart {
                                                    id,
                                                    name,
                                                });
                                            }
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(&event_data) {
                                        if delta.delta.r#type == "text_delta" {
                                            if let Some(text) = delta.delta.text {
                                                if !text.is_empty() {
                                                    yield Ok(crate::provider::ProviderStreamEvent::Text(text));
                                                }
                                            }
                                        } else if delta.delta.r#type == "input_json_delta" {
                                            if let Some(partial) = delta.delta.partial_json {
                                                if let Some(id) = &current_tool_id {
                                                    yield Ok(crate::provider::ProviderStreamEvent::ToolCallChunk {
                                                        id: id.clone(),
                                                        arguments: partial,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                "message_delta" => {
                                    if let Ok(delta) = serde_json::from_str::<MessageDelta>(&event_data) {
                                        let usage = delta.usage.map(|u| Usage {
                                            prompt_tokens: None,
                                            completion_tokens: Some(u.output_tokens),
                                            total_tokens: None,
                                        });

                                        if delta.delta.stop_reason.is_some() || usage.is_some() {
                                            yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                finish_reason: delta.delta.stop_reason,
                                                usage,
                                            });
                                        }
                                    }
                                }
                                "message_start" => {
                                    if let Ok(start) = serde_json::from_str::<MessageStart>(&event_data) {
                                        let usage = Usage {
                                            prompt_tokens: Some(start.message.usage.input_tokens),
                                            completion_tokens: Some(start.message.usage.output_tokens),
                                            total_tokens: Some(start.message.usage.input_tokens + start.message.usage.output_tokens),
                                        };
                                        // Anthropic splits usage between message_start (input) and message_delta (output)
                                        // But we can just send an early Done or wait for message_delta to send final usage.
                                    }
                                }
                                "error" => {
                                    if let Ok(error) = serde_json::from_str::<StreamError>(&event_data) {
                                        yield Err(Error::Request {
                                            provider: ProviderKind::Anthropic,
                                            message: error.error.message,
                                        });
                                        return;
                                    }
                                }
                                _ => {
                                    debug!(event_type = %event_type, "Ignoring SSE event");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(Error::Stream(e.to_string()));
                        return;
                    }
                }
            }
        }
    }"""

content = content.replace("    fn parse_sse_stream", parse_func + "\n    fn parse_sse_stream")

# Replace structs
structs = """#[derive(Debug, Deserialize)]
struct ContentBlockStart {
    content_block: ContentBlockStartDetail,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartDetail {
    r#type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDelta {
    delta: ContentDelta,
}

#[derive(Debug, Deserialize)]
struct ContentDelta {
    r#type: String,
    text: Option<String>,
    partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    delta: MessageDeltaContent,
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageStart {
    message: MessageStartDetail,
}

#[derive(Debug, Deserialize)]
struct MessageStartDetail {
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaContent {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    output_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    error: AnthropicErrorDetail,
}
"""

content = re.sub(r'#\[derive\(Debug, Deserialize\)\]\nstruct ContentBlockDelta \{[\s\S]*?error: AnthropicErrorDetail,\n\}', structs, content)

# Remove unused parse_sse_stream entirely
content = re.sub(r'    fn parse_sse_stream[\s\S]*?        }\n    }\n', '', content)

with open("src/provider/anthropic.rs", "w") as f:
    f.write(content)

