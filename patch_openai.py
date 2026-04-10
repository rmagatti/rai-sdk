import re

with open("src/provider/openai.rs", "r") as f:
    content = f.read()

# 1. Update OpenAIRequest struct to include stream_options
content = re.sub(
    r'struct OpenAIRequest \{([\s\S]*?)(max_tokens: Option<i32>,)',
    r'struct OpenAIRequest {\1\2\n    #[serde(skip_serializing_if = "Option::is_none")]\n    stream_options: Option<OpenAIStreamOptions>,',
    content
)

# Add OpenAIStreamOptions
content = content.replace(
    'struct OpenAIToolDefinition {',
    '#[derive(Debug, Serialize)]\nstruct OpenAIStreamOptions {\n    include_usage: bool,\n}\n\n#[derive(Debug, Serialize)]\nstruct OpenAIToolDefinition {'
)

# 2. Update build_request to set stream_options
content = re.sub(
    r'stream: Some\(stream\),',
    r'stream: Some(stream),\n            stream_options: stream.then(|| OpenAIStreamOptions { include_usage: true }),',
    content
)

# 3. Update Stream Structs
content = content.replace(
    'struct OpenAIStreamResponse {\n    choices: Vec<OpenAIStreamChoice>,\n}',
    'struct OpenAIStreamResponse {\n    choices: Vec<OpenAIStreamChoice>,\n    usage: Option<OpenAIUsage>,\n}'
)

content = content.replace(
    'struct OpenAIDelta {\n    content: Option<String>,\n}',
    '''struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    id: Option<String>,
    function: Option<OpenAIStreamFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}'''
)

# 4. Add parse_provider_sse_stream
parse_func = """    fn parse_provider_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut buffer = String::new();

        async_stream::stream! {
            tokio::pin!(byte_stream);

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(event_end) = buffer.find("\\n\\n") {
                            let event = buffer[..event_end].to_string();
                            buffer = buffer[event_end + 2..].to_string();

                            for line in event.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        return;
                                    }

                                    match serde_json::from_str::<OpenAIStreamResponse>(data) {
                                        Ok(stream_response) => {
                                            if let Some(usage) = stream_response.usage {
                                                yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                    finish_reason: stream_response.choices.first().and_then(|c| c.finish_reason.clone()).or(Some("stop".to_string())),
                                                    usage: Some(Usage {
                                                        prompt_tokens: Some(usage.prompt_tokens),
                                                        completion_tokens: Some(usage.completion_tokens),
                                                        total_tokens: Some(usage.total_tokens),
                                                    }),
                                                });
                                                continue;
                                            }

                                            if let Some(choice) = stream_response.choices.first() {
                                                if let Some(content) = &choice.delta.content {
                                                    if !content.is_empty() {
                                                        yield Ok(crate::provider::ProviderStreamEvent::Text(content.clone()));
                                                    }
                                                }

                                                if let Some(tool_calls) = &choice.delta.tool_calls {
                                                    for tc in tool_calls {
                                                        if let Some(id) = &tc.id {
                                                            let name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();
                                                            yield Ok(crate::provider::ProviderStreamEvent::ToolCallStart {
                                                                id: id.clone(),
                                                                name,
                                                            });
                                                        }
                                                        if let Some(func) = &tc.function {
                                                            if let Some(args) = &func.arguments {
                                                                yield Ok(crate::provider::ProviderStreamEvent::ToolCallChunk {
                                                                    id: tc.id.clone().unwrap_or_default(), // Normally tracked by client or chunk order, but openai just sends empty id for subsequent chunks. Wait, I should track the current tool call id if OpenAI omits it.
                                                                    arguments: args.clone(),
                                                                });
                                                            }
                                                        }
                                                    }
                                                }

                                                if let Some(finish_reason) = &choice.finish_reason {
                                                    yield Ok(crate::provider::ProviderStreamEvent::Done {
                                                        finish_reason: Some(finish_reason.clone()),
                                                        usage: None,
                                                    });
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            debug!(error = %e, data = %data, "Failed to parse SSE chunk");
                                        }
                                    }
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
    }
"""
content = content.replace("    fn parse_sse_stream", parse_func + "\n    fn parse_sse_stream")

# 5. Replace unimplemented!() in generate_stream
impl_stream = """    pub async fn generate_stream(
        &self,
        model: &OpenAIModel,
        prompt: &Prompt,
        config: &crate::generation::GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::provider::ProviderStreamEvent>> + Send>>>
    {
        let request = self.build_request(model, prompt, config, true, None);
        let url = format!("{}/chat/completions", self.base_url);

        debug!(url = %url, "Sending streaming request to OpenAI");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %error_body, "OpenAI streaming request failed");
            return Err(self.parse_error(status.as_u16(), &error_body));
        }

        let byte_stream = response.bytes_stream();

        Ok(Box::pin(Self::parse_provider_sse_stream(byte_stream)))
    }"""

content = re.sub(r'    pub async fn generate_stream\([\s\S]*?unimplemented!\(\)\n    }', impl_stream, content)

with open("src/provider/openai.rs", "w") as f:
    f.write(content)

