import re

with open("src/provider/openai.rs", "r") as f:
    content = f.read()

old_func = """    fn parse_provider_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut buffer = String::new();

        async_stream::stream! {
            tokio::pin!(byte_stream);

            while let Some(chunk_result) = byte_stream.next().await {"""

new_func = """    fn parse_provider_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<crate::provider::ProviderStreamEvent>>
    where
        S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
    {
        let mut buffer = String::new();

        async_stream::stream! {
            tokio::pin!(byte_stream);
            let mut current_tool_id: Option<String> = None;

            while let Some(chunk_result) = byte_stream.next().await {"""

content = content.replace(old_func, new_func)

old_tc = """                                                    for tc in tool_calls {
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
                                                    }"""

new_tc = """                                                    for tc in tool_calls {
                                                        if let Some(id) = &tc.id {
                                                            current_tool_id = Some(id.clone());
                                                            let name = tc.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default();
                                                            yield Ok(crate::provider::ProviderStreamEvent::ToolCallStart {
                                                                id: id.clone(),
                                                                name,
                                                            });
                                                        }
                                                        if let Some(func) = &tc.function {
                                                            if let Some(args) = &func.arguments {
                                                                let id_to_use = current_tool_id.clone().unwrap_or_default();
                                                                yield Ok(crate::provider::ProviderStreamEvent::ToolCallChunk {
                                                                    id: id_to_use,
                                                                    arguments: args.clone(),
                                                                });
                                                            }
                                                        }
                                                    }"""

content = content.replace(old_tc, new_tc)

# Also remove `parse_sse_stream` since it's unused now.
content = re.sub(r'    fn parse_sse_stream[\s\S]*?        }\n    }', '', content)

with open("src/provider/openai.rs", "w") as f:
    f.write(content)

