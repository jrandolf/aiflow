use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestDeveloperMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionToolArgs,
    ChatCompletionToolType, FunctionCall, FunctionObjectArgs,
};

use crate::{Message, Tool, message};

/// Conversion from `Message` to a list of OpenAI-compatible chat completion request messages.
impl TryFrom<Message> for Vec<ChatCompletionRequestMessage> {
    type Error = anyhow::Error;

    fn try_from(val: Message) -> Result<Self, Self::Error> {
        let mut messages = Self::new();
        let mut tool_messages = Vec::new();
        for part in val.parts {
            match part {
                message::Part::Text(text_part) => {
                    if !tool_messages.is_empty() {
                        for tool_message in core::mem::take(&mut tool_messages) {
                            messages.push(ChatCompletionRequestMessage::Tool(tool_message));
                        }
                    }
                    messages.push(match val.role {
                        message::Role::Developer => {
                            ChatCompletionRequestMessage::Developer(text_part.into())
                        }
                        message::Role::User => ChatCompletionRequestMessage::User(text_part.into()),
                        message::Role::Assistant => {
                            ChatCompletionRequestMessage::Assistant(text_part.into())
                        }
                    });
                }
                message::Part::Tool(tool_part) => {
                    if val.role != message::Role::Assistant {
                        return Err(anyhow::anyhow!("Tool part must be an assistant message"));
                    }
                    let (tool_call, tool_message) = tool_part.into();

                    // Merge consecutive tool calls.
                    if let Some(&mut ChatCompletionRequestMessage::Assistant(
                        ref mut assistant_message,
                    )) = messages.last_mut()
                    {
                        assistant_message
                            .tool_calls
                            .get_or_insert_default()
                            .push(tool_call);
                    } else {
                        messages.push(ChatCompletionRequestMessage::Assistant(
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .tool_calls(vec![tool_call])
                                .build()?,
                        ));
                    }

                    if let Some(tool_message) = tool_message {
                        tool_messages.push(tool_message);
                    }
                }
                message::Part::Error(error_part) => {
                    if val.role != message::Role::Developer {
                        return Err(anyhow::anyhow!("Error part must be a developer message"));
                    }
                    if !tool_messages.is_empty() {
                        for tool_message in core::mem::take(&mut tool_messages) {
                            messages.push(ChatCompletionRequestMessage::Tool(tool_message));
                        }
                    }
                    messages.push(ChatCompletionRequestMessage::Developer(error_part.into()));
                }
            }
        }
        if !tool_messages.is_empty() {
            for tool_message in core::mem::take(&mut tool_messages) {
                messages.push(ChatCompletionRequestMessage::Tool(tool_message));
            }
        }
        Ok(messages)
    }
}

/// Macro to implement `From` for `TextPart` for various OpenAI message types.
#[macro_export]
macro_rules! impl_from_text_part {
    ($target_type:ty, $args_type:ty) => {
        impl From<message::TextPart> for $target_type {
            fn from(val: message::TextPart) -> Self {
                <$args_type>::default()
                    .content(val.text)
                    .build()
                    .expect("failed to build text part")
            }
        }
    };
}

impl_from_text_part!(
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs
);

impl_from_text_part!(
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestDeveloperMessageArgs
);

impl_from_text_part!(
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageArgs
);

/// Conversion from `ToolPart` to OpenAI tool call and optional tool message.
impl From<message::ToolPart>
    for (
        ChatCompletionMessageToolCall,
        Option<ChatCompletionRequestToolMessage>,
    )
{
    fn from(val: message::ToolPart) -> Self {
        val.tool.into()
    }
}

/// Conversion from `ToolCall` to OpenAI tool call and optional tool message.
impl From<message::ToolCall>
    for (
        ChatCompletionMessageToolCall,
        Option<ChatCompletionRequestToolMessage>,
    )
{
    fn from(val: message::ToolCall) -> Self {
        (
            ChatCompletionMessageToolCall {
                id: val.id.clone(),
                r#type: ChatCompletionToolType::Function,
                function: FunctionCall {
                    name: val.name,
                    arguments: val.args.to_string(),
                },
            },
            if let Some(result) = val.result {
                Some(
                    ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(val.id)
                        .content(ChatCompletionRequestToolMessageContent::Text(
                            result.to_string(),
                        ))
                        .build()
                        .expect("failed to build tool message"),
                )
            } else {
                None
            },
        )
    }
}

/// Conversion from `ErrorPart` to an OpenAI developer message.
impl From<message::ErrorPart> for ChatCompletionRequestDeveloperMessage {
    fn from(val: message::ErrorPart) -> Self {
        ChatCompletionRequestDeveloperMessageArgs::default()
            .content(val.error.to_string())
            .build()
            .expect("failed to build error part")
    }
}

/// Conversion from `Tool` to an OpenAI chat completion tool definition.
impl From<&Tool> for ChatCompletionTool {
    fn from(val: &Tool) -> Self {
        FunctionObjectArgs::default()
            .name(val.name().to_owned())
            .description(val.description().to_owned())
            .parameters(val.parameters().clone())
            .strict(true)
            .build()
            .and_then(|function| {
                ChatCompletionToolArgs::default()
                    .r#type(ChatCompletionToolType::Function)
                    .function(function)
                    .build()
            })
            .expect("failed to build tool")
    }
}
