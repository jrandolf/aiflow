use openai_responses::types::{
    ContentInput, FunctionCall, FunctionCallOutput, InputItem, InputListItem, InputMessage, Role,
    Tool as ResponsesTool,
};

use crate::{Message, Tool, message};

/// Conversion from `Message` to a list of OpenAI-compatible `InputListItem`s.
impl TryFrom<Message> for Vec<InputListItem> {
    type Error = anyhow::Error;

    fn try_from(val: Message) -> Result<Self, Self::Error> {
        let mut items = Self::new();
        for part in val.parts {
            match part {
                message::Part::Text(text_part) => {
                    items.push(match val.role {
                        message::Role::Developer => InputListItem::Message(InputMessage {
                            role: Role::Developer,
                            content: ContentInput::Text(text_part.text),
                        }),
                        message::Role::User => InputListItem::Message(InputMessage {
                            role: Role::User,
                            content: ContentInput::Text(text_part.text),
                        }),
                        message::Role::Assistant => InputListItem::Message(InputMessage {
                            role: Role::Assistant,
                            content: ContentInput::Text(text_part.text),
                        }),
                    });
                }
                message::Part::Tool(tool_part) => {
                    if val.role != message::Role::Assistant {
                        return Err(anyhow::anyhow!("Tool part must be an assistant message"));
                    }
                    items.extend(Self::from(tool_part));
                }
                message::Part::Error(error_part) => {
                    if val.role != message::Role::Developer {
                        return Err(anyhow::anyhow!("Error part must be a developer message"));
                    }
                    items.push(error_part.into());
                }
            }
        }

        Ok(items)
    }
}

/// Conversion from `ToolPart` to a list of OpenAI-compatible `InputListItem`s.
impl From<message::ToolPart> for Vec<InputListItem> {
    fn from(val: message::ToolPart) -> Self {
        val.tool.into()
    }
}

/// Conversion from `ToolCall` to a list of OpenAI-compatible `InputListItem`s.
impl From<message::ToolCall> for Vec<InputListItem> {
    fn from(val: message::ToolCall) -> Self {
        let mut items = vec![InputListItem::Item(InputItem::FunctionCall(FunctionCall {
            call_id: val.id.clone(),
            name: val.name,
            arguments: val.args.to_string(),
            id: None,
            status: None,
        }))];

        if let Some(result) = val.result {
            items.push(InputListItem::Item(InputItem::FunctionCallOutput(
                FunctionCallOutput {
                    call_id: val.id,
                    output: result.to_string(),
                    id: None,
                    status: None,
                },
            )));
        }

        items
    }
}

/// Conversion from `ErrorPart` to an OpenAI-compatible `InputListItem`.
impl From<message::ErrorPart> for InputListItem {
    fn from(val: message::ErrorPart) -> Self {
        Self::Message(InputMessage {
            role: Role::Developer,
            content: ContentInput::Text(val.error.to_string()),
        })
    }
}

/// Conversion from `Tool` to an OpenAI-compatible tool definition.
impl From<&Tool> for ResponsesTool {
    fn from(val: &Tool) -> Self {
        Self::Function {
            name: val.name().to_owned(),
            parameters: val.parameters().clone().into(),
            strict: true,
            description: Some(val.description().to_owned()),
        }
    }
}
