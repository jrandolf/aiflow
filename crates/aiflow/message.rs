use serde::{Deserialize, Serialize};

/// Represents a message exchanged in the AI conversation, including its role and content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Unique identifier for the message.
    pub id: String,
    /// The role of the message sender (developer, user, assistant).
    pub role: Role,
    /// The content parts of the message (text, tool, or error).
    pub parts: Vec<Part>,
}

impl Message {
    /// Returns an iterator over mutable tool calls in the message parts.
    pub fn tool_calls(&mut self) -> impl Iterator<Item = &mut ToolCall> {
        self.parts.iter_mut().filter_map(|part| {
            if let &mut Part::Tool(ref mut tool) = part {
                Some(&mut tool.tool)
            } else {
                None
            }
        })
    }
}

/// The role of a message sender in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// Message from the developer.
    Developer,
    /// Message from the user.
    User,
    /// Message from the assistant.
    Assistant,
}

/// A part of a message, which can be text, a tool call, or an error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Part {
    /// Text content part.
    Text(TextPart),
    /// Tool call part.
    Tool(ToolPart),
    /// Error part. Only occurs when the stream fails.
    Error(ErrorPart),
}

/// Text content for a message part.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextPart {
    /// The text content.
    pub text: String,
}

/// Tool call content for a message part.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPart {
    /// The tool call details.
    pub tool: ToolCall,
}

/// Represents a tool call, including its arguments and result.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for the tool call.
    pub id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Arguments for the tool call.
    pub args: serde_json::Value,
    /// Optional result of the tool call.
    pub result: Option<serde_json::Value>,
}

/// Represents an error part in a message, wrapping an error value.
#[derive(Debug)]
pub struct ErrorPart {
    /// The error value.
    pub error: anyhow::Error,
}

impl Clone for ErrorPart {
    fn clone(&self) -> Self {
        Self {
            error: anyhow::Error::from_boxed(self.error.to_string().into()),
        }
    }
}

impl PartialEq for ErrorPart {
    fn eq(&self, other: &Self) -> bool {
        self.error.to_string() == other.error.to_string()
    }
}

impl Eq for ErrorPart {}

impl Serialize for ErrorPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.error.to_string())
    }
}

impl<'de> Deserialize<'de> for ErrorPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            error: anyhow::Error::from_boxed(String::deserialize(deserializer)?.into()),
        })
    }
}
