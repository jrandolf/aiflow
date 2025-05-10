//! AI message streaming and tool integration for OpenAI-compatible models.

extern crate alloc;

mod util;

pub mod message;
pub mod openai;
pub use message::Message;
pub mod tool;
use openai_responses::{
    StreamError,
    types::{Input, Request},
};
use tokio::{sync::Mutex, task::JoinSet};
pub use tool::{Tool, ToolBuilder};

use assert2::let_assert;
use async_openai::types::{
    ChatCompletionStreamOptions, ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs,
};
use bigdecimal::{BigDecimal, FromPrimitive as _};
use genawaiter::sync::Gen;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use alloc::collections::BTreeMap;
use alloc::collections::btree_map;
use alloc::sync::Arc;
use core::fmt::{self, Display};
use futures::{FutureExt as _, Stream, StreamExt as _};
use util::parse_incomplete_json;
use uuid::Uuid;

/// Tracks token usage for a session, including cached, input, and output tokens.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of cached input tokens.
    pub cached_input_tokens: BigDecimal,
    /// Number of input tokens.
    pub input_tokens: BigDecimal,
    /// Number of output tokens.
    pub output_tokens: BigDecimal,
}

/// Represents an AI session, tracking the cursor and cost.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Cursor for the previous response, if any.
    pub cursor: Option<String>,
    /// Total cost of the session.
    pub cost: BigDecimal,
}

/// Supported AI models for message generation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    /// GPT-4.1 model (default).
    #[default]
    #[serde(rename = "gpt-4.1")]
    Gpt4_1,
    /// GPT-4.1 Mini model.
    #[serde(rename = "gpt-4.1-mini")]
    Gpt4_1Mini,
    /// GPT-4.1 Nano model.
    #[serde(rename = "gpt-4.1-nano")]
    Gpt4_1Nano,
    /// O3 model.
    #[serde(rename = "o3")]
    O3,
    /// O4 Mini model.
    #[serde(rename = "o4-mini")]
    O4Mini,
}

impl Model {
    fn cost_per_input_token(self) -> BigDecimal {
        match self {
            Self::Gpt4_1 => BigDecimal::from_f64(2.0),
            Self::Gpt4_1Mini => BigDecimal::from_f64(0.4),
            Self::Gpt4_1Nano => BigDecimal::from_f64(0.1),
            Self::O3 => BigDecimal::from_f64(10.0),
            Self::O4Mini => BigDecimal::from_f64(1.1),
        }
        .expect("failed to convert to bigdecimal")
            / BigDecimal::from_f64(1_000_000.0).expect("failed to convert to bigdecimal")
    }

    fn cost_per_cached_input_token(self) -> BigDecimal {
        match self {
            Self::Gpt4_1 => BigDecimal::from_f64(0.5),
            Self::Gpt4_1Mini => BigDecimal::from_f64(0.1),
            Self::Gpt4_1Nano => BigDecimal::from_f64(0.025),
            Self::O3 => BigDecimal::from_f64(2.5),
            Self::O4Mini => BigDecimal::from_f64(0.275),
        }
        .expect("failed to convert to bigdecimal")
            / BigDecimal::from_f64(1_000_000.0).expect("failed to convert to bigdecimal")
    }

    fn cost_per_output_token(self) -> BigDecimal {
        match self {
            Self::Gpt4_1 => BigDecimal::from_f64(8.0),
            Self::Gpt4_1Mini => BigDecimal::from_f64(1.6),
            Self::Gpt4_1Nano => BigDecimal::from_f64(0.4),
            Self::O3 => BigDecimal::from_f64(40.0),
            Self::O4Mini => BigDecimal::from_f64(4.4),
        }
        .expect("failed to convert to bigdecimal")
            / BigDecimal::from_f64(1_000_000.0).expect("failed to convert to bigdecimal")
    }

    #[must_use]
    pub fn cost(self, usage: &Usage) -> BigDecimal {
        self.cost_per_input_token() * usage.input_tokens.clone()
            + self.cost_per_cached_input_token() * usage.cached_input_tokens.clone()
            + self.cost_per_output_token() * usage.output_tokens.clone()
    }
}

impl Display for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Gpt4_1 => write!(formatter, "gpt-4.1"),
            Self::Gpt4_1Mini => write!(formatter, "gpt-4.1-mini"),
            Self::Gpt4_1Nano => write!(formatter, "gpt-4.1-nano"),
            Self::O3 => write!(formatter, "o3"),
            Self::O4Mini => write!(formatter, "o4-mini"),
        }
    }
}

pub mod config {
    use serde::{Deserialize, Serialize};

    /// Tool selection strategy for AI message generation.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub enum ToolChoice {
        /// Let the model choose tools automatically (default).
        #[default]
        Auto,
        /// Require tool usage.
        Required,
        /// Do not use tools.
        None,
    }
}

/// Configuration for generating AI messages.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateConfig {
    /// Model to use for generation.
    pub model: Model,
    /// Tool selection strategy.
    pub tool_choice: config::ToolChoice,
}

/// Streams AI-generated messages based on the input messages and tools using the Responses API.
///
/// # Arguments
///
/// * `session` - Mutable reference to the session state.
/// * `messages` - Vector of input messages to process.
/// * `tools` - Set of tools available for the AI to use.
/// * `config` - Optional configuration for message generation.
///
/// # Returns
///
/// A stream of `Result<Arc<Mutex<Message>>>` items representing the AI's responses.
///
/// # Panics
///
/// This function may panic if the `OpenAI` API key is invalid or if there are
/// issues with the tool configuration.
pub fn responses_stream(
    session: &mut Session,
    messages: &[Message],
    tools: tool::Set,
    config: Option<GenerateConfig>,
) -> impl Stream<Item = anyhow::Result<Arc<Mutex<Message>>>> {
    let config = config.unwrap_or_default();

    let openai = openai_responses::Client::from_env().expect("failed to create openai client");

    let thread = messages
        .iter()
        .cloned()
        .map(Message::try_into)
        .collect::<Result<Vec<Vec<_>>, _>>()
        .expect("to convert messages")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let tool_parameters = tools.values().map(Into::into).collect::<Vec<_>>();

    let model = config.model.to_string();

    Gen::new(|co| async move {
        let assistant_message = Arc::new(Mutex::new(Message {
            id: Uuid::now_v7().to_string(),
            role: message::Role::Assistant,
            parts: Vec::new(),
        }));
        co.yield_(Ok(Arc::clone(&assistant_message))).await;

        loop {
            let mut current_thread = thread.clone();
            {
                let assistant_message = assistant_message.lock().await;
                if !assistant_message.parts.is_empty() {
                    current_thread.extend(
                        Vec::try_from(assistant_message.clone()).expect("to convert message"),
                    );
                }
            }

            let request = Request::builder()
                .model(model.clone())
                .input(Input::List(current_thread.clone()))
                .previous_response_id_optional(session.cursor.clone())
                .tools(tool_parameters.clone())
                .tool_choice(match config.tool_choice {
                    config::ToolChoice::Auto => openai_responses::types::ToolChoice::Auto,
                    config::ToolChoice::Required => openai_responses::types::ToolChoice::Required,
                    config::ToolChoice::None => openai_responses::types::ToolChoice::None,
                })
                .parallel_tool_calls(false)
                .build();

            let mut stream = openai.stream(request);

            let mut deltas = BTreeMap::new();
            let mut tool_executions = JoinSet::new();

            while let Some(result) = stream.next().await {
                let event = match result {
                    Ok(response) => response,
                    Err(error) => {
                        if let StreamError::Stream(reqwest_eventsource::Error::InvalidStatusCode(
                            _,
                            response,
                        )) = error
                        {
                            co.yield_(Err(anyhow::anyhow!(
                                "Failed to create stream: {}",
                                response.json::<Value>().await.expect("to get json")
                            )))
                            .await;
                        } else {
                            co.yield_(Err(anyhow::anyhow!("Failed to create stream: {error}")))
                                .await;
                        }
                        return;
                    }
                };

                #[expect(clippy::wildcard_enum_match_arm, reason = "there are a lot of events")]
                match event {
                    openai_responses::types::Event::ResponseCompleted { response } => {
                        if let Some(previous_response_id) = response.previous_response_id {
                            session.cursor = Some(previous_response_id);
                        }
                        if let Some(responses_usage) = response.usage {
                            session.cost += config.model.cost(&Usage {
                                cached_input_tokens: responses_usage
                                    .input_tokens_details
                                    .cached_tokens
                                    .into(),
                                input_tokens: responses_usage
                                    .input_tokens
                                    .saturating_sub(
                                        responses_usage.input_tokens_details.cached_tokens,
                                    )
                                    .into(),
                                output_tokens: responses_usage.output_tokens.into(),
                            });
                        }
                    }
                    openai_responses::types::Event::OutputItemAdded {
                        item: openai_responses::types::OutputItem::FunctionCall(function_call),
                        output_index,
                    } => {
                        let mut assistant_message = assistant_message.lock().await;
                        deltas.insert(
                            (output_index, 0),
                            (function_call.arguments, assistant_message.parts.len()),
                        );
                        assistant_message
                            .parts
                            .push(message::Part::Tool(message::ToolPart {
                                tool: message::ToolCall {
                                    id: function_call.call_id,
                                    name: function_call.name,
                                    args: Value::Null,
                                    result: None,
                                },
                            }));
                    }
                    openai_responses::types::Event::ContentPartAdded {
                        part:
                            openai_responses::types::OutputContent::Text { text, .. }
                            | openai_responses::types::OutputContent::Refusal { refusal: text, .. },
                        content_index,
                        output_index,
                        ..
                    } => {
                        let mut assistant_message = assistant_message.lock().await;
                        deltas.insert(
                            (output_index, content_index),
                            (text.clone(), assistant_message.parts.len()),
                        );
                        assistant_message
                            .parts
                            .push(message::Part::Text(message::TextPart { text }));
                    }

                    openai_responses::types::Event::RefusalDelta {
                        delta,
                        content_index,
                        output_index,
                        ..
                    }
                    | openai_responses::types::Event::OutputTextDelta {
                        delta,
                        content_index,
                        output_index,
                        ..
                    } => {
                        let &mut (ref mut text, part_index) = deltas
                            .get_mut(&(output_index, content_index))
                            .expect("openai to correctly order events");
                        text.push_str(&delta);

                        let mut assistant_message = assistant_message.lock().await;
                        let part = assistant_message
                            .parts
                            .get_mut(part_index)
                            .expect("part to exist");
                        let_assert!(
                            &mut message::Part::Text(ref mut text_part) = part,
                        );
                        text.clone_into(&mut text_part.text);
                        drop(assistant_message);
                    }
                    openai_responses::types::Event::FunctionCallArgumentsDelta {
                        delta,
                        output_index,
                        ..
                    } => {
                        let &mut (ref mut args, part_index) = deltas
                            .get_mut(&(output_index, 0))
                            .expect("openai to correctly order events");
                        args.push_str(&delta);

                        let mut assistant_message = assistant_message.lock().await;
                        let part = assistant_message
                            .parts
                            .get_mut(part_index)
                            .expect("part to exist");
                        let_assert!(&mut message::Part::Tool(message::ToolPart { ref mut tool }) = part);
                        tool.args = parse_incomplete_json(args).unwrap_or_default();

                        let Some(tool_executor) = tools.get(&tool.name) else {
                            continue;
                        };
                        if !tool_executor.is_streamable() {
                            continue;
                        }

                        if let Some(future) =
                            tool_executor.execute(tool.id.clone(), tool.args.clone())
                        {
                            tool_executions.spawn(future.map(move |result| (part_index, result)));
                        }
                        drop(assistant_message);
                        continue;
                    }

                    // As soon as the function call arguments are done, we can execute tools, if they are available.
                    openai_responses::types::Event::FunctionCallArgumentsDone {
                        output_index,
                        ..
                    } => {
                        let &mut (_, part_index) = deltas
                            .get_mut(&(output_index, 0))
                            .expect("openai to correctly order events");

                        let mut assistant_message = assistant_message.lock().await;
                        let part = assistant_message
                            .parts
                            .get_mut(part_index)
                            .expect("part to exist");
                        let_assert!(
                            &mut message::Part::Tool(message::ToolPart {
                                ref mut tool
                            }) = part
                        );

                        let Some(tool_executor) = tools.get(&tool.name) else {
                            tool.result = Some(json!(format!("No such tool: {}", tool.name)));
                            continue;
                        };
                        if tool_executor.is_streamable() {
                            continue;
                        }

                        if let Some(future) =
                            tool_executor.execute(tool.id.clone(), tool.args.clone())
                        {
                            tool_executions.spawn(future.map(move |result| (part_index, result)));
                        }
                        drop(assistant_message);
                        continue;
                    }
                    _ => {
                        // Ignore other events
                        continue;
                    }
                }

                co.yield_(Ok(Arc::clone(&assistant_message))).await;
            }

            if tool_executions.is_empty() {
                return;
            }

            while let Some((part_index, result)) = tool_executions
                .join_next()
                .await
                .transpose()
                .expect("tool to be executed")
            {
                let result = match result {
                    Ok(result) => Some(result),
                    Err(error) => Some(json!(format!("Error: {error}"))),
                };
                let mut assistant_message = assistant_message.lock().await;
                let part = assistant_message
                    .parts
                    .get_mut(part_index)
                    .expect("part to exist");
                let_assert!(
                    &mut message::Part::Tool(message::ToolPart {
                        ref mut tool,
                    }) = part,
                );
                tool.result = result;
                drop(assistant_message);
            }

            co.yield_(Ok(Arc::clone(&assistant_message))).await;

            if assistant_message
                .lock()
                .await
                .tool_calls()
                .filter(|tool_call| tool_call.result.is_none())
                .count()
                > 0
            {
                // There are some client tool calls that need to be executed.
                return;
            }
        }
    })
}

/// Streams AI-generated messages based on the input messages and tools.
///
/// # Arguments
///
/// * `messages` - Vector of input messages to process
/// * `tools` - Set of tools available for the AI to use
/// * `config` - Optional configuration for message generation
///
/// # Returns
///
/// A stream of `Result<Message>` items representing the AI's responses
///
/// # Panics
///
/// This function may panic if the `OpenAI` API key is invalid or if there are
/// issues with the tool configuration.
pub fn stream(
    session: &mut Session,
    messages: &[Message],
    tools: tool::Set,
    config: Option<GenerateConfig>,
) -> impl Stream<Item = anyhow::Result<Arc<Mutex<Message>>>> {
    let config = config.unwrap_or_default();

    let openai = async_openai::Client::new();

    let thread = messages
        .iter()
        .cloned()
        .map(Message::try_into)
        .collect::<Result<Vec<Vec<_>>, _>>()
        .expect("to convert messages")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let tool_parameters = tools.values().map(Into::into).collect::<Vec<_>>();

    let model = config.model.to_string();

    Gen::new(|co| async move {
        let assistant_message = Arc::new(Mutex::new(Message {
            id: Uuid::now_v7().to_string(),
            role: message::Role::Assistant,
            parts: Vec::new(),
        }));
        co.yield_(Ok(Arc::clone(&assistant_message))).await;

        loop {
            let mut current_thread = thread.clone();
            {
                let assistant_message = assistant_message.lock().await;
                if !assistant_message.parts.is_empty() {
                    current_thread.extend(
                        Vec::try_from(assistant_message.clone()).expect("to convert message"),
                    );
                }
            }

            let request = CreateChatCompletionRequestArgs::default()
                .model(model.clone())
                .messages(current_thread.clone())
                .tools(tool_parameters.clone())
                .parallel_tool_calls(false)
                .tool_choice(match config.tool_choice {
                    config::ToolChoice::Auto => ChatCompletionToolChoiceOption::Auto,
                    config::ToolChoice::Required => ChatCompletionToolChoiceOption::Required,
                    config::ToolChoice::None => ChatCompletionToolChoiceOption::None,
                })
                .stream_options(ChatCompletionStreamOptions {
                    include_usage: true,
                })
                .build()
                .expect("failed to build request");

            let mut stream = match openai.chat().create_stream(request.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    co.yield_(Err(anyhow::anyhow!("Failed to create stream: {error}")))
                        .await;
                    return;
                }
            };

            let mut tool_deltas: BTreeMap<u32, (String, usize)> = BTreeMap::new();
            let mut tool_executions = JoinSet::new();

            while let Some(result) = stream.next().await {
                let response = match result {
                    Ok(response) => response,
                    Err(error) => {
                        co.yield_(Err(anyhow::anyhow!("Failed to create stream: {error}")))
                            .await;
                        return;
                    }
                };

                let is_last_chunk = response.choices.is_empty();
                if let Some(chat_choice) = response.choices.into_iter().next() {
                    // Accumulate text
                    if let Some(content) = chat_choice.delta.content {
                        if let Some(&mut message::Part::Text(ref mut text_part)) =
                            assistant_message.lock().await.parts.last_mut()
                        {
                            text_part.text.push_str(&content);
                        } else {
                            assistant_message
                                .lock()
                                .await
                                .parts
                                .push(message::Part::Text(message::TextPart { text: content }));
                        }
                    }

                    // Accumulate tool calls
                    if let Some(tool_call_deltas) = chat_choice.delta.tool_calls {
                        for tool_call_delta in tool_call_deltas {
                            match tool_deltas.entry(tool_call_delta.index) {
                                btree_map::Entry::Vacant(vacant_entry) => {
                                    let index = assistant_message.lock().await.parts.len();
                                    assistant_message
                                        .lock()
                                        .await
                                        .parts
                                        .push(message::Part::Tool(message::ToolPart {
                                            tool: message::ToolCall {
                                                id: tool_call_delta.id.unwrap_or_default(),
                                                name: tool_call_delta
                                                    .function
                                                    .expect("first part always has a function")
                                                    .name
                                                    .unwrap_or_default(),
                                                args: Value::Null,
                                                result: None,
                                            },
                                        }));
                                    vacant_entry.insert((String::new(), index));
                                }
                                btree_map::Entry::Occupied(mut occupied_entry) => {
                                    let &mut (ref mut args, part_index) = occupied_entry.get_mut();

                                    let mut assistant_message = assistant_message.lock().await;
                                    let part = assistant_message
                                        .parts
                                        .get_mut(part_index)
                                        .expect("part to exist");
                                    let_assert!(&mut message::Part::Tool(message::ToolPart { ref mut tool }) = part);
                                    if let Some(function) = tool_call_delta.function {
                                        args.push_str(&function.arguments.unwrap_or_default());
                                        tool.args = parse_incomplete_json(args).unwrap_or_default();
                                    }

                                    let Some(tool_executor) = tools.get(&tool.name) else {
                                        continue;
                                    };
                                    if !tool_executor.is_streamable() {
                                        continue;
                                    }

                                    if let Some(future) =
                                        tool_executor.execute(tool.id.clone(), tool.args.clone())
                                    {
                                        tool_executions
                                            .spawn(future.map(move |result| (part_index, result)));
                                    }
                                    drop(assistant_message);
                                }
                            }
                        }
                    }

                    co.yield_(Ok(Arc::clone(&assistant_message))).await;
                }

                if is_last_chunk {
                    if let Some(completion_usage) = response.usage {
                        let cached_input_tokens = completion_usage
                            .prompt_tokens_details
                            .and_then(|details| details.cached_tokens)
                            .unwrap_or_default();
                        session.cost += config.model.cost(&Usage {
                            cached_input_tokens: cached_input_tokens.into(),
                            input_tokens: completion_usage
                                .prompt_tokens
                                .saturating_sub(cached_input_tokens)
                                .into(),
                            output_tokens: completion_usage.completion_tokens.into(),
                        });
                    }
                }
            }

            for (_, (_, part_index)) in tool_deltas {
                let mut assistant_message = assistant_message.lock().await;
                let part = assistant_message
                    .parts
                    .get_mut(part_index)
                    .expect("part to exist");
                let_assert!(
                    &mut message::Part::Tool(message::ToolPart {
                        ref mut tool,
                    }) = part
                );

                let Some(tool_executor) = tools.get(&tool.name) else {
                    tool.result = Some(json!(format!("No such tool: {}", tool.name)));
                    continue;
                };
                if tool_executor.is_streamable() {
                    continue;
                }

                if let Some(future) = tool_executor.execute(tool.id.clone(), tool.args.clone()) {
                    tool_executions.spawn(future.map(move |result| (part_index, result)));
                }
                drop(assistant_message);
            }

            if tool_executions.is_empty() {
                return;
            }

            while let Some((part_index, result)) = tool_executions
                .join_next()
                .await
                .transpose()
                .expect("tool to be executed")
            {
                let result = match result {
                    Ok(result) => Some(result),
                    Err(error) => Some(json!(format!("Error: {error}"))),
                };
                let mut assistant_message = assistant_message.lock().await;
                let part = assistant_message
                    .parts
                    .get_mut(part_index)
                    .expect("part to exist");
                let_assert!(
                    &mut message::Part::Tool(message::ToolPart {
                        ref mut tool,
                    }) = part,
                );
                tool.result = result;
                drop(assistant_message);
            }

            co.yield_(Ok(Arc::clone(&assistant_message))).await;

            if assistant_message
                .lock()
                .await
                .tool_calls()
                .filter(|tool_call| tool_call.result.is_none())
                .count()
                > 0
            {
                // There are some client tool calls that need to be executed.
                return;
            }
        }
    })
}
