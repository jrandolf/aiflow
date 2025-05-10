# aiflow

[![crates.io](https://img.shields.io/crates/v/aiflow?style=flat-square)](https://crates.io/crates/aiflow)
[![license](https://img.shields.io/crates/l/aiflow?style=flat-square)](https://github.com/jrandolf/aiflow)
[![ci](https://img.shields.io/github/actions/workflow/status/jrandolf/aiflow/ci.yaml?label=ci&style=flat-square)](https://github.com/jrandolf/aiflow/actions/workflows/ci.yaml)
[![docs](https://img.shields.io/docsrs/aiflow?style=flat-square)](https://docs.rs/aiflow/latest/aiflow/)

**aiflow** is a Rust library for AI message streaming and tool integration, designed for use with OpenAI-compatible models. It provides abstractions for managing conversations, tool calls, and message streaming, making it easy to build advanced AI-driven applications.

## Features

- **AI Message Streaming**: Stream responses from OpenAI-compatible models in real time.
- **Tool Integration**: Define and register custom tools that the AI can call during conversations.
- **Session and Usage Tracking**: Track token usage and session costs.
- **Extensible**: Easily add new tools and extend message handling.
- **Strongly Typed**: Leverages Rust's type system for safety and clarity.

## Usage

Add `aiflow` as a dependency in your `Cargo.toml` (see [crates/aiflow/Cargo.toml](crates/aiflow/Cargo.toml) for details). Example usage:

```rust
use aiflow::{GenerateConfig, Message, Session, responses_stream, tool};
use serde_json::json;
use tool::SetExt as _;
use tool::ToolBuilder;
use tool::extract::{Args, Context, Id};

// Create a session and messages
let mut session = Session::default();
let messages = vec![/* ... */];
let tools = tool::Set::default();
let config = Some(GenerateConfig::default());

// Create a tool.
fn say_hello() -> anyhow::Result<Value> {
    println!("hello");
    Ok(json!({ "success": true }))
}
tools.add(
    ToolBuilder::default()
        .name("say_hello")
        .executor(say_hello)
        .build()
        .expect("to build tool"),
);

// Create a tool with extractors.
fn say_hello_to_someone(
    Id(tool_call_id): Id,
    Args(name): Args<String>,
) -> anyhow::Result<Value> {
    println!("hello {name}");
    Ok(json!({ "success": true }))
}
tools.add(
    ToolBuilder::default()
        .name("say_hello")
        .executor(say_hello)
        .parameters::<String>()
        .build()
        .expect("to build tool"),
);

// Create a tool with a context.
fn say_hello_to_someone_with_context(
    Id(tool_call_id): Id,
    Args(name): Args<String>,
    Context(greeting): Context<&'static str>,
) -> anyhow::Result<Value> {
    println!("{greeting} {name}", greeting = greeting.as_str());
    Ok(json!({ "success": true }))
}
tools.add(
    ToolBuilder::default()
        .name("say_hello")
        .executor(say_hello)
        .parameters::<String>()
        .context("hello")
        .build()
        .expect("to build tool"),
);

// Create a client tool. You will need to modify the tool call
// yourself and set the result.
tools.add(
    ToolBuilder::default()
        .name("say_hello")
        .build()
        .expect("to build tool"),
);

// Stream AI responses
tokio::spawn(async move {
    let mut stream = responses_stream(&mut session, &messages, tools, config);
    while let Some(response) = stream.next().await {
        // Handle each streamed message
    }
});
```

## Development

- Requires Rust 2024 or later.
- Uses [tokio](https://tokio.rs/) for async runtime.
- Linting and formatting are enforced via Clippy and rustfmt.

### Running Tests

```
cargo test --workspace
```

### Linting

```
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
