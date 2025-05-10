pub mod executor;
pub mod extract;

use core::any::Any;

use alloc::sync::Arc;

use derive_builder::Builder;
use executor::Executor;
use futures::{FutureExt as _, future::BoxFuture};
use rustc_hash::FxHashMap;
use schemars::Schema;
use schemars::transform::RecursiveTransform;
use schemars::{JsonSchema, schema_for, transform::Transform as _};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Represents a callable tool that can be used by the AI, including its name, description, parameters, and execution logic.
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct Tool {
    /// Name of the tool.
    #[builder(setter(into))]
    name: String,
    /// Description of the tool.
    #[builder(setter(into), default)]
    description: String,
    /// JSON schema for the tool's parameters.
    ///
    /// Can be left empty for tools that don't require parameters.
    #[builder(
        setter(custom),
        default = r#"schemars::json_schema!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
            "required": []
        })"#
    )]
    parameters: Schema,
    /// Whether the tool supports streaming.
    ///
    /// Tools that support streaming will be called for each chunk of tool output.
    #[builder(default)]
    stream: bool,
    /// Optional context for the tool.
    #[builder(setter(custom), default)]
    context: Option<Context>,
    /// Optional executor for the tool's logic.
    #[builder(setter(custom), default)]
    execute: Option<CallExecutor>,
}

impl Tool {
    /// Returns the name of the tool.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the description of the tool.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the JSON schema for the tool's parameters.
    #[must_use]
    pub const fn parameters(&self) -> &Schema {
        &self.parameters
    }

    /// Returns true if the tool supports streaming.
    #[must_use]
    pub const fn is_streamable(&self) -> bool {
        self.stream
    }

    /// Executes the tool with the given id and arguments, if an executor is set.
    #[must_use]
    pub fn execute(
        &self,
        id: String,
        args: Value,
    ) -> Option<BoxFuture<'static, anyhow::Result<Value>>> {
        self.execute.as_ref().map(|executor| {
            executor.execute((Call {
                context: self.context.clone(),
                id: Some(id),
                args: Some(args),
            },))
        })
    }
}

impl ToolBuilder {
    /// Sets the executor for the tool for automatic tooling.
    #[expect(private_bounds, reason = "internal")]
    #[must_use]
    pub fn executor<R, F, A, E>(mut self, executor: E) -> Self
    where
        R: Serialize,
        A: for<'re> TryFrom<&'re mut Call, Error = anyhow::Error> + Send,
        F: Future<Output = anyhow::Result<R>> + Send,
        E: Executor<A, Output = F> + Clone + Sync + Send + 'static,
    {
        self.execute = Some(Some(Arc::new(move |call_state| {
            let executor = executor.clone();
            (async move |executor: E, mut call_state| -> Result<Value, anyhow::Error> {
                let args = A::try_from(&mut call_state)?;
                let result = executor.execute(args).await?;
                Ok(serde_json::to_value(result)?)
            })(executor, call_state)
            .boxed()
        })));
        self
    }

    /// Sets the parameters schema for the tool using a type that implements `JsonSchema` and `DeserializeOwned`.
    #[must_use]
    pub fn parameters<P: JsonSchema + DeserializeOwned + Send>(mut self) -> Self {
        let mut transform = RecursiveTransform(|schema: &mut Schema| {
            if let Some(schema) = schema.as_object_mut() {
                schema.remove("$schema");
                schema.remove("title");
                if schema.get("type").is_some() {
                    schema.remove("format");
                }
            }
        });

        let mut parameter_schema = schema_for!(P);
        transform.transform(&mut parameter_schema);

        self.parameters = Some(parameter_schema);
        self
    }

    /// Sets the context for the tool.
    #[must_use]
    pub fn context<T: Any + Sync + Send>(mut self, context: T) -> Self {
        self.context = Some(Some(Arc::new(context)));
        self
    }
}

/// Type alias for a shared context object.
type Context = Arc<dyn Any + Sync + Send>;

/// Represents a call to a tool, including context, id, and arguments.
struct Call {
    context: Option<Context>,
    id: Option<String>,
    args: Option<Value>,
}

/// Type alias for a callable tool executor.
type CallExecutor =
    Arc<dyn Executor<(Call,), Output = BoxFuture<'static, anyhow::Result<Value>>> + Sync + Send>;

/// A set of tools, indexed by name.
pub type Set = FxHashMap<String, Tool>;

/// Extension trait for adding tools to a set.
pub trait SetExt {
    /// Adds a tool to the set.
    fn add(&mut self, tool: Tool);
}

impl SetExt for Set {
    fn add(&mut self, tool: Tool) {
        self.insert(tool.name.clone(), tool);
    }
}
