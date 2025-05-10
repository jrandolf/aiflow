#![expect(unused_variables, reason = "This is a macro")]
use alloc::sync::Arc;
use core::any::Any;

use serde::de::DeserializeOwned;

use super::Call;

/// Wrapper for extracting an ID from a tool call.
pub struct Id(pub String);

impl TryFrom<&mut Call> for Id {
    type Error = anyhow::Error;

    #[expect(clippy::unwrap_in_result, reason = "Developer error")]
    fn try_from(state: &mut Call) -> Result<Self, Self::Error> {
        Ok(Self(state.id.take().expect("id to exist")))
    }
}

/// Wrapper for extracting arguments of type `T` from a tool call.
pub struct Args<T>(pub T);

impl<T> TryFrom<&mut Call> for Args<T>
where
    T: DeserializeOwned,
{
    type Error = anyhow::Error;

    #[expect(clippy::unwrap_in_result, reason = "Developer error")]
    fn try_from(state: &mut Call) -> Result<Self, Self::Error> {
        serde_json::from_value(state.args.take().expect("args to exist"))
            .map(Self)
            .map_err(Into::into)
    }
}

/// Wrapper for extracting a context of type `T` from a tool call.
pub struct Context<T>(pub Arc<T>);

impl<T: Any + Sync + Send> TryFrom<&mut Call> for Context<T> {
    type Error = anyhow::Error;

    #[expect(clippy::unwrap_in_result, reason = "Developer error")]
    fn try_from(state: &mut Call) -> Result<Self, Self::Error> {
        Ok(Self(
            state
                .context
                .take()
                .expect("context to exist")
                .downcast()
                .expect("to be a T"),
        ))
    }
}

macro_rules! impl_tryfrom_call_tuple {
    ($($T:ident),*) => {
        impl<$($T),*> TryFrom<&mut Call> for ($($T,)*)
        where
            $($T: for<'re> TryFrom<&'re mut Call, Error = anyhow::Error>,)*
        {
            type Error = anyhow::Error;

            fn try_from(state: &mut Call) -> Result<Self, Self::Error> {
                Ok(($( $T::try_from(state)?, )*))
            }
        }
    };
}

impl_tryfrom_call_tuple!();
impl_tryfrom_call_tuple!(T0);
impl_tryfrom_call_tuple!(T0, T1);
impl_tryfrom_call_tuple!(T0, T1, T2);
impl_tryfrom_call_tuple!(T0, T1, T2, T3);
impl_tryfrom_call_tuple!(T0, T1, T2, T3, T4);
impl_tryfrom_call_tuple!(T0, T1, T2, T3, T4, T5);
impl_tryfrom_call_tuple!(T0, T1, T2, T3, T4, T5, T6);
impl_tryfrom_call_tuple!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_tryfrom_call_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
