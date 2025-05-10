#![expect(non_snake_case, reason = "This is a macro")]

/// Emulates the `Fn` trait for custom executor logic.
pub trait Executor<Args> {
    /// The output type of the executor.
    type Output;
    /// Executes the logic with the given arguments.
    fn execute(&self, args: Args) -> Self::Output;
}

macro_rules! impl_executor {
    ($($T:ident),*) => {
        impl<'re, F, R, $($T),*> Executor<($($T,)*)> for F
        where
            F: Fn($($T,)*) -> R,
        {
            type Output = R;
            fn execute(&self, ($($T,)*): ($($T,)*)) -> Self::Output {
                (self)($($T,)*)
            }
        }
    };
}

impl_executor!();
impl_executor!(T0);
impl_executor!(T0, T1);
impl_executor!(T0, T1, T2);
impl_executor!(T0, T1, T2, T3);
impl_executor!(T0, T1, T2, T3, T4);
impl_executor!(T0, T1, T2, T3, T4, T5);
impl_executor!(T0, T1, T2, T3, T4, T5, T6);
impl_executor!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_executor!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
