//! Code formatter: token-stream reformatting with trivia preservation.

pub mod Comment;
pub mod Engine;
pub mod Rules;

pub use Engine::{format, tokenize_with_trivia, FmtConfig, FmtToken, Trivia};
