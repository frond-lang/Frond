#![allow(non_snake_case)]
//! Fmt — code formatter (token-stream reformatting with trivia preservation).
//!
//! Aggregates three submodules:
//! - [`Comment`]: comment/trivia extraction and classification
//! - [`Engine`]: formatting engine (format / tokenize_with_trivia / FmtConfig)
//! - [`Rules`]: formatting rules (indentation, spacing, line breaks)

pub mod Comment;
pub mod Engine;
pub mod Rules;

pub use Engine::{format, tokenize_with_trivia, FmtConfig, FmtToken, Trivia};
