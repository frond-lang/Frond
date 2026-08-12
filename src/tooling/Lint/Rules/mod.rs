#![allow(non_snake_case)]
//! Rules — lint rule implementations by category.
//!
//! Aggregates four rule submodules:
//! - [`Correctness`]: correctness rules (non-exhaustive match, unreachable code, dead var/func/param)
//! - [`Style`]: style rules (naming, unused import, redundant paren)
//! - [`Perf`]: performance rules (memoizable, inlineable, stack_allocable)
//! - [`Idioms`]: idiom rules (prefer val, string interpolation)

pub mod Correctness;
pub mod Style;
pub mod Perf;
pub mod Idioms;
