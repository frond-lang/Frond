#![allow(non_snake_case)]
//! sema — Semantic analysis modules.
//!
//! Aggregates the Sema pipeline submodules:
//! - `Sema`: Core type system data structures (Ty / TypeArena / SemaResult).
//! - `Relations`: Type relation checks (equality / subtype / numeric promotion).
//! - `Inference`: Type inference and constraint solving.
//! - `Monomorph`: Monomorphization instance collection.
//!
//! Note: `Analyzer` (Sema-post static analysis) now lives in `crate::pass`.

pub mod Sema;
pub mod Relations;
pub mod Inference;
pub mod Monomorph;
