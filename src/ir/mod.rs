#![allow(non_snake_case)]
//! ir — Intermediate representation, IR builder, and compute functions.
//!
//! Aggregates three IR-related submodules:
//! - [`Ir`]: IR data structures (Node, Frame, SubGraph, DataFlowGraph, ComputeFn table).
//! - [`Builder`]: IR builder (IrBuilder + all compile_* methods + build() entry point).
//! - [`Compute`]: compute_fn implementations (node execution semantics).
//!
//! 二进制产物序列化见顶层 [`crate::resin`] 模块。

pub mod Ir;
pub mod Builder;
pub mod Compute;
