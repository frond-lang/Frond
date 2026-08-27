//! Builder — IR builder (AST + SemaResult -> DataFlowGraph).
//!
//! Aggregates the builder submodule tree. [`Core`] holds the IrBuilder struct,
//! state/scope helpers and the compile_expr/build() entry points; the sibling
//! modules hold the remaining compile_* methods by construct family
//! (Assign / Call / ControlFlow / Function / Ops / ...).
//!
//! [`Core`]: crate::ir::Builder::Core

use crate::ir::Ir::*;
use std::sync::Arc;

mod Core;
mod Access;
mod Assign;
mod Call;
mod Captures;
mod Const;
mod ControlFlow;
mod Function;
mod Lambda;
mod Loops;
mod Ops;
mod Recursion;
mod Stmt;
mod Literal;
mod Versioning;
use Core::*;
pub use Core::IrBuilder;
use Literal::*;
