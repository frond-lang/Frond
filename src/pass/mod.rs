#![allow(non_snake_case)]
//! pass — Post-processing passes (Sema-post and IR-post).
//!
//! Aggregates two post-processing pass modules:
//! - [`Analyzer`]: Sema-post static analysis (dead code / dead var / dead func
//!   + memoization strategies) + IR-post loop analysis (invariants / unroll).
//!   Consumes [`SemaResult`] + AST, produces [`AnalysisReport`] consumed by
//!   IrBuilder and Optimizer.
//! - [`Optimizer`]: IR-post graph optimization. Pass pipeline (per round):
//!   LICM → LoopUnroll → Inline → ConstFold → CSE → CopyProp → DCE.
//!   Structural transforms (LICM/Unroll/Inline) run before traditional opts;
//!   the loop transform + inline passes (formerly separate LoopAnalysis /
//!   LoopOptimizer / InlineOptimizer modules) are now merged into Analyzer /
//!   Optimizer respectively. Consumes and transforms [`DataFlowGraph`] in place.
//!
//! Both are independent post-processing passes sitting between major pipeline
//! stages: Analyzer runs between Sema and IR build (and again, via
//! `analyze_loops`, after IR build); Optimizer runs between IR build and
//! Engine execution.
//!
//! [`SemaResult`]: crate::sema::Sema::SemaResult
//! [`AnalysisReport`]: crate::pass::Analyzer::AnalysisReport
//! [`DataFlowGraph`]: crate::ir::Ir::DataFlowGraph

pub mod Analyzer;
pub mod Optimizer;
