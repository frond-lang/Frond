//! Inference — type inference algorithm layer.
//!
//! Depends on crate::Sema (type system foundations) + crate::Relations (type
//! relation checks). Responsibilities: type inference, constraint solving,
//! flow-sensitive narrowing.
//!
//! Aggregates the inference submodule tree: [`Core`] (InferContext state +
//! drivers), ExprInfer / CallInfer / Match / Solver / FlowNarrow / Captures /
//! Unify / TypeAst / Subst / StmtInfer / ModuleEnv / Helpers.
//! Monomorphization instance collection lives in crate::Monomorph.
//!
//! [`Core`]: crate::sema::Inference::Core

use crate::sema::Sema::*;
use crate::sema::Relations::*;
use crate::ast::Ast::{
    AstArena, BinaryOp, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module,
    Pattern, PatternId, PatternLiteral, PatternRef, Stmt, StmtId,
    TypeNode, TypeRef as AstTypeRef,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Generates the match arm for numeric literal inference (shared structure for IntLit/FloatLit).
/// `$suffix_fn` is the suffix→TypeHandle method, `$predicate` is the expected-type predicate, and `$fallback` is the default Type variant.
macro_rules! numeric_lit {
    ($self:expr, $suffix:expr, $expected:expr, $suffix_fn:ident, $predicate:ident, $fallback:ident) => {{
        if let Some(suf) = $suffix {
            if let Some(ty) = $self.$suffix_fn(suf) {
                return ty;
            }
        }
        if let Some(exp) = $expected {
            let resolved = $self.arena.resolve(exp);
            if $self.arena.get(resolved).$predicate() {
                return exp;
            }
        }
        $self.make_builtin(Type::$fallback)
    }};
}

mod Core;
mod CallInfer;
mod Captures;
mod ExprInfer;
mod FlowNarrow;
mod Helpers;
mod Match;
mod ModuleEnv;
mod Solver;
mod StmtInfer;
mod Subst;
mod TypeAst;
mod Unify;
pub use Core::InferContext;
use Solver::*;
use FlowNarrow::*;
use Helpers::*;
use Subst::*;
