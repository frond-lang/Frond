//! Relations.rs — type relation judgment layer
//!
//! Split from Sema.rs. Depends on crate::Sema (type system foundation).
//! Responsibilities: type equality, subtype relations, numeric promotion,
//! peer-type resolution, module structural matching.

use crate::sema::Sema::*;
use crate::ast::Ast::{AstArena, MethodDecl, TypeNode, TypeRef as AstTypeRef};

// =========================================================================
// phase5: subtype_check — subtype relation judgment
//
// Rust port of `src/sema/subtype_check.zig`.
// Determines subtype relations between the various types in the Kuzo
// language: null/nullable, record structural subtyping, ADT error subtyping,
// Throw subtyping, trait structural subtyping.
// =========================================================================

/// Structural equality of named type argument lists: name match + length
/// match + element-wise recursive equality.
/// Shared by the three named composite types: Adt / Generic / Trait.
#[inline]
fn named_args_equal(
    arena: &TypeArena,
    na: &str,
    ta: &[TypeHandle],
    nb: &str,
    tb: &[TypeHandle],
) -> bool {
    na == nb
        && ta.len() == tb.len()
        && ta.iter().zip(tb.iter()).all(|(&x, &y)| types_equal(arena, x, y))
}

/// Recursively determine whether two types are structurally equal (compares
/// `Ty` contents after `resolve`).
///
/// Authoritative implementation: `InferContext::types_structurally_equal`
/// delegates to this function.
/// Used by standalone checkers such as `is_subtype`.
pub fn types_equal(arena: &TypeArena, a: TypeHandle, b: TypeHandle) -> bool {
    let ra = arena.resolve(a);
    let rb = arena.resolve(b);
    if ra == rb {
        return true;
    }
    let a_ct = arena.get(ra);
    let b_ct = arena.get(rb);
    if std::mem::discriminant(&a_ct) != std::mem::discriminant(&b_ct) {
        return false;
    }
    match (a_ct, b_ct) {
        (Ty::TypeVar(ia), Ty::TypeVar(ib)) => ia == ib,
        (Ty::Fn(_), Ty::Fn(_)) => {
            let (pa, rpa) = arena.fn_parts(ra);
            let (pb, rpb) = arena.fn_parts(rb);
            pa.len() == pb.len()
                && pa.iter().zip(pb.iter()).all(|(&x, &y)| types_equal(arena, x, y))
                && types_equal(arena, rpa, rpb)
        }
        (Ty::Record(_), Ty::Record(_)) => {
            let fa = arena.record_fields(ra);
            let fb = arena.record_fields(rb);
            if fa.len() != fb.len() {
                return false;
            }
            for (x, y) in fa.iter().zip(fb.iter()) {
                let names_match = match (x.name.as_deref(), y.name.as_deref()) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => true,
                    _ => false,
                };
                if !names_match || !types_equal(arena, x.ty, y.ty) {
                    return false;
                }
            }
            true
        }
        (Ty::Adt(_), Ty::Adt(_)) => {
            let (na, ta) = arena.adt_parts(ra);
            let (nb, tb) = arena.adt_parts(rb);
            named_args_equal(arena, na, ta, nb, tb)
        }
        (Ty::Generic(_), Ty::Generic(_)) => {
            let (na, ta) = arena.generic_parts(ra);
            let (nb, tb) = arena.generic_parts(rb);
            named_args_equal(arena, na, ta, nb, tb)
        }
        (Ty::Array(_), Ty::Array(_)) => {
            let (ea, sa) = arena.array_parts(ra);
            let (eb, sb) = arena.array_parts(rb);
            sa == sb && types_equal(arena, ea, eb)
        }
        (Ty::Throw(_), Ty::Throw(_)) => {
            let (va, ea) = arena.throw_parts(ra);
            let (vb, eb) = arena.throw_parts(rb);
            types_equal(arena, va, vb) && types_equal(arena, ea, eb)
        }
        // Single-parameter builtin generics: equal iff element types are equal
        (Ty::Channel(_), Ty::Channel(_)) => {
            types_equal(arena, arena.channel_elem(ra), arena.channel_elem(rb))
        }
        (Ty::Async(_), Ty::Async(_)) => {
            types_equal(arena, arena.async_value(ra), arena.async_value(rb))
        }
        (Ty::Lazy(_), Ty::Lazy(_)) => {
            types_equal(arena, arena.lazy_value(ra), arena.lazy_value(rb))
        }
        (Ty::Atomic(_), Ty::Atomic(_)) => {
            types_equal(arena, arena.atomic_elem(ra), arena.atomic_elem(rb))
        }
        (Ty::Sender(_), Ty::Sender(_)) => {
            types_equal(arena, arena.sender_elem(ra), arena.sender_elem(rb))
        }
        (Ty::Receiver(_), Ty::Receiver(_)) => {
            types_equal(arena, arena.receiver_elem(ra), arena.receiver_elem(rb))
        }
        (Ty::Trait(_), Ty::Trait(_)) => {
            let (na, ta) = arena.trait_parts(ra);
            let (nb, tb) = arena.trait_parts(rb);
            named_args_equal(arena, na, ta, nb, tb)
        }
        (Ty::TraitObject(_), Ty::TraitObject(_)) => {
            let (na, ma) = arena.trait_object_parts(ra);
            let (nb, mb) = arena.trait_object_parts(rb);
            na == nb
                && ma.len() == mb.len()
                && ma.iter().zip(mb.iter()).all(|(a, b)| {
                    a.name == b.name && a.param_count == b.param_count
                })
        }
        (Ty::Nullable(_), Ty::Nullable(_)) => {
            let ia = arena.nullable_inner(ra);
            let ib = arena.nullable_inner(rb);
            types_equal(arena, ia, ib)
        }
        (Ty::Ref(_), Ty::Ref(_)) => {
            let (ia, ra_raw) = arena.ref_parts(ra);
            let (ib, rb_raw) = arena.ref_parts(rb);
            ra_raw == rb_raw && types_equal(arena, ia, ib)
        }
        // Scalar unit variants (I32, Bool, Str, ...), Never, Unknown, Null,
        // Void — discriminant match implies equality
        _ => true,
    }
}

/// Subtype judgment rule. Each rule matches a specific `(sub, sup)` shape
/// combination; returns `Some(bool)` on a hit, `None` to defer to the next
/// rule.
///
/// Unifies the scattered if-let dispatch inside `is_subtype`. Adding a new
/// subtype rule only requires implementing `SubtypeRule` on a fieldless
/// struct and registering it in `SUBTYPE_RULES`.
trait SubtypeRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool>;
}

/// Reflexivity: same type or structurally equal.
struct ReflexiveRule;
impl SubtypeRule for ReflexiveRule {
    #[inline]
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let r_sub = arena.resolve(sub);
        let r_sup = arena.resolve(sup);
        if r_sub == r_sup || types_equal(arena, r_sub, r_sup) {
            Some(true)
        } else {
            None
        }
    }
}

/// The `Null` literal is assignable to any `nullable` type.
struct NullToNullableRule;
impl SubtypeRule for NullToNullableRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let sub_ct = arena.get(arena.resolve(sub));
        let sup_ct = arena.get(arena.resolve(sup));
        if matches!(sub_ct, Ty::Null) && matches!(sup_ct, Ty::Nullable(_)) {
            Some(true)
        } else {
            None
        }
    }
}

/// `sub <: Nullable(inner)` ⟹ `sub <: inner`。
struct NullableInnerRule;
impl SubtypeRule for NullableInnerRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let sup_resolved = arena.resolve(sup);
        let sup_ct = arena.get(sup_resolved);
        if let Ty::Nullable(_) = sup_ct {
            let inner = arena.nullable_inner(sup_resolved);
            Some(is_subtype(arena, sub, inner))
        } else {
            None
        }
    }
}

/// Record structural subtype: `sub_fields` covers every field of `sup_fields`
/// with compatible types.
struct RecordSubtypeRule;
impl SubtypeRule for RecordSubtypeRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let sub_resolved = arena.resolve(sub);
        let sup_resolved = arena.resolve(sup);
        let sub_ct = arena.get(sub_resolved);
        let sup_ct = arena.get(sup_resolved);
        if let (Ty::Record(_), Ty::Record(_)) = (sub_ct, sup_ct) {
            let sub_fields = arena.record_fields(sub_resolved);
            let sup_fields = arena.record_fields(sup_resolved);
            Some(is_record_subtype(arena, sub_fields, sup_fields))
        } else {
            None
        }
    }
}

/// ADT same-name subtype: compares type names directly.
struct AdtNameRule;
impl SubtypeRule for AdtNameRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let sub_resolved = arena.resolve(sub);
        let sup_resolved = arena.resolve(sup);
        let sub_ct = arena.get(sub_resolved);
        let sup_ct = arena.get(sup_resolved);
        if let (Ty::Adt(_), Ty::Adt(_)) = (sub_ct, sup_ct) {
            let (sub_name, _) = arena.adt_parts(sub_resolved);
            let (sup_name, _) = arena.adt_parts(sup_resolved);
            Some(sub_name == sup_name)
        } else {
            None
        }
    }
}

/// `Throw<V1, E1> <: Throw<V2, E2>` ⟹ `V1 <: V2 ∧ E1 <: E2`。
struct ThrowSubtypeRule;
impl SubtypeRule for ThrowSubtypeRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool> {
        let sub_resolved = arena.resolve(sub);
        let sup_resolved = arena.resolve(sup);
        let sub_ct = arena.get(sub_resolved);
        let sup_ct = arena.get(sup_resolved);
        if let (Ty::Throw(_), Ty::Throw(_)) = (sub_ct, sup_ct) {
            let (sv, se) = arena.throw_parts(sub_resolved);
            let (pv, pe) = arena.throw_parts(sup_resolved);
            Some(is_throw_subtype(arena, sv, se, pv, pe))
        } else {
            None
        }
    }
}

/// Subtype rule chain: tries each rule in order; the first hit (returning
/// `Some`) decides.
/// Order matches the original if-let chain: reflexive → null→nullable →
/// nullable inner → record → ADT same-name → throw. The `trait structural`
/// subtype needs `sema_result` and is determined by the caller via
/// `is_trait_structural_subtype`.
const SUBTYPE_RULES: &[&dyn SubtypeRule] = &[
    &ReflexiveRule,
    &NullToNullableRule,
    &NullableInnerRule,
    &RecordSubtypeRule,
    &AdtNameRule,
    &ThrowSubtypeRule,
];

/// Determine whether `sub` is a subtype of `sup`.
///
/// Dispatches through the `SUBTYPE_RULES` chain: reflexivity, null→nullable,
/// nullable inner, record structural subtyping, ADT same-name, Throw subtype.
/// Returns the result of the first matching rule; returns `false` if no rules
/// match. The `trait structural` subtype needs `sema_result` and is determined
/// by the caller via `is_trait_structural_subtype`.
pub fn is_subtype(arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> bool {
    for rule in SUBTYPE_RULES.iter() {
        if let Some(ok) = rule.check(arena, sub, sup) {
            return ok;
        }
    }
    false
}

/// Record field matching core: whether the `provided` field set covers every
/// field of `required`.
/// Matches by field name (positional fields with `name == None` are treated as
/// same-named); after a name match, invokes `check` to verify field-type
/// compatibility. `check`'s signature is `|arena, provided_ty, required_ty|`.
///
/// Eliminates duplicated field traversal between `is_record_subtype` and
/// `record_arg_satisfies`.
fn match_record_fields<F>(
    arena: &TypeArena,
    required: &[FieldType],
    provided: &[FieldType],
    mut check: F,
) -> bool
where
    F: FnMut(&TypeArena, TypeHandle, TypeHandle) -> bool,
{
    for req_field in required.iter() {
        let mut found = false;
        for prov_field in provided.iter() {
            let names_match = match (prov_field.name.as_deref(), req_field.name.as_deref()) {
                (Some(a), Some(b)) => a == b,
                (None, None) => true,
                _ => false,
            };
            if names_match {
                if !check(arena, prov_field.ty, req_field.ty) {
                    return false;
                }
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

/// Record structural subtype judgment: whether `sub_fields` covers every
/// field of `sup_fields`.
/// Matches by field name and recursively verifies that field types satisfy the
/// subtype relation (width + depth subtyping).
pub fn is_record_subtype(
    arena: &TypeArena,
    sub_fields: &[FieldType],
    sup_fields: &[FieldType],
) -> bool {
    // sub_fields provides, sup_fields requires; sub field types must be subtypes
    // of the corresponding sup field types.
    match_record_fields(arena, sup_fields, sub_fields, is_subtype)
}

/// Throw subtype judgment: value type and error type must both satisfy the
/// subtype relation.
pub fn is_throw_subtype(
    arena: &TypeArena,
    sub_val: TypeHandle,
    sub_err: TypeHandle,
    sup_val: TypeHandle,
    sup_err: TypeHandle,
) -> bool {
    is_subtype(arena, sub_val, sup_val) && is_subtype(arena, sub_err, sup_err)
}

/// Trait structural subtype judgment: the method set of `sub_name` must cover
/// every method name of `super_name`.
pub fn is_trait_structural_subtype(
    sema_result: &SemaResult,
    sub_name: &str,
    sup_name: &str,
) -> bool {
    let sub_def = match sema_result.get_trait_def(sub_name) {
        Some(d) => d,
        None => return false,
    };
    let sup_def = match sema_result.get_trait_def(sup_name) {
        Some(d) => d,
        None => return false,
    };
    for sup_method in sup_def.methods.iter() {
        let found = sub_def
            .methods
            .iter()
            .any(|m| m.name.as_ref() == sup_method.name.as_ref());
        if !found {
            return false;
        }
    }
    true
}

/// Determine whether the argument record `arg` satisfies the field
/// requirements of the parameter record `param`.
/// Non-record types are considered to satisfy; record types recursively verify
/// that every parameter field exists and its type is satisfied.
pub fn record_arg_satisfies(arena: &TypeArena, param: TypeHandle, arg: TypeHandle) -> bool {
    let rp = arena.resolve(param);
    let ra = arena.resolve(arg);
    let param_ct = arena.get(rp);
    let arg_ct = arena.get(ra);
    match (param_ct, arg_ct) {
        (Ty::Record(_), Ty::Record(_)) => {
            let pf = arena.record_fields(rp);
            let af = arena.record_fields(ra);
            // pf requires, af provides; verify recursively.
            match_record_fields(arena, pf, af, |a, prov, req| record_arg_satisfies(a, req, prov))
        }
        _ => true,
    }
}

// =========================================================================
// phase5: throw_check helpers — numeric widening and return-type unification
//
// Rust port of the numeric widening helpers in `src/sema/throw_check.zig`.
// InferContext methods (unify_return_type / try_widen_unify / check_propagate /
// check_throw_stmt) are implemented in the InferContext impl block below.
// =========================================================================

/// Returns the integer's rank (for widening comparison). Signed and unsigned
/// integers of the same width share the same rank; non-integers return 0.
#[inline]
pub fn int_type_rank(ty: &Ty) -> u8 {
    ty.int_rank().unwrap_or(0)
}

/// Returns the float type's rank (for widening comparison); non-floats return 0.
#[inline]
pub fn float_type_rank(ty: &Ty) -> u8 {
    ty.float_bit_width().map(|b| b as u8).unwrap_or(0)
}

/// Determines whether this is a signed integer (free-function version,
/// equivalent to `Ty::is_signed_int`).
#[inline]
pub fn is_signed_int_ct(ty: &Ty) -> bool {
    ty.is_signed_int()
}

/// Determine whether the numeric type `from` can be implicitly widened to the
/// numeric type `to`.
/// Covers int→int, float→float, and int→float widening rules.
pub fn can_coerce_numeric(arena: &TypeArena, to: TypeHandle, from: TypeHandle) -> bool {
    let to_ct = arena.get(arena.resolve(to));
    let from_ct = arena.get(arena.resolve(from));
    let to_int = int_type_rank(&to_ct);
    let from_int = int_type_rank(&from_ct);
    let to_float = float_type_rank(&to_ct);
    let from_float = float_type_rank(&from_ct);

    // Between integers: widening allowed when ranks are equal or the target rank is larger
    if to_int > 0 && from_int > 0 {
        let to_signed = is_signed_int_ct(&to_ct);
        let from_signed = is_signed_int_ct(&from_ct);
        if to_int == from_int && to_signed == from_signed {
            return true;
        }
        if to_signed == from_signed {
            return to_int >= from_int;
        }
        // signed → unsigned requires the target rank to be strictly larger to hold the sign bit
        return to_int > from_int;
    }
    // Between floats: widening allowed when ranks are equal or the target rank is larger (no narrowing)
    if to_float > 0 && from_float > 0 {
        return to_float >= from_float;
    }
    // int → float: allowed
    if to_float > 0 && from_int > 0 {
        return true;
    }
    false
}

// =========================================================================
// phase5: kind_check — type kind checking
//
// Rust port of `src/sema/kind_check.zig`.
// Verifies that type-constructor usage in type annotations is consistent with
// its kind (arity).
// =========================================================================

/// Returns the number of type arguments (kind arity) expected by the type
/// constructor named `name`.
/// Built-in higher-kinded types use a fixed arity; user-defined ADTs use their
/// declared type-parameter count; bare type names have arity 0.
pub fn arity_of_type_name(sema_result: &SemaResult, name: &str) -> usize {
    if let Some(arity) = generic_type_arity(name) {
        return arity as usize;
    }
    match sema_result.get_type_def(name) {
        Some(def) => def.type_params.len(),
        None => 0,
    }
}

/// Determines whether `name` is a type parameter in the current scope.
fn is_type_param(name: &str, type_param_names: &[&str]) -> bool {
    type_param_names.contains(&name)
}

/// Recursively checks each type constructor's usage in the type-node tree
/// against its kind arity.
/// `type_param_names` lists the type parameter names valid in the current
/// scope (usable as concrete types).
/// On mismatch, an error is appended to `errors`.
pub fn check_type_node(
    sema_result: &SemaResult,
    ast: &AstArena<'_>,
    node: AstTypeRef,
    type_param_names: &[&str],
    errors: &mut Vec<SemaError>,
) {
    let tn = &ast.ty(node).node;
    let span = ast.ty(node).span;
    match tn {
        TypeNode::Named { name } => {
            if is_type_param(name, type_param_names) {
                return;
            }
            let arity = arity_of_type_name(sema_result, name);
            if arity > 0 {
                errors.push(SemaError::new(
                    &format!(
                        "kind mismatch: type constructor '{}' expects {} type argument(s) but is used as a concrete type",
                        name, arity
                    ),
                    span.line,
                    span.column,
                ));
            }
        }
        TypeNode::SelfType => {}
        TypeNode::Generic { name, args } => {
            if !is_type_param(name, type_param_names) {
                let arity = arity_of_type_name(sema_result, name);
                if arity != 0 && arity != args.len() {
                    errors.push(SemaError::new(
                        &format!(
                            "kind mismatch: type constructor '{}' expects {} type argument(s) but got {}",
                            name,
                            arity,
                            args.len()
                        ),
                        span.line,
                        span.column,
                    ));
                }
            }
            for &arg in args.iter() {
                check_type_node(sema_result, ast, arg, type_param_names, errors);
            }
        }
        TypeNode::Nullable { inner } => {
            check_type_node(sema_result, ast, *inner, type_param_names, errors);
        }
        TypeNode::RefType { inner } => {
            check_type_node(sema_result, ast, *inner, type_param_names, errors);
        }
        TypeNode::RawPtr { inner } => {
            check_type_node(sema_result, ast, *inner, type_param_names, errors);
        }
        TypeNode::Function { params, return_type } => {
            for &p in params.iter() {
                check_type_node(sema_result, ast, p, type_param_names, errors);
            }
            check_type_node(sema_result, ast, *return_type, type_param_names, errors);
        }
        TypeNode::Record { fields } => {
            for f in fields.iter() {
                check_type_node(sema_result, ast, f.ty, type_param_names, errors);
            }
        }
        TypeNode::Array { element_type, .. } => {
            check_type_node(sema_result, ast, *element_type, type_param_names, errors);
        }
        TypeNode::KindAnnotated { inner, .. } => {
            check_type_node(sema_result, ast, *inner, type_param_names, errors);
        }
    }
}

/// Computes the remaining kind arity of a type node: how many more type
/// arguments are needed before it becomes a concrete type.
/// Bare type names return their arity; partially applied higher-kinded types
/// return (arity - number of supplied arguments).
pub fn kind_arity_of_type_node(sema_result: &SemaResult, ast: &AstArena<'_>, node: AstTypeRef) -> usize {
    let tn = &ast.ty(node).node;
    match tn {
        TypeNode::Named { name } => arity_of_type_name(sema_result, name),
        TypeNode::Generic { name, args } => {
            let head = arity_of_type_name(sema_result, name);
            if args.len() >= head {
                0
            } else {
                head - args.len()
            }
        }
        _ => 0,
    }
}

// =========================================================================
// phase5: module_check — module structural checking
//
// Rust port of `src/sema/module_check.zig`.
// Provides a signature summary of a module's member methods and a judgment of
// whether a module structurally satisfies the method set required by a trait.
// =========================================================================

/// Signature summary of a module's member method: method name and parameter count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodSig {
    pub name: Box<str>,
    pub arity: usize,
}

/// Reason for a failed module structural match against a trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchReason {
    /// Match succeeded
    Ok,
    /// Method missing
    Missing,
    /// Parameter count mismatch
    ArityMismatch,
}

/// Result of a module structural match against a trait.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    pub ok: bool,
    pub missing_method: Option<Box<str>>,
    pub arity_expected: usize,
    pub arity_got: usize,
    pub reason: MatchReason,
}

impl MatchResult {
    /// Match succeeded.
    pub fn ok() -> Self {
        MatchResult {
            ok: true,
            missing_method: None,
            arity_expected: 0,
            arity_got: 0,
            reason: MatchReason::Ok,
        }
    }
}

/// Module checker: determines whether a set of provided method signatures
/// structurally satisfies a set of required method signatures.
#[derive(Debug, Default)]
pub struct ModuleChecker;

impl ModuleChecker {
    pub fn new() -> Self {
        ModuleChecker
    }

    /// Verifies, for each method in `required`, that it exists in `provided`
    /// with a matching parameter count.
    /// Any missing method or parameter-count mismatch yields a failure result
    /// carrying the reason.
    pub fn structurally_satisfies(
        &self,
        provided: &[MethodSig],
        required: &[MethodSig],
    ) -> MatchResult {
        for req in required.iter() {
            let found = provided.iter().find(|p| p.name.as_ref() == req.name.as_ref());
            match found {
                Some(prov) => {
                    if prov.arity != req.arity {
                        return MatchResult {
                            ok: false,
                            missing_method: Some(req.name.clone()),
                            arity_expected: req.arity,
                            arity_got: prov.arity,
                            reason: MatchReason::ArityMismatch,
                        };
                    }
                }
                None => {
                    return MatchResult {
                        ok: false,
                        missing_method: Some(req.name.clone()),
                        arity_expected: 0,
                        arity_got: 0,
                        reason: MatchReason::Missing,
                    };
                }
            }
        }
        MatchResult::ok()
    }

    /// Collects the signatures of trait methods that have no body (i.e. need to
    /// be implemented).
    pub fn required_methods(&self, trait_methods: &[MethodDecl<'_>]) -> Vec<MethodSig> {
        let mut list = Vec::new();
        for m in trait_methods.iter() {
            if m.body.is_some() {
                continue;
            }
            list.push(MethodSig {
                name: m.name.into(),
                arity: m.params.len(),
            });
        }
        list
    }
}


// =========================================================================
// sema v2: Peer Type Resolution — unified type unification entry point
//
// Design rationale:
// Unifies literal_promotion + try_widen_unify + if-branch unification into a
// single peer_type entry point.
// Given N types, finds the most compatible common type (join / least upper
// bound).
//
// Rules (in priority order):
// 1. Empty list → Unknown
// 2. Single element → that type
// 3. All identical → that type
// 4. Contains Never → recurse after filtering (divergent types contribute no value)
// 5. Numeric types → take the widest (int→int widest, float preferred, int→float)
// 6. nullable propagation → Nullable<peer(inner types)>
// 7. throw propagation → Throw<peer(value types), peer(error types)>
// 8. ADT → consult error_newtype subtype relation
// 9. No common type → Unknown (records an error)
// =========================================================================

/// Compute the common type (join / least upper bound) of multiple types.
///
/// Unified entry point: replaces scattered literal_promotion / try_widen_unify
/// / if-branch unification.
/// Returns the most compatible type that can accommodate all input types.
///
/// **Rules** (in priority order):
/// 1. Empty list → `Unknown`
/// 2. Single element → that type
/// 3. Contains `Never` → recurse after filtering (divergent paths contribute no value)
/// 4. All identical (structurally equal) → that type
/// 5. All numeric → take the widest (int→int largest bit width, float prioritized over int, int→float widening)
/// 6. All nullable → `Nullable<peer(inners)>`
/// 7. All throw → `Throw<peer(values), peer(errors)>`
/// 8. Contains nullable + non-nullable → `Nullable<peer(all inners)>`
/// 9. ADT error subtype → take the supertype
/// 10. No common type → `Unknown`
pub fn peer_type(arena: &mut TypeArena, types: &[TypeHandle]) -> TypeHandle {
    if types.is_empty() {
        return arena.make(Ty::Unknown);
    }
    if types.len() == 1 {
        return types[0];
    }

    // Filter out Never and Void (divergent / value-less types contribute no value)
    // Never: diverging (return/throw/break/continue)
    // Void: no meaningful value (e.g. if-then without else and `then` is a statement)
    let non_trivial: Vec<TypeHandle> = types
        .iter()
        .filter(|&&t| {
            !matches!(
                arena.get(arena.resolve(t)),
                Ty::Never | Ty::Void
            )
        })
        .copied()
        .collect();
    if non_trivial.is_empty() {
        // All branches are Never/Void. Only when EVERY branch is Never (every
        // path diverges) is the result Never. If any Void is present, there is
        // a non-diverging fall-through path (e.g. `if c { return }` without an
        // else — the implicit else falls through), so the result is Void, not
        // Never. This prevents false "unreachable" warnings for code after a
        // single-sided diverging if.
        if types
            .iter()
            .all(|&t| matches!(arena.get(arena.resolve(t)), Ty::Never))
        {
            return arena.make(Ty::Never);
        }
        return arena.make(Ty::Void);
    }
    if non_trivial.len() == 1 {
        return non_trivial[0];
    }

    // All identical (structurally equal) → return the first
    let first = non_trivial[0];
    if non_trivial[1..].iter().all(|&t| types_equal(arena, first, t)) {
        return first;
    }

    // All numeric → take the widest
    let all_numeric = non_trivial.iter().all(|&t| {
        let ct = arena.get(arena.resolve(t));
        ct.is_int() || ct.is_float()
    });
    if all_numeric {
        return peer_numeric(arena, &non_trivial);
    }

    // All nullable → Nullable<peer(inners)>
    let all_nullable = non_trivial.iter().all(|&t| {
        matches!(arena.get(arena.resolve(t)), Ty::Nullable(_))
    });
    if all_nullable {
        let inners: Vec<TypeHandle> = non_trivial
            .iter()
            .map(|&t| {
                let resolved = arena.resolve(t);
                match arena.get(resolved) {
                    Ty::Nullable(_) => arena.nullable_inner(resolved),
                    _ => unreachable!(),
                }
            })
            .collect();
        let peer_inner = peer_type(arena, &inners);
        return arena.make_nullable(peer_inner);
    }

    // Contains nullable + non-nullable → Nullable<peer(all inners)>
    let has_nullable = non_trivial.iter().any(|&t| {
        matches!(arena.get(arena.resolve(t)), Ty::Nullable(_))
    });
    if has_nullable {
        let inners: Vec<TypeHandle> = non_trivial
            .iter()
            .map(|&t| {
                let resolved = arena.resolve(t);
                match arena.get(resolved) {
                    Ty::Nullable(_) => arena.nullable_inner(resolved),
                    other => {
                        if matches!(other, Ty::Null) {
                            arena.make(Ty::Unknown)
                        } else {
                            t
                        }
                    }
                }
            })
            .collect();
        let peer_inner = peer_type(arena, &inners);
        return arena.make_nullable(peer_inner);
    }

    // All throw → Throw<peer(values), peer(errors)>
    let all_throw = non_trivial.iter().all(|&t| {
        matches!(arena.get(arena.resolve(t)), Ty::Throw(_))
    });
    if all_throw {
        let (values, errors): (Vec<TypeHandle>, Vec<TypeHandle>) = non_trivial
            .iter()
            .map(|&t| {
                let resolved = arena.resolve(t);
                match arena.get(resolved) {
                    Ty::Throw(_) => arena.throw_parts(resolved),
                    _ => unreachable!(),
                }
            })
            .unzip();
        let peer_val = peer_type(arena, &values);
        let peer_err = peer_type(arena, &errors);
        return arena.make_throw(peer_val, peer_err);
    }

    // Incompatible → Unknown
    arena.make(Ty::Unknown)
}

/// Peer-type resolution for binary operations (internalizes literal
/// promotion rules).
///
/// Unified replacement for `literal_promotion`: the literal-promotion rules are
/// internalized as part of this function, eliminating the dual-track design of
/// literal_promotion vs. peer_type.
///
/// **Rules** (in priority order):
/// 1. One side literal, the other a variable → return the variable's type
///    (literal promoted to the variable's type)
/// 2. Both sides literals → `peer_numeric` takes the widest
/// 3. Both sides variables → `peer_type` computes the common type (numeric
///    widest, nullable propagation, etc.)
pub fn peer_type_binary(
    arena: &mut TypeArena,
    left: TypeHandle,
    right: TypeHandle,
    left_is_literal: bool,
    right_is_literal: bool,
) -> TypeHandle {
    // Rule 1: literal promoted to the variable's type
    if left_is_literal && !right_is_literal {
        return arena.resolve(right);
    }
    if !left_is_literal && right_is_literal {
        return arena.resolve(left);
    }

    // Rules 2 & 3: both sides literals or both sides variables → peer_type
    peer_type(arena, &[left, right])
}

/// Peer resolution for numeric types: take the widest type.
///
/// Rules:
/// 1. Contains float → take the widest float (largest bit width)
/// 2. All int → take the widest int (same sign: largest bit width; mixed sign:
///    take the unsigned type with the largest bit width)
fn peer_numeric(arena: &mut TypeArena, types: &[TypeHandle]) -> TypeHandle {
    let resolved: Vec<Ty> = types
        .iter()
        .map(|&t| arena.get(arena.resolve(t)))
        .collect();

    // Contains float → take the widest float
    let has_float = resolved.iter().any(|ct| ct.is_float());
    if has_float {
        let mut widest = Ty::F16;
        let mut widest_bits: u16 = 16;
        for ct in &resolved {
            if let Some(bits) = ct.float_bit_width() {
                if bits > widest_bits {
                    widest_bits = bits;
                    widest = match ct {
                        Ty::F16 => Ty::F16,
                        Ty::F32 => Ty::F32,
                        Ty::F64 => Ty::F64,
                        Ty::F128 => Ty::F128,
                        _ => unreachable!(),
                    };
                }
            }
        }
        return arena.make(widest);
    }

    // All int → take the widest
    let mut widest = Ty::I8;
    let mut widest_bits: u16 = 8;
    for ct in &resolved {
        if let Some(bits) = ct.int_bit_width() {
            if bits > widest_bits {
                widest_bits = bits;
                widest = *ct;
            }
        }
    }
    arena.make(widest)
}
