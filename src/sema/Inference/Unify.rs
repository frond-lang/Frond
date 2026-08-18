//! Unify — Unification, widening, propagate/throw checking. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Side-effect-free structural type equality check: does not modify any TypeVar and triggers no unify side effects.
    /// Used when matching trait method signatures to compare parameter and return types.
    ///
    /// Delegates to the free function `types_equal` to avoid maintaining two copies of the structural-equality logic.
    pub fn types_structurally_equal(&self, a: TypeHandle, b: TypeHandle) -> bool {
        types_equal(self.arena, a, b)
    }

    // ── throw_check methods ──

    /// Unifies a function's declared return type with the type inferred from its body.
    /// Applies special relaxation for nullable/throw return types: when the body returns void (early exit / throw),
    /// it is not treated as a mismatch; otherwise it tries a widening unification and falls back to a strict unify on failure.
    pub fn unify_return_type(
        &mut self,
        declared: TypeHandle,
        inferred: TypeHandle,
    ) -> Result<(), UnifyError> {
        let r_declared = self.arena.resolve(declared);
        let r_inferred = self.arena.resolve(inferred);

        let declared_ty = self.arena.get(r_declared);
        let inferred_ty = self.arena.get(r_inferred);

        // async function: the declared return type should be Async<X> and the body infers Async<Y>;
        // recursively unify the inner types X and Y.
        if let (Type::Async(_), Type::Async(_)) = (declared_ty, inferred_ty) {
            let da = self.arena.async_value(r_declared);
            let ia = self.arena.async_value(r_inferred);
            return self.unify_return_type(da, ia);
        }
        // async function body directly returns the inner value (not Async-wrapped):
        // declared Async<X>, body infers Y → recursively unify X with Y.
        if let Type::Async(_) = declared_ty {
            let da = self.arena.async_value(r_declared);
            return self.unify_return_type(da, r_inferred);
        }

        match declared_ty {
            Type::Nullable(_) => match inferred_ty {
                Type::Nullable(_) => self.arena.unify(declared, inferred),
                Type::Void => Ok(()), // The body produced no value; compatible with nullable.
                _ => {
                    let inner_ty = self.arena.nullable_inner(r_declared);
                    match self.try_widen_unify(inner_ty, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(inner_ty, r_inferred),
                    }
                }
            },
            Type::Throw(_) => match inferred_ty {
                Type::Throw(_) => {
                    match self.try_widen_unify(declared, inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(declared, inferred),
                    }
                }
                Type::Void => Ok(()), // The body produced no value; compatible with throw.
                _ => {
                    let (vt, _) = self.arena.throw_parts(r_declared);
                    match self.try_widen_unify(vt, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(vt, r_inferred),
                    }
                }
            },
            _ => {
                match self.try_widen_unify(r_declared, r_inferred) {
                    Ok(_) => Ok(()),
                    Err(_) => self.arena.unify(declared, inferred),
                }
            }
        }
    }

    /// Immediately unifies two types; on failure, registers an Equality constraint for the fixpoint iteration to retry.
    ///
    /// Replaces the `let _ = self.arena.unify(t1, t2)` pattern:
    /// - unify succeeds → bind immediately (preserves inference-order advantages).
    /// - unify fails → register an Equality constraint with the solver, retried by the fixpoint iteration
    ///   (other constraints may bind the relevant TypeVars first, making a later unify succeed).
    #[inline]
    pub fn unify_or_constrain(&mut self, t1: TypeHandle, t2: TypeHandle) {
        // Instantiation mode: skip HM constraint solving (types were already checked in the sema HM stage).
        if self.instantiation_ctx.is_some() {
            return;
        }
        // Lazy<T> subsumption: Lazy<T> auto-unwraps to T when the context expects T.
        // This makes `lazy(1i32) + 3i32` type-check (Lazy<i32> unwraps to i32).
        {
            let InferContext { arena, .. } = self;
            let ra = arena.resolve(t1);
            let rb = arena.resolve(t2);
            let a_is_lazy = matches!(arena.get(ra), Type::Lazy(_));
            let b_is_lazy = matches!(arena.get(rb), Type::Lazy(_));
            if a_is_lazy && !b_is_lazy {
                let inner = arena.lazy_value(ra);
                self.unify_or_constrain(inner, t2);
                return;
            }
            if b_is_lazy && !a_is_lazy {
                let inner = arena.lazy_value(rb);
                self.unify_or_constrain(t1, inner);
                return;
            }
        }
        // Record candidate before unify so finalize_solution can detect ambiguity when
        // the same TypeVar is required to bind to multiple distinct concrete types
        // (Bug #83: `pair(1i32, 2i64)` silently bound T to i32). Without this, only
        // failed unifies recorded candidates, so a TypeVar bound by the first
        // successful unify + a conflicting failed unify would show a single candidate.
        let InferContext { arena, solver, .. } = self;
        solver.record_candidate(arena, t1, t2);
        if arena.unify(t1, t2).is_err() {
            solver.add_equality(t1, t2);
        }
    }

    /// Call-site argument check with the hard-concrete rule: when BOTH the
    /// parameter and the argument are fully concrete (no TypeVar anywhere) and
    /// cannot unify, no solver iteration can ever reconcile them — report the
    /// mismatch immediately. This closes the silent str-through-Path hole
    /// (`File.open("x", ..)` compiled and shifted every argument one ABI slot
    /// at runtime). TypeVars/literals keep the lenient constraint path (the
    /// soft-unify wall stdlib relies on stays intact).
    pub fn unify_call_arg(&mut self, param: TypeHandle, arg: TypeHandle, line: u32, column: u32) {
        let (concrete_pair, hard_fail) = {
            let InferContext { arena, .. } = self;
            let rp = arena.resolve(param);
            let ra = arena.resolve(arg);
            use super::CallInfer::type_contains_typevar;
            let concrete_pair = !type_contains_typevar(arena, rp)
                && !type_contains_typevar(arena, ra);
            let hard_fail = arena.unify(rp, ra).is_err();
            (concrete_pair, hard_fail)
        };
        // Numeric pairs are exempt: implicit width widening (i64 -> i128 etc.)
        // is an established lenient path the stdlib leans on; this sentinel is
        // for STRUCTURAL mismatches (str into Path, Path into str?, ...).
        let both_numeric = self.arena.get(self.arena.resolve(param)).is_numeric()
            && self.arena.get(self.arena.resolve(arg)).is_numeric();
        // The builtin `Error` interface accepts ANY error type (throw-family
        // covariance): exempt it from the structural check.
        let param_is_error_iface = matches!(
            self.arena.get(self.arena.resolve(param)),
            crate::types::Ty::Type::Adt(_)
        ) && {
            let (name, _) = self.arena.adt_parts(self.arena.resolve(param));
            name == "Error"
        };
        if concrete_pair && !both_numeric && !param_is_error_iface && hard_fail {
            // Nullable/throw structural promotion (T -> T?) is legitimate for
            // concrete pairs; only when BOTH the plain unify and the widening
            // path fail is it a real mismatch (stdlib passes str into str?
            // parameters, and str? unwraps into str receivers).
            let widened = self.try_widen_unify(param, arg).is_ok();
            if !widened {
                let p_str = format!("{}", self.arena.display(self.arena.resolve(param)));
                let a_str = format!("{}", self.arena.display(self.arena.resolve(arg)));
                self.add_error_at(
                    &format!(
                        "argument type mismatch: cannot pass '{}' where '{}' is expected",
                        a_str, p_str
                    ),
                    line,
                    column,
                );
                return;
            }
        }
        self.unify_or_constrain(param, arg);
    }

    /// Attempts a widening unification of two types and returns the unified type.
    /// First tries a strict unify; on failure, if both are numeric, picks one per the widening rules;
    /// otherwise handles structural compatibility for nullable/throw vs. ordinary types, void, etc.
    pub fn try_widen_unify(
        &mut self,
        t1: TypeHandle,
        t2: TypeHandle,
    ) -> Result<TypeHandle, UnifyError> {
        let r1 = self.arena.resolve(t1);
        let r2 = self.arena.resolve(t2);

        // never unifies with any type as the other type.
        if matches!(self.arena.get(r1), Type::Never) {
            return Ok(r2);
        }
        if matches!(self.arena.get(r2), Type::Never) {
            return Ok(r1);
        }

        // First try a strict unify.
        if self.arena.unify(r1, r2).is_ok() { return Ok(r1) }

        let c1 = self.arena.get(r1);
        let c2 = self.arena.get(r2);

        // Async unfolding: Async<X> with Y (non-Async) → recursively unify X with Y.
        // Scenario: inside an async function body, Ok(void) returns Throw<void, '_E>,
        // expected is Async<Throw<void, IOError>>; the Async layer must be unfolded to solve '_E.
        if let Type::Async(_) = c1 {
            let inner = self.arena.async_value(r1);
            return self.try_widen_unify(inner, r2);
        }
        if let Type::Async(_) = c2 {
            let inner = self.arena.async_value(r2);
            return self.try_widen_unify(r1, inner);
        }

        // Bug #60 (Plan A fully strict): numeric widening is removed — different numeric types
        // are no longer implicitly promoted; an explicit cast is required. Strict unify has already
        // been attempted above and failed; numeric type pairs fall through to the match's _ branch and return TypeMismatch.

        match (c1, c2) {
            (Type::Nullable(_), _) => match c2 {
                Type::Nullable(_) => {
                    let i1 = self.arena.resolve(self.arena.nullable_inner(r1));
                    let i2 = self.arena.resolve(self.arena.nullable_inner(r2));
                    match self.arena.unify(i1, i2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
                Type::Void => Ok(r1), // void can be treated as the "empty" value of a nullable.
                _ => {
                    // nullable<T> is compatible with T.
                    let inner1_ty = self.arena.nullable_inner(r1);
                    match self.arena.unify(inner1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
            },
            (Type::Throw(_), _) => match c2 {
                Type::Throw(_) => {
                    let (vt1, et1) = self.arena.throw_parts(r1);
                    let (vt2, et2) = self.arena.throw_parts(r2);
                    let v1 = self.arena.resolve(vt1);
                    let v2 = self.arena.resolve(vt2);
                    let e1 = self.arena.resolve(et1);
                    let e2 = self.arena.resolve(et2);
                    self.arena.unify(e1, e2)?;
                    match self.arena.unify(v1, v2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            // Recursively call try_widen_unify to handle structural compatibility for Nullable/Throw etc.,
                            // but no longer performs numeric widening (Plan A fully strict).
                            match self.try_widen_unify(v1, v2) {
                                Ok(_) => Ok(r1),
                                Err(_) => Err(UnifyError::TypeMismatch),
                            }
                        }
                    }
                }
                Type::Void => Ok(r1), // void can be treated as throw's "no value taken".
                _ => {
                    // Throw<T, E> is compatible with T (value dimension only).
                    let (vt1_ty, _) = self.arena.throw_parts(r1);
                    match self.arena.unify(vt1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => Err(UnifyError::TypeMismatch),
                    }
                }
            },
            (Type::Void, _) => match c2 {
                Type::Nullable(_) | Type::Throw(_) => Ok(r2),
                _ => Err(UnifyError::TypeMismatch),
            },
            (_, Type::Nullable(_)) => {
                // T is compatible with nullable<T>; unify as nullable.
                let inner2_ty = self.arena.nullable_inner(r2);
                self.arena.unify(r1, inner2_ty)?;
                Ok(r2)
            }
            (_, Type::Throw(_)) => {
                // T is compatible with Throw<T, E>; unify as throw.
                let (vt2_ty, _) = self.arena.throw_parts(r2);
                self.arena.unify(r1, vt2_ty)?;
                Ok(r2)
            }
            _ => Err(UnifyError::TypeMismatch),
        }
    }

    /// Checks the legality of the propagation operator `?` on an expression and returns the unfolded type.
    ///
    /// `expected_return` is the enclosing function's return type (possibly `Async<Throw<V, E>>` or `Throw<V, E>`),
    /// used to unify error_type so that throw propagation types correctly.
    ///
    /// - nullable: unfolds to the inner type.
    /// - throw: unfolds to the value type and unifies error_type with the enclosing function's error_type.
    /// - TypeVar: deferred to the solver; returns a fresh_type_var to avoid cascading false positives.
    /// - other types: reports an error and returns the original type.
    pub fn check_propagate(
        &mut self,
        resolved_inner: TypeHandle,
        inner_ty: TypeHandle,
        expected_return: Option<TypeHandle>,
        line: u32,
        column: u32,
    ) -> TypeHandle {
        let ct = self.arena.get(resolved_inner);
        match ct {
            Type::Nullable(_) => self.arena.nullable_inner(resolved_inner),
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved_inner);
                // Unify error_type with the enclosing function's error_type (when the outer function is a throwing one).
                // Frond allows `?` in non-throwing functions (panics/exits on failure); in that case error_type is not propagated.
                if let Some(er) = expected_return {
                    let er_resolved = self.arena.resolve(er);
                    let er_ty = self.arena.get(er_resolved);
                    // async function: expected_return may be Async<Throw<V', E'>>.
                    let outer_throw_handle = if let Type::Async(_) = er_ty {
                        Some(self.arena.resolve(self.arena.async_value(er_resolved)))
                    } else {
                        None
                    };
                    let outer_resolved = outer_throw_handle.unwrap_or(er_resolved);
                    if let Type::Throw(_) = self.arena.get(outer_resolved) {
                        let (_, outer_err) = self.arena.throw_parts(outer_resolved);
                        self.unify_or_constrain(error_type, outer_err);
                    }
                    // Non-Throw outer (e.g. void) or TypeVar: silently skip, no error.
                }
                value_type
            }
            Type::TypeVar(_) => {
                // Operand type not yet determined; defer to the solver for later judgment.
                // Return a fresh_type_var to avoid cascading false positives in downstream lookups.
                self.arena.fresh_type_var()
            }
            _ => {
                self.add_error_at(
                    "propagation operator '?' cannot be used on a non-nullable, non-throw expression",
                    line,
                    column,
                );
                inner_ty
            }
        }
    }

    /// Checks the expression type of a throw statement.
    /// Frond has no try-catch; throw is a general-purpose raising mechanism that accepts any ADT/Record/Throw/TypeVar.
    pub fn check_throw_stmt(&mut self, thrown_ty: TypeHandle, _line: u32, _column: u32) {
        let resolved = self.arena.resolve(thrown_ty);
        let ct = self.arena.get(resolved);
        match ct {
            Type::TypeVar(_) => return,   // Deferred to the unification stage.
            Type::Throw(_) => return, // throw Error("...") returns Throw; legal.
            Type::Adt(_) | Type::Generic(_) => return, // Error type (ordinary ADT).
            _ => return, // Permissive: throw is a general-purpose mechanism.
        }
    }

    // ── infer_expr / infer_stmt / infer_pattern placeholders (implemented below) ──

}
