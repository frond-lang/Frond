//! Match — Pattern/match inference and exhaustiveness (usefulness matrix). Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Performs GADT type refinement for a constructor pattern.
    ///
    /// **Semantics** (ported from `src/sema/gadt_check.zig` refineConstructorPattern):
    /// 1. Look up the constructor definition (CtorDefInfo) from sema_result.
    /// 2. Unify the constructor's return type with expected_ty to refine type variables.
    /// 3. Recursively infer sub-patterns against the constructor's field types.
    ///
    /// **Return value**: `true` means this function handled it (the constructor is registered);
    /// `false` means the constructor is unregistered and falls back to regular pattern inference.
    ///
    /// **Throw error branch**: when expected_ty is a Throw type and the constructor is an error_newtype
    /// ADT constructor, the `is_throw_error_branch` flag is true, and the constructor's return type and
    /// sub-patterns are unified and bound to error_type. This flag flows through both the return-type
    /// resolution and sub-pattern binding steps, with no separate early-exit branch; it shares the same
    /// control flow as the regular GADT path.
    pub fn refine_constructor_pattern(
        &mut self,
        ctor_name: &str,
        sub_patterns: &[PatternRef],
        expected_ty: TypeHandle,
        ast: &AstArena<'_>,
        env: EnvId,
        line: u32,
        column: u32,
    ) -> bool {
        // Use field_type_reprs (self-contained TypeRepr) instead of field_type_nodes (AST references),
        // to avoid AST arena mismatch when used cross-module, which would make TypeRef indices point at the wrong type nodes.
        // return_type_node still uses AstTypeRef (GADT cases are rare and usually intra-module).
        type CtorInfoSnapshot = (Box<str>, bool, Option<AstTypeRef>, Box<[TypeRepr]>);
        let resolved_expected = self.arena.resolve(expected_ty);

        // find_ctor_def returns an owned snapshot (S5 diagnostics mutate
        // mid-lookup); destructure into the local tuple shape.
        let ctor_info: Option<CtorInfoSnapshot> =
            self.find_ctor_def(ctor_name, expected_ty, line, column).map(|c| {
                (
                    c.type_name.clone(),
                    c.is_newtype,
                    c.return_type_node,
                    c.field_type_reprs.clone(),
                )
            });

        // Throw<T, E> builtin type variant matching:
        // Throw is a builtin sum type; its variants Ok(T) / Error(E) are not registered as CtorDefInfo.
        // - Ok (or any non-error constructor) → value variant → sub-patterns bind to value_type.
        // - Error / Err (by name) or any registered error ADT constructor → error variant → sub-patterns bind to error_type.
        //   (When the constructor name collides with the Throw error-variant name "Error"/"Err",
        //    regardless of what error_type is, the pattern matches the Throw error variant
        //    and sub-patterns bind to error_type rather than the constructor's field types.)
        if let Type::Throw(_) = self.arena.get(resolved_expected) {
            let (value_type, error_type) = self.arena.throw_parts(resolved_expected);
            let is_error_variant = ctor_name == crate::ir::Compute::CTOR_ERR
                || ctor_name == crate::ir::Compute::CTOR_ERR_ALT
                || ctor_info.is_some();
            let branch_ty = if is_error_variant { error_type } else { value_type };
            for &sub_pat in sub_patterns.iter() {
                self.infer_pattern(sub_pat, ast, branch_ty, env);
            }
            return true;
        }

        let (type_name, is_newtype, return_type_node, field_type_reprs) = match ctor_info {
            Some(info) => info,
            None => return false,
        };
        let _ = is_newtype;

        // Resolve the constructor's return type (GADT → return_type_node; regular ADT → the Adt for type_name).
        // Goes through InferContext's full type resolution (type_from_ast), unifying handling of all TypeNode variants,
        // to eliminate type loss caused by the simplified resolve_type_node_to_handle falling back to fresh_type_var for complex types.
        let ctor_return_ty = if let Some(rtn) = return_type_node {
            self.type_from_ast(rtn, ast)
        } else {
            self.arena.make_adt(type_name, Box::new([]))
        };

        // Unify the constructor's return type with the expected type to perform GADT type refinement.
        // On failure, register a constraint for the fixpoint iteration to retry.
        // A constructor pattern matches the ADT INSIDE a Nullable scrutinee
        // (`match v: J? { DObj(e) => .., _ => null }` — null routes to other
        // arms): peel the Nullable before the compatibility unify, otherwise
        // plain arena.unify fails and sub-patterns wrongly bind to J?.
        let unify_target = {
            let r = self.arena.resolve(expected_ty);
            match self.arena.get(r) {
                Type::Nullable(_) => self.arena.nullable_inner(r),
                _ => expected_ty,
            }
        };
        let ctor_compatible = self.arena.unify(ctor_return_ty, unify_target).is_ok();
        if !ctor_compatible {
            self.unify_or_constrain(ctor_return_ty, unify_target);
        }

        // Recursively infer sub-patterns against the constructor's field types and bind variables.
        // When the constructor's return type is incompatible with the expected type (e.g. an Error ADT used to unwrap a Throw's error_type),
        // sub-patterns bind to expected_ty rather than the constructor's field types, ensuring pattern variables get the correct runtime type.
        for (i, &sub_pat) in sub_patterns.iter().enumerate() {
            let sub_ty = if !ctor_compatible {
                expected_ty
            } else if i < field_type_reprs.len() {
                self.type_repr_to_handle(&field_type_reprs[i])
            } else {
                self.arena.fresh_type_var()
            };
            self.infer_pattern(sub_pat, ast, sub_ty, env);
        }

        true
    }

    /// Looks up a constructor definition by name from sema_result.
    /// When multiple types define the same constructor name, uses `expected_ty`
    /// to disambiguate (type-oriented pattern constructor resolution).
    /// S5 (zero-silence): when ≥2 candidates survive the type-oriented pass,
    /// the historical first-candidate fallback silently misbound sub-patterns;
    /// it is now an error (qualify the spelling: `Module.Ctor`).
    pub(super) fn find_ctor_def(
        &mut self,
        ctor_name: &str,
        expected_ty: TypeHandle,
        line: u32,
        column: u32,
    ) -> Option<crate::sema::Sema::CtorDefInfo> {
        // Owned snapshot: the S5 diagnostics mutate self mid-lookup, so a
        // borrowed return would fight the borrow checker for no gain.
        // Error-domain exemption: the builtin anonymous error interface
        // (bare "Error") is an OPEN type — nested ctor patterns name
        // implementations across types by design (throw/catch idioms:
        // `Error(NotFound(_))` against `Throw<str, Error>`); cross-owner
        // candidates there are the intended semantics, not ambiguity.
        let expected_is_error_iface = {
            let r = self.arena.resolve(expected_ty);
            match self.arena.get(r) {
                Type::Adt(_) => self.arena.adt_parts(r).0 == "Error",
                _ => false,
            }
        };
        // Qualified spelling `Module.Ctor`: narrow the bare-name candidates to
        // the qualifier's module, then keep the expected-type disambiguation
        // among same-module candidates.
        if ctor_name.contains('.') {
            let bare = ctor_name.rsplit('.').next().unwrap_or(ctor_name);
            let qual = &ctor_name[..ctor_name.len() - bare.len() - 1];
            let anchor = self
                .sema_result
                .resolve_module_qualifier(&self.sema_result.current_module_name, qual);
            let candidates: Vec<&CtorDefInfo> = self
                .sema_result
                .get_ctor_defs(bare)
                .into_iter()
                .filter(|c| match &anchor {
                    // std/builtin: owning type names are bare.
                    Some((_, true)) => !c.type_name.contains('.'),
                    Some((a, false)) => c
                        .type_name
                        .strip_prefix(a.as_str())
                        .is_some_and(|r| r.starts_with('.')),
                    None => false,
                })
                .collect();
            if candidates.len() == 1 {
                return Some(candidates[0].clone());
            }
            if candidates.len() > 1 {
                let exp_resolved = self.arena.resolve(expected_ty);
                if let Type::Adt(_) = self.arena.get(exp_resolved) {
                    let (exp_type_name, _) = self.arena.adt_parts(exp_resolved);
                    let matches: Vec<&CtorDefInfo> = candidates
                        .iter()
                        .copied()
                        .filter(|c| c.type_name.as_ref() == exp_type_name)
                        .collect();
                    if matches.len() == 1 {
                        return Some((*matches[0]).clone());
                    }
                }
            }
            // S5: extract the pick + owners first so the sema_result borrow
            // ends before the diagnostic's &mut self.
            let ambiguous = candidates.len() > 1;
            let owners: Vec<String> = candidates
                .iter()
                .map(|c| c.type_name.as_ref().to_string())
                .collect();
            let pick = candidates.into_iter().next().cloned();
            if ambiguous && !expected_is_error_iface {
                self.ambiguous_pattern_error(ctor_name, &owners, line, column);
            }
            return pick;
        }
        let candidates = self.sema_result.get_ctor_defs(ctor_name);
        if candidates.len() <= 1 {
            return candidates.into_iter().next().cloned();
        }
        // Type-oriented disambiguation: select by the Adt type_name of expected_ty
        let exp_resolved = self.arena.resolve(expected_ty);
        if let Type::Adt(_) = self.arena.get(exp_resolved) {
            let (exp_type_name, _) = self.arena.adt_parts(exp_resolved);
            let matches: Vec<_> = candidates.iter()
                .filter(|c| c.type_name.as_ref() == exp_type_name)
                .collect();
            if matches.len() == 1 {
                return Some((*matches[0]).clone());
            }
        }
        // S5: ambiguity survives the type-oriented pass — diagnose instead
        // of silently binding the first candidate's fields.
        let owners: Vec<String> = candidates
            .iter()
            .map(|c| c.type_name.as_ref().to_string())
            .collect();
        let pick = candidates.into_iter().next().cloned();
        if !expected_is_error_iface {
            self.ambiguous_pattern_error(ctor_name, &owners, line, column);
        }
        pick
    }

    /// S5: report a pattern-position constructor ambiguity (owners listed;
    /// qualified spelling is the escape hatch). Reports and continues with
    /// the first candidate so downstream inference still has a shape.
    fn ambiguous_pattern_error(
        &mut self,
        ctor_name: &str,
        owners: &[String],
        line: u32,
        column: u32,
    ) {
        self.add_error_at(
            &format!(
                "ambiguous constructor pattern '{}': defined by types [{}]; use Module.{} to disambiguate",
                ctor_name,
                owners.join(", "),
                ctor_name.rsplit('.').next().unwrap_or(ctor_name),
            ),
            line,
            column,
        );
    }

    // ── Usefulness algorithm (Maranget) for match exhaustiveness checking ──

    /// Convert a `TypeRepr` to a `TypeHandle`, substituting type parameters with
    /// the actual type arguments from the scrutinee type (for generic ADTs).
    pub(super) fn ctor_field_type(
        &mut self,
        repr: &TypeRepr,
        params: &[Box<str>],
        args: &[TypeHandle],
    ) -> TypeHandle {
        if let TypeRepr::Named(name) = repr {
            if let Some(idx) = params.iter().position(|p| p.as_ref() == name.as_ref()) {
                if idx < args.len() {
                    return args[idx];
                }
            }
        }
        self.type_repr_to_handle(repr)
    }

    /// Get the arity and field types of a constructor, given the column type
    /// (used to disambiguate same-named constructors across types and to substitute
    /// generic type parameters).
    pub(super) fn ctor_arity_and_fields(
        &mut self,
        col_type: TypeHandle,
        ctor: &PatCtor,
    ) -> (usize, Vec<TypeHandle>) {
        // Throw<T, E> builtin: Ok has arity 1 (field = value_type),
        // Error (or any registered error-variant constructor) has arity 1
        // (field = error_type). These constructors are NOT registered as
        // CtorDefInfo (see refine_constructor_pattern), so they need explicit
        // handling to avoid arity-0 fallback that loses sub-pattern information.
        if let PatCtor::Adt(name) = ctor {
            let resolved = self.arena.resolve(col_type);
            if let Type::Throw(_) = self.arena.get(resolved) {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                let field_ty = if name.as_ref() == "Ok" { value_type } else { error_type };
                return (1, vec![field_ty]);
            }
        }

        // Phase 1: collect data via immutable borrows (all cloned to release borrows).
        let collected: Option<(Box<[TypeRepr]>, Box<[Box<str>]>, Vec<TypeHandle>)> = match ctor {
            PatCtor::Adt(name) => {
                let resolved = self.arena.resolve(col_type);
                match self.arena.get(resolved) {
                    Type::Adt(_) => {
                        let (type_name, type_args) = self.arena.adt_parts(resolved);
                        self.sema_result.get_type_def(type_name).and_then(|td| {
                            td.constructors.iter()
                                .find(|c| c.name.as_ref() == name.as_ref())
                                .map(|c| {
                                    (c.field_type_reprs.clone(), td.type_params.clone(), type_args.to_vec())
                                })
                        })
                    }
                    _ => self.sema_result.get_ctor_def(name).map(|c| {
                        (c.field_type_reprs.clone(), Box::new([]) as Box<[Box<str>]>, Vec::new())
                    }),
                }
            }
            _ => None,
        };

        // Phase 2: convert field type reprs to handles (mutable borrow).
        match collected {
            Some((reprs, params, args)) => {
                let field_types: Vec<TypeHandle> = reprs.iter()
                    .map(|r| self.ctor_field_type(r, &params, &args))
                    .collect();
                (field_types.len(), field_types)
            }
            None => (0, Vec::new()),
        }
    }

    /// Return all constructors of a type if it has a finite (complete) signature.
    /// Returns `None` for types with infinite value spaces (int, float, char, str,
    /// nullable, etc.).
    pub(super) fn type_all_ctors(&self, col_type: TypeHandle) -> Option<Vec<PatCtor>> {
        let resolved = self.arena.resolve(col_type);
        match self.arena.get(resolved) {
            Type::Adt(_) => {
                let (type_name, _) = self.arena.adt_parts(resolved);
                let type_def = self.sema_result.get_type_def(type_name)?;
                if type_def.constructors.is_empty() {
                    return None;
                }
                Some(type_def.constructors.iter()
                    .map(|c| PatCtor::Adt(c.name.clone()))
                    .collect())
            }
            Type::Bool => Some(vec![PatCtor::Bool(true), PatCtor::Bool(false)]),
            _ => None,
        }
    }

    /// Core usefulness check: is `query` useful w.r.t. `matrix`?
    /// Implements Maranget's algorithm U(M, q).
    pub(super) fn is_useful(
        &mut self,
        matrix: &[Vec<NormPat>],
        col_types: &[TypeHandle],
        query: &[NormPat],
        depth: usize,
    ) -> bool {
        // Base case: no columns → useful iff the matrix has no rows.
        if query.is_empty() {
            return matrix.is_empty();
        }
        if col_types.is_empty() {
            return false; // defensive: no type info
        }
        // Safety valve: prevent exponential blowup on pathological nesting.
        if depth > 48 {
            return false;
        }

        let col_type = col_types[0];
        match &query[0] {
            NormPat::Wild => {
                // Collect distinct constructors appearing in the first column.
                let mut seen: Vec<PatCtor> = Vec::new();
                for row in matrix {
                    if !row.is_empty() {
                        if let NormPat::Ctor(c, _) = &row[0] {
                            if !seen.contains(c) {
                                seen.push(c.clone());
                            }
                        }
                    }
                }

                if seen.is_empty() {
                    // No constructors in column → check default matrix.
                    let (dm, dt) = default_matrix(matrix, col_types);
                    return self.is_useful(&dm, &dt, &query[1..], depth + 1);
                }

                // Try each seen constructor: if any makes the query useful, done.
                for ctor in &seen {
                    let (arity, field_types) = self.ctor_arity_and_fields(col_type, ctor);
                    let (sm, st) = specialize_matrix(matrix, col_types, ctor, arity, &field_types);
                    let mut sq = vec![NormPat::Wild; arity];
                    sq.extend(query[1..].iter().cloned());
                    if self.is_useful(&sm, &st, &sq, depth + 1) {
                        return true;
                    }
                }

                // All seen constructors failed; check if the signature is complete.
                let is_complete = self.type_all_ctors(col_type)
                    .map(|all| all.iter().all(|c| seen.contains(c)))
                    .unwrap_or(false);
                if is_complete {
                    return false;
                }
                // Incomplete signature → wildcard is useful via the default matrix.
                let (dm, dt) = default_matrix(matrix, col_types);
                self.is_useful(&dm, &dt, &query[1..], depth + 1)
            }
            NormPat::Ctor(target, sub) => {
                let (arity, field_types) = self.ctor_arity_and_fields(col_type, target);
                let (sm, st) = specialize_matrix(matrix, col_types, target, arity, &field_types);
                let mut sq = sub.clone();
                if sq.len() > arity {
                    sq.truncate(arity);
                }
                while sq.len() < arity {
                    sq.push(NormPat::Wild);
                }
                sq.extend(query[1..].iter().cloned());
                self.is_useful(&sm, &st, &sq, depth + 1)
            }
        }
    }

    /// Generate a human-readable witness for a non-exhaustive match (what's missing).
    pub(super) fn witness(
        &mut self,
        matrix: &[Vec<NormPat>],
        col_types: &[TypeHandle],
        depth: usize,
    ) -> String {
        if col_types.is_empty() || depth > 8 {
            return String::new();
        }
        let col_type = col_types[0];
        let resolved = self.arena.resolve(col_type);
        let ty = self.arena.get(resolved);

        let seen: Vec<PatCtor> = matrix.iter()
            .filter_map(|r| {
                if !r.is_empty() {
                    if let NormPat::Ctor(c, _) = &r[0] { Some(c.clone()) } else { None }
                } else { None }
            })
            .collect();
        let has_wild = matrix.iter().any(|r| !r.is_empty() && matches!(r[0], NormPat::Wild));

        match ty {
            Type::Adt(_) => {
                let (type_name, _) = self.arena.adt_parts(resolved);
                // Collect constructor names (cloned to release the borrow).
                let ctor_names: Vec<Box<str>> = self.sema_result.get_type_def(type_name)
                    .map(|td| td.constructors.iter().map(|c| c.name.clone()).collect())
                    .unwrap_or_default();

                // Check for missing top-level constructors.
                let missing: Vec<&str> = ctor_names.iter()
                    .map(|n| n.as_ref())
                    .filter(|n| !seen.iter().any(|c| matches!(c, PatCtor::Adt(a) if a.as_ref() == *n)))
                    .collect();
                if !missing.is_empty() {
                    return format!(": missing {}", missing.join(", "));
                }

                // All top-level constructors present but nested issue; recurse.
                for name in &ctor_names {
                    let pc = PatCtor::Adt(name.clone());
                    if seen.contains(&pc) {
                        let (arity, ft) = self.ctor_arity_and_fields(col_type, &pc);
                        let (sm, st) = specialize_matrix(matrix, col_types, &pc, arity, &ft);
                        let q = vec![NormPat::Wild; arity];
                        if self.is_useful(&sm, &st, &q, 0) {
                            return format!(": `{}` not exhaustive{}", name,
                                self.witness(&sm, &st, depth + 1));
                        }
                    }
                }
                String::new()
            }
            Type::Bool => {
                if !seen.iter().any(|c| matches!(c, PatCtor::Bool(true))) {
                    ": missing `true`".to_string()
                } else if !seen.iter().any(|c| matches!(c, PatCtor::Bool(false))) {
                    ": missing `false`".to_string()
                } else {
                    String::new()
                }
            }
            _ => {
                if !has_wild {
                    ": missing catch-all `_`".to_string()
                } else {
                    String::new()
                }
            }
        }
    }

    /// Check match exhaustiveness using the usefulness algorithm (Maranget).
    ///
    /// Reports a non-exhaustive error if any value of the scrutinee type is not
    /// covered by the (unguarded) match arms. Also warns on unreachable arms.
    pub(super) fn check_match_exhaustive(
        &mut self,
        ast: &AstArena<'_>,
        scrutinee: crate::ast::Ast::ExprId,
        scrutinee_ty: TypeHandle,
        arms: &[crate::ast::Ast::MatchArm],
    ) {
        let col_types = vec![scrutinee_ty];

        // An arm is "guarded" if it has an arm-level guard OR an inline Pattern::Guard.
        // Guarded arms do not guarantee coverage (the guard may fail) and do not make
        // subsequent arms unreachable.
        let is_arm_guarded = |arm: &crate::ast::Ast::MatchArm| -> bool {
            arm.guard.is_some() || matches!(ast.pattern(arm.pattern).node, Pattern::Guard { .. })
        };

        // ── Exhaustiveness check ──
        // Build matrix from unguarded arms only (guarded arms don't guarantee coverage).
        let exhaust_matrix: Vec<Vec<NormPat>> = arms.iter()
            .filter(|arm| !is_arm_guarded(arm))
            .flat_map(|arm| {
                let pat = unwrap_guard_pat(ast, arm.pattern);
                normalize_pattern(ast, pat).into_iter().map(|p| vec![p])
            })
            .collect();

        // Only report non-exhaustive for types with finite constructor sets (ADT, Bool).
        // For infinite types (int, str, char, ...), preserve existing lenient behavior.
        let resolved = self.arena.resolve(scrutinee_ty);
        let is_finite = matches!(self.arena.get(resolved), Type::Adt(_) | Type::Bool);

        if is_finite && self.is_useful(&exhaust_matrix, &col_types, &[NormPat::Wild], 0) {
            let span = ast.expr(scrutinee).span;
            let witness = self.witness(&exhaust_matrix, &col_types, 0);
            self.add_error_at(
                &format!("non-exhaustive match{}", witness),
                span.line,
                span.column,
            );
        }

        // ── Unreachable arm detection ──
        // Arm i is unreachable iff its pattern is not useful given the matrix of
        // previous *unguarded* arms (guarded arms don't block subsequent arms).
        let mut prev_matrix: Vec<Vec<NormPat>> = Vec::new();
        for arm in arms.iter() {
            let pat = unwrap_guard_pat(ast, arm.pattern);
            let alternatives = normalize_pattern(ast, pat);

            let any_useful = alternatives.iter()
                .any(|alt| self.is_useful(&prev_matrix, &col_types, &[alt.clone()], 0));

            if !any_useful && !alternatives.is_empty() {
                let span = ast.pattern(arm.pattern).span;
                self.add_warning_at("unreachable match arm", span.line, span.column);
            }

            // Add this arm's patterns to the previous matrix (only if unguarded).
            if !is_arm_guarded(arm) {
                for alt in alternatives {
                    prev_matrix.push(vec![alt]);
                }
            }
        }
    }

    /// Check if `Type.Ctor` is a qualified constructor access.
    /// Returns `Some((type_name, field_type_reprs))` when `type_name` is a
    /// registered type and `ctor_name` is one of its constructors.
    pub(super) fn check_qualified_ctor(
        &self,
        type_name: &str,
        ctor_name: &str,
    ) -> Option<(Box<str>, Box<[TypeRepr]>)> {
        // Module-scoped: the AST type segment resolves to its canonical key.
        let canonical = self.sema_result.resolve_type_key(type_name);
        let type_idx = self.sema_result.type_def_idx(&canonical)?;
        let type_def = &self.sema_result.type_defs[&type_idx];
        let ctor = type_def
            .constructors
            .iter()
            .find(|c| c.name.as_ref() == ctor_name)?;
        Some((ctor.type_name.clone(), ctor.field_type_reprs.clone()))
    }

    /// Infer an `Expr::Match` expression (extracted from `infer_expr_inner`).
    pub(super) fn infer_match_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr(*scrutinee, ast, env, None);
                let resolved_scrutinee = self.arena.resolve(scrutinee_ty);

                // Throw type widening: when a match contains both Ok and Error/error-constructor patterns,
                // the scrutinee should be a Throw<T, E>. If the scrutinee is an Err implementer (e.g. an Error ADT)
                // rather than a Throw, widen it to Throw<fresh, scrutinee_ty> (with scrutinee as error_type),
                // so that Ok(v) matches the value variant and Error(e) matches the error variant and binds the whole error value (not the constructor's fields).
                let resolved_scrutinee = {
                    let is_throw = matches!(
                        self.arena.get(resolved_scrutinee),
                        Type::Throw(_)
                    );
                    let has_ok_arm = arms.iter().any(|arm| {
                        match &ast.pattern(arm.pattern).node {
                            Pattern::Constructor { name, .. } => *name == crate::ir::Compute::CTOR_OK,
                            Pattern::Variable { name } => *name == crate::ir::Compute::CTOR_OK,
                            _ => false,
                        }
                    });
                    if !is_throw && has_ok_arm {
                        // Check whether the scrutinee implements the Err trait (via the witness table).
                        let implements_err = self.arena.type_name(resolved_scrutinee)
                            .and_then(|tn| {
                                let canonical = self.sema_result.resolve_type_key(tn);
                                self.sema_result.type_def_idx(&canonical)
                            })
                            .map(|idx| self.witness_table.implements("Err", dynamic_type_id(idx)))
                            .unwrap_or(false);
                        let widened = if implements_err {
                            // scrutinee is an error type → Throw<fresh_val, scrutinee>.
                            let fresh_val = self.arena.fresh_type_var();
                            self.arena.make_throw(fresh_val, resolved_scrutinee)
                        } else {
                            // scrutinee is a value type → Throw<scrutinee, fresh_err>.
                            let fresh_err = self.arena.fresh_type_var();
                            self.arena.make_throw(resolved_scrutinee, fresh_err)
                        };
                        self.unify_or_constrain(scrutinee_ty, widened);
                        self.arena.resolve(widened)
                    } else {
                        resolved_scrutinee
                    }
                };

                // sema v2: extract the scrutinee's path (used for ConstructorMatch narrowing).
                let scrutinee_path = expr_path(ast, *scrutinee);

                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                // Nullable arm narrowing: on a Nullable scrutinee, a binding
                // arm (`name => ...`) that follows an explicit `null` arm is
                // known non-null — narrow the binding to the inner type
                // (stdlib idiom `match fname { null => .., name => .. }`).
                let scrutinee_nullable =
                    matches!(self.arena.get(resolved_scrutinee), Type::Nullable(_));
                let mut null_arm_seen = false;
                for arm in arms.iter() {
                    let child_env = self.sema_result.env.child(env);
                    let is_null_arm = matches!(
                        &ast.pattern(arm.pattern).node,
                        Pattern::Literal(PatternLiteral::Null)
                    );
                    let arm_pattern_ty = if scrutinee_nullable && null_arm_seen && !is_null_arm {
                        self.arena.nullable_inner(resolved_scrutinee)
                    } else {
                        resolved_scrutinee
                    };
                    if is_null_arm {
                        null_arm_seen = true;
                    }

                    // sema v2: enter the match-arm scope and apply ConstructorMatch narrowing.
                    self.flow_ctx.push_scope();
                    if let Some(ref path) = scrutinee_path {
                        // Check whether this is a constructor pattern; if so, add a ConstructorMatch fact.
                        if let Some((ctor_name, bound_vars)) =
                            extract_constructor_pattern(&ast.pattern(arm.pattern).node, ast)
                        {
                            // Constructor match: the scrutinee is narrowed to this constructor's type.
                            let narrowed_ty = self.arena.make_adt(
                                ctor_name.into(),
                                Box::new([]),
                            );
                            self.flow_ctx.add_fact(FlowFact {
                                path: path.clone().into(),
                                narrowed_ty,
                                kind: NarrowKind::ConstructorMatch {
                                    ctor_name: ctor_name.into(),
                                    bound_vars: bound_vars.into(),
                                },
                            });
                        }
                    }

                    self.infer_pattern(arm.pattern, ast, arm_pattern_ty, child_env);
                    if let Some(guard) = arm.guard {
                        let _ = self.infer_expr(guard, ast, child_env, None);
                    }
                    // Propagate the match's expected type to the arm body,
                    // so expressions that depend on the expected constraint (e.g. NullLit) infer correctly.
                    let body_ty = self.infer_expr(arm.body, ast, child_env, expected);
                    self.flow_ctx.pop_scope();

                    arm_tys.push(body_ty);
                }

                // v2 convergence: use only peer_type to unify all arm types (eliminates the per-arm widen dual-track scheme).
                // peer_type handles single-arm (return directly), multi-arm (join), and all-Never/Void (return Never/Void).
                let result_ty = if arm_tys.is_empty() {
                    self.make_builtin(Type::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                };
                // Constrain the match result against the outer expected type, so that when the match is the RHS of a let,
                // pending TypeVars in the result type can be solved in reverse.
                if let Some(exp) = expected {
                    self.unify_or_constrain(result_ty, exp);
                }

                // ── Exhaustiveness check ──
                // A match is exhaustive if it has a wildcard `_` or variable-binding arm
                // (without guard). For ADT scrutinees, check that all constructors are covered.
                self.check_match_exhaustive(ast, *scrutinee, resolved_scrutinee, arms);

                result_ty
            }
            _ => unreachable!("infer_match_expr called on non-Match expression"),
        }
    }

    /// Infers the pattern type and introduces bound variables into the environment.
    pub fn infer_pattern(
        &mut self,
        pat: PatternId,
        ast: &AstArena<'_>,
        expected_ty: TypeHandle,
        env: EnvId,
    ) {
        let node = &ast.pattern(pat).node;
        match node {
            Pattern::Wildcard => {}
            Pattern::Literal(lit) => {
                let lit_ty = match lit {
                    PatternLiteral::Int(_) => Some(self.make_builtin(Type::I32)),
                    PatternLiteral::Float(_) => Some(self.make_builtin(Type::F64)),
                    PatternLiteral::Bool(_) => Some(self.make_builtin(Type::Bool)),
                    PatternLiteral::Char(_) => Some(self.make_builtin(Type::Char)),
                    PatternLiteral::String(_) => Some(self.make_builtin(Type::Str)),
                    PatternLiteral::Null => None,
                };
                if let Some(lt) = lit_ty {
                    let resolved = self.arena.resolve(expected_ty);
                    let ct = self.arena.get(resolved).clone();
                    let is_int_expected = ct.is_int();
                    let is_int_lit = matches!(lit, PatternLiteral::Int(_));
                    if !(is_int_lit && is_int_expected) {
                        self.unify_or_constrain(lt, expected_ty);
                    }
                }
            }
            Pattern::Variable { name } => {
                // Upper-case leading char → zero-argument constructor.
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let sub_pats: Vec<PatternRef> = Vec::new();
                    self.refine_constructor_pattern(name, &sub_pats, expected_ty, ast, env, ast.pattern(pat).span.line, ast.pattern(pat).span.column);
                    // Store disambiguation result for the IR builder (same-named constructors).
                    if let Some(ctor) = self.find_ctor_def(name, expected_ty, ast.pattern(pat).span.line, ast.pattern(pat).span.column) {
                        self.sema_result.pattern_ctor_types.insert(
                            (self.current_module_name.clone(), pat.0),
                            ctor.type_name.clone(),
                        );
                    }
                } else {
                    self.sema_result.env.define(env, name, expected_ty);
                }
            }
            Pattern::Constructor { name, patterns } => {
                if !self.refine_constructor_pattern(name, patterns, expected_ty, ast, env, ast.pattern(pat).span.line, ast.pattern(pat).span.column) {
                    // Regular constructor fallback: use field_type_reprs (self-contained TypeRepr)
                    // instead of field_type_nodes (AST reference) to avoid cross-module AST arena
                    // mismatches.
                    let field_type_reprs: Box<[TypeRepr]> = self
                        .sema_result
                        .get_ctor_def(name)
                        .map(|c| c.field_type_reprs.clone())
                        .unwrap_or_else(|| Box::new([]));
                    if field_type_reprs.is_empty() && self.sema_result.get_ctor_def(name).is_none() {
                        // Unknown constructor name (e.g. Rust-style `Some(x)`/`None` —
                        // nullable here has no constructors; match `null => ...` and bind
                        // the value in the other arm instead). Reject at sema time:
                        // the old silent fallback bound pattern vars to fresh TypeVars,
                        // deferring the breakage to method-dispatch misses at IR time.
                        let span = ast.pattern(pat).span;
                        self.add_error_at(
                            &format!(
                                "unknown constructor pattern '{}': no constructor of this name exists (T? matches with `null => ...` plus a binding arm)",
                                name
                            ),
                            span.line,
                            span.column,
                        );
                    }
                    for (i, &sub_pat) in patterns.iter().enumerate() {
                        let sub_ty = if i < field_type_reprs.len() {
                            self.type_repr_to_handle(&field_type_reprs[i])
                        } else {
                            self.arena.fresh_type_var()
                        };
                        self.infer_pattern(sub_pat, ast, sub_ty, env);
                    }
                }
                // Store disambiguation result for the IR builder (same-named constructors).
                if let Some(ctor) = self.find_ctor_def(name, expected_ty, ast.pattern(pat).span.line, ast.pattern(pat).span.column) {
                    self.sema_result.pattern_ctor_types.insert(
                        (self.current_module_name.clone(), pat.0),
                        ctor.type_name.clone(),
                    );
                }
            }
            Pattern::Record { fields } => {
                for field in fields.iter() {
                    let field_ty = self.arena.fresh_type_var();
                    self.infer_pattern(field.pattern, ast, field_ty, env);
                }
            }
            Pattern::OrPattern { left, right } => {
                self.infer_pattern(*left, ast, expected_ty, env);
                self.infer_pattern(*right, ast, expected_ty, env);
            }
            Pattern::Guard { pattern, condition } => {
                self.infer_pattern(*pattern, ast, expected_ty, env);
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);
            }
        }
    }

    // ── register_builtins ──

}

// =========================================================================
// Usefulness algorithm (Maranget) — pattern matrix exhaustiveness checking
// =========================================================================

/// Constructor identifier used by the usefulness algorithm.
#[derive(Clone, PartialEq)]
pub(super) enum PatCtor {
    Adt(Box<str>),
    Bool(bool),
    Int(Box<str>),
    Float(Box<str>),
    Char(u32),
    Str(Box<str>),
    Null,
}

/// Normalized pattern (arena-independent) for the usefulness algorithm.
/// Or-patterns are expanded into multiple alternatives during normalization.
#[derive(Clone)]
pub(super) enum NormPat {
    Wild,
    Ctor(PatCtor, Vec<NormPat>),
}

/// Unwrap an inline `Pattern::Guard` to retrieve the inner pattern.
pub(super) fn unwrap_guard_pat(ast: &AstArena<'_>, pat: PatternRef) -> PatternRef {
    match &ast.pattern(pat).node {
        Pattern::Guard { pattern, .. } => *pattern,
        _ => pat,
    }
}

/// Normalize an AST pattern into one or more `NormPat` alternatives.
/// Or-patterns expand to multiple alternatives; sub-pattern or-patterns produce
/// the cartesian product of alternatives.
pub(super) fn normalize_pattern(ast: &AstArena<'_>, pat: PatternRef) -> Vec<NormPat> {
    match &ast.pattern(pat).node {
        Pattern::Wildcard => vec![NormPat::Wild],
        Pattern::Variable { name } => {
            if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                vec![NormPat::Ctor(PatCtor::Adt(name.to_string().into_boxed_str()), Vec::new())]
            } else {
                vec![NormPat::Wild]
            }
        }
        Pattern::Constructor { name, patterns } => {
            // Qualified spellings (`A.TEf`) normalize to the bare constructor
            // name so exhaustiveness sees one constructor across spellings
            // (runtime constructor names are bare).
            let bare = name.rsplit('.').next().unwrap_or(name);
            let ctor = PatCtor::Adt(bare.to_string().into_boxed_str());
            // Cartesian product of sub-pattern alternatives.
            let mut alternatives: Vec<Vec<NormPat>> = vec![Vec::new()];
            for &sub_pat in patterns.iter() {
                let sub_alts = normalize_pattern(ast, sub_pat);
                let mut next = Vec::new();
                for existing in &alternatives {
                    for sub_alt in &sub_alts {
                        let mut combined = existing.clone();
                        combined.push(sub_alt.clone());
                        next.push(combined);
                    }
                }
                alternatives = next;
            }
            alternatives.into_iter()
                .map(|subs| NormPat::Ctor(ctor.clone(), subs))
                .collect()
        }
        Pattern::Literal(lit) => {
            let c = match lit {
                PatternLiteral::Bool(b) => PatCtor::Bool(*b),
                PatternLiteral::Int(s) => PatCtor::Int(s.to_string().into_boxed_str()),
                PatternLiteral::Float(s) => PatCtor::Float(s.to_string().into_boxed_str()),
                PatternLiteral::Char(c) => PatCtor::Char(*c),
                PatternLiteral::String(s) => PatCtor::Str(s.to_string().into_boxed_str()),
                PatternLiteral::Null => PatCtor::Null,
            };
            vec![NormPat::Ctor(c, Vec::new())]
        }
        // Record patterns always match the single record constructor; treat as catch-all
        // (field sub-pattern exhaustiveness is not tracked — conservative but safe).
        Pattern::Record { .. } => vec![NormPat::Wild],
        Pattern::OrPattern { left, right } => {
            let mut result = normalize_pattern(ast, *left);
            result.extend(normalize_pattern(ast, *right));
            result
        }
        Pattern::Guard { pattern, .. } => normalize_pattern(ast, *pattern),
    }
}

/// Specialize a pattern matrix with respect to a constructor `target` of arity `arity`.
///
/// For each row: wildcard → expand to `arity` wildcards + remaining columns;
/// matching constructor → extract subpatterns (padded/truncated to `arity`) + remaining;
/// non-matching constructor → drop the row.
pub(super) fn specialize_matrix(
    matrix: &[Vec<NormPat>],
    col_types: &[TypeHandle],
    target: &PatCtor,
    arity: usize,
    field_types: &[TypeHandle],
) -> (Vec<Vec<NormPat>>, Vec<TypeHandle>) {
    let mut new_matrix = Vec::new();
    for row in matrix {
        if row.is_empty() {
            continue;
        }
        match &row[0] {
            NormPat::Wild => {
                let mut new_row = vec![NormPat::Wild; arity];
                new_row.extend(row[1..].iter().cloned());
                new_matrix.push(new_row);
            }
            NormPat::Ctor(c, sub) if c == target => {
                let mut new_row = sub.clone();
                if new_row.len() > arity {
                    new_row.truncate(arity);
                }
                while new_row.len() < arity {
                    new_row.push(NormPat::Wild);
                }
                new_row.extend(row[1..].iter().cloned());
                new_matrix.push(new_row);
            }
            _ => {} // different constructor: drop row
        }
    }
    let mut new_col_types = field_types.to_vec();
    new_col_types.extend(col_types[1..].iter().copied());
    (new_matrix, new_col_types)
}

/// Default matrix: keep only wildcard rows, dropping the first column.
pub(super) fn default_matrix(
    matrix: &[Vec<NormPat>],
    col_types: &[TypeHandle],
) -> (Vec<Vec<NormPat>>, Vec<TypeHandle>) {
    let mut new_matrix = Vec::new();
    for row in matrix {
        if row.is_empty() {
            new_matrix.push(Vec::new());
        } else if matches!(row[0], NormPat::Wild) {
            new_matrix.push(row[1..].to_vec());
        }
    }
    (new_matrix, col_types[1..].to_vec())
}

