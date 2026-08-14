//! ExprInfer — Expression inference core (infer_expr_inner and simple expr kinds). Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Internal implementation of expression type inference (does not store ExprInfo).
    pub(super) fn infer_expr_inner(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let node = &ast.expr(expr).node;
        match node {
            // ── Literals ──
            Expr::IntLit { raw, suffix } => {
                // Range-check suffixed integer literals at sema time (Bug #72: stage consistency with IR Builder).
                if let Some(suf) = suffix {
                    if let Some(tag) = crate::types::ValueTag::from_name(suf) {
                        if let Some(err) = check_int_literal_range(raw, tag) {
                            self.add_error(&err);
                        }
                    }
                }
                numeric_lit!(self, suffix, expected, int_suffix_to_type, is_int, I32)
            }
            Expr::FloatLit { suffix, .. } => numeric_lit!(self, suffix, expected, float_suffix_to_type, is_float, F64),
            Expr::BoolLit(_) => self.make_builtin(Type::Bool),
            Expr::CharLit(_) => self.make_builtin(Type::Char),
            Expr::StrLit(_) => self.make_builtin(Type::Str),
            Expr::StrInterp(parts) => {
                // Recursively infer the sub-expressions inside the interpolation so their ExprInfo is registered in expr_types.
                // Otherwise the IR compiler's `select_binary_compute_fn` falls back to "i32" when it cannot find the type,
                // and mis-dispatches bool/str (non-integer) types to CF_EQ_I32 (as_i32 on bool is always 0).
                for p in parts {
                    if let InterpolationPart::Expression(e) = p {
                        let _ = self.infer_expr(*e, ast, env, None);
                    }
                }
                self.make_builtin(Type::Str)
            }
            Expr::NullLit => {
                // The null literal has type Nullable<T>, where T is solved via the expected constraint.
                // try_widen_unify handles all expected types (Nullable<T> unifies the inner type;
                // other types try to widen or report an error), so no type-specific special-casing of expected is needed.
                let tv = self.arena.fresh_type_var();
                let ty = self.arena.make_nullable(tv);
                if let Some(exp) = expected {
                    if let Err(e) = self.try_widen_unify(exp, ty) {
                        self.add_error(&format!("null literal incompatible with expected type: {}", e));
                    }
                }
                ty
            }
            Expr::VoidLit => self.make_builtin(Type::Void),

            // ── Identifiers ──
            Expr::Ident(_) => {
                // Bug K: a bare nullary constructor (e.g. `None` for `type Opt<T> = | Some(T) | None`)
                // is registered in the env as the ADT value `Opt<rigid_T>`. Because `freshen_type`
                // deliberately skips rigid TypeVars, looking it up leaves the rigid `T` unbound, so
                // `val x: Opt<i32> = None` fails to infer `T = i32`. When an `expected` type is a
                // concrete generic ADT matching this constructor's owning type, instantiate the
                // constructor with the expected type's type arguments directly.
                if let Some(expected) = expected {
                    if let Expr::Ident(name) = &ast.expr(expr).node {
                        if let Some(ty) = self.infer_nullary_ctor_with_expected(name, expected) {
                            return ty;
                        }
                    }
                }
                self.infer_ident_expr(expr, ast, env)
            }

            // ── Assignment ──
            Expr::Assign { target, value } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                self.make_builtin(Type::Void)
            }
            Expr::CompoundAssign { target, value, .. } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                target_ty
            }

            // ── Binary operations ──
            Expr::Binary { .. } => self.infer_binary_expr(expr, ast, env),

            // ── Unary operations ──
            Expr::Unary { operand, .. } => {
                let _ = self.infer_expr(*operand, ast, env, None);
                // ! / ~ / - all return the operand's type.
                self.infer_expr(*operand, ast, env, None)
            }

            // ── Type cast `expr as T` ──
            Expr::As { expr: src, target } => {
                let _ = self.infer_expr(*src, ast, env, None);
                self.type_from_ast(*target, ast)
            }

            // ── Reference / dereference ──
            Expr::RefOf(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_ref(inner_ty, false)
            }
            Expr::Deref(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Type::Ref(_) => self.arena.ref_parts(resolved).0,
                    _ => operand_ty, // Dereferencing a non-reference: return the original type.
                }
            }

            // ── Function calls ──
            Expr::Call { .. } => self.infer_call_expr(expr, ast, env, expected),

            // ── Method calls ──
            Expr::MethodCall { .. }
            | Expr::SafeMethodCall { .. } => self.infer_method_call_expr(expr, ast, env, expected),

            // ── Field access ──
            Expr::FieldAccess { recv, field } => {
                // Qualified-name syntax: Type.Ctor (qualified access of a zero-argument constructor)
                if let Expr::Ident(type_name) = &ast.expr(*recv).node {
                    if let Some((ctor_type_name, field_type_reprs)) =
                        self.check_qualified_ctor(type_name, field)
                    {
                        if field_type_reprs.is_empty() {
                            // Zero-argument constructor: return Adt(type_name)
                            return self.arena.make_adt(ctor_type_name, Box::new([]));
                        }
                        // Constructor with arguments in FieldAccess: report an error
                        let span = ast.expr(expr).span;
                        self.add_error_at(
                            &format!(
                                "constructor '{}' of type '{}' requires arguments; use {}('{}') syntax",
                                field, type_name, field, type_name
                            ),
                            span.line,
                            span.column,
                        );
                        return self.arena.fresh_type_var();
                    }
                }

                let recv_ty = self.infer_expr(*recv, ast, env, None);
                // Detect a ModuleRef receiver: cross-module constant access such as Math.PI.
                // On hit, record recv's expr key → mangled name (module_path.field) into
                // module_const_recv_exprs, so IR compilation skips recv and emits a global_load directly.
                let recv_resolved = self.arena.resolve(recv_ty);
                if let Type::ModuleRef(_) = self.arena.get(recv_resolved) {
                    let (path, module_env) = self.arena.module_ref_parts(recv_resolved);
                    if self.env.lookup_local(module_env, field).is_some() {
                        let mangled = format!("{}.{}", path, field);
                        let recv_key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            recv.0 as u64,
                        );
                        self.sema_result.module_const_recv_exprs.insert(recv_key, mangled);
                    }
                }
                let span = ast.expr(expr).span;
                self.lookup_field_type(recv_ty, field, span.line, span.column)
            }
            Expr::SafeAccess { recv, field } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                // SafeAccess `?.` is only meaningful for Nullable/Ref; for other types it degrades to an ordinary field access.
                let is_nullable = matches!(self.arena.get(resolved), Type::Nullable(_));
                let inner = self.unwrap_ref(recv_ty);
                let span = ast.expr(expr).span;
                let field_ty = self.lookup_field_type(inner, field, span.line, span.column);
                // For a Nullable receiver, the field-access result should also be Nullable (propagating the None semantic).
                if is_nullable {
                    self.arena.make_nullable(field_ty)
                } else {
                    field_ty
                }
            }

            // ── Index / slice ──
            Expr::Index { recv, index } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*index, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                match self.arena.get(resolved) {
                    Type::Array(_) => self.arena.array_parts(resolved).0,
                    // Str indexing returns Char (stdlib uses patterns like normalized[0] == '/').
                    Type::Str => self.arena.make(Type::Char),
                    // Unknown/TypeVar/Generic/Adt, etc. do not report errors:
                    // sema v2 does not always infer variable types precisely (e.g. u8[] may be unified as Unknown);
                    // until sema type inference matures, permissively allow these to avoid cascading false positives.
                    _ => self.arena.fresh_type_var(),
                }
            }
            Expr::Slice { recv, start, end, .. } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*start, ast, env, None);
                let _ = self.infer_expr(*end, ast, env, None);
                recv_ty // A slice returns the same type.
            }

            // ── Propagation ──
            Expr::Propagate(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(inner_ty);
                let span = ast.expr(expr).span;
                self.check_propagate(resolved, inner_ty, self.expected_return, span.line, span.column)
            }
            Expr::NonNullAssert(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Type::Nullable(_) => self.arena.nullable_inner(resolved),
                    _ => operand_ty,
                }
            }
            Expr::Elvis { lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let rl = self.arena.resolve(left_ty);
                if let Type::Nullable(_) = self.arena.get(rl) {
                    let inner = self.arena.nullable_inner(rl);
                    if let Err(e) = self.try_widen_unify(inner, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Nullable inner type: {}", e));
                    }
                    inner
                } else if let Type::Throw(_) = self.arena.get(rl) {
                    // Throw<T,E> ?? rhs → returns T, symmetric with Nullable (Bug #28).
                    let value_ty = self.arena.throw_parts(rl).0;
                    if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                    }
                    value_ty
                } else {
                    left_ty
                }
            }

            // ── Array literals ──
            Expr::ArrayLit { elements, fill } => {
                // Extract the element type from expected so literal elements can be promoted per the annotation.
                // (e.g. in `val data: u8[] = [72, 101]`, 72 should be promoted to u8 rather than the default i32.)
                let expected_elem = expected.and_then(|exp| {
                    let r = self.arena.resolve(exp);
                    match self.arena.get(r) {
                        Type::Array(_) => Some(self.arena.array_parts(r).0),
                        _ => None,
                    }
                });
                // Array fill syntax: [value, ..count] — infer value and count, return runtime-sized array
                if let Some((value, count)) = fill {
                    let value_ty = self.infer_expr(*value, ast, env, expected_elem);
                    // Infer count to register its ExprInfo; length is runtime-determined
                    let _count_ty = self.infer_expr(*count, ast, env, None);
                    return self.arena.make_array(value_ty, None);
                }
                if elements.is_empty() {
                    let elem_ty = expected_elem.unwrap_or_else(|| self.arena.fresh_type_var());
                    return self.arena.make_array(elem_ty, None);
                }
                let first_ty = self.infer_expr(elements[0], ast, env, expected_elem);
                for &e in elements.iter().skip(1) {
                    let elem_ty = self.infer_expr(e, ast, env, expected_elem);
                    if let Err(e_err) = self.try_widen_unify(first_ty, elem_ty) {
                        self.add_error(&format!("array element type mismatch: {}", e_err));
                    }
                }
                self.arena.make_array(first_ty, Some(elements.len() as u64))
            }

            // ── Record literals ──
            Expr::RecordLit(fields) => {
                let field_types: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.infer_expr(f.value, ast, env, None),
                    })
                    .collect();
                self.arena.make_record(field_types.into_boxed_slice(), None)
            }
            Expr::RecordExtend { base, updates } => {
                let base_ty = self.infer_expr(*base, ast, env, None);
                let resolved = self.arena.resolve(base_ty);
                match self.arena.get(resolved) {
                    Type::Record(_) => {
                        let base_fields = self.arena.record_fields(resolved);
                        let name = self.arena.record_name(resolved).map(|s| s.into());
                        let mut all_fields: Vec<FieldType> = base_fields.to_vec();
                        for update in updates.iter() {
                            let update_ty = self.infer_expr(update.value, ast, env, None);
                            let mut found = false;
                            for f in all_fields.iter_mut() {
                                if f.name.as_deref() == Some(update.name) {
                                    f.ty = update_ty;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                all_fields.push(FieldType {
                                    name: Some(update.name.into()),
                                    ty: update_ty,
                                });
                            }
                        }
                        self.arena.make_record(all_fields.into_boxed_slice(), name)
                    }
                    _ => {
                        let span = ast.expr(expr).span;
                        self.add_error_at("record extend requires record type", span.line, span.column);
                        self.arena.fresh_type_var()
                    }
                }
            }

            // ── Lambda ──
            Expr::Lambda { params, body, is_async, return_type } => {
                // Lambda requires an explicit return type annotation.
                if return_type.is_none() {
                    self.add_error("lambda requires an explicit return type annotation: fun(params): T { ... }");
                }
                let child_env = self.env.child(env);
                let param_types: Vec<TypeHandle> = params
                    .iter()
                    .map(|p| {
                        let param_ty = match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, ast),
                            None => self.arena.fresh_type_var(),
                        };
                        self.env.define(child_env, p.name, param_ty);
                        param_ty
                    })
                    .collect();
                let body_ty = match body {
                    LambdaBody::Block(b) => self.infer_expr(*b, ast, child_env, None),
                    LambdaBody::Expression(e) => self.infer_expr(*e, ast, child_env, None),
                };
                // ── Unified capture analysis ──
                // Record the capture list for this lambda scope. The IR builder
                // consumes this (replacing its own `collect_free_idents_expr`
                // re-scan); the per-capture mode drives by-val vs by-ref at
                // runtime. Self-reference detection: a named nested function
                // referencing its own name is excluded from captures.
                {
                    let body_expr_id = match body {
                        LambdaBody::Block(b) => *b,
                        LambdaBody::Expression(e) => *e,
                    };
                    let mut param_names: Vec<&str> = params.iter().map(|p| p.name).collect();
                    // No name available at the Lambda expr level (named nested
                    // functions go through `Stmt::LocalDecl`); self-upvalue is
                    // handled there.
                    let _ = &mut param_names;
                    let captures = self.compute_captures(ast, body_expr_id, &param_names, false);
                    if !captures.is_empty() || self.instantiation_ctx.is_none() {
                        // Only record during the HM pass (instantiation mode
                        // reuses the HM-pass capture table).
                        if self.instantiation_ctx.is_none() {
                            let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                            self.sema_result.put_captures(key, &self.current_module_name, captures);
                        }
                    }
                }
                let effective_body_ty = if let Some(rt) = return_type {
                    let annot_ty = self.type_from_ast(*rt, ast);
                    if let Err(e) = self.try_widen_unify(annot_ty, body_ty) {
                        self.add_error(&format!("lambda body type incompatible with declared return type: {}", e));
                    }
                    annot_ty
                } else {
                    body_ty
                };
                let ret_ty = if *is_async {
                    self.arena.make_async(effective_body_ty)
                } else {
                    effective_body_ty
                };
                self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
            }

            // ── if expressions ──
            Expr::If { cond, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(*cond, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);

                // sema v2: extract flow facts (nullable narrowing).
                let (then_facts, else_facts) = analyze_null_check_facts(
                    self.arena,
                    ast,
                    *cond,
                    env,
                    &self.env,
                );

                let then_env = self.env.child(env);
                // Enter the then scope and apply the then facts.
                self.flow_ctx.push_scope();
                for fact in &then_facts {
                    self.flow_ctx.add_fact(fact.clone());
                }
                let then_ty = self.infer_expr(*then_branch, ast, then_env, expected);
                self.flow_ctx.pop_scope();

                if let Some(else_br) = else_branch {
                    let else_env = self.env.child(env);
                    // Enter the else scope and apply the else facts.
                    self.flow_ctx.push_scope();
                    for fact in &else_facts {
                        self.flow_ctx.add_fact(fact.clone());
                    }
                    let else_ty = self.infer_expr(*else_br, ast, else_env, expected);
                    self.flow_ctx.pop_scope();

                    // v2 convergence: use only peer_type to unify branch types (eliminates the try_widen_unify dual-track scheme).
                    // peer_type already inlines Never/Void filtering, numeric widening, and nullable/throw propagation.
                    peer_type(self.arena, &[then_ty, else_ty])
                } else {
                    // No else branch: the implicit else falls through as Void.
                    // peer_type(then, Void) ensures a diverging then (Never) does
                    // not make the whole if diverge — the fall-through path is
                    // reachable. Only an explicit `else { diverge }` yields Never.
                    let void_ty = self.make_builtin(Type::Void);
                    peer_type(self.arena, &[then_ty, void_ty])
                }
            }

            // ── Block expressions ──
            Expr::Block { stmts, trailing } => {
                let child_env = self.env.child(env);
                let mut diverges = false;
                for &stmt in stmts.iter() {
                    if diverges {
                        // Bug #84: code after a diverging statement is unreachable.
                        // Report a warning but continue inferring so the IR builder has
                        // ExprInfo for all expressions (it processes all statements
                        // independently of sema's divergence analysis).
                        let span = ast.stmt(stmt).span;
                        self.add_warning_at("unreachable code after throw/return/break/continue", span.line, span.column);
                    }
                    let stmt_ty = self.infer_stmt(stmt, ast, child_env);
                    // Detect divergence: direct control-flow exits (return/throw/break/
                    // continue) or statements whose inferred type is Never (e.g. an
                    // if/match/block expression where all branches diverge).
                    let is_direct_exit = matches!(
                        &ast.stmt(stmt).node,
                        Stmt::Return { .. } | Stmt::Throw { .. } | Stmt::Break | Stmt::Continue
                    );
                    let is_never = stmt_ty
                        .map(|t| matches!(self.arena.get(self.arena.resolve(t)), Type::Never))
                        .unwrap_or(false);
                    if !diverges && (is_direct_exit || is_never) {
                        diverges = true;
                    }
                }
                if let Some(te) = trailing {
                    if diverges {
                        // Trailing expression after a diverging statement is unreachable.
                        let span = ast.expr(*te).span;
                        self.add_warning_at("unreachable code after throw/return/break/continue", span.line, span.column);
                        // Still infer the trailing expression so the IR builder has
                        // ExprInfo for it (it processes all expressions independently
                        // of sema's divergence analysis). The block's type is Never
                        // regardless of the trailing expression's type.
                        let _ = self.infer_expr(*te, ast, child_env, expected);
                        self.make_builtin(Type::Never)
                    } else {
                        self.infer_expr(*te, ast, child_env, expected)
                    }
                } else if diverges {
                    self.make_builtin(Type::Never)
                } else {
                    self.make_builtin(Type::Void)
                }
            }

            // ── match expressions ──
            Expr::Match { .. } => self.infer_match_expr(expr, ast, env, expected),

            // ── Atomic / Lazy ──
            Expr::Atomic(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_atomic(inner_ty)
            }
            Expr::Lazy(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_lazy(inner_ty)
            }

            // ── select expressions: Go-style channel multiplexing ──
            //
            // Iterate over all arms:
            //   receive arm: create a child env, infer Channel<T> from channel_expr,
            //                extract the element type T for the binding (if any), and infer the body type.
            //   timeout arm: directly infer the body type.
            // Use peer_type to join all body types (consistent with Match; more robust than the Zig side, which only takes the first).
            Expr::Select(arms) => {
                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                for arm in arms.iter() {
                    let child_env = self.env.child(env);
                    self.flow_ctx.push_scope();
                    match arm {
                        crate::ast::Ast::SelectArm::Receive { channel_expr, binding, body } => {
                            // Infer the channel expression's type and extract the element type for the binding.
                            let chan_ty = self.infer_expr(*channel_expr, ast, child_env, None);
                            let resolved = self.arena.resolve(chan_ty);
                            let elem_ty = match self.arena.get(resolved) {
                                // Nullable(Channel<T>) → take Channel's T.
                                Type::Nullable(_) => {
                                    let inner = self.arena.nullable_inner(resolved);
                                    let inner_resolved = self.arena.resolve(inner);
                                    match self.arena.get(inner_resolved) {
                                        Type::Channel(_) => self.arena.channel_elem(inner_resolved),
                                        _ => chan_ty,
                                    }
                                }
                                // Channel<T> → take T.
                                Type::Channel(_) => self.arena.channel_elem(resolved),
                                _ => chan_ty,
                            };
                            if let Some(name) = binding {
                                let _ = self.env.define(child_env, name, elem_ty);
                            }
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                        crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                            // Infer the duration expression too — without this, its ExprInfo
                            // is never written and the IR builder reports "missing ExprInfo".
                            let _ = self.infer_expr(*duration, ast, child_env, None);
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                    }
                    self.flow_ctx.pop_scope();
                }
                if arm_tys.is_empty() {
                    self.make_builtin(Type::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                }
            }

            // ── inline_trait values: construct a TraitObject type ──
            Expr::InlineTrait(_) => self.infer_inline_trait_expr(expr, ast, env, expected),
        }
    }

    /// Bug K: when a bare nullary constructor (e.g. `None`) is used in a context with a
    /// concrete expected ADT type (e.g. `val x: Opt<i32> = None`), instantiate the
    /// constructor's owning type with the expected type's type arguments so the generic
    /// parameter is inferred (rather than left as an unbound rigid var).
    ///
    /// Returns `Some(ty)` when `name` is a registered nullary constructor and `expected`
    /// is a concrete `Type::Adt` whose name matches the constructor's owning type;
    /// otherwise returns `None` so the caller falls back to normal identifier inference.
    pub(super) fn infer_nullary_ctor_with_expected(
        &mut self,
        name: &str,
        expected: TypeHandle,
    ) -> Option<TypeHandle> {
        // Only consider names registered as constructors.
        let ctors = self.sema_result.get_ctor_defs(name);
        if ctors.is_empty() {
            return None;
        }
        // Only nullary constructors (zero fields) are bare values; constructors with
        // fields go through the Call path and infer via argument unification.
        let is_nullary = ctors.iter().any(|c| c.field_type_reprs.is_empty());
        if !is_nullary {
            return None;
        }
        let exp_resolved = self.arena.resolve(expected);
        let (exp_type_name, exp_args) = match self.arena.get(exp_resolved) {
            Type::Adt(_) => self.arena.adt_parts(exp_resolved),
            // Generic is the alias/parameterized form used for some type aliases; treat it
            // like Adt for instantiation purposes.
            Type::Generic(_) => self.arena.generic_parts(exp_resolved),
            _ => return None,
        };
        // Disambiguate same-named nullary constructors across types: require the owning
        // type name to match the expected type name.
        let matching = ctors
            .iter()
            .find(|c| c.field_type_reprs.is_empty() && c.type_name.as_ref() == exp_type_name);
        let type_name = match matching {
            Some(c) => c.type_name.clone(),
            // No name match: only safe to instantiate when there is a single candidate,
            // to avoid guessing among ambiguous same-named constructors.
            None => {
                if ctors.len() == 1 {
                    ctors[0].type_name.clone()
                } else {
                    return None;
                }
            }
        };
        // Build the concrete instantiation: OwnerType<expected_args...>.
        // exp_args is already resolved/concrete (it came from the type annotation).
        Some(self.arena.make_adt(type_name, exp_args.to_vec().into_boxed_slice()))
    }

    /// Infer an `Expr::Ident` expression (extracted from `infer_expr_inner`).
    pub(super) fn infer_ident_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Ident(name) => {
                // sema v2: prefer the flow-narrowing result (path-sensitive type refinement).
                if let Some(narrowed_ty) = self.flow_ctx.lookup_narrowed(name) {
                    return narrowed_ty;
                }
                // Resolution order inside a method body (current_this_type non-empty):
                //   1. lookup_local  — local variables and parameters only (no parent traversal)
                //   2. For concrete types: try_implicit_this_field — fields before methods
                //      (prevents same-named methods in the parent env from shadowing fields)
                //   3. env.lookup    — full chain (methods, top-level functions)
                //   4. For trait default methods (TypeVar): try_implicit_this_field — permissive
                //      fallback for fields that can't be verified at trait definition time
                let this_ty_opt = self.current_this_type();
                if let Some(this_ty) = this_ty_opt {
                    // 1. Local variables and parameters only.
                    if let Some(scheme) = self.env.lookup_local(env, name) {
                        return self.freshen_type(scheme);
                    }
                    let is_typevar = matches!(
                        self.arena.get(self.arena.resolve(this_ty)),
                        Type::TypeVar(_)
                    );
                    // 2. Concrete types: fields take precedence over same-named methods.
                    if !is_typevar {
                        if let Some(field_ty) = self.try_implicit_this_field(this_ty, name) {
                            self.pending_implicit_this = Some((
                                expr,
                                crate::sema::Sema::ImplicitThisAccess::Field((*name).to_string().into_boxed_str()),
                            ));
                            return field_ty;
                        }
                    }
                    // 3. Full lookup (methods registered in parent env, top-level functions).
                    if let Some(scheme) = self.env.lookup(env, name) {
                        return self.freshen_type(scheme);
                    }
                    // 4. Trait default methods: permissive field fallback (TypeVar can't
                    //    verify field existence; deferred to monomorphization).
                    if is_typevar {
                        if let Some(field_ty) = self.try_implicit_this_field(this_ty, name) {
                            self.pending_implicit_this = Some((
                                expr,
                                crate::sema::Sema::ImplicitThisAccess::Field((*name).to_string().into_boxed_str()),
                            ));
                            return field_ty;
                        }
                    }
                } else {
                    // Outside methods: full env lookup.
                    if let Some(scheme) = self.env.lookup(env, name) {
                        return self.freshen_type(scheme);
                    }
                }
                // Instantiation mode: the temporary InferContext's env does not contain module-level declarations;
                // query sema_result instead (already resolved in the HM stage).
                if self.instantiation_ctx.is_some() {
                    // Look up from expr_types (the expression's type was already resolved in the HM stage).
                    let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                    if let Some(info) = self.sema_result.get_expr(key) {
                        return info.ty;
                    }
                    // In instantiation mode, do not report an error; return a fresh_type_var.
                    return self.arena.fresh_type_var();
                }
                let span = ast.expr(expr).span;
                self.add_error_at(&format!("undefined variable '{}'", name), span.line, span.column);
                self.arena.fresh_type_var()
            }
            _ => unreachable!("infer_ident_expr called on non-Ident expression"),
        }
    }

    /// Infer an `Expr::Binary` expression (extracted from `infer_expr_inner`).
    pub(super) fn infer_binary_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Binary { op, lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let left_is_lit = Self::expr_is_literal(ast, *lhs);
                let right_is_lit = Self::expr_is_literal(ast, *rhs);
                let bin_span = ast.expr(expr).span;
                // Lazy<T> subsumption: unwrap Lazy to inner type for binary operations.
                // `lazy(1i32) + 3i32` treats the left operand as i32.
                let left_unwrapped = {
                    let rl = self.arena.resolve(left_ty);
                    if matches!(self.arena.get(rl), Type::Lazy(_)) {
                        self.arena.lazy_value(rl)
                    } else {
                        left_ty
                    }
                };
                let right_unwrapped = {
                    let rr = self.arena.resolve(right_ty);
                    if matches!(self.arena.get(rr), Type::Lazy(_)) {
                        self.arena.lazy_value(rr)
                    } else {
                        right_ty
                    }
                };
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                        let rl = self.arena.resolve(left_unwrapped);
                        let rr = self.arena.resolve(right_unwrapped);
                        // Bug I: arrays must use `++` (ConcatList) for concatenation, not `+`.
                        // `+` on arrays previously type-checked (returning an array type) but produced
                        // garbage at runtime (len=0). Note: `*` is a legitimate array-repeat idiom
                        // (e.g. `[0u8] * 4096`), so only `Add` is rejected here.
                        if *op == BinaryOp::Add {
                            let left_is_array = matches!(self.arena.get(rl), Type::Array(_));
                            let right_is_array = matches!(self.arena.get(rr), Type::Array(_));
                            if left_is_array || right_is_array {
                                self.add_error_at(
                                    "cannot use + on arrays; use ++ for concatenation",
                                    bin_span.line,
                                    bin_span.column,
                                );
                                return left_unwrapped;
                            }
                        }
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // Bug #73/#74: strict numeric type checking.
                            // - Bare literals (no suffix) can be promoted freely.
                            // - Explicitly typed operands (suffixed literals or variables)
                            //   require explicit cast for different bit widths or int/float crossing.
                            self.check_numeric_binop_compat(ast, *lhs, *rhs, rl, rr, bin_span);
                            // v2 convergence: peer_type_binary replaces literal_promotion;
                            // literal promotion rules are inlined into peer_type_binary.
                            return peer_type_binary(
                                self.arena,
                                left_unwrapped,
                                right_unwrapped,
                                left_is_lit,
                                right_is_lit,
                            );
                        }
                        self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        left_unwrapped
                    }
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::RefEq | BinaryOp::RefNeq
                    | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                        let rl = self.arena.resolve(left_unwrapped);
                        let rr = self.arena.resolve(right_unwrapped);
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // Bug #73/#74: same strict checking for comparison ops.
                            self.check_numeric_binop_compat(ast, *lhs, *rhs, rl, rr, bin_span);
                            // v2 convergence: comparison ops use peer_type_binary to unify operand types.
                            let _ = peer_type_binary(
                                self.arena,
                                left_unwrapped,
                                right_unwrapped,
                                left_is_lit,
                                right_is_lit,
                            );
                        } else {
                            self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        }
                        self.make_builtin(Type::Bool)
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        let bool_ty = self.make_builtin(Type::Bool);
                        self.unify_or_constrain(left_unwrapped, bool_ty);
                        self.unify_or_constrain(right_unwrapped, bool_ty);
                        bool_ty
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                    | BinaryOp::Shl | BinaryOp::Shr => {
                        self.unify_or_constrain(left_unwrapped, right_unwrapped);
                        left_unwrapped
                    }
                    BinaryOp::ConcatList => {
                        // Array concatenation a ++ b: left and right element types must match; the result reuses the left operand's element type.
                        // Avoids creating an orphan fresh_type_var (res_elem would have no constraint to the inputs).
                        let left_elem = self.arena.fresh_type_var();
                        let left_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(left_unwrapped, left_arr);
                        let right_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(right_unwrapped, right_arr);
                        self.arena.make_array(left_elem, None)
                    }
                    BinaryOp::Range | BinaryOp::RangeInclusive => {
                        // Range expressions a..b / a..=b return a RangeIterator type
                        // (Range is itself an iterator; For loops statically dispatch through RangeIterator.next).
                        let i64_ty = self.make_builtin(Type::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, left_unwrapped) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        let i64_ty = self.make_builtin(Type::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, right_unwrapped) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        self.arena.make_generic(
                            "RangeIterator".into(),
                            Box::new([]),
                        )
                    }
                    BinaryOp::Elvis => {
                        let rl = self.arena.resolve(left_ty);
                        if let Type::Nullable(_) = self.arena.get(rl) {
                            return self.arena.nullable_inner(rl);
                        }
                        // Throw<T,E> ?? rhs → returns T (the Ok value type), symmetric with Nullable (Bug #28).
                        if let Type::Throw(_) = self.arena.get(rl) {
                            let value_ty = self.arena.throw_parts(rl).0;
                            // Unify rhs with value_ty to ensure the default value's type is compatible.
                            if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                                self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                            }
                            return value_ty;
                        }
                        left_ty
                    }
                }
            }
            _ => unreachable!("infer_binary_expr called on non-Binary expression"),
        }
    }

    /// Infer an `Expr::InlineTrait` expression (extracted from `infer_expr_inner`).
    ///
    /// Obtain the trait name from the expected type (the val_decl's type annotation),
    /// verify method completeness, and produce TraitObject { trait_name, method_sigs }.
    /// With no expected type, report an error and return a fresh_type_var (an inline_trait without an annotation is not allowed).
    pub(super) fn infer_inline_trait_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::InlineTrait(methods) => {
                // Obtain the trait name from the expected type.
                let trait_name: Option<Box<str>> = if let Some(exp) = expected {
                    let resolved = self.arena.resolve(exp);
                    match self.arena.get(resolved) {
                        Type::Trait(_) => {
                            let (name, _) = self.arena.trait_parts(resolved);
                            Some(name.into())
                        }
                        Type::TraitObject(_) => {
                            let (trait_name, _) = self.arena.trait_object_parts(resolved);
                            Some(trait_name.into())
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // Collect the inline_trait's method signatures.
                let method_sigs: Vec<TraitMethodSig> = methods
                    .iter()
                    .map(|m| {
                        let return_type = match m.return_type {
                            Some(rt) => self.type_from_ast(rt, ast),
                            None => self.arena.make(Type::Void),
                        };
                        TraitMethodSig {
                            name: m.name.into(),
                            param_count: m.params.len() as u8,
                            return_type,
                            is_async: m.is_async,
                            has_body: m.body.is_some(),
                        }
                    })
                    .collect();

                if let Some(tname) = trait_name {
                    // Verify method completeness: every required method (without a body) in trait_def must appear in the inline_trait.
                    let missing: Vec<String> = if let Some(trait_def) = self.sema_result.get_trait_def(&tname) {
                        trait_def
                            .methods
                            .iter()
                            .filter(|req| !req.has_body)
                            .filter(|req| {
                                !method_sigs
                                    .iter()
                                    .any(|m| m.name == req.name && m.param_count == req.param_count)
                            })
                            .map(|req| {
                                format!(
                                    "inline_trait missing required method {} of trait {} (param count {})",
                                    tname, req.name, req.param_count
                                )
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let span = ast.expr(expr).span;
                    for msg in missing {
                        self.sema_result.errors.push(SemaError::new(&msg, span.line, span.column));
                    }

                    // Type-check each method body: bind types for parameters (use the annotation if present, otherwise a fresh_type_var),
                    // set expected_return, and call infer_expr to populate expr_types for sub-expressions inside the body.
                    // This is the data source for IR-compile-time type queries (e.g. str + str → concat).
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            let method_env = self.env.child(env);
                            for param in m.params.iter() {
                                let param_ty = match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                };
                                self.env.define(method_env, param.name, param_ty);
                            }
                            let prev_return = self.expected_return;
                            self.expected_return =
                                m.return_type.map(|rt| self.type_from_ast(rt, ast));
                            let _ = self.infer_expr(body, ast, method_env, self.expected_return);
                            self.expected_return = prev_return;
                        }
                    }

                    self.arena.make_trait_object(
                        tname,
                        method_sigs.into_boxed_slice(),
                    )
                } else {
                    let span = ast.expr(expr).span;
                    self.sema_result.errors.push(SemaError::new(
                        "inline_trait cannot infer trait name: explicit type annotation required",
                        span.line,
                        span.column,
                    ));
                    self.arena.fresh_type_var()
                }
            }
            _ => unreachable!("infer_inline_trait_expr called on non-InlineTrait expression"),
        }
    }

}
