//! Core — InferContext struct + constructor/state helpers and the top-level
//! drivers (check_module / infer_expr dispatch). Extracted from the former
//! Inference.rs (no logic changes).

use super::*;

pub struct InstantiationCtx {
    /// Function name of the current instance (used to short-circuit generic recursion: func_name == cur_func_name).
    pub func_name: Box<str>,
    /// type_args of the current instance (same length as type_params, positionally aligned).
    pub type_args: Box<[TypeHandle]>,
    /// Type-parameter name → type_args index (fast lookup).
    pub type_param_map: FxHashMap<String, u16>,
    /// Module name of the callee (for cross-module cases the expr_types key must match what the IR Builder looks up).
    pub module_name: String,
    /// Staged expression type table (key = module_expr_key(module_name, expr_id)).
    pub local_expr_types: FxHashMap<u64, ExprInfo>,
    /// Staged field_accesses metadata (key = module_expr_key).
    pub local_field_accesses: FxHashMap<u64, FieldAccessInfo>,
    /// Cycle detection: cache_key currently being instantiated → instance_id (supports forward references).
    pub in_progress: FxHashMap<u64, u32>,
}

/// type_binding_stack and this_binding_stack push/pop as impl/trait/fn blocks are entered and exited.
/// env is the local variable environment (EnvArena); expected_return drives reverse inference of the return type.
pub struct InferContext<'a> {
    pub arena: &'a mut TypeArena,
    pub sema_result: &'a mut SemaResult,
    pub type_binding_stack: TypeBindingStack,
    pub this_binding_stack: ThisBindingStack,
    /// Pending implicit-this access marker, set by Ident/Call fallback when a bare
    /// identifier/call resolves to an instance field/method. Consumed by infer_expr
    /// after store_expr_info to update the staged ExprInfo.
    pub pending_implicit_this: Option<(ExprId, crate::sema::Sema::ImplicitThisAccess)>,
    /// Expected return type of the current function (used for reverse inference of throw expressions, etc.).
    pub expected_return: Option<TypeHandle>,
    /// sema v2: constraint solver (lazy solving + snapshot/rollback).
    pub solver: ConstraintSolver,
    /// sema v2: flow-sensitive narrowing context (path-sensitive type refinement).
    pub flow_ctx: FlowContext,
    /// sema v2: witness table (static dispatch table for trait implementations).
    pub witness_table: WitnessTable,
    /// Logical path of the module currently being checked (e.g. "Math.Geometry"), used to register mangled names.
    /// Set at the start of check_module_with_env for use by methods like infer_stmt that do not take a module parameter.
    pub current_module_logical_path: Option<String>,
    /// Module-specific EnvId of the module currently being checked.
    /// Looked up from the shared `sema_result.module_envs` at the start of
    /// check_module_with_env; used to register symbols during
    /// predeclare_declarations.
    pub current_module_env: Option<EnvId>,
    /// Filename of the module currently being checked (e.g. "Math/Geometry.frond"), used as part of the expr_types composite key.
    /// Prevents ExprIds from different modules from colliding in the global expr_types.
    pub current_module_name: String,
    /// Diagnostic trace table: records (TypeHandle, Span) for each expression's inference result, used to trace unresolved TypeVars back to their source locations.
    /// Only populated when FROND_SEMA_TRACE is enabled, to avoid memory overhead during normal compilation.
    pub type_trace: Vec<(TypeHandle, crate::ast::Ast::Span)>,
    /// Constructor short name → module EnvId where the type is defined (Zig-style @This semantics).
    ///
    /// When `import std.time.Duration` and the module defines `pub type Duration`,
    /// predefine uses redefine to overwrite the ModuleRef with the constructor Fn. This map retains
    /// "type name → source module env", allowing the MethodCall path 0b to fall back to free functions in the module
    /// (when the type name equals the filename, the type is treated as a module namespace).
    pub ctor_module_envs: FxHashMap<String, EnvId>,
    /// Instantiation-mode context: None = HM mode, Some = instantiation mode.
    /// Set to Some when resolving types in a monomorphized function body; None during HM type checking.
    pub instantiation_ctx: Option<InstantiationCtx>,
    /// Tracks local binding mutability per environment scope: (env_id, name) → is_mutable.
    /// Used to detect val→var / var→val mutability-changing shadowing (Bug #76).
    pub local_mutability: FxHashMap<(u32, String), bool>,
    /// Name of the trait whose default methods are currently being inferred (None outside trait blocks).
    /// Used by lookup_method_type to resolve bare method calls (implicit this) inside trait default
    /// methods, where current_this_type() is a rigid TypeVar that has no method table of its own.
    pub current_trait_name: Option<Box<str>>,
    /// Trait names implemented by the type declaration whose methods are currently
    /// being inferred (None outside a type block). Drives `super.method(...)`
    /// resolution: super statically targets the bound trait-default layer of the
    /// enclosing type (explicit delegate or unique provider).
    pub current_type_decl_traits: Option<Vec<Box<str>>>,
}

/// Checks whether a type references any unresolved TypeVar (in unresolved_set).
/// Used during diagnostics to trace unresolved TypeVars back to their expression locations.

impl<'a> InferContext<'a> {
    pub fn new(arena: &'a mut TypeArena, sema_result: &'a mut SemaResult) -> Self {
        InferContext {
            arena,
            sema_result,
            type_binding_stack: TypeBindingStack::new(),
            this_binding_stack: ThisBindingStack::new(),
            pending_implicit_this: None,
            expected_return: None,
            solver: ConstraintSolver::new(),
            flow_ctx: FlowContext::new(),
            witness_table: WitnessTable::new(),
            current_module_logical_path: None,
            current_module_env: None,
            current_module_name: String::new(),
            type_trace: Vec::new(),
            ctor_module_envs: FxHashMap::default(),
            instantiation_ctx: None,
            local_mutability: FxHashMap::default(),
            current_trait_name: None,
            current_type_decl_traits: None,
        }
    }

    /// Construct from existing TypeArena and SemaResult (for incremental recheck).
    /// Preserves global state (witness_table) from previous run.
    /// Unlike `new()`, this clones the witness_table from sema_result to preserve
    /// clean modules' witness entries.
    pub fn from_existing(
        arena: &'a mut TypeArena,
        sema_result: &'a mut SemaResult,
    ) -> Self {
        // Clone witness_table before moving sema_result into the struct to avoid
        // a simultaneous mutable + immutable borrow conflict.
        let witness_table = sema_result.witness_table.clone();
        InferContext {
            arena,
            sema_result,
            type_binding_stack: TypeBindingStack::new(),
            this_binding_stack: ThisBindingStack::new(),
            pending_implicit_this: None,
            expected_return: None,
            solver: ConstraintSolver::new(),
            flow_ctx: FlowContext::new(),
            witness_table,
            current_module_logical_path: None,
            current_module_env: None,
            current_module_name: String::new(),
            type_trace: Vec::new(),
            ctor_module_envs: FxHashMap::default(),
            instantiation_ctx: None,
            local_mutability: FxHashMap::default(),
            current_trait_name: None,
            current_type_decl_traits: None,
        }
    }

    // ── Type binding stack operations ──

    /// Enters a generic scope: allocates a rigid var for each type parameter and pushes it onto the stack.
    /// Parameters without a declared kind default to Star; declared kinds are used for HKT checks.
    pub fn push_type_bindings(&mut self, type_params: &[(&str, Option<SemKind>)]) {
        self.type_binding_stack.push();
        for &(name, ref kind_opt) in type_params {
            let var = match kind_opt {
                Some(kind) => self.arena.fresh_rigid_var_with_kind(kind.clone()),
                None => self.arena.fresh_rigid_var(),
            };
            self.type_binding_stack.insert_top(name, var);
        }
    }

    /// Leaves a generic scope: pops the top stack frame.
    pub fn pop_type_bindings(&mut self) {
        self.type_binding_stack.pop();
    }

    /// Looks up a type-parameter binding.
    pub fn lookup_type_binding(&self, name: &str) -> Option<TypeHandle> {
        self.type_binding_stack.lookup(name)
    }

    // ── Instantiation mode (monomorphized function-body type resolution) ──

    /// Enters instantiation mode: replaces type-parameter bindings with concrete type_args.
    ///
    /// push_type_bindings (rigid var) should already have been called; this method replaces the rigid vars in the top stack frame
    /// with concrete TypeHandles from type_args (insert_top internally uses HashMap::insert to overwrite the same-name key).
    pub fn enter_instantiation_mode(
        &mut self,
        func_name: Box<str>,
        type_args: Box<[TypeHandle]>,
        type_param_names: &[&str],
        module_name: String,
        in_progress: FxHashMap<u64, u32>,
    ) {
        // Replace the rigid vars at the top of type_binding_stack with concrete type_args.
        for (i, &name) in type_param_names.iter().enumerate() {
            if i < type_args.len() {
                self.type_binding_stack.insert_top(name, type_args[i]);
            }
        }

        // Build type_param_map.
        let mut type_param_map: FxHashMap<String, u16> = FxHashMap::default();
        for (i, &name) in type_param_names.iter().enumerate() {
            type_param_map.insert(name.to_string(), i as u16);
        }

        self.instantiation_ctx = Some(InstantiationCtx {
            func_name,
            type_args,
            type_param_map,
            module_name,
            local_expr_types: FxHashMap::default(),
            local_field_accesses: FxHashMap::default(),
            in_progress,
        });
    }

    /// Leaves instantiation mode: takes out the staged local_expr_types and local_field_accesses.
    ///
    /// The caller is responsible for transferring the returned values into MonomorphInstance.
    pub fn leave_instantiation_mode(
        &mut self,
    ) -> Option<(
        FxHashMap<u64, ExprInfo>,
        FxHashMap<u64, FieldAccessInfo>,
        FxHashMap<u64, u32>,
    )> {
        self.instantiation_ctx.take().map(|ctx| {
            (
                ctx.local_expr_types,
                ctx.local_field_accesses,
                ctx.in_progress,
            )
        })
    }

    // ── Self binding stack operations ──

    /// Enters a type block: binds Self to a concrete type.
    /// `self_ty` should be in `Adt { name, type_args }` form, where type_args reference vars in the TypeBindingStack.
    pub fn push_this_type(&mut self, self_ty: TypeHandle) {
        self.this_binding_stack.push(self_ty);
    }

    /// Enters a trait default method: binds Self to a fresh_rigid_var (to be solved by unification at impl time).
    /// Using a rigid var marks Self as a template parameter and is automatically excluded from diagnostics (only unbound non-rigid TypeVars are reported as errors).
    pub fn push_this_type_var(&mut self) -> TypeHandle {
        let var = self.arena.fresh_rigid_var();
        self.this_binding_stack.push(var);
        var
    }

    /// Leaves a type/trait block: pops the Self binding.
    pub fn pop_this_type(&mut self) {
        self.this_binding_stack.pop();
    }

    /// Current Self type (top of stack).
    pub fn current_this_type(&self) -> Option<TypeHandle> {
        self.this_binding_stack.current()
    }

    // ── Error recording ──

    pub fn add_error(&mut self, message: &str) {
        // line=0/column=0 means no location info (the sema inference stage has not yet associated an AST location).
        self.sema_result.add_error(SemaError::new(message, 0, 0));
    }

    /// Adds an error with location info (for call sites that have AST span context).
    pub fn add_error_at(&mut self, message: &str, line: u32, column: u32) {
        self.sema_result.add_error(SemaError::new(message, line, column));
    }

    /// Adds a warning with location info (does not set has_error; does not stop compilation).
    pub fn add_warning_at(&mut self, message: &str, line: u32, column: u32) {
        self.sema_result.add_warning(SemaError::new(message, line, column));
    }

    // ── self parameter resolution (phase3b) ──

    /// Determines whether a parameter's type_annotation is ThisType (or RefType<ThisType>).
    ///
    /// The parser auto-fills a ThisType annotation for `self`/`&self` parameters of methods inside type/trait blocks;
    /// Sema uses this type node to detect a self parameter, rather than relying on the parameter name.
    pub(super) fn is_this_param(&self, type_annotation: Option<AstTypeRef>, ast: &AstArena<'_>) -> bool {
        match type_annotation {
            Some(ta) => match &ast.ty(ta).node {
                crate::ast::Ast::TypeNode::ThisType => true,
                crate::ast::Ast::TypeNode::RefType { inner } => {
                    matches!(ast.ty(*inner).node, crate::ast::Ast::TypeNode::ThisType)
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Infers the type of a self parameter.
    ///
    /// **Semantic rules (Rust port, intentionally refined)**:
    /// - `self` may only appear in methods inside type/trait blocks (ThisBindingStack must be non-empty).
    /// - `self` parameters disallow type annotations (the parser auto-fills ThisType or RefType<ThisType>).
    /// - A top-level fun with a self parameter → error.
    /// - A self parameter with an explicit `: Type` annotation → error.
    ///
    /// **Return value**:
    /// - `self` (no annotation, inside a type block) → the scope type.
    /// - `&self` (no annotation, inside a type block) → `Ref<scope type>`.
    /// - Illegal usage → reports an error and returns a fresh_type_var (error recovery).
    pub fn infer_this_param(
        &mut self,
        type_annotation: Option<AstTypeRef>,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let self_ty = match self.current_this_type() {
            Some(ty) => ty,
            None => {
                // Get the span from type_annotation if present, otherwise no location info.
                let (line, column) = type_annotation
                    .map(|ta| {
                        let s = ast.ty(ta).span;
                        (s.line, s.column)
                    })
                    .unwrap_or((0, 0));
                self.add_error_at(
                    "this parameter requires enclosing type or trait block",
                    line,
                    column,
                );
                return self.arena.fresh_type_var();
            }
        };

        // Check the type annotation: self parameters disallow explicit annotations.
        // The parser auto-fills ThisType for `self` (no `:`) and RefType<ThisType> for `&self`.
        // A user-written `self: Foo` goes through parse_param's `:` branch, where type_annotation is the user's type.
        match type_annotation {
            None => {
                // No annotation (theoretically unreachable; the parser always fills one for self).
                self_ty
            }
            Some(ta) => {
                let tn = &ast.ty(ta).node;
                let span = ast.ty(ta).span;
                match tn {
                    // `self` (parser auto-fills ThisType) → return the scope type.
                    TypeNode::ThisType => self_ty,
                    // `&self` (parser auto-fills RefType<ThisType>) → return Ref<scope type>.
                    TypeNode::RefType { inner } => {
                        if matches!(ast.ty(*inner).node, TypeNode::ThisType) {

                            self.arena.make_ref(self_ty, false)
                        } else {
                            // `&self: &Foo` with an explicit reference annotation written by the user → error.
                            self.add_error_at(
                                "this parameter does not allow explicit type annotation",
                                span.line,
                                span.column,
                            );
                            self.arena.fresh_type_var()
                        }
                    }
                    // `self: Foo` with an explicit annotation written by the user → error.
                    _ => {
                        self.add_error_at(
                            "this parameter does not allow explicit type annotation",
                            span.line,
                            span.column,
                        );
                        self.arena.fresh_type_var()
                    }
                }
            }
        }
    }

    /// Infers the type of an expression. This is the core entry point of type checking and recursively handles all expression variants.
    /// After inference, stores the ExprInfo into sema_result.
    pub fn infer_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let ty = self.infer_expr_inner(expr, ast, env, expected);
        self.store_expr_info(expr, ty);
        // Flush pending implicit-this marker into the staged ExprInfo.
        if let Some((eid, access)) = self.pending_implicit_this.take() {
            let key = if let Some(ref ictx) = self.instantiation_ctx {
                module_expr_key(&ictx.module_name, eid.0 as u64)
            } else {
                module_expr_key(&self.current_module_name, eid.0 as u64)
            };
            let info = if let Some(ref mut ictx) = self.instantiation_ctx {
                ictx.local_expr_types.get_mut(&key)
            } else {
                self.sema_result.expr_types.get_mut(&key)
            };
            if let Some(info) = info {
                info.implicit_this = Some(access);
            }
        }
        // Diagnostic trace: only record (TypeHandle, Span) when FROND_SEMA_TRACE is enabled.
        if std::env::var("FROND_SEMA_TRACE").is_ok() {
            let span = ast.expr(expr).span;
            self.type_trace.push((ty, span));
        }
        ty
    }

    /// Module check entry point: orchestrates populate → predeclare → infer → kind_check → monomorph.
    ///
    /// Returns true when there are no errors. Steps:
    /// 1. populate_module fills the SemaResult definition tables.
    /// 2. Create the root environment and register builtins.
    /// 3. Predeclare functions and type constructors.
    /// 4. Infer expression declarations and function bodies.
    /// 5. Run kind_check.
    /// 6. Collect monomorphization instances.
    pub fn check_module(&mut self, module: &Module<'_>) -> bool {
        // Single-module check: create a new root_env, register builtins, check the module.
        self.reset_state();
        let root_env = self.sema_result.env.root();
        self.register_builtins(root_env);
        let all_modules = [module];
        self.check_module_with_env(module, root_env, &all_modules)
    }

    /// Multi-module shared-env check entry point.
    ///
    /// Accepts an externally shared `root_env` (already populated with builtins and prior
    /// modules' symbols), and on top of it processes imports, predeclarations, and checks the
    /// current module.
    /// Cross-module symbols are resolved through the shared env chain.
    pub fn check_module_with_env<'m>(
        &mut self,
        module: &'m Module<'m>,
        root_env: EnvId,
        all_modules: &[&'m Module<'m>],
    ) -> bool {
        // 1. Populate the definition tables (if not already populated).
        populate_module(self.arena, self.sema_result, module, all_modules);

        // 1b. Check for cyclic type aliases (Bug #80).
        self.check_alias_cycles();

        // 1c. Duplicate constructor names across types are now allowed at the
        // definition level (disambiguated by type context or `Type.Ctor`).
        // Ambiguity is reported only at use sites when disambiguation fails
        // (see infer_call). (Bug #81.)

        // 1d. Check for duplicate named fields within a constructor (Bug #82).
        self.check_duplicate_ctor_fields();

        // 2. Reset state (do not reset env; preserve the shared root_env).
        self.reset_state();
        self.reset_per_module_state();
        // Snapshot current type_vars/types length: arena is shared across modules and not reset;
        // diagnostics only count the TypeVars newly added by this module.
        let type_vars_baseline = self.arena.type_vars_len();
        let types_baseline = self.arena.len();
        self.current_module_logical_path = module_logical_path(module.name);
        self.current_module_name = module.name.to_string();
        // Sync the SemaResult-side module context used by the free-function
        // name resolvers (resolve_type_key / resolve_named_type_resolved).
        self.sema_result.current_module_name = module.name.to_string();

        // 3. Process import declarations: register module reference aliases + import aliases.
        self.process_import_decls(module, root_env);

        // 4. Predeclare functions and type constructors (including extern functions).
        self.predeclare_declarations(module, root_env);

        // 5. Populate the witness table (iterate trait impls).
        self.populate_witness_table(module);

        // 6. Infer declarations.
        // Use module_env as the base environment (rather than root_env) so function bodies can
        // resolve same-module functions via the env chain (predeclare_declarations registers
        // into module_env), while still reaching root_env's global builtins and constructors
        // through the parent chain.
        let check_env = self.current_module_env.unwrap_or(root_env);
        for decl in module.declarations.iter() {
            self.check_decl(&decl.node, decl.span, &module.arena, check_env);
        }

        // 7. kind_check all type annotations.
        self.run_kind_checks(module);

        // 8. Collect monomorphization instances (generic function instances; does not depend on witness_table).
        crate::sema::Monomorph::collect_monomorph_instances(module, all_modules, self.sema_result, self.arena);

        // 9. Solve deferred constraints (with witness table support for trait bound solving).
        // Split borrows of self's different fields: arena as mutable borrow, witness_table as shared borrow.
        let InferContext { arena, solver, witness_table, type_trace, .. } = self;
        solver.solve_with_witness(arena, Some(witness_table));

        // 9.1 Report solver ambiguity errors (Bug #83: generic parameter type
        // unification failures were silently dropped — solver.errors() was never
        // consulted, so e.g. `pair(1i32, 2i64)` silently bound T to i32 and accepted
        // the i64 argument).
        // Only report ambiguity errors (from finalize_solution: a TypeVar was required
        // to bind to multiple distinct concrete types). "type mismatch" errors from
        // the fixpoint loop are NOT reported — they are often false positives because
        // `unify_or_constrain` does strict unify only, while other paths use
        // `try_widen_unify` (widening/nullable/async unfolding) which accepts those
        // same type pairs.
        {
            let solver_errors: Vec<ConstraintError> = solver.errors().to_vec();
            for ce in solver_errors {
                if !ce.reason.as_ref().contains("ambiguous") {
                    continue;
                }
                let (t1, t2) = match &ce.constraint {
                    Constraint::Equality(t1, t2) => (*t1, *t2),
                    _ => continue,
                };
                let r1 = arena.resolve(t1);
                let r2 = arena.resolve(t2);
                let s1 = arena.display(r1);
                let s2 = arena.display(r2);
                self.sema_result.add_error(crate::sema::Sema::SemaError::new(
                    &format!("type mismatch: {} does not unify with {} ({})", s1, s2, ce.reason),
                    ce.line, ce.column,
                ));
            }
        }

        // 9.4 Default unbound non-rigid TypeVars to void (root cause F).
        //
        // Having unbound non-rigid TypeVars after constraint solving is a normal type inference
        // phenomenon:
        // - E in Throw<T, E> is unconstrained (when the function never throws, E has no source of info).
        // - T in ArrayIter<T> is unconstrained (when the iterator is never consumed).
        // - TypeVars produced by generic function instantiation are not constrained by call sites.
        //
        // These TypeVars have no candidate types (not ambiguity); void as the unit type is a safe
        // default: unconstrained means any type satisfies, and void introduces no extra constraint.
        // This is the standard type-inference defaulting technique (ML's defaulting rule), not an
        // error fallback.
        // Real type errors are caught by unify failures and ambiguity detection, independent of
        // this diagnostic.
        let void_ty = arena.make(Type::Void);
        for i in type_vars_baseline..arena.type_vars.len() {
            let tv = &mut arena.type_vars[i];
            if !tv.is_rigid && tv.bound.is_none() {
                tv.bound = Some(void_ty);
            }
        }

        // 9.5 Global residual TypeVar diagnostic: unbound non-rigid TypeVars after solving indicate
        // type inference failure.
        // Only count TypeVars newly added by this module (arena is shared across modules; entries
        // before baseline belong to prior modules).
        let unresolved: Vec<u32> = arena.type_vars.iter().enumerate()
            .skip(type_vars_baseline)
            .filter(|(_, tv)| !tv.is_rigid && tv.bound.is_none())
            .map(|(i, _)| i as u32)
            .collect();

        // Verbose logging (controlled by FROND_SEMA_TRACE env var): print unresolved TypeVar details
        // for easier diagnosis.
        if !unresolved.is_empty() && std::env::var("FROND_SEMA_TRACE").is_ok() {
            let unresolved_set: FxHashSet<u32> = unresolved.iter().copied().collect();
            eprintln!(
                "[sema] {} unresolved type variable(s) after constraint solving:",
                unresolved.len()
            );
            for &idx in unresolved.iter().take(50) {
                let tv = &arena.type_vars[idx as usize];
                eprintln!("  TypeVar({}) kind={:?}", idx, tv.kind);
            }
            if unresolved.len() > 50 {
                eprintln!("  ... and {} more", unresolved.len() - 50);
            }
            // Print a sample of type slots containing unresolved TypeVars (up to 30).
            // Only iterate over type slots newly added by this module (entries before baseline
            // belong to prior modules).
            eprintln!("  sample referencing types (baseline={}):", types_baseline);
            let mut shown = 0u32;
            for i in types_baseline..arena.types.len() {
                let h = TypeHandle(i as u32);
                let s = format!("{}", arena.display(h));
                if s.contains("'_") {
                    eprintln!("    types[{}] = {}", i, s);
                    shown += 1;
                    if shown >= 30 { break; }
                }
            }
            // Reverse lookup: iterate type_trace to find expression spans referencing unresolved TypeVars.
            eprintln!("  referencing expression spans:");
            let mut span_shown = 0u32;
            for &(ty, span) in type_trace.iter() {
                if type_contains_any_unresolved(ty, arena, &unresolved_set) {
                    let s = format!("{}", arena.display(ty));
                    eprintln!("    {}:{}  {}", span.line, span.column, s);
                    span_shown += 1;
                    if span_shown >= 50 { break; }
                }
            }
            if span_shown == 0 {
                eprintln!("    (no direct expression references found — TypeVar may be inside fn signature)");
            }
        }

        // 9z. ExprInfo type_name backfill: fixpoint stores snapshot the type
        // BEFORE the solver finalizes; entries whose stored type_name is None
        // but whose ty handle NOW resolves to a concrete named type get the
        // name backfilled (method-sugar dispatch reads the name — without
        // this, zero-arg ctor receivers degrade to `_` and lose dispatch).
        {
            let keys: Vec<u64> = self
                .sema_result
                .expr_types
                .iter()
                .filter(|(_, info)| info.type_name.is_none())
                .map(|(k, _)| *k)
                .collect();
            for k in keys {
                let ty = self.sema_result.expr_types[&k].ty;
                if let Some(name) = self.arena.type_name_concrete(self.arena.resolve(ty)) {
                    if let Some(info) = self.sema_result.expr_types.get_mut(&k) {
                        info.type_name = Some(name.into_boxed_str());
                    }
                }
            }
        }

        // 10. Mirror witness_table into sema_result (so the IR layer can access trait method
        // dispatch info).
        // witness_table accumulates across modules; sync the latest state after each check.
        self.sema_result.witness_table = witness_table.clone();

        // 10a. Validate the override/delegate bindings of this module's type
        // declarations (override keyword semantics, ambiguous inherited
        // defaults, delegate targets), then collect trait default-method
        // monomorphization instances (depends on the mirrored witness_table).
        crate::sema::Monomorph::validate_trait_method_bindings(module, self.sema_result);
        crate::sema::Monomorph::validate_trait_inheritance(module, self.sema_result);
        crate::sema::Monomorph::collect_trait_default_instances(module, self.sema_result);

        // 11. Report global residual TypeVar diagnostics.
        if !unresolved.is_empty() {
            self.add_error_at(
                &format!("{} unresolved type variable(s) after constraint solving", unresolved.len()),
                0, 0,
            );
        }

        !self.sema_result.has_error
    }

    /// Resets the check state (env is not reset; the shared root_env is preserved).
    pub fn reset_state(&mut self) {
        self.expected_return = None;
        self.type_binding_stack = TypeBindingStack::new();
        self.this_binding_stack = ThisBindingStack::new();
        self.solver.reset();
        self.flow_ctx.reset();
        self.type_trace.clear();
        // witness_table is not reset (accumulates across modules, supports multi-module trait impls).
    }

    /// Reset per-module state that reset_state misses.
    /// Called before incremental recheck of a module.
    pub fn reset_per_module_state(&mut self) {
        self.local_mutability.clear();
        self.instantiation_ctx = None;
    }

}
