//! ModuleEnv — Module-level checking: builtins, imports, predeclare, check_decl. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Registers builtin functions into the environment.
    pub fn register_builtins(&mut self, env: EnvId) {
        // Panic: (str) -> void
        let str_ty = self.make_builtin(Type::Str);
        let void_ty = self.make_builtin(Type::Void);
        let panic_fn = self.arena.make_fn(
            vec![str_ty].into_boxed_slice(),
            void_ty,
        );
        self.sema_result.env.define(env, "Panic", panic_fn);

        // type/type_name has been converted to a frond wrapper (see Reflect.frond::type_name).
        // Sema no longer registers the `type` builtin.

        // Ok: ∀T,E. (T) -> Throw<T, E>
        // Registered with rigid vars (generic parameters); at call sites instantiate_fn_type
        // instantiates them to fresh non-rigid vars.
        let val_ty = self.arena.fresh_rigid_var();
        let err_ty = self.arena.fresh_rigid_var();
        let throw_ty = self.arena.make_throw(val_ty, err_ty);
        let ok_fn = self.arena.make_fn(
            vec![val_ty].into_boxed_slice(),
            throw_ty,
        );
        self.sema_result.env.define(env, "Ok", ok_fn);

        // Numeric type constructors: i8/i16/.../f64 etc. as ∀T. (T) -> Self.
        // Registered with rigid vars; instantiated by instantiate_fn_type at call sites.
        for (name, ct) in numeric_builtin_names() {
            let param = self.arena.fresh_rigid_var();
            let ret_ty = self.make_builtin(ct);
            let fn_ty = self.arena.make_fn(
                vec![param].into_boxed_slice(),
                ret_ty,
            );
            self.sema_result.env.define(env, name, fn_ty);
        }

        // channel<T>(capacity: usize) -> Channel<T>
        // Builtin channel constructor: creates a Channel<T> with the given capacity.
        let usize_ty = self.make_builtin(Type::Usize);
        let t_var3 = self.arena.fresh_rigid_var();
        let chan_ret = self.arena.make_channel(t_var3);
        let chan_fn = self.arena.make_fn(
            vec![usize_ty].into_boxed_slice(),
            chan_ret,
        );
        self.sema_result.env.define(env, "channel", chan_fn);
    }

    // ── check_module ──

    /// Gets or creates the EnvId dedicated to a module path.
    ///
    /// Creates envs level-by-level along path segments, forming a hierarchy:
    ///   "std.io.File" → create env_std (parent=root_env)
    ///                 → env_std_io (parent=env_std)
    ///                 → env_std_io_file (parent=env_std_io)
    ///
    /// Each env level registers the child module's short name → ModuleRef so that
    /// path-based field access can be resolved structurally through the env chain.
    /// Existing path envs are reused (idempotent).
    ///
    /// Returns the EnvId corresponding to the given path.
    pub(super) fn ensure_module_env(&mut self, full_path: &str, root_env: EnvId) -> EnvId {
        // Cached: return directly.
        if let Some(&eid) = self.sema_result.module_envs.get(full_path) {
            return eid;
        }
        let segments: Vec<&str> = full_path.split('.').collect();
        let mut current_path = String::new();
        let mut parent_env = root_env;
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                current_path.push('.');
            }
            current_path.push_str(seg);
            // Env for the current path segment: reuse if present, otherwise create.
            let env_id = if let Some(&eid) = self.sema_result.module_envs.get(&current_path) {
                eid
            } else {
                let eid = self.sema_result.env.child(parent_env);
                self.sema_result.module_envs.insert(current_path.clone(), eid);
                eid
            };
            // Register the current segment's short name → ModuleRef in the parent env so that
            // level-by-level field access can resolve it.
            // First segment is registered in root_env; subsequent segments in their parent path env.
            let mod_ref_ty = self.arena.make_module_ref(
                current_path.clone().into_boxed_str(),
                env_id,
            );
            // Do not overwrite existing bindings (user's explicit import / constructor takes priority).
            self.sema_result.env.define(parent_env, seg, mod_ref_ty);
            parent_env = env_id;
        }
        parent_env
    }

    /// Directory module semantics: look up a function in sibling modules' envs.
    ///
    /// When `sqrt` in `Math.sqrt` is defined in `Power.frond` (rather than `Math.frond`),
    /// derives the directory prefix (e.g. "std.math") from `mod_path` (e.g. "std.math.Math"),
    /// then iterates over the envs of all sibling modules in the same directory
    /// ("std.math.Power", "std.math.Trig", ...) looking up the function by its bare `method` name.
    /// Skips its own env (already checked by the caller via lookup_local).
    pub(super) fn lookup_sibling_module_fn(
        &self,
        mod_path: &str,
        self_env: EnvId,
        method: &str,
    ) -> Option<TypeHandle> {
        // Derive the directory prefix: the part of mod_path before the last '.'.
        let dot_pos = mod_path.rfind('.')?;
        let dir_prefix = &mod_path[..dot_pos]; // e.g. "std.math"
        let sibling_prefix = format!("{}.", dir_prefix); // "std.math."

        // Iterate over sibling modules in module_envs that start with "std.math.".
        for (path, &env_id) in self.sema_result.module_envs.iter() {
            if !path.starts_with(&sibling_prefix) {
                continue;
            }
            if path == mod_path {
                continue; // Skip self.
            }
            if env_id == self_env {
                continue; // Skip self env.
            }
            if let Some(ty) = self.sema_result.env.lookup_local(env_id, method) {
                return Some(ty);
            }
        }
        None
    }

    /// Registers module path aliases into the environment (for same-package module symbol visibility).
    ///
    /// Creates a hierarchical env for each module path and registers the trailing short name
    /// → ModuleRef in root_env, so that same-package modules can be accessed directly via their
    /// short name (e.g. `Calendar` → `ModuleRef("std.time.Calendar", env)`).
    /// Existing bindings are not overwritten (user's explicit import takes priority).
    pub fn register_module_aliases(&mut self, root_env: EnvId, module_paths: &[String]) {
        for path in module_paths {
            if path.is_empty() {
                continue;
            }
            // Ensure the module hierarchy env exists (including intermediate path prefixes).
            let module_env = self.ensure_module_env(path, root_env);
            // Register the trailing short name in root_env (for same-package short-name access).
            if let Some(last_seg) = path.rsplit('.').next() {
                if !last_seg.is_empty() && path.contains('.') {
                    // Do not overwrite existing bindings.
                    if self.sema_result.env.lookup(root_env, last_seg).is_none() {
                        let mod_ref_ty = self.arena.make_module_ref(
                            path.clone().into_boxed_str(),
                            module_env,
                        );
                        self.sema_result.env.define(root_env, last_seg, mod_ref_ty);
                    }
                }
            }
        }
    }

    /// Processes ImportDecls in the module:
    /// - Full-path import `import std.io.File` → ensure the module hierarchy env exists; the
    ///   first segment is registered as a ModuleRef (field access builds the path level by level:
    ///   std → std.io → std.io.File, resolved via the env chain).
    /// - Selective import `import std.io.File { open }` → look up the symbol in the target
    ///   module's env and register it as an alias.
    pub(super) fn process_import_decls(&mut self, module: &Module<'_>, env: EnvId) {
        // Register the current module's own path prefix (e.g. std/io/Path.frond → std.io.Path)
        // so in-module self-references (e.g. std.io.Path.last_index_of) can resolve.
        if let Some(logical_path) = module_logical_path(module.name) {
            // ensure_module_env creates the hierarchy env and registers the first-segment
            // ModuleRef in the parent env.
            self.ensure_module_env(&logical_path, env);
        }

        for decl in module.declarations.iter() {
            if let Decl::ImportDecl { module_path, items, .. } = &decl.node {
                if module_path.is_empty() {
                    continue;
                }
                let full_path = module_path.join(".");
                // Ensure the imported module's hierarchy env exists (including intermediate path
                // prefixes and first-segment ModuleRef registration).
                let module_env = self.ensure_module_env(&full_path, env);

                // Selective import: look up symbols in the target module env and register them
                // into the current env.
                if let Some(items) = items {
                    for item in items.iter() {
                        // Look up the symbol by bare name in the module env (does not traverse
                        // the parent env, to avoid importing global symbols).
                        if let Some(sym_ty) = self.sema_result.env.lookup_local(module_env, item.name) {
                            self.sema_result.env.define(env, item.alias.unwrap_or(item.name), sym_ty);
                            // Register the alias for the IR binding layer
                            // (sema.import_aliases → IrBuilder's func_subgraphs /
                            // global_var_slots alias keys). Without this the
                            // symbol type-checks but the IR call cannot bind
                            // (the "selective imports never worked at the IR
                            // level" gap — previously masked by the bare-key
                            // last-writer-wins slot). A duplicate alias bound
                            // to a DIFFERENT target is a hard error: writer-wins
                            // here would silently rebind every bare call.
                            let local = item.alias.unwrap_or(item.name);
                            let target_mangled = format!("{}.{}", full_path, item.name);
                            let existing = self.sema_result.get_import_alias(local).cloned();
                            match existing {
                                Some(prev) => {
                                    let prev_desc = match &prev {
                                        crate::sema::Sema::AliasTarget::Symbol(m) => {
                                            if m.as_ref() == target_mangled {
                                                continue; // same target re-imported: idempotent
                                            }
                                            format!("'{}'", m)
                                        }
                                        crate::sema::Sema::AliasTarget::Module(m) => {
                                            format!("module '{}'", m)
                                        }
                                    };
                                    self.sema_result.add_error(
                                        crate::sema::Sema::SemaError::new(
                                            &format!(
                                                "import alias '{}' is already bound to {} — \
                                                 aliasing two different symbols under one name \
                                                 is ambiguous; import one and qualify the other",
                                                local, prev_desc
                                            ),
                                            0,
                                            0,
                                        ),
                                    );
                                }
                                None => {
                                    self.sema_result.put_import_alias(
                                        local,
                                        crate::sema::Sema::AliasTarget::Symbol(
                                            target_mangled.into_boxed_str(),
                                        ),
                                        module.name,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Populates the witness table: iterates trait impls in the module and registers them.
    ///
    /// For each `impl Trait for Type`, extracts trait_name and type_name, looks up type_def to
    /// obtain the type_id, and registers the methods into the witness table.
    pub(super) fn populate_witness_table(&mut self, module: &Module<'_>) {
        type TraitImplInfo = (String, String, Vec<(String, u16)>);
        // Collect trait impl info up front to avoid borrowing module while iterating and holding
        // &mut self simultaneously.
        let mut impls: Vec<TraitImplInfo> = Vec::new();

        for decl in module.declarations.iter() {
            if let Decl::TypeDecl { name, implemented_traits, methods, .. } = &decl.node {
                // Look up type_id (using type_def_index + FIRST_DYNAMIC_TYPE_ID offset).
                let type_id = self
                    .sema_result
                    .type_def_index
                    .get(*name)
                    .map(|&idx| dynamic_type_id(idx));

                if let Some(tid) = type_id {
                    // Register a witness entry for each implemented trait.
                    for impl_trait in implemented_traits.iter() {
                        let trait_name = impl_trait.trait_name.to_string();
                        // Collect method slots: method_name → method_idx (position in TypeDefInfo.methods).
                        let method_slots: Vec<(String, u16)> = methods
                            .iter()
                            .enumerate()
                            .map(|(i, m)| (m.name.to_string(), i as u16))
                            .collect();
                        impls.push((trait_name, name.to_string(), method_slots));
                        let _ = tid; // tid is used in the loop below.
                    }
                }
            }
        }

        // Register into the witness table.
        for (trait_name, type_name, method_slots_vec) in impls {
            // Re-query type_id (the previous borrow has been released).
            let type_id = self
                .sema_result
                .type_def_index
                .get(type_name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                let mut slots = FxHashMap::default();
                for (method_name, method_idx) in method_slots_vec {
                    slots.insert(method_name.into_boxed_str(), method_idx);
                }
                self.witness_table
                    .register(&trait_name, tid, &type_name, slots);
                // Record module ownership for incremental purge (witness key).
                let mod_name = self.current_module_name.clone();
                self.sema_result.module_ownership.witness_keys
                    .entry(mod_name)
                    .or_default()
                    .insert((trait_name.clone().into_boxed_str(), tid));
            }
        }
    }

    /// Wraps the return type into `Async<T>` as needed.
    ///
    /// The return type `X` declared by an async function/method actually denotes `Async<X>`
    /// (aligned with the Lambda/Zig implementation).
    /// If the user already wrote `Async<X>` explicitly, do not double-wrap; otherwise wrap as
    /// `Async<ret_ty>`.
    /// Used uniformly by predeclare_declarations, check_decl(FunDecl), type-block methods, and
    /// trait-block methods, to avoid omissions caused by repeated inlining (root cause D:
    /// type/trait-block methods previously missed the wrap).
    pub(super) fn wrap_async_return(&mut self, ret_ty: TypeHandle, is_async: bool) -> TypeHandle {
        if !is_async {
            return ret_ty;
        }
        let resolved = self.arena.resolve(ret_ty);
        let already_async = matches!(self.arena.get(resolved), Type::Async(_));
        if already_async {
            ret_ty
        } else {
            self.arena.make_async(ret_ty)
        }
    }

    /// Predeclares the module's functions and type constructors into the environment.
    ///
    /// Functions and type constructors are registered into the module-dedicated env
    /// (module_env) rather than root_env.
    /// The module env's parent points to root_env (or the parent path env), so the module can
    /// still access global builtins.
    /// Callers resolve symbols directly by bare name in the module env via the env reference
    /// carried by ModuleRef, without needing mangled names.
    pub fn predeclare_declarations(&mut self, module: &Module<'_>, root_env: EnvId) {
        let module_path = module_logical_path(module.name);
        // Get or create the module-dedicated env (idempotent: ensure_module_env reuses existing envs).
        let module_env = match &module_path {
            Some(mp) => self.ensure_module_env(mp, root_env),
            None => root_env,
        };
        // Record the current module env for use by check_decl (e.g. let-bindings).
        self.current_module_env = Some(module_env);
        // Same-module duplicate function definitions: `define` is first-wins and
        // silently ignores the second registration, so a redefinition used to
        // compile quietly with confusing resolution (Bug #94). Track names within
        // this predeclare pass (immune to any module being processed twice).
        let mut seen_fns: FxHashSet<&str> = FxHashSet::default();
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { name, type_params, params, return_type, is_async, .. } => {
                    if !seen_fns.insert(name) {
                        self.add_error_at(
                            &format!("duplicate definition of function '{}' in this module", name),
                            decl.span.line,
                            decl.span.column,
                        );
                    }
                    // Top-level functions disallow a self parameter (detected via the ThisType
                    // type node, not by parameter name).
                    if !params.is_empty() && self.is_this_param(params[0].type_annotation, &module.arena) {
                        self.add_error_at(
                            "this parameter is not allowed in top-level function",
                            decl.span.line,
                            decl.span.column,
                        );
                    }
                    // Generic function: push type_bindings so type_from_ast resolves type
                    // parameters as rigid vars; the predeclared type matches check_decl
                    // (avoids generic params being mis-resolved as Adt).
                    if !type_params.is_empty() {
                        self.push_type_bindings(
                            &type_params.iter().map(|tp| {
                                (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                            }).collect::<Vec<_>>(),
                        );
                    }
                    // All functions are predeclared (including generics): generic functions use
                    // rigid vars as type parameters, solving forward-reference issues (function
                    // bodies can reference later-defined same-module functions).
                    let param_types: Vec<TypeHandle> = params
                        .iter()
                        .map(|p| match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, &module.arena),
                            None => self.arena.fresh_type_var(),
                        })
                        .collect();
                    // async function: the user-declared return type X actually denotes
                    // Async<X> (aligned with check_decl/Lambda).
                    // - If the user already wrote Async<X> explicitly, do not double-wrap.
                    // - Otherwise wrap as Async<ret_ty_raw>.
                    let ret_ty_raw = match return_type {
                        Some(rt) => self.type_from_ast(*rt, &module.arena),
                        None => self.arena.fresh_type_var(),
                    };
                    let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                    let fn_ty = self.arena.make_fn(
                        param_types.into_boxed_slice(),
                        ret_ty,
                    );
                    // Register into the module-dedicated env (bare name); ModuleRef lookup uses
                    // lookup_local in this env.
                    // Also register into root_env to make it globally visible (cross-module
                    // bare-name reference compatibility):
                    //   define does not overwrite existing bindings; the first registration wins.
                    self.sema_result.env.define(module_env, name, fn_ty);
                    self.sema_result.env.define(root_env, name, fn_ty);
                    // Generic function: pop type_bindings (symmetric with check_decl).
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                }
                Decl::TypeDecl { name, type_params, def, .. } => {
                    // Predeclare the type constructor.
                    let self_ty = if type_params.is_empty() {
                        self.arena.make_adt((*name).into(), Box::new([]))
                    } else {
                        // Generic type: predeclare with a rigid var.
                        self.arena.fresh_rigid_var()
                    };
                    // Constructors are registered into root_env (not module_env):
                    // constructors are companion symbols of the type, at the same naming level,
                    // and must use redefine to overwrite the ModuleRef alias previously registered
                    // by register_module_aliases, so `DateTime(...)` resolves to the constructor
                    // rather than the ModuleRef.
                    match def {
                        crate::ast::Ast::TypeDef::Adt { constructors } => {
                            for ctor in constructors.iter() {
                                let ctor_fn_ty = self.build_ctor_fn_type(ctor, name, &module.arena);
                                self.sema_result.env.redefine(root_env, ctor.name, ctor_fn_ty);
                                // Record constructor short name → module env (Zig @This semantics),
                                // so `TypeName.free_func(args)` can fall back to lookup a
                                // in-module free function.
                                self.ctor_module_envs.insert(ctor.name.to_string(), module_env);
                            }
                        }
                        crate::ast::Ast::TypeDef::Newtype { name: ctor_name, inner } => {
                            // newtype constructor: (inner) -> Self.
                            let inner_ty = self.type_from_ast(*inner, &module.arena);
                            let ctor_fn_ty = self.arena.make_fn(
                                vec![inner_ty].into_boxed_slice(),
                                self_ty,
                            );
                            self.sema_result.env.redefine(root_env, ctor_name, ctor_fn_ty);
                            // Record constructor short name → module env (Zig @This semantics),
                            // so `TypeName.free_func(args)` can fall back to lookup a
                            // in-module free function.
                            self.ctor_module_envs.insert(ctor_name.to_string(), module_env);
                        }
                        _ => {}
                    }
                    let _ = self_ty;
                }
                _ => {}
            }
        }
    }

    /// Builds the function type for a constructor.
    pub(super) fn build_ctor_fn_type(
        &mut self,
        ctor: &crate::ast::Ast::ConstructorDef<'_>,
        type_name: &str,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let param_types: Vec<TypeHandle> = ctor
            .fields
            .iter()
            .map(|f| self.type_from_ast(f.ty, ast))
            .collect();
        let ret_ty = match ctor.return_type {
            Some(rt) => self.type_from_ast(rt, ast),
            None => self.arena.make_adt(type_name.into(), Box::new([])),
        };
        // Zero-argument variants are values, not functions: Leaf's type should be Tree, not
        // () -> Tree.
        if param_types.is_empty() {
            return ret_ty;
        }
        self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
    }

    /// Check for cyclic type aliases (Bug #80).
    ///
    /// Iterates through all type definitions; for each alias, follows the `target_type_name`
    /// chain to detect cycles. The existing `visiting`-based cycle detection in
    /// `resolve_name_to_type` / `resolve_named_type_resolved` is ineffective because the
    /// `target_type` short-circuit (returning the pre-resolved TypeHandle directly) bypasses
    /// the recursive `target_type_name` path where cycle detection lives.
    ///
    /// This function reports a `cyclic type alias` error for each cycle found, e.g.:
    /// `type A = B` + `type B = A` → `cyclic type alias: A -> B -> A`.
    pub(super) fn check_alias_cycles(&mut self) {
        use std::collections::HashSet;
        // Collect already-reported cycle messages to avoid duplicates across modules
        // (sema_result.type_defs is cumulative across all modules).
        let existing: HashSet<String> = self.sema_result.errors.iter()
            .filter_map(|e| {
                e.message.as_ref().strip_prefix("cyclic type alias: ").map(String::from)
            })
            .collect();
        let mut reported: HashSet<String> = existing;
        let mut cycles_to_report: Vec<String> = Vec::new();
        for td in self.sema_result.type_defs.values() {
            if td.kind != crate::sema::Sema::TypeDefKind::Alias {
                continue;
            }
            let mut chain: Vec<String> = Vec::new();
            let mut visiting: HashSet<String> = HashSet::new();
            let start_name = td.name.to_string();
            chain.push(start_name.clone());
            visiting.insert(start_name.clone());
            let mut current = td.target_type_name.as_deref().map(String::from);
            loop {
                match current {
                    None => break, // No target_name: not a cycle (or target is a non-named type).
                    Some(ref target_name) => {
                        if visiting.contains(target_name) {
                            // Cycle detected: build the cycle path for the error message.
                            chain.push(target_name.clone());
                            let cycle_start = chain.iter().position(|n| n == target_name)
                                .unwrap_or(0);
                            let cycle_path = chain[cycle_start..].join(" -> ");
                            if reported.insert(cycle_path.clone()) {
                                cycles_to_report.push(cycle_path);
                            }
                            break;
                        }
                        // Look up the target's type def to continue the chain.
                        let target_td = self.sema_result.get_type_def(target_name);
                        match target_td {
                            Some(t) if t.kind == crate::sema::Sema::TypeDefKind::Alias => {
                                visiting.insert(target_name.clone());
                                chain.push(target_name.clone());
                                current = t.target_type_name.as_deref().map(String::from);
                            }
                            _ => break, // Target is not an alias: no cycle.
                        }
                    }
                }
            }
        }
        for cycle_path in cycles_to_report {
            self.sema_result.add_error(crate::sema::Sema::SemaError::new(
                &format!("cyclic type alias: {}", cycle_path),
                0, 0,
            ));
        }
    }

    /// Check for duplicate named fields within a single constructor (Bug #82).
    ///
    /// ADT/record constructors may have positional (unnamed) fields, which are
    /// allowed to repeat; only named fields must be unique. Pattern matching and
    /// field access rely on unique names to disambiguate.
    pub(super) fn check_duplicate_ctor_fields(&mut self) {
        use std::collections::HashSet;
        // Collect (ctor_name, field_name) pairs already reported, to dedupe across
        // modules (type_defs is cumulative).
        let mut reported: HashSet<(String, String)> = self.sema_result.errors.iter()
            .filter_map(|e| {
                e.message.as_ref().strip_prefix("duplicate field '").and_then(|s| {
                    let mut it = s.split("' in constructor ");
                    let field = it.next()?.to_string();
                    let ctor = it.next()?.to_string();
                    Some((field, ctor))
                })
            })
            .collect();
        let mut new_errors: Vec<String> = Vec::new();
        for td in self.sema_result.type_defs.values() {
            for ctor in td.constructors.iter() {
                let mut seen: HashSet<String> = HashSet::new();
                for fname_opt in ctor.field_names.iter() {
                    if let Some(fname) = fname_opt {
                        let fname_s = fname.to_string();
                        let key = (fname_s.clone(), ctor.name.to_string());
                        if reported.contains(&key) {
                            continue;
                        }
                        if !seen.insert(fname_s.clone()) {
                            new_errors.push(format!(
                                "duplicate field '{}' in constructor {}",
                                fname_s, ctor.name
                            ));
                            reported.insert(key);
                        }
                    }
                }
            }
        }
        for msg in new_errors {
            self.sema_result.add_error(crate::sema::Sema::SemaError::new(&msg, 0, 0));
        }
    }

    /// Checks a single declaration (infers function body / expression).
    ///
    /// Takes `&Decl` and `decl_span` as separate parameters: top-level declarations get
    /// span+node from `Spanned<Decl>`, while nested `LocalDecl`'s `Box<Decl>` has no span and
    /// the caller supplies it from the enclosing Stmt.
    pub(super) fn check_decl(&mut self, decl: &Decl<'_>, decl_span: crate::ast::Ast::Span, ast: &AstArena<'_>, env: EnvId) {
        match decl {
            Decl::FunDecl { name, type_params, params, return_type, body, extern_c_body, is_async, attributes, .. } => {
                // ── FFI permission check ──
                // `@extern` / `@c_include` / `#{ }#` are only allowed in builtin (stdlib) modules;
                // `@internal` marks stdlib implementation primitives (callable only from
                // `builtin/**` / `std/**` — enforced at call binding in the IR builder) and may
                // be declared anywhere inside the stdlib, but never in user code.
                // The builtin check is based on the module-name path prefix
                // (`current_module_name` is set from `module.name` in `check_module_with_env`;
                // builtin module names look like "builtin/io/Raw.frond").
                let is_builtin = self.current_module_name.starts_with("builtin/");
                let is_stdlib = is_builtin || self.current_module_name.starts_with("std/");
                if !is_builtin {
                    for attr in attributes {
                        match attr.name {
                            crate::ffi::ATTR_INTERNAL if !is_stdlib => self.add_error_at(
                                "attribute '@internal' is reserved for the standard library implementation",
                                decl_span.line,
                                decl_span.column,
                            ),
                            crate::ffi::ATTR_EXTERN | crate::ffi::ATTR_C_INCLUDE => self.add_error_at(
                                &format!("attribute '@{}' is only allowed in builtin (stdlib) modules", attr.name),
                                decl_span.line,
                                decl_span.column,
                            ),
                            _ => {}
                        }
                    }
                    if extern_c_body.is_some() {
                        self.add_error_at(
                            "inline C body (#{ }#) is only allowed in builtin (stdlib) modules",
                            decl_span.line,
                            decl_span.column,
                        );
                    }
                }
                // Top-level functions disallow a self parameter (detected via the ThisType type
                // node, not by parameter name; self is only allowed inside type/trait block methods).
                if !params.is_empty() && self.is_this_param(params[0].type_annotation, ast) {
                    self.add_error_at(
                        "this parameter is not allowed in top-level function",
                        decl_span.line,
                        decl_span.column,
                    );
                }
                // Create a child environment for the function.
                let fn_env = self.sema_result.env.child(env);
                // Type parameter bindings.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // @extern("C") function: register the signature but skip body type checking
                // (the body is C code, not a Frond expression).
                if extern_c_body.is_some() {
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                    let _ = name;
                    return;
                }
                // Parameter bindings (also collect parameter types to build the function type).
                let param_types: Vec<TypeHandle> = params.iter().map(|p| {
                    let param_ty = match p.type_annotation {
                        Some(ta) => self.type_from_ast(ta, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    self.sema_result.env.define(fn_env, p.name, param_ty);
                    param_ty
                }).collect();
                // Return type (use fresh_type_var when unannotated; later unified with the body type).
                // async function: the user-declared return type X actually denotes Async<X>
                // (aligned with the Lambda/Zig implementation).
                // - If the user already wrote Async<X> explicitly, do not double-wrap.
                // - Otherwise wrap as Async<ret_ty_raw>.
                let ret_ty_raw = match return_type {
                    Some(rt) => self.type_from_ast(*rt, ast),
                    None => self.arena.fresh_type_var(),
                };
                let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                // Build the function type and register it into fn_env (supports recursive
                // self-reference) and env (supports subsequent references).
                // Top-level functions were already predeclared by predeclare_declarations; define
                // returns false and does not overwrite.
                let fn_ty = self.arena.make_fn(
                    param_types.into_boxed_slice(),
                    ret_ty,
                );
                self.sema_result.env.define(fn_env, *name, fn_ty);
                self.sema_result.env.define(env, *name, fn_ty);
                // Set the expected return type.
                let prev_return = self.expected_return;
                self.expected_return = Some(ret_ty);
                // Infer the function body.
                let body_ty = self.infer_expr(*body, ast, fn_env, self.expected_return);
                // Restore.
                self.expected_return = prev_return;
                // Unify the return type with the body type:
                // - Unannotated return type: ret_ty is a fresh TypeVar; bind via unify_or_constrain.
                // - Annotated return type: unify via unify_return_type, handling async unfolding
                //   (declared Async<Throw<T, E>>, body returns Throw<T', E'> directly; must unfold
                //   the Async layer to unify the inner Throw so TypeVars in E' get solved).
                //   On failure, register an Equality constraint for the solver to retry later.
                if return_type.is_none() {
                    self.unify_or_constrain(ret_ty, body_ty);
                } else if self.unify_return_type(ret_ty, body_ty).is_err() {
                    self.solver.add_equality(ret_ty, body_ty);
                }
                // Non-void declared return type with no trailing expression and no
                // return/throw would silently return garbage at runtime — reject it.
                if return_type.is_some() {
                    self.check_missing_return_value(
                        &format!("function '{}'", name),
                        ret_ty,
                        *body,
                        ast,
                        decl_span.line,
                        decl_span.column,
                    );
                    // Sync Throw-returning function with a bare non-Throw tail:
                    // the from_datetime_utc/scanln leak class. ret_ty arrives
                    // Async-wrapped for async funs, which the helper filters out.
                    self.check_throw_tail_wrapped(
                        &format!("function '{}'", name),
                        ret_ty,
                        *body,
                        body_ty,
                        ast,
                        decl_span.line,
                        decl_span.column,
                    );
                }
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = name;
            }
            Decl::ExprDecl { expr, stmt, .. } => {
                if let Some(s) = stmt {
                    let _ = self.infer_stmt(*s, ast, env);
                } else {
                    let _ = self.infer_expr(*expr, ast, env, None);
                }
            }
            Decl::TypeDecl { name, type_params, def, implemented_traits, methods, .. } => {
                // Register the nested type definition into sema_result (so constructor calls are
                // recognized during type checking).
                ast_type_decl_to_type_def(self.arena, self.sema_result, *name, type_params, def, ast, decl_span, &self.current_module_name);
                // Register methods into TypeDefInfo.methods (indexed by method_idx).
                // This mirrors what `populate_module` does for top-level TypeDecls. Without this,
                // `lookup_method_idx` cannot find local-type methods and IR dispatch returns void.
                for method in methods.iter() {
                    crate::sema::Sema::ast_method_to_func_sig_pub(self.arena, self.sema_result, *name, method, ast);
                }
                // Type parameter bindings (including kind registration): so references to the
                // generic parameter T inside the type block can be resolved from
                // type_binding_stack.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // Build the ADT type handle.
                let self_ty = if type_params.is_empty() {
                    self.arena.make_adt((*name).into(), Box::new([]))
                } else {
                    // Generic type: build Adt { name, type_args: [rigid_T, ...] }.
                    // Use the rigid var from type_binding_stack as type_args, to avoid producing
                    // unresolved TypeVars when fresh_type_var is used as self_ty.
                    let type_args: Vec<TypeHandle> = type_params.iter()
                        .map(|tp| self.lookup_type_binding(tp.name)
                            .unwrap_or_else(|| self.arena.fresh_type_var()))
                        .collect();
                    self.arena.make_adt((*name).into(), type_args.into_boxed_slice())
                };
                // Register constructor function types into the current env (so Call expressions
                // can find the constructors).
                match def {
                    crate::ast::Ast::TypeDef::Record { fields } => {
                        let param_types: Vec<TypeHandle> = fields.iter().map(|f| {
                            self.type_from_ast(f.ty, ast)
                        }).collect();
                        let fn_ty = self.arena.make_fn(
                            param_types.into_boxed_slice(),
                            self_ty,
                        );
                        self.sema_result.env.define(env, *name, fn_ty);
                    }
                    crate::ast::Ast::TypeDef::Adt { constructors } => {
                        for ctor in constructors {
                            let param_types: Vec<TypeHandle> = ctor.fields.iter().map(|f| {
                                self.type_from_ast(f.ty, ast)
                            }).collect();
                            let fn_ty = if param_types.is_empty() {
                                // Zero-argument variants are values, not functions.
                                self_ty
                            } else {
                                self.arena.make_fn(
                                    param_types.into_boxed_slice(),
                                    self_ty,
                                )
                            };
                            self.sema_result.env.define(env, ctor.name, fn_ty);
                        }
                    }
                    crate::ast::Ast::TypeDef::Alias { .. } | crate::ast::Ast::TypeDef::Newtype { .. } => {}
                }
                // Type method checking.
                self.push_this_type(self_ty);
                // Expose the implemented trait names to method-body inference:
                // `super.method(...)` resolves against the trait-default layer of
                // these traits (see infer_super_method_call).
                self.current_type_decl_traits = Some(
                    implemented_traits.iter().map(|t| t.trait_name.into()).collect(),
                );
                // First register all methods as functions into env (supports bare-name method
                // call syntax `method(recv, args)`), then check method bodies (avoids
                // forward-reference issues).
                for method in methods.iter() {
                    let m_param_types: Vec<TypeHandle> = method.params.iter().map(|p| {
                        if self.is_this_param(p.type_annotation, ast) {
                            self_ty
                        } else {
                            match p.type_annotation {
                                Some(ta) => self.type_from_ast(ta, ast),
                                None => self.arena.fresh_type_var(),
                            }
                        }
                    }).collect();
                    let m_ret_ty_raw = match method.return_type {
                        Some(rt) => self.type_from_ast(rt, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    // async method: wrap the return type as Async<T> (aligned with top-level
                    // FunDecl/predeclare; root cause D).
                    let m_ret_ty = self.wrap_async_return(m_ret_ty_raw, method.is_async);
                    let m_fn_ty = self.arena.make_fn(
                        m_param_types.into_boxed_slice(),
                        m_ret_ty,
                    );
                    self.sema_result.env.define(env, method.name, m_fn_ty);
                }
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.sema_result.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_this_param(param.type_annotation, ast) {
                                self.infer_this_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.sema_result.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async method: wrap the return type as Async<T> (aligned with FunDecl;
                        // root cause D); unify_return_type unfolds the Async layer to unify the
                        // inner type with the body type.
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // Non-void declared return type with no trailing expression and no
                        // return/throw would silently return garbage at runtime — reject it.
                        if let Some(r) = ret_ty {
                            self.check_missing_return_value(
                                &format!("method '{}'", method.name),
                                r,
                                body,
                                ast,
                                decl_span.line,
                                decl_span.column,
                            );
                        }
                        // Unify the method body type with the declared return type (aligned with
                        // FunDecl):
                        // - Unannotated return type: ret_ty is None → fresh_type_var; bind via
                        //   unify_or_constrain.
                        // - Annotated return type: unify via unify_return_type (handles async
                        //   unfolding); on failure, register an Equality constraint for the
                        //   solver to retry later.
                        // This lets fresh vars produced by expressions that don't depend on
                        // `expected` (e.g. FieldAccess) be solved via ret_ty, avoiding orphan
                        // TypeVars.
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                        // Sync Throw-returning method with a bare non-Throw tail
                        // (the from_datetime_utc/scanln leak class). ret_ty arrives
                        // Async-wrapped for async methods, which the helper filters.
                        if let Some(r) = ret_ty {
                            self.check_throw_tail_wrapped(
                                &format!("method '{}'", method.name),
                                r,
                                body,
                                body_ty,
                                ast,
                                decl_span.line,
                                decl_span.column,
                            );
                        }
                    }
                }
                self.current_type_decl_traits = None;
                self.pop_this_type();
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
            }
            Decl::TraitDecl { name, type_params, methods, .. } => {
                // Register the nested trait definition into sema_result (so trait type
                // annotations are recognized).
                ast_trait_decl_to_trait_def(self.arena, self.sema_result, name, methods, ast);
                // Type parameter bindings (including kind registration): so references to
                // generic parameters inside the trait block can be resolved from
                // type_binding_stack.
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                let self_var = self.push_this_type_var();
                self.current_trait_name = Some((*name).to_string().into_boxed_str());
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.sema_result.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_this_param(param.type_annotation, ast) {
                                self.infer_this_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.sema_result.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async default method: wrap the return type as Async<T> (aligned with
                        // FunDecl; root cause D); unify_return_type unfolds the Async layer to
                        // unify the inner type with the body type.
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // Non-void declared return type with no trailing expression and no
                        // return/throw would silently return garbage at runtime — reject it.
                        if let Some(r) = ret_ty {
                            self.check_missing_return_value(
                                &format!("method '{}'", method.name),
                                r,
                                body,
                                ast,
                                decl_span.line,
                                decl_span.column,
                            );
                        }
                        // Unify the method body type with the declared return type (aligned with
                        // FunDecl):
                        // - Unannotated return type: ret_ty is None → fresh_type_var; bind via
                        //   unify_or_constrain.
                        // - Annotated return type: unify via unify_return_type (handles async
                        //   unfolding); on failure, register an Equality constraint for the
                        //   solver to retry later.
                        // This lets fresh vars produced by expressions that don't depend on
                        // `expected` (e.g. FieldAccess) be solved via ret_ty, avoiding orphan
                        // TypeVars.
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                        // Sync Throw-returning method with a bare non-Throw tail
                        // (the from_datetime_utc/scanln leak class). ret_ty arrives
                        // Async-wrapped for async methods, which the helper filters.
                        if let Some(r) = ret_ty {
                            self.check_throw_tail_wrapped(
                                &format!("method '{}'", method.name),
                                r,
                                body,
                                body_ty,
                                ast,
                                decl_span.line,
                                decl_span.column,
                            );
                        }
                    }
                }
                self.pop_this_type();
                self.current_trait_name = None;
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = (name, self_var);
            }
            _ => {}
        }
    }

    /// Runs kind_check on all type annotations in the module.
    pub(super) fn run_kind_checks(&mut self, module: &Module<'_>) {
        let mut errors = Vec::new();
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { params, return_type, .. } => {
                    for p in params.iter() {
                        if let Some(ta) = p.type_annotation {
                            check_type_node(self.sema_result, &module.arena, ta, &[], &mut errors);
                        }
                    }
                    if let Some(rt) = return_type {
                        check_type_node(self.sema_result, &module.arena, *rt, &[], &mut errors);
                    }
                }
                Decl::TypeDecl { def: crate::ast::Ast::TypeDef::Adt { constructors }, .. } => {
                    for ctor in constructors.iter() {
                        for f in ctor.fields.iter() {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                f.ty,
                                &[],
                                &mut errors,
                            );
                        }
                        if let Some(rt) = ctor.return_type {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                rt,
                                &[],
                                &mut errors,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        for e in errors {
            self.sema_result.add_error(e);
        }
    }

}
