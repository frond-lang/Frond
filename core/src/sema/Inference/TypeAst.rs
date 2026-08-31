//! TypeAst — AST-to-Type conversion (type_from_ast family). Mechanically split from Inference.rs (no logic changes).

// =========================================================================
// phase5: InferContext extensions — type resolution, freshening, structural equality, throw checks
//
// Adds InferContext methods ported from `src/sema/type_check.zig` and `throw_check.zig`.
// =========================================================================


use super::*;

impl<'a> InferContext<'a> {
    /// Resolves an AST TypeNode to a TypeHandle (convenience overload, with no type-parameter map).
    pub fn type_from_ast(&mut self, type_ref: AstTypeRef, ast: &AstArena<'_>) -> TypeHandle {
        let empty = FxHashMap::default();
        self.type_from_ast_with_params(type_ref, ast, &empty)
    }

    /// Resolves a name to a TypeHandle (alias unfolding + cycle detection).
    ///
    /// This is the core of Named type resolution: type_param_map → type_binding → builtin scalars →
    /// trait → recursive Alias unfolding in type_defs → user-defined Adt.
    /// `visiting` is used for alias cycle detection (A→B→A); on a cycle it returns Adt(name) to terminate.
    pub(super) fn resolve_name_to_type(
        &mut self,
        name: &str,
        type_param_map: &FxHashMap<String, TypeHandle>,
        visiting: &mut FxHashSet<String>,
    ) -> TypeHandle {
        // 1. Type-parameter map.
        if let Some(ty) = type_param_map.get(name) {
            return *ty;
        }
        // 2. Type binding stack (generic scope).
        if let Some(ty) = self.lookup_type_binding(name) {
            return ty;
        }
        // 3. Builtin scalars + str/null/void: derived from BUILTIN_TABLE.
        if let Some(ct) = name_to_concrete(name) {
            return self.arena.make(ct);
        }
        // 3.5 Opaque nongeneric builtins (Lib): dedicated Type variant, not an Adt.
        if name == crate::types::NAME_LIB {
            return self.arena.make(Type::Lib);
        }
        // 4. trait definition → Trait type (module-scoped canonical key:
        // own-module traits shadow std bare names).
        if let Some(td) = self.sema_result.get_trait_def(name) {
            let canonical = self.sema_result.resolve_trait_key(name);
            let _ = td;
            return self.arena.make_trait(canonical.into(), Box::new([]));
        }
        // Module-scoped canonical key: own module shadows std, imports follow,
        // std bare keys then unique user fallbacks (resolve_type_key).
        let canonical = self.sema_result.resolve_type_key(name);
        // Alias cycle detection.
        if visiting.contains(canonical.as_str()) {
            return self.arena.make_adt(canonical.into(), Box::new([]));
        }
        visiting.insert(canonical.to_string());
        // 5. Alias unfolding: type Name = T → resolve T.
        // Prefer the already-resolved target_type (TypeHandle), which covers non-named targets like functions/Records/Arrays;
        // fall back to target_type_name (a named target, e.g. type A = B).
        let alias_info = self
            .sema_result
            .get_type_def(&canonical)
            .filter(|td| td.kind == TypeDefKind::Alias)
            .map(|td| (td.target_type, td.target_type_name.as_deref().map(String::from), td.type_params.len()));
        if let Some((alias_target_ty, alias_target_name, param_count)) = alias_info {
            if param_count > 0 {
                // A GENERIC alias must be applied, not used bare — the stored
                // target still carries name-keyed placeholders; returning it
                // as-is would leak `Adt("T")` junk into signatures.
                self.add_error(&format!(
                    "generic type alias '{}' requires type arguments (e.g. '{}<arg>)",
                    name, name
                ));
                visiting.remove(canonical.as_str());
                return self.arena.make(Type::Unknown);
            }
            if let Some(inner_ty) = alias_target_ty {
                visiting.remove(canonical.as_str());
                return inner_ty;
            }
            if let Some(target_name) = alias_target_name {
                let result = self.resolve_name_to_type(&target_name, type_param_map, visiting);
                visiting.remove(canonical.as_str());
                return result;
            }
        }
        visiting.remove(canonical.as_str());
        // 6. User-defined type → Adt (canonical, module-qualified for user
        // modules — the runtime identity).
        self.arena.make_adt(canonical.into(), Box::new([]))
    }

    /// Whether a source-qualified type spelling resolves to something known:
    /// a registered type/trait (after module-qualifier mapping) or a builtin
    /// generic/scalar reachable through the path (`std.Async` style).
    fn known_qualified(&self, name: &str) -> bool {
        if self
            .sema_result
            .get_type_def(&self.sema_result.resolve_type_key(name))
            .is_some()
        {
            return true;
        }
        if self.sema_result.get_trait_def(name).is_some() {
            return true;
        }
        let tail = name.rsplit('.').next().unwrap_or(name);
        is_builtin_generic_type(tail) || name_to_concrete(tail).is_some()
    }

    /// Resolves an AST TypeNode to a TypeHandle (full version, with a type-parameter map).
    ///
    /// Handles all TypeNode variants: Named, ThisType, Generic, Nullable, RefType, RawPtr,
    /// Function, Record, Array, KindAnnotated. Builtin scalars go through from_scalar_name;
    /// the generic Throw is special-cased into a Throw type; other builtin generics become Generic;
    /// user-defined ADTs become Adt; traits become Trait.
    pub fn type_from_ast_with_params(
        &mut self,
        type_ref: AstTypeRef,
        ast: &AstArena<'_>,
        type_param_map: &FxHashMap<String, TypeHandle>,
    ) -> TypeHandle {
        let tn = &ast.ty(type_ref).node;
        match tn {
            TypeNode::Named { name } => {
                // Qualified spelling sanity gate: unlike bare names (which
                // fall through to an Adt for forward-reference recovery), a
                // dotted path that maps to no module and no definition is
                // always a real error — fail loudly with the spelling.
                if name.contains('.') && !self.known_qualified(name) {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at(
                        &format!(
                            "unknown type '{}': no module of that name is imported or loaded",
                            name
                        ),
                        span.line,
                        span.column,
                    );
                    return self.arena.make(Type::Unknown);
                }
                // Delegate to resolve_name_to_type: builtin scalar → trait → alias unfolding → Adt.
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name, type_param_map, &mut visiting)
            }
            TypeNode::ThisType => match self.current_this_type() {
                Some(ty) => ty,
                None => {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at("This type can only be used within type or trait methods", span.line, span.column);
                    self.arena.make(Type::Void)
                }
            },
            TypeNode::Generic { name, args } => {
                // Recursively resolve type arguments.
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.type_from_ast_with_params(a, ast, type_param_map))
                    .collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // Qualified spelling sanity gate (see the Named arm).
                if name.contains('.') && !self.known_qualified(name) {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at(
                        &format!(
                            "unknown type '{}': no module of that name is imported or loaded",
                            name
                        ),
                        span.line,
                        span.column,
                    );
                    return self.arena.make(Type::Unknown);
                }

                // Higher-kinded type (HKT) in the type-parameter map: F<T> where F is a type parameter.
                if let Some(&param_handle) = type_param_map.get(*name) {
                    // Kind check: verify that F's kind matches the number and kinds of the arguments.
                    let constructor_kind = self.arena.kind_of(param_handle);
                    // If constructor_kind is not Star (i.e. F is a type constructor),
                    // or args is non-empty (i.e. an F<T> application), perform the kind check.
                    if !matches!(constructor_kind, SemKind::Star) || !args_box.is_empty() {
                        let arg_kinds: Vec<SemKind> = args_box
                            .iter()
                            .map(|&a| self.arena.kind_of(a))
                            .collect();
                        if let Err(kind_err) = self.arena.check_kind_application(&constructor_kind, &arg_kinds) {
                            // Error recovery: record the error but keep constructing the type.
                            let span = ast.ty(type_ref).span;
                            self.add_error_at(&kind_err, span.line, span.column);
                        }
                    }
                    return self.arena.make_generic((*name).into(), args_box);
                }
                // Builtin generic types (Throw/Atomic/Async/Channel, etc.) construct dedicated Type variants
                // and do not go through the Type::Generic path — this avoids later matching builtin generics by string name.
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic((*name).into(), args_box);
                }
                // trait definition → Trait type (module-scoped canonical).
                if self.sema_result.get_trait_def(name).is_some() {
                    let canonical = self.sema_result.resolve_trait_key(name);
                    return self.arena.make_trait(canonical.into(), args_box);
                }
                // Module-scoped canonical key for the user-type lookups below
                // (own module shadows std bare names).
                let canonical = self.sema_result.resolve_type_key(name);
                // Generic TYPE ALIAS application (`CmpFn<i32>`): unfold to the
                // alias target with the params substituted by the applied
                // arguments. Previously this became an opaque Adt that never
                // unfolded — a lambda could not be passed where `CmpFn<i32>`
                // was expected, and the stored target had its params erased.
                if let Some(td) = self.sema_result.get_type_def(&canonical) {
                    if td.kind == TypeDefKind::Alias {
                        if let Some(target) = td.target_type {
                            if td.type_params.len() == args_box.len() {
                                let pairs: Vec<(Box<str>, TypeHandle)> = td
                                    .type_params
                                    .iter()
                                    .zip(args_box.iter())
                                    .map(|(n, &a)| (n.clone(), a))
                                    .collect();
                                return self.substitute_named_adts(target, &pairs);
                            }
                            let span = ast.ty(type_ref).span;
                            self.add_error_at(
                                &format!(
                                    "generic type alias '{}' takes {} type argument(s), got {}",
                                    name,
                                    td.type_params.len(),
                                    args_box.len()
                                ),
                                span.line,
                                span.column,
                            );
                            return self.arena.make(Type::Unknown);
                        }
                    }
                }
                // User-defined generic ADT.
                let has_type_params = self
                    .sema_result
                    .get_type_def(&canonical)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt(canonical.into(), args_box);
                }
                // Fallback: construct a Generic (may be undefined or a forward reference; reported on later use).
                self.arena.make_generic((*name).into(), args_box)
            }
            TypeNode::Nullable { inner } => {
                // Surface syntax `T??` is rejected (AST shape: a Nullable node whose
                // inner node is also Nullable): nullable has no Some-constructor, so a
                // nested nullable carries no meaning and silently collapsing it would
                // let Kotlin/TS users keep wrong expectations (Some(null) semantics).
                // The type machinery still collapses nested nullables produced by
                // generic instantiation (`T?` with T := X?) inside make_nullable —
                // only literally written double-`?` is an error. `Alias?` where Alias
                // resolves to a nullable stays legal (named inner node, not Nullable).
                if self.instantiation_ctx.is_none()
                    && matches!(&ast.ty(*inner).node, TypeNode::Nullable { .. })
                {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at(
                        "double nullable 'T??' is not allowed: nullable has no Some-constructor, so nesting adds no meaning. Use a single '?', or an ADT for two-level absence (e.g. `type Hit<T> = | Missing | Found(T?)`)",
                        span.line,
                        span.column,
                    );
                }
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_nullable(inner_ty)
            }
            TypeNode::RefType { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, false)
            }
            TypeNode::RawPtr { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, true)
            }
            TypeNode::Function { params, return_type } => {
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.type_from_ast_with_params(p, ast, type_param_map))
                    .collect();
                let new_ret = self.type_from_ast_with_params(*return_type, ast, type_param_map);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            TypeNode::Record { fields } => {
                if fields.is_empty() {
                    return self.arena.make(Type::Void);
                }
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.type_from_ast_with_params(f.ty, ast, type_param_map),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), None)
            }
            TypeNode::Array { element_type, size } => {
                let elem_ty = self.type_from_ast_with_params(*element_type, ast, type_param_map);
                self.arena.make_array(elem_ty, *size)
            }
            TypeNode::KindAnnotated { inner, .. } => {
                self.type_from_ast_with_params(*inner, ast, type_param_map)
            }
        }
    }

    // ── freshen_type / apply_type_subst ──

}
