//! Subst — Type-variable substitution and instantiation. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Recursively collects every TypeVar idx in a type, inserting it into subst (with a placeholder value TypeHandle(0); only the key matters).
    pub(super) fn collect_type_vars(&self, ty: TypeHandle, subst: &mut FxHashMap<u32, TypeHandle>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                subst.entry(idx).or_insert(TypeHandle(0));
            }
            // All composite types (incl. Channel/Async/Lazy/Atomic/Sender/Receiver) delegate
            // their child traversal to `for_each_child`, the single source of truth.
            _ => self
                .arena
                .for_each_child(resolved, |c| self.collect_type_vars(c, subst)),
        }
    }

    /// Instantiates a function type: creates fresh non-rigid copies of every unbound TypeVar in the signature.
    ///
    /// Polymorphic built-in functions (e.g. Ok/i8 registered as generics with rigid vars) must be instantiated on every call,
    /// otherwise type constraints from different calls would clash (after the first call permanently binds a var, subsequent calls cannot unify).
    /// Non-polymorphic functions (signatures with no TypeVar) are returned as-is.
    pub(super) fn instantiate_fn_type(&mut self, fn_ty: TypeHandle) -> TypeHandle {
    let resolved = self.arena.resolve(fn_ty);
    // Collect all unbound TypeVar idxs in the function signature (collect_type_vars follows resolve;
    // already-bound TypeVars are not collected).
    let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
    if !matches!(self.arena.get(resolved), Type::Fn(_)) {
        return resolved;
    }
    {
        let (params, return_type) = self.arena.fn_parts(resolved);
        for &p in params.iter() {
            self.collect_type_vars(p, &mut subst);
        }
        self.collect_type_vars(return_type, &mut subst);
    }
    if subst.is_empty() {
        return resolved;
    }
    // Create a fresh non-rigid var for each idx (the collect_type_vars borrow has been released, so a mutable borrow is safe).
    let indices: Vec<u32> = subst.keys().copied().collect();
    for idx in indices {
        let fresh = self.arena.fresh_type_var();
        subst.insert(idx, fresh);
    }
    self.substitute_type(resolved, &subst)
}

    /// Type substitution: replaces the specified TypeVars (by idx) in a type with the bindings from the substitution table.
    ///
    /// Recursively traverses compound types, replacing matching TypeVars. Used to replace a formal parameter's rigid var
    /// with a fresh non-rigid var at the call site, so it can be bound by unification.
    pub(super) fn substitute_type(&mut self, ty: TypeHandle, subst: &FxHashMap<u32, TypeHandle>) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                // Hit in the substitution table → return the substituted type; otherwise leave as-is.
                subst.get(&idx).copied().unwrap_or(resolved)
            }
            Type::Fn(_) => {
                let (params, return_type) = self.arena.fn_parts(resolved);
                let params: Vec<TypeHandle> = params.to_vec();
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.substitute_type(p, subst))
                    .collect();
                let new_ret = self.substitute_type(return_type, subst);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            Type::Record(_) => {
                let fields = self.arena.record_fields(resolved).to_vec();
                let name = self.arena.record_name(resolved).map(|s| s.into());
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: f.name.clone(),
                        ty: self.substitute_type(f.ty, subst),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), name)
            }
            Type::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_adt(name, new_args.into_boxed_slice())
            }
            Type::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_nullable(new_inner)
            }
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                let name: Box<str> = name.into();
                let args: Vec<TypeHandle> = args.to_vec();
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_generic(name, new_args.into_boxed_slice())
            }
            Type::Array(_) => {
                let (element_type, size) = self.arena.array_parts(resolved);
                let new_elem = self.substitute_type(element_type, subst);
                self.arena.make_array(new_elem, size)
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                let new_v = self.substitute_type(value_type, subst);
                let new_e = self.substitute_type(error_type, subst);
                self.arena.make_throw(new_v, new_e)
            }
            Type::Trait(_) => {
                let (name, type_args) = self.arena.trait_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_trait(name, new_args.into_boxed_slice())
            }
            Type::TraitObject(_) => {
                let (trait_name, method_sigs) = self.arena.trait_object_parts(resolved);
                self.arena.make_trait_object(trait_name.into(), method_sigs.to_vec().into_boxed_slice())
            }
            Type::Ref(_) => {
                let (inner, is_raw) = self.arena.ref_parts(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_ref(new_inner, is_raw)
            }
            // Single-element builtin generics — kept in lockstep with `for_each_child` so that
            // TypeVars nested inside them are substituted (otherwise instantiate_fn_type would
            // collect a var via collect_type_vars but fail to replace it here).
            Type::Channel(_) => {
                let elem = self.arena.channel_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_channel(new_elem)
            }
            Type::Async(_) => {
                let value = self.arena.async_value(resolved);
                let new_value = self.substitute_type(value, subst);
                self.arena.make_async(new_value)
            }
            Type::Lazy(_) => {
                let value = self.arena.lazy_value(resolved);
                let new_value = self.substitute_type(value, subst);
                self.arena.make_lazy(new_value)
            }
            Type::Atomic(_) => {
                let elem = self.arena.atomic_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_atomic(new_elem)
            }
            Type::Sender(_) => {
                let elem = self.arena.sender_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_sender(new_elem)
            }
            Type::Receiver(_) => {
                let elem = self.arena.receiver_elem(resolved);
                let new_elem = self.substitute_type(elem, subst);
                self.arena.make_receiver(new_elem)
            }
            Type::ForeignFn(_) => {
                let ret = self.arena.foreign_fn_ret(resolved);
                let new_ret = self.substitute_type(ret, subst);
                self.arena.make_foreign_fn(new_ret)
            }
            // Scalars, Never, Unknown, Void, Null, TraitObject, ModuleRef, Timer have no sub-nodes → return as-is.
            _ => resolved,
        }
    }

    // ── Literal promotion ──
    // v2 convergence: literal_promotion has been replaced by peer_type_binary,
    // literal promotion rules are inlined into peer_type_binary, eliminating the dual-track scheme.

    // ── GADT inference (phase3e) ──

    /// Freshens a type: replaces unbound TypeVars in the type with new TypeVars.
    /// Used when looking up a generic function type from the environment, to keep each call independent (replaces the old HM instantiate).
    pub fn freshen_type(&mut self, ty: TypeHandle) -> TypeHandle {
        // 1. Collect all unbound TypeVar idxs.
        let mut free_vars: Vec<u32> = Vec::new();
        self.collect_free_vars(ty, &mut free_vars);
        if free_vars.is_empty() {
            return ty;
        }
        // 2. Allocate a fresh var for each free var and build the substitution table.
        let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
        for idx in free_vars.iter() {
            let fresh = self.arena.fresh_type_var();
            subst.insert(*idx, fresh);
        }
        // 3. Apply the substitution.
        self.apply_type_subst(ty, &subst)
    }

    /// Recursively collects the idxs of unbound TypeVars in a type (deduplicated).
    ///
    /// Note: Fn types do not collect their internal TypeVars. A function type is a "type scheme";
    /// instantiation of its free variables is handled uniformly by `instantiate_fn_type` at the call site.
    /// If freshen_type also instantiated Fn-internal variables, it would duplicate the work of instantiate_fn_type,
    /// leaving the first set of fresh copies orphaned (referenced by no unify) and ultimately reported as unresolved TypeVars.
    pub(super) fn collect_free_vars(&self, ty: TypeHandle, free_vars: &mut Vec<u32>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::TypeVar(idx) => {
                // A rigid var represents a generic parameter declaration (e.g. T in type ArrayIter<T>);
                // it is fixed within the current scope and must not be freshened/instantiated.
                // Collect only unbound non-rigid TypeVars (local inference variables).
                if !self.arena.type_var(idx).is_rigid && !free_vars.contains(&idx) {
                    free_vars.push(idx);
                }
            }
            // Skip Fn types: instantiation is handled by instantiate_fn_type at the call site.
            Type::Fn(_) => {}
            // All other composite types delegate child traversal to `for_each_child`.
            _ => self
                .arena
                .for_each_child(resolved, |c| self.collect_free_vars(c, free_vars)),
        }
    }

    /// Substitutes TypeVars in a type using the substitution table (by idx). Side-effect free; returns a new type.
    /// Delegates to the existing `substitute_type` implementation.
    pub fn apply_type_subst(
        &mut self,
        ty: TypeHandle,
        subst: &FxHashMap<u32, TypeHandle>,
    ) -> TypeHandle {
        self.substitute_type(ty, subst)
    }

    // ── types_structurally_equal ──

}

/// Checks whether a type references any unresolved TypeVar (in unresolved_set).
/// Used during diagnostics to trace unresolved TypeVars back to their expression locations.
pub(super) fn type_contains_any_unresolved(
    ty: TypeHandle,
    arena: &TypeArena,
    unresolved_set: &FxHashSet<u32>,
) -> bool {
    let resolved = arena.resolve(ty);
    match arena.get(resolved) {
        Type::TypeVar(idx) => unresolved_set.contains(&idx),
        Type::Fn(_) => {
            let (params, return_type) = arena.fn_parts(resolved);
            params.iter().any(|&p| type_contains_any_unresolved(p, arena, unresolved_set))
                || type_contains_any_unresolved(return_type, arena, unresolved_set)
        }
        Type::Record(_) => arena.record_fields(resolved)
            .iter()
            .any(|f| type_contains_any_unresolved(f.ty, arena, unresolved_set)),
        Type::Adt(_) => {
            let (_, type_args) = arena.adt_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Type::Nullable(_) => {
            let inner = arena.nullable_inner(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Type::Ref(_) => {
            let (inner, _) = arena.ref_parts(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Type::Generic(_) => {
            let (_, args) = arena.generic_parts(resolved);
            args.iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Type::Array(_) => {
            let (element_type, _) = arena.array_parts(resolved);
            type_contains_any_unresolved(element_type, arena, unresolved_set)
        }
        Type::Throw(_) => {
            let (value_type, error_type) = arena.throw_parts(resolved);
            type_contains_any_unresolved(value_type, arena, unresolved_set)
                || type_contains_any_unresolved(error_type, arena, unresolved_set)
        }
        Type::Trait(_) => {
            let (_, type_args) = arena.trait_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        _ => false,
    }
}

