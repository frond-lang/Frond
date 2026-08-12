// =========================================================================
// Arena — type allocator + unify/occurs/resolve.
// =========================================================================

use super::Tag::*;
use super::Ty::*;
use super::Display::TypeDisplay;

/// Arena-based `Type` allocator: manages type slots, structural details, type variables,
/// and kind variables.
///
/// Every `Type` is allocated via `make()` and returns a `TypeHandle`; composite types are
/// allocated a `DetailId` via the `make_*` methods, which store their structural data in
/// the `details` table. Methods such as `unify`/`occurs`/`resolve`/`kind_of` need to
/// access `details`, so they are methods on `TypeArena`.
pub struct TypeArena {
    /// Type slots: the `Type` enum (`Copy`), indexed by `TypeHandle`.
    pub types: Vec<Type>,
    /// Structural detail table: structural data for composite types, indexed by `DetailId`.
    pub details: Vec<TypeDetail>,
    /// Type variable table: indexed by the `TypeVar(u32)` payload.
    pub type_vars: Vec<TypeVar>,
    /// Kind variable binding table: indexed by `SemKind::Var(idx)`.
    kind_vars: Vec<Option<SemKind>>,
}

impl Default for TypeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeArena {
    pub fn new() -> Self {
        TypeArena {
            types: Vec::new(),
            details: Vec::new(),
            type_vars: Vec::new(),
            kind_vars: Vec::new(),
        }
    }

    // -- Basic access --

    #[inline]
    pub fn len(&self) -> usize {
        self.types.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
    #[inline]
    pub fn type_vars_len(&self) -> usize {
        self.type_vars.len()
    }
    #[inline]
    pub fn get(&self, h: TypeHandle) -> Type {
        self.types[h.0 as usize]
    }

    /// Allocate a `Type` and return its handle.
    pub fn make(&mut self, ty: Type) -> TypeHandle {
        let h = TypeHandle(self.types.len() as u32);
        self.types.push(ty);
        h
    }

    /// Allocate a `TypeDetail` and return its `DetailId`.
    fn make_detail(&mut self, detail: TypeDetail) -> DetailId {
        let id = DetailId(self.details.len() as u32);
        self.details.push(detail);
        id
    }

    /// Get a reference to a detail (panics for variants where `has_detail()` is false).
    #[inline]
    pub fn detail(&self, id: DetailId) -> &TypeDetail {
        &self.details[id.0 as usize]
    }

    // -- Composite type constructors (allocate detail + make Type) --

    pub fn make_throw(&mut self, value: TypeHandle, error: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Throw {
            value_type: value,
            error_type: error,
        });
        self.make(Type::Throw(id))
    }
    pub fn make_channel(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Channel { elem });
        self.make(Type::Channel(id))
    }
    pub fn make_async(&mut self, value: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Async { value });
        self.make(Type::Async(id))
    }
    pub fn make_lazy(&mut self, value: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Lazy { value });
        self.make(Type::Lazy(id))
    }
    pub fn make_atomic(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Atomic { elem });
        self.make(Type::Atomic(id))
    }
    pub fn make_sender(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Sender { elem });
        self.make(Type::Sender(id))
    }
    pub fn make_receiver(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Receiver { elem });
        self.make(Type::Receiver(id))
    }
    pub fn make_array(&mut self, elem: TypeHandle, size: Option<u64>) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Array { elem, size });
        self.make(Type::Array(id))
    }
    pub fn make_ref(&mut self, inner: TypeHandle, is_raw: bool) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Ref { inner, is_raw });
        self.make(Type::Ref(id))
    }
    pub fn make_fn(
        &mut self,
        params: Box<[TypeHandle]>,
        return_type: TypeHandle,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Fn { params, return_type });
        self.make(Type::Fn(id))
    }
    pub fn make_nullable(&mut self, inner: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Nullable { inner });
        self.make(Type::Nullable(id))
    }
    pub fn make_adt(
        &mut self,
        name: Box<str>,
        type_args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Adt { name, type_args });
        self.make(Type::Adt(id))
    }
    pub fn make_record(
        &mut self,
        fields: Box<[FieldType]>,
        name: Option<Box<str>>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Record { fields, name });
        self.make(Type::Record(id))
    }
    pub fn make_trait(
        &mut self,
        name: Box<str>,
        type_args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Trait { name, type_args });
        self.make(Type::Trait(id))
    }
    pub fn make_trait_object(
        &mut self,
        trait_name: Box<str>,
        method_sigs: Box<[TraitMethodSig]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::TraitObject {
            trait_name,
            method_sigs,
        });
        self.make(Type::TraitObject(id))
    }
    pub fn make_module_ref(&mut self, path: Box<str>, env: EnvId) -> TypeHandle {
        let id = self.make_detail(TypeDetail::ModuleRef { path, env });
        self.make(Type::ModuleRef(id))
    }
    pub fn make_generic(
        &mut self,
        name: Box<str>,
        args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Generic { name, args });
        self.make(Type::Generic(id))
    }

    // -- Structural accessors (replace inline access like ConcreteType::Fn{params, return_type}) --

    /// Extract the `DetailId` from a `Type` (panics when `has_detail` is false).
    #[inline]
    pub fn detail_id_of(&self, ty: Type) -> DetailId {
        match ty {
            Type::Throw(id)
            | Type::Channel(id)
            | Type::Async(id)
            | Type::Lazy(id)
            | Type::Atomic(id)
            | Type::Sender(id)
            | Type::Receiver(id)
            | Type::Array(id)
            | Type::Ref(id)
            | Type::Fn(id)
            | Type::Nullable(id)
            | Type::Adt(id)
            | Type::Record(id)
            | Type::Trait(id)
            | Type::TraitObject(id)
            | Type::ModuleRef(id)
            | Type::Generic(id) => id,
            _ => panic!("Type {:?} does not carry a DetailId", ty),
        }
    }

    /// The `(value_type, error_type)` of a `Throw`.
    #[inline]
    pub fn throw_parts(&self, h: TypeHandle) -> (TypeHandle, TypeHandle) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Throw {
                value_type,
                error_type,
            } => (*value_type, *error_type),
            _ => unreachable!(),
        }
    }
    /// The element type of a `Channel`.
    #[inline]
    pub fn channel_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Channel { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// The value type of an `Async`.
    #[inline]
    pub fn async_value(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Async { value } => *value,
            _ => unreachable!(),
        }
    }
    /// The value type of a `Lazy`.
    #[inline]
    pub fn lazy_value(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Lazy { value } => *value,
            _ => unreachable!(),
        }
    }
    /// The element type of an `Atomic`.
    #[inline]
    pub fn atomic_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Atomic { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// The element type of a `Sender`.
    #[inline]
    pub fn sender_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Sender { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// The element type of a `Receiver`.
    #[inline]
    pub fn receiver_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Receiver { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// The `(elem, size)` of an `Array`.
    #[inline]
    pub fn array_parts(&self, h: TypeHandle) -> (TypeHandle, Option<u64>) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Array { elem, size } => (*elem, *size),
            _ => unreachable!(),
        }
    }
    /// The `(inner, is_raw)` of a `Ref`.
    #[inline]
    pub fn ref_parts(&self, h: TypeHandle) -> (TypeHandle, bool) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Ref { inner, is_raw } => (*inner, *is_raw),
            _ => unreachable!(),
        }
    }
    /// The parameter slice and return type of an `Fn`.
    #[inline]
    pub fn fn_parts(&self, h: TypeHandle) -> (&[TypeHandle], TypeHandle) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Fn { params, return_type } => (params.as_ref(), *return_type),
            _ => unreachable!(),
        }
    }
    /// The inner type of a `Nullable`.
    #[inline]
    pub fn nullable_inner(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Nullable { inner } => *inner,
            _ => unreachable!(),
        }
    }
    /// The `(name, type_args)` of an `Adt`.
    #[inline]
    pub fn adt_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Adt { name, type_args } => (name.as_ref(), type_args.as_ref()),
            _ => unreachable!(),
        }
    }
    /// The `fields` slice of a `Record`.
    #[inline]
    pub fn record_fields(&self, h: TypeHandle) -> &[FieldType] {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Record { fields, .. } => fields.as_ref(),
            _ => unreachable!(),
        }
    }
    /// The `name` of a `Record`.
    #[inline]
    pub fn record_name(&self, h: TypeHandle) -> Option<&str> {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Record { name, .. } => name.as_deref(),
            _ => unreachable!(),
        }
    }
    /// The `(name, type_args)` of a `Trait`.
    #[inline]
    pub fn trait_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Trait { name, type_args } => (name.as_ref(), type_args.as_ref()),
            _ => unreachable!(),
        }
    }
    /// The `(trait_name, method_sigs)` of a `TraitObject`.
    #[inline]
    pub fn trait_object_parts(&self, h: TypeHandle) -> (&str, &[TraitMethodSig]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::TraitObject {
                trait_name,
                method_sigs,
            } => (trait_name.as_ref(), method_sigs.as_ref()),
            _ => unreachable!(),
        }
    }
    /// The `(path, env)` of a `ModuleRef`.
    #[inline]
    pub fn module_ref_parts(&self, h: TypeHandle) -> (&str, EnvId) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::ModuleRef { path, env } => (path.as_ref(), *env),
            _ => unreachable!(),
        }
    }
    /// The `(name, args)` of a `Generic`.
    #[inline]
    pub fn generic_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Generic { name, args } => (name.as_ref(), args.as_ref()),
            _ => unreachable!(),
        }
    }

    /// Visits every direct child `TypeHandle` of `ty` via `f`, in a fixed canonical order.
    ///
    /// This is the single source of truth for "which sub-types does a type contain":
    /// traversal-style walkers (`collect_type_vars`, `collect_free_vars`,
    /// `resolve_has_type_var`) delegate here so that adding a new composite variant only
    /// requires updating this one match instead of every walker. The set of children
    /// visited matches [`TypeArena::occurs`] / [`TypeArena::unify`] / `types_equal`.
    ///
    /// `TypeVar` and leaf types (scalars, `Str`/`Null`/`Void`/`Never`/`Unknown`,
    /// `TraitObject`, `ModuleRef`, `Timer`) carry no child `TypeHandle`s and visit nothing.
    pub fn for_each_child<F: FnMut(TypeHandle)>(&self, h: TypeHandle, mut f: F) {
        let resolved = self.resolve(h);
        match self.get(resolved) {
            Type::Fn(_) => {
                let (params, return_type) = self.fn_parts(resolved);
                for &p in params {
                    f(p);
                }
                f(return_type);
            }
            Type::Record(_) => {
                for field in self.record_fields(resolved) {
                    f(field.ty);
                }
            }
            Type::Adt(_) => {
                let (_, args) = self.adt_parts(resolved);
                for &a in args {
                    f(a);
                }
            }
            Type::Generic(_) => {
                let (_, args) = self.generic_parts(resolved);
                for &a in args {
                    f(a);
                }
            }
            Type::Trait(_) => {
                let (_, args) = self.trait_parts(resolved);
                for &a in args {
                    f(a);
                }
            }
            Type::Nullable(_) => f(self.nullable_inner(resolved)),
            Type::Ref(_) => f(self.ref_parts(resolved).0),
            Type::Array(_) => f(self.array_parts(resolved).0),
            Type::Throw(_) => {
                let (value_type, error_type) = self.throw_parts(resolved);
                f(value_type);
                f(error_type);
            }
            Type::Channel(_) => f(self.channel_elem(resolved)),
            Type::Async(_) => f(self.async_value(resolved)),
            Type::Lazy(_) => f(self.lazy_value(resolved)),
            Type::Atomic(_) => f(self.atomic_elem(resolved)),
            Type::Sender(_) => f(self.sender_elem(resolved)),
            Type::Receiver(_) => f(self.receiver_elem(resolved)),
            // TypeVar, scalars, Str/Null/Void/Never/Unknown, TraitObject, ModuleRef, Timer: no children.
            _ => {}
        }
    }

    // -- Type variable construction --

    pub fn fresh_type_var(&mut self) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new(false));
        self.make(Type::TypeVar(idx))
    }
    pub fn fresh_type_var_with_kind(&mut self, kind: SemKind) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new_with_kind(false, kind));
        self.make(Type::TypeVar(idx))
    }
    pub fn fresh_rigid_var(&mut self) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new(true));
        self.make(Type::TypeVar(idx))
    }
    pub fn fresh_rigid_var_with_kind(&mut self, kind: SemKind) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new_with_kind(true, kind));
        self.make(Type::TypeVar(idx))
    }
    pub fn fresh_kind_var(&mut self) -> SemKind {
        let idx = self.kind_vars.len() as u32;
        self.kind_vars.push(None);
        SemKind::Var(idx)
    }

    #[inline]
    pub fn type_var_kind(&self, idx: u32) -> &SemKind {
        &self.type_vars[idx as usize].kind
    }

    /// Compute the kind of an arbitrary `TypeHandle`.
    pub fn kind_of(&self, ty: TypeHandle) -> SemKind {
        match self.get(ty) {
            Type::TypeVar(idx) => self.type_vars[idx as usize].kind.clone(),
            _ => SemKind::Star,
        }
    }

    // -- resolve / occurs / unify / kind unification --
    // (The original sema/Sema.rs implementation was migrated wholesale;
    // ConcreteType -> Type, and structural access now uses accessor APIs.)

    /// Resolve a kind variable to its binding (analogous to type resolution).
    pub fn resolve_kind(&self, mut kind: SemKind) -> SemKind {
        while let SemKind::Var(idx) = &kind {
            if let Some(Some(bound)) = self.kind_vars.get(*idx as usize) {
                kind = bound.clone();
            } else {
                break;
            }
        }
        kind
    }

    /// Kind unification: attempt to unify two kinds; on success, binds the kind variable.
    pub fn unify_kind(&mut self, k1: &SemKind, k2: &SemKind) -> Result<(), ()> {
        let r1 = self.resolve_kind(k1.clone());
        let r2 = self.resolve_kind(k2.clone());

        if r1 == r2 {
            return Ok(());
        }

        match (&r1, &r2) {
            (SemKind::Var(idx), _) => {
                self.kind_vars[*idx as usize] = Some(r2);
                Ok(())
            }
            (_, SemKind::Var(idx)) => {
                self.kind_vars[*idx as usize] = Some(r1);
                Ok(())
            }
            (
                SemKind::Arrow { param: p1, result: r1 },
                SemKind::Arrow { param: p2, result: r2 },
            ) => {
                self.unify_kind(p1, p2)?;
                self.unify_kind(r1, r2)
            }
            _ => Err(()),
        }
    }

    /// Check kind consistency of a type application.
    pub fn check_kind_application(
        &mut self,
        constructor_kind: &SemKind,
        arg_kinds: &[SemKind],
    ) -> Result<SemKind, String> {
        let resolved_ck = self.resolve_kind(constructor_kind.clone());

        if arg_kinds.is_empty() {
            return Ok(resolved_ck);
        }

        match &resolved_ck {
            SemKind::Star => Err(format!(
                "kind mismatch: type of kind '*' cannot be applied to {} type argument(s)",
                arg_kinds.len()
            )),
            SemKind::Var(_) => {
                let mut expected_kind = SemKind::Star;
                for arg_kind in arg_kinds.iter().rev() {
                    expected_kind = SemKind::Arrow {
                        param: Box::new(arg_kind.clone()),
                        result: Box::new(expected_kind),
                    };
                }
                self.unify_kind(&resolved_ck, &expected_kind)
                    .map(|_| SemKind::Star)
                    .map_err(|_| {
                        format!(
                            "kind mismatch: cannot infer kind for type constructor with {} argument(s)",
                            arg_kinds.len()
                        )
                    })
            }
            SemKind::Arrow { param, result } => {
                let arg_kind_resolved = self.resolve_kind(arg_kinds[0].clone());
                let param_resolved = self.resolve_kind((**param).clone());
                if param_resolved != arg_kind_resolved {
                    if self.unify_kind(&param_resolved, &arg_kind_resolved).is_err() {
                        return Err(format!(
                            "kind mismatch: expected argument of kind {:?}, found {:?}",
                            param_resolved, arg_kind_resolved
                        ));
                    }
                }
                self.check_kind_application(result, &arg_kinds[1..])
            }
        }
    }

    /// Get a reference to a type variable.
    #[inline]
    pub fn type_var(&self, idx: u32) -> &TypeVar {
        &self.type_vars[idx as usize]
    }

    /// Resolve a `TypeHandle` to a non-`TypeVar` or its bound target.
    pub fn resolve(&self, h: TypeHandle) -> TypeHandle {
        let mut cur = h;
        loop {
            match self.get(cur) {
                Type::TypeVar(idx) => {
                    if let Some(bound) = self.type_vars[idx as usize].bound {
                        if bound != cur {
                            cur = bound;
                            continue;
                        }
                    }
                    return cur;
                }
                _ => return cur,
            }
        }
    }

    /// Resolve with path compression: flattens the TypeVar chain so subsequent
    /// resolves are O(1). Use in mutable contexts (unify, constraint solving)
    /// for amortized near-constant-time resolution.
    pub fn resolve_mut(&mut self, h: TypeHandle) -> TypeHandle {
        // First pass: find the root (immutable borrow).
        let root = self.resolve(h);
        if root == h {
            return root;
        }
        // Second pass: compress — point each intermediate var directly at root.
        let mut cur = h;
        loop {
            match self.get(cur) {
                Type::TypeVar(idx) => {
                    let bound = self.type_vars[idx as usize].bound;
                    match bound {
                        Some(b) if b != root && b != cur => {
                            self.type_vars[idx as usize].bound = Some(root);
                            cur = b;
                            continue;
                        }
                        _ => break,
                    }
                }
                _ => break,
            }
        }
        root
    }

    /// Occurs check: whether type variable `var_idx` occurs within `ty` (prevents infinite types).
    pub fn occurs(&self, var_idx: u32, ty: TypeHandle) -> bool {
        let t = self.get(ty);
        match t {
            Type::TypeVar(idx) => idx == var_idx,
            Type::Fn(_) => {
                let (params, return_type) = self.fn_parts(ty);
                params.iter().any(|&p| self.occurs(var_idx, p))
                    || self.occurs(var_idx, return_type)
            }
            Type::Record(_) => self
                .record_fields(ty)
                .iter()
                .any(|f| self.occurs(var_idx, f.ty)),
            Type::Nullable(_) => self.occurs(var_idx, self.nullable_inner(ty)),
            Type::Ref(_) => self.occurs(var_idx, self.ref_parts(ty).0),
            Type::Adt(_) => {
                let (_, type_args) = self.adt_parts(ty);
                type_args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.throw_parts(ty);
                self.occurs(var_idx, value_type) || self.occurs(var_idx, error_type)
            }
            Type::Channel(_) => self.occurs(var_idx, self.channel_elem(ty)),
            Type::Async(_) => self.occurs(var_idx, self.async_value(ty)),
            Type::Lazy(_) => self.occurs(var_idx, self.lazy_value(ty)),
            Type::Atomic(_) => self.occurs(var_idx, self.atomic_elem(ty)),
            Type::Sender(_) => self.occurs(var_idx, self.sender_elem(ty)),
            Type::Receiver(_) => self.occurs(var_idx, self.receiver_elem(ty)),
            Type::Generic(_) => {
                let (_, args) = self.generic_parts(ty);
                args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Type::Trait(_) => {
                let (_, type_args) = self.trait_parts(ty);
                type_args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Type::Array(_) => self.occurs(var_idx, self.array_parts(ty).0),
            Type::TraitObject(_) => false,
            _ => false,
        }
    }

    /// Unify two types (in-place mutation of `type_var.bound` or overwriting `never`/`unknown` slots).
    pub fn unify(&mut self, t1: TypeHandle, t2: TypeHandle) -> Result<(), UnifyError> {
        let a = self.resolve_mut(t1);
        let b = self.resolve_mut(t2);
        if a == b {
            return Ok(());
        }

        let a_ty = self.get(a);
        let b_ty = self.get(b);

        // -- type_var binding (a side) --
        if let Type::TypeVar(idx) = a_ty {
            let is_rigid = self.type_vars[idx as usize].is_rigid;
            if is_rigid {
                if let Type::TypeVar(bidx) = b_ty {
                    if bidx == idx {
                        return Ok(());
                    }
                    // b side is a non-rigid var: bind b to rigid a
                    if !self.type_vars[bidx as usize].is_rigid {
                        if self.occurs(bidx, a) {
                            return Err(UnifyError::OccursCheckFailed);
                        }
                        let b_kind = self.type_vars[bidx as usize].kind.clone();
                        let a_kind = self.kind_of(a);
                        if self.unify_kind(&b_kind, &a_kind).is_err() {
                            return Err(UnifyError::TypeMismatch);
                        }
                        self.type_vars[bidx as usize].bound = Some(a);
                        return Ok(());
                    }
                }
                return Err(UnifyError::TypeMismatch);
            }
            if self.occurs(idx, b) {
                return Err(UnifyError::OccursCheckFailed);
            }
            let var_kind = self.type_vars[idx as usize].kind.clone();
            let target_kind = self.kind_of(b);
            if self.unify_kind(&var_kind, &target_kind).is_err() {
                return Err(UnifyError::TypeMismatch);
            }
            self.type_vars[idx as usize].bound = Some(b);
            return Ok(());
        }

        // -- type_var binding (b side) --
        if let Type::TypeVar(idx) = b_ty {
            let is_rigid = self.type_vars[idx as usize].is_rigid;
            if is_rigid {
                return Err(UnifyError::TypeMismatch);
            }
            if self.occurs(idx, a) {
                return Err(UnifyError::OccursCheckFailed);
            }
            let var_kind = self.type_vars[idx as usize].kind.clone();
            let target_kind = self.kind_of(a);
            if self.unify_kind(&var_kind, &target_kind).is_err() {
                return Err(UnifyError::TypeMismatch);
            }
            self.type_vars[idx as usize].bound = Some(a);
            return Ok(());
        }

        // -- never / unknown unify with any type as the other side (in-place overwrite of the original slot) --
        match a_ty {
            Type::Never | Type::Unknown => {
                self.types[t1.0 as usize] = b_ty;
                return Ok(());
            }
            _ => {}
        }
        match b_ty {
            Type::Never | Type::Unknown => {
                self.types[t2.0 as usize] = a_ty;
                return Ok(());
            }
            _ => {}
        }

        // -- Structural unification --
        // Type is Copy, so no clone is needed; composite type fields are obtained via accessors.
        // For fields requiring recursive unify, TypeHandle pairs are first collected into a
        // Vec to avoid borrow conflicts.
        match (a_ty, b_ty) {
            // Payload-less unit variants: `==` is sufficient (scalars + Str/Null/Void +
            // Never/Unknown). DetailId-carrying variants are excluded by is_atomic_unit.
            (a_ty, b_ty) if a_ty.is_atomic_unit() && a_ty == b_ty => Ok(()),

            (Type::Fn(_), Type::Fn(_)) => {
                let (pa, ra) = self.fn_parts(a);
                let (pb, rb) = self.fn_parts(b);
                if pa.len() != pb.len() {
                    return Err(UnifyError::TypeMismatch);
                }
                let pairs: Vec<(TypeHandle, TypeHandle)> =
                    pa.iter().copied().zip(pb.iter().copied()).collect();
                for (x, y) in pairs {
                    self.unify(x, y)?;
                }
                self.unify(ra, rb)
            }

            (Type::Record(_), Type::Record(_)) => {
                let fa = self.record_fields(a);
                let fb = self.record_fields(b);
                if fa.len() != fb.len() {
                    return Err(UnifyError::TypeMismatch);
                }
                let pairs: Vec<(TypeHandle, TypeHandle)> =
                    fa.iter().zip(fb.iter()).map(|(x, y)| (x.ty, y.ty)).collect();
                for (x, y) in pairs {
                    self.unify(x, y)?;
                }
                Ok(())
            }

            (Type::Nullable(_), Type::Nullable(_)) => {
                self.unify(self.nullable_inner(a), self.nullable_inner(b))
            }

            (Type::Ref(_), Type::Ref(_)) => {
                let (ia, ra) = self.ref_parts(a);
                let (ib, rb) = self.ref_parts(b);
                if ra != rb {
                    return Err(UnifyError::TypeMismatch);
                }
                self.unify(ia, ib)
            }

            (Type::Adt(_), Type::Adt(_)) => self.unify_named_args(a, b, Self::adt_parts),

            (Type::Generic(_), Type::Generic(_)) => self.unify_named_args(a, b, Self::generic_parts),

            (Type::Trait(_), Type::Trait(_)) => self.unify_named_args(a, b, Self::trait_parts),

            (Type::TraitObject(_), Type::TraitObject(_)) => {
                let (na, ma) = self.trait_object_parts(a);
                let (nb, mb) = self.trait_object_parts(b);
                if na != nb || ma.len() != mb.len() {
                    return Err(UnifyError::TypeMismatch);
                }
                for (x, y) in ma.iter().zip(mb.iter()) {
                    if x != y {
                        return Err(UnifyError::TypeMismatch);
                    }
                }
                Ok(())
            }

            // A Trait type and a TraitObject with the same name can unify.
            (Type::Trait(_), Type::TraitObject(_)) => {
                let na = self.trait_parts(a).0;
                let nb = self.trait_object_parts(b).0;
                if na == nb {
                    Ok(())
                } else {
                    Err(UnifyError::TypeMismatch)
                }
            }
            (Type::TraitObject(_), Type::Trait(_)) => {
                let na = self.trait_object_parts(a).0;
                let nb = self.trait_parts(b).0;
                if na == nb {
                    Ok(())
                } else {
                    Err(UnifyError::TypeMismatch)
                }
            }

            (Type::Array(_), Type::Array(_)) => {
                let ea = self.array_parts(a).0;
                let eb = self.array_parts(b).0;
                self.unify(ea, eb)
            }

            (Type::Throw(_), Type::Throw(_)) => {
                let (va, ea) = self.throw_parts(a);
                let (vb, eb) = self.throw_parts(b);
                self.unify(va, vb)?;
                self.unify(ea, eb)
            }

            // Single-parameter builtin generics: recursively unify the element/value type.
            (Type::Channel(_), Type::Channel(_)) => {
                self.unify(self.channel_elem(a), self.channel_elem(b))
            }
            (Type::Async(_), Type::Async(_)) => {
                self.unify(self.async_value(a), self.async_value(b))
            }
            (Type::Lazy(_), Type::Lazy(_)) => {
                self.unify(self.lazy_value(a), self.lazy_value(b))
            }
            (Type::Atomic(_), Type::Atomic(_)) => {
                self.unify(self.atomic_elem(a), self.atomic_elem(b))
            }
            (Type::Sender(_), Type::Sender(_)) => {
                self.unify(self.sender_elem(a), self.sender_elem(b))
            }
            (Type::Receiver(_), Type::Receiver(_)) => {
                self.unify(self.receiver_elem(a), self.receiver_elem(b))
            }

            _ => Err(UnifyError::TypeMismatch),
        }
    }

    /// Unify two named-and-arg'd types (Adt / Generic / Trait) via a shared accessor.
    ///
    /// `parts_fn` extracts the `(name, type_args)` pair from each side; names and arg
    /// counts must match, then args are unified pairwise. The `pairs` vec is collected
    /// up-front so the `&mut self` borrows in the recursive `unify` calls do not alias
    /// the `&self` borrows held by the accessor.
    fn unify_named_args(
        &mut self,
        a: TypeHandle,
        b: TypeHandle,
        parts_fn: impl Fn(&TypeArena, TypeHandle) -> (&str, &[TypeHandle]),
    ) -> Result<(), UnifyError> {
        let (na, ta) = parts_fn(self, a);
        let (nb, tb) = parts_fn(self, b);
        if na != nb || ta.len() != tb.len() {
            return Err(UnifyError::TypeMismatch);
        }
        let pairs: Vec<(TypeHandle, TypeHandle)> =
            ta.iter().copied().zip(tb.iter().copied()).collect();
        for (x, y) in pairs {
            self.unify(x, y)?;
        }
        Ok(())
    }

    /// Extract the type name (used for `ExprInfo.type_name`).
    /// Scalars return their static name; adt/generic/trait return their name;
    /// ref/nullable recursively take the inner name; all others return `None`.
    /// Recursive cases require arena access to child nodes.
    pub fn type_name(&self, ty: TypeHandle) -> Option<&str> {
        let t = self.get(ty);
        // Builtin scalars + Str/Null/Void + builtin generic names: return the static name.
        if t.is_scalar()
            || matches!(
                t,
                Type::Str | Type::Null | Type::Void | Type::Throw(_) | Type::Channel(_)
                    | Type::Async(_) | Type::Lazy(_) | Type::Atomic(_)
                    | Type::Sender(_) | Type::Receiver(_)
            )
        {
            return Some(t.name());
        }
        match t {
            Type::Adt(_) => Some(self.adt_parts(ty).0),
            Type::Generic(_) => Some(self.generic_parts(ty).0),
            Type::Trait(_) => Some(self.trait_parts(ty).0),
            Type::TraitObject(_) => Some(self.trait_object_parts(ty).0),
            Type::Ref(_) => self.type_name(self.ref_parts(ty).0),
            Type::Nullable(_) => self.type_name(self.nullable_inner(ty)),
            _ => None,
        }
    }

    /// Construct a `TypeDisplay` wrapper, usable as `format!("{}", arena.display(h))`.
    #[inline]
    pub fn display(&self, ty: TypeHandle) -> TypeDisplay<'_> {
        TypeDisplay { arena: self, ty }
    }

    // -- Builtin scalar name lookup (formerly from_scalar_name) --

    /// Construct a `Type` (builtin type) from a scalar type name; unknown names return `Unknown`.
    pub fn from_scalar_name(&mut self, name: &str) -> TypeHandle {
        match Type::from_type_name(name) {
            Some(ty) => self.make(ty),
            None => self.make(Type::Unknown),
        }
    }
}
