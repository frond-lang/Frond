// =========================================================================
// Arena — 类型分配器 + unify/occurs/resolve
// =========================================================================

use super::Tag::*;
use super::ty::*;
use super::Display::TypeDisplay;

/// Ty 分配器：arena-based，管理类型槽、结构详情、类型变量与 kind 变量。
///
/// 所有 Ty 通过 `make()` 分配返回 `TypeHandle`；复合类型通过 `make_*` 方法
/// 分配 `DetailId` 并存结构数据到 `details` 表。`unify`/`occurs`/`resolve`/
/// `kind_of` 等方法需访问 `details`，故为 `TypeArena` 方法。
pub struct TypeArena {
    /// 类型槽：Ty 枚举（Copy）按 TypeHandle 索引。
    pub types: Vec<Ty>,
    /// 结构详情表：复合类型的结构数据按 DetailId 索引。
    pub details: Vec<TypeDetail>,
    /// 类型变量表：TypeVar(u32) 载荷索引此表。
    pub type_vars: Vec<TypeVar>,
    /// kind 变量绑定表：SemKind::Var(idx) 索引此表。
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

    // ── 基础访问 ──

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
    pub fn get(&self, h: TypeHandle) -> Ty {
        self.types[h.0 as usize]
    }

    /// 分配一个 Ty，返回句柄。
    pub fn make(&mut self, ty: Ty) -> TypeHandle {
        let h = TypeHandle(self.types.len() as u32);
        self.types.push(ty);
        h
    }

    /// 分配一个 TypeDetail，返回 DetailId。
    fn make_detail(&mut self, detail: TypeDetail) -> DetailId {
        let id = DetailId(self.details.len() as u32);
        self.details.push(detail);
        id
    }

    /// 获取 detail 引用（has_detail() 为 false 的变体调用会 panic）。
    #[inline]
    pub fn detail(&self, id: DetailId) -> &TypeDetail {
        &self.details[id.0 as usize]
    }

    // ── 复合类型构造器（分配 detail + make Ty）──

    pub fn make_throw(&mut self, value: TypeHandle, error: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Throw {
            value_type: value,
            error_type: error,
        });
        self.make(Ty::Throw(id))
    }
    pub fn make_channel(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Channel { elem });
        self.make(Ty::Channel(id))
    }
    pub fn make_async(&mut self, value: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Async { value });
        self.make(Ty::Async(id))
    }
    pub fn make_lazy(&mut self, value: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Lazy { value });
        self.make(Ty::Lazy(id))
    }
    pub fn make_atomic(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Atomic { elem });
        self.make(Ty::Atomic(id))
    }
    pub fn make_sender(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Sender { elem });
        self.make(Ty::Sender(id))
    }
    pub fn make_receiver(&mut self, elem: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Receiver { elem });
        self.make(Ty::Receiver(id))
    }
    pub fn make_array(&mut self, elem: TypeHandle, size: Option<u64>) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Array { elem, size });
        self.make(Ty::Array(id))
    }
    pub fn make_ref(&mut self, inner: TypeHandle, is_raw: bool) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Ref { inner, is_raw });
        self.make(Ty::Ref(id))
    }
    pub fn make_fn(
        &mut self,
        params: Box<[TypeHandle]>,
        return_type: TypeHandle,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Fn { params, return_type });
        self.make(Ty::Fn(id))
    }
    pub fn make_nullable(&mut self, inner: TypeHandle) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Nullable { inner });
        self.make(Ty::Nullable(id))
    }
    pub fn make_adt(
        &mut self,
        name: Box<str>,
        type_args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Adt { name, type_args });
        self.make(Ty::Adt(id))
    }
    pub fn make_record(
        &mut self,
        fields: Box<[FieldType]>,
        name: Option<Box<str>>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Record { fields, name });
        self.make(Ty::Record(id))
    }
    pub fn make_trait(
        &mut self,
        name: Box<str>,
        type_args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Trait { name, type_args });
        self.make(Ty::Trait(id))
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
        self.make(Ty::TraitObject(id))
    }
    pub fn make_module_ref(&mut self, path: Box<str>, env: EnvId) -> TypeHandle {
        let id = self.make_detail(TypeDetail::ModuleRef { path, env });
        self.make(Ty::ModuleRef(id))
    }
    pub fn make_generic(
        &mut self,
        name: Box<str>,
        args: Box<[TypeHandle]>,
    ) -> TypeHandle {
        let id = self.make_detail(TypeDetail::Generic { name, args });
        self.make(Ty::Generic(id))
    }

    // ── 结构化访问器（替代 ConcreteType::Fn{params, return_type} 内联访问）──

    /// 从 Ty 取出 DetailId（has_detail 为 false 时 panic）。
    #[inline]
    pub fn detail_id_of(&self, ty: Ty) -> DetailId {
        match ty {
            Ty::Throw(id)
            | Ty::Channel(id)
            | Ty::Async(id)
            | Ty::Lazy(id)
            | Ty::Atomic(id)
            | Ty::Sender(id)
            | Ty::Receiver(id)
            | Ty::Array(id)
            | Ty::Ref(id)
            | Ty::Fn(id)
            | Ty::Nullable(id)
            | Ty::Adt(id)
            | Ty::Record(id)
            | Ty::Trait(id)
            | Ty::TraitObject(id)
            | Ty::ModuleRef(id)
            | Ty::Generic(id) => id,
            _ => panic!("Ty {:?} does not carry a DetailId", ty),
        }
    }

    /// Throw 的 (value_type, error_type)。
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
    /// Channel 元素类型。
    #[inline]
    pub fn channel_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Channel { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// Async 的值类型。
    #[inline]
    pub fn async_value(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Async { value } => *value,
            _ => unreachable!(),
        }
    }
    /// Lazy 的值类型。
    #[inline]
    pub fn lazy_value(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Lazy { value } => *value,
            _ => unreachable!(),
        }
    }
    /// Atomic 元素类型。
    #[inline]
    pub fn atomic_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Atomic { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// Sender 元素类型。
    #[inline]
    pub fn sender_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Sender { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// Receiver 元素类型。
    #[inline]
    pub fn receiver_elem(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Receiver { elem } => *elem,
            _ => unreachable!(),
        }
    }
    /// Array 的 (elem, size)。
    #[inline]
    pub fn array_parts(&self, h: TypeHandle) -> (TypeHandle, Option<u64>) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Array { elem, size } => (*elem, *size),
            _ => unreachable!(),
        }
    }
    /// Ref 的 (inner, is_raw)。
    #[inline]
    pub fn ref_parts(&self, h: TypeHandle) -> (TypeHandle, bool) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Ref { inner, is_raw } => (*inner, *is_raw),
            _ => unreachable!(),
        }
    }
    /// Fn 的参数切片与返回类型。
    #[inline]
    pub fn fn_parts(&self, h: TypeHandle) -> (&[TypeHandle], TypeHandle) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Fn { params, return_type } => (params.as_ref(), *return_type),
            _ => unreachable!(),
        }
    }
    /// Nullable 的内部类型。
    #[inline]
    pub fn nullable_inner(&self, h: TypeHandle) -> TypeHandle {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Nullable { inner } => *inner,
            _ => unreachable!(),
        }
    }
    /// Adt 的 (name, type_args)。
    #[inline]
    pub fn adt_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Adt { name, type_args } => (name.as_ref(), type_args.as_ref()),
            _ => unreachable!(),
        }
    }
    /// Record 的 fields 切片。
    #[inline]
    pub fn record_fields(&self, h: TypeHandle) -> &[FieldType] {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Record { fields, .. } => fields.as_ref(),
            _ => unreachable!(),
        }
    }
    /// Record 的 name。
    #[inline]
    pub fn record_name(&self, h: TypeHandle) -> Option<&str> {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Record { name, .. } => name.as_deref(),
            _ => unreachable!(),
        }
    }
    /// Trait 的 (name, type_args)。
    #[inline]
    pub fn trait_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Trait { name, type_args } => (name.as_ref(), type_args.as_ref()),
            _ => unreachable!(),
        }
    }
    /// TraitObject 的 (trait_name, method_sigs)。
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
    /// ModuleRef 的 (path, env)。
    #[inline]
    pub fn module_ref_parts(&self, h: TypeHandle) -> (&str, EnvId) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::ModuleRef { path, env } => (path.as_ref(), *env),
            _ => unreachable!(),
        }
    }
    /// Generic 的 (name, args)。
    #[inline]
    pub fn generic_parts(&self, h: TypeHandle) -> (&str, &[TypeHandle]) {
        match self.detail(self.detail_id_of(self.get(h))) {
            TypeDetail::Generic { name, args } => (name.as_ref(), args.as_ref()),
            _ => unreachable!(),
        }
    }

    // ── 类型变量构造 ──

    pub fn fresh_type_var(&mut self) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new(false));
        self.make(Ty::TypeVar(idx))
    }
    pub fn fresh_type_var_with_kind(&mut self, kind: SemKind) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new_with_kind(false, kind));
        self.make(Ty::TypeVar(idx))
    }
    pub fn fresh_rigid_var(&mut self) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new(true));
        self.make(Ty::TypeVar(idx))
    }
    pub fn fresh_rigid_var_with_kind(&mut self, kind: SemKind) -> TypeHandle {
        let idx = self.type_vars.len() as u32;
        self.type_vars.push(TypeVar::new_with_kind(true, kind));
        self.make(Ty::TypeVar(idx))
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

    /// 计算任意 TypeHandle 的 kind。
    pub fn kind_of(&self, ty: TypeHandle) -> SemKind {
        match self.get(ty) {
            Ty::TypeVar(idx) => self.type_vars[idx as usize].kind.clone(),
            _ => SemKind::Star,
        }
    }

    // ── resolve / occurs / unify / kind 统一 ──
    //（原 sema/Sema.rs 实现整体迁移，ConcreteType → Ty，结构访问改为访问器 API）

    /// 解析 kind 变量到其绑定值（类似 type resolve）。
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

    /// kind 统一：尝试统一两个 kind，成功则绑定 kind 变量。
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

    /// 检查类型应用（type application）的 kind 一致性。
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

    /// 获取类型变量引用。
    #[inline]
    pub fn type_var(&self, idx: u32) -> &TypeVar {
        &self.type_vars[idx as usize]
    }

    /// 解析 TypeHandle 到非 TypeVar 或已绑定的目标。
    pub fn resolve(&self, h: TypeHandle) -> TypeHandle {
        let mut cur = h;
        loop {
            match self.get(cur) {
                Ty::TypeVar(idx) => {
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

    /// occurs check：类型变量 `var_idx` 是否出现在 `ty` 中（防止无限类型）。
    pub fn occurs(&self, var_idx: u32, ty: TypeHandle) -> bool {
        let t = self.get(ty);
        match t {
            Ty::TypeVar(idx) => idx == var_idx,
            Ty::Fn(_) => {
                let (params, return_type) = self.fn_parts(ty);
                params.iter().any(|&p| self.occurs(var_idx, p))
                    || self.occurs(var_idx, return_type)
            }
            Ty::Record(_) => self
                .record_fields(ty)
                .iter()
                .any(|f| self.occurs(var_idx, f.ty)),
            Ty::Nullable(_) => self.occurs(var_idx, self.nullable_inner(ty)),
            Ty::Ref(_) => self.occurs(var_idx, self.ref_parts(ty).0),
            Ty::Adt(_) => {
                let (_, type_args) = self.adt_parts(ty);
                type_args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.throw_parts(ty);
                self.occurs(var_idx, value_type) || self.occurs(var_idx, error_type)
            }
            Ty::Channel(_) => self.occurs(var_idx, self.channel_elem(ty)),
            Ty::Async(_) => self.occurs(var_idx, self.async_value(ty)),
            Ty::Lazy(_) => self.occurs(var_idx, self.lazy_value(ty)),
            Ty::Atomic(_) => self.occurs(var_idx, self.atomic_elem(ty)),
            Ty::Sender(_) => self.occurs(var_idx, self.sender_elem(ty)),
            Ty::Receiver(_) => self.occurs(var_idx, self.receiver_elem(ty)),
            Ty::Generic(_) => {
                let (_, args) = self.generic_parts(ty);
                args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Ty::Trait(_) => {
                let (_, type_args) = self.trait_parts(ty);
                type_args.iter().any(|&a| self.occurs(var_idx, a))
            }
            Ty::Array(_) => self.occurs(var_idx, self.array_parts(ty).0),
            Ty::TraitObject(_) => false,
            _ => false,
        }
    }

    /// 统一两个类型（就地修改 `type_var.bound` 或覆写 `never`/`unknown` 槽位）。
    pub fn unify(&mut self, t1: TypeHandle, t2: TypeHandle) -> Result<(), UnifyError> {
        let a = self.resolve(t1);
        let b = self.resolve(t2);
        if a == b {
            return Ok(());
        }

        let a_ty = self.get(a);
        let b_ty = self.get(b);

        // ── type_var 绑定（a 侧）──
        if let Ty::TypeVar(idx) = a_ty {
            let is_rigid = self.type_vars[idx as usize].is_rigid;
            if is_rigid {
                if let Ty::TypeVar(bidx) = b_ty {
                    if bidx == idx {
                        return Ok(());
                    }
                    // b 侧是非 rigid var：把 b 绑定到 rigid a
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

        // ── type_var 绑定（b 侧）──
        if let Ty::TypeVar(idx) = b_ty {
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

        // ── never / unknown 与任意类型统一为对方（就地覆写原槽位）──
        match a_ty {
            Ty::Never | Ty::Unknown => {
                self.types[t1.0 as usize] = b_ty;
                return Ok(());
            }
            _ => {}
        }
        match b_ty {
            Ty::Never | Ty::Unknown => {
                self.types[t2.0 as usize] = a_ty;
                return Ok(());
            }
            _ => {}
        }

        // ── 结构化统一 ──
        // Ty is Copy，无需 clone；复合类型字段通过访问器获取。
        // 对于需要递归 unify 的字段，先收集 TypeHandle 对到 Vec 以避免借用冲突。
        match (a_ty, b_ty) {
            (Ty::I8, Ty::I8)
            | (Ty::I16, Ty::I16)
            | (Ty::I32, Ty::I32)
            | (Ty::I64, Ty::I64)
            | (Ty::I128, Ty::I128)
            | (Ty::U8, Ty::U8)
            | (Ty::U16, Ty::U16)
            | (Ty::U32, Ty::U32)
            | (Ty::U64, Ty::U64)
            | (Ty::U128, Ty::U128)
            | (Ty::Isize, Ty::Isize)
            | (Ty::Usize, Ty::Usize)
            | (Ty::F16, Ty::F16)
            | (Ty::F32, Ty::F32)
            | (Ty::F64, Ty::F64)
            | (Ty::F128, Ty::F128)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Str, Ty::Str)
            | (Ty::Char, Ty::Char)
            | (Ty::Null, Ty::Null)
            | (Ty::Void, Ty::Void) => Ok(()),

            (Ty::Fn(_), Ty::Fn(_)) => {
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

            (Ty::Record(_), Ty::Record(_)) => {
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

            (Ty::Nullable(_), Ty::Nullable(_)) => {
                self.unify(self.nullable_inner(a), self.nullable_inner(b))
            }

            (Ty::Ref(_), Ty::Ref(_)) => {
                let (ia, ra) = self.ref_parts(a);
                let (ib, rb) = self.ref_parts(b);
                if ra != rb {
                    return Err(UnifyError::TypeMismatch);
                }
                self.unify(ia, ib)
            }

            (Ty::Adt(_), Ty::Adt(_)) => {
                let (na, ta) = self.adt_parts(a);
                let (nb, tb) = self.adt_parts(b);
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

            (Ty::Generic(_), Ty::Generic(_)) => {
                let (na, ta) = self.generic_parts(a);
                let (nb, tb) = self.generic_parts(b);
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

            (Ty::Trait(_), Ty::Trait(_)) => {
                let (na, ta) = self.trait_parts(a);
                let (nb, tb) = self.trait_parts(b);
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

            (Ty::TraitObject(_), Ty::TraitObject(_)) => {
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

            // Trait 类型与 TraitObject（同名）可统一
            (Ty::Trait(_), Ty::TraitObject(_)) => {
                let na = self.trait_parts(a).0;
                let nb = self.trait_object_parts(b).0;
                if na == nb {
                    Ok(())
                } else {
                    Err(UnifyError::TypeMismatch)
                }
            }
            (Ty::TraitObject(_), Ty::Trait(_)) => {
                let na = self.trait_object_parts(a).0;
                let nb = self.trait_parts(b).0;
                if na == nb {
                    Ok(())
                } else {
                    Err(UnifyError::TypeMismatch)
                }
            }

            (Ty::Array(_), Ty::Array(_)) => {
                let ea = self.array_parts(a).0;
                let eb = self.array_parts(b).0;
                self.unify(ea, eb)
            }

            (Ty::Throw(_), Ty::Throw(_)) => {
                let (va, ea) = self.throw_parts(a);
                let (vb, eb) = self.throw_parts(b);
                self.unify(va, vb)?;
                self.unify(ea, eb)
            }

            // 单参数内置泛型：递归统一元素/值类型
            (Ty::Channel(_), Ty::Channel(_)) => {
                self.unify(self.channel_elem(a), self.channel_elem(b))
            }
            (Ty::Async(_), Ty::Async(_)) => {
                self.unify(self.async_value(a), self.async_value(b))
            }
            (Ty::Lazy(_), Ty::Lazy(_)) => {
                self.unify(self.lazy_value(a), self.lazy_value(b))
            }
            (Ty::Atomic(_), Ty::Atomic(_)) => {
                self.unify(self.atomic_elem(a), self.atomic_elem(b))
            }
            (Ty::Sender(_), Ty::Sender(_)) => {
                self.unify(self.sender_elem(a), self.sender_elem(b))
            }
            (Ty::Receiver(_), Ty::Receiver(_)) => {
                self.unify(self.receiver_elem(a), self.receiver_elem(b))
            }

            _ => Err(UnifyError::TypeMismatch),
        }
    }

    /// 提取类型名（用于 `ExprInfo.type_name`）。
    /// 标量返回静态名；adt/generic/trait 返回其名；ref/nullable 递归取 inner 名；
    /// 其余返回 `None`。递归场景需 arena 访问子节点。
    pub fn type_name(&self, ty: TypeHandle) -> Option<&str> {
        let t = self.get(ty);
        // 内置标量 + Str/Null/Void + 内置泛型名：返回静态名
        if t.is_scalar()
            || matches!(
                t,
                Ty::Str | Ty::Null | Ty::Void | Ty::Throw(_) | Ty::Channel(_)
                    | Ty::Async(_) | Ty::Lazy(_) | Ty::Atomic(_)
                    | Ty::Sender(_) | Ty::Receiver(_)
            )
        {
            return Some(t.name());
        }
        match t {
            Ty::Adt(_) => Some(self.adt_parts(ty).0),
            Ty::Generic(_) => Some(self.generic_parts(ty).0),
            Ty::Trait(_) => Some(self.trait_parts(ty).0),
            Ty::TraitObject(_) => Some(self.trait_object_parts(ty).0),
            Ty::Ref(_) => self.type_name(self.ref_parts(ty).0),
            Ty::Nullable(_) => self.type_name(self.nullable_inner(ty)),
            _ => None,
        }
    }

    /// 构造一个 `TypeDisplay` 包装器，可用于 `format!("{}", arena.display(h))`。
    #[inline]
    pub fn display(&self, ty: TypeHandle) -> TypeDisplay<'_> {
        TypeDisplay { arena: self, ty }
    }

    // ── 内置标量名查找（原 from_scalar_name）──

    /// 从标量类型名反向构造 Ty（内置类型），未知名返回 Unknown。
    pub fn from_scalar_name(&mut self, name: &str) -> TypeHandle {
        match Ty::from_type_name(name) {
            Some(ty) => self.make(ty),
            None => self.make(Ty::Unknown),
        }
    }
}
