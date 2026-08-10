//! Rel.rs — 类型关系判定层
//!
//! 从 Sema.rs 拆分。依赖 crate::Sema（类型系统基础）。
//! 职责：类型等价、子类型关系、数值提升、peer type 解析、模块结构化匹配。

use crate::sema::Sema::*;
use crate::ast::Ast::{AstArena, MethodDecl, TypeNode, TypeRef as AstTypeRef};

// =========================================================================
// phase5: subtype_check — 子类型关系判定
//
// 对 `src/sema/subtype_check.zig` 的 Rust 移植。
// 判定 Kuzo 语言中各种类型之间的子类型关系：null/nullable、record 结构子类型、
// ADT 错误子类型、Throw 子类型、trait 结构化子类型。
// =========================================================================

/// 命名类型实参列表结构相等：名称匹配 + 长度匹配 + 逐元素递归相等。
/// 供 Adt/Generic/Trait 三种命名复合类型共用。
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

/// 递归判定两个类型是否结构相等（resolve 后比较 Ty 内容）。
///
/// 作为权威实现，`InferContext::types_structurally_equal` 委托本函数。
/// 供 `is_subtype` 等独立 checker 使用。
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
        // 单参数内置泛型：元素类型相等才相等
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
        // 标量单元变体（I32, Bool, Str, ...）、Never, Unknown, Null, Void —
        // discriminant 已匹配即相等
        _ => true,
    }
}

/// 子类型判定规则。每条规则匹配特定 `(sub, sup)` 形状组合，命中时返回
/// `Some(bool)`，未命中返回 `None` 交由下一条规则处理。
///
/// 统一 `is_subtype` 内部散落的 if-let 分派，新增子类型规则只需添加一个
/// impl SubtypeRule 的无字段 struct 并注册到 `SUBTYPE_RULES`。
trait SubtypeRule {
    fn check(&self, arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> Option<bool>;
}

/// 自反性：同一类型或结构相等。
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

/// `Null` 字面量可赋值给任意 `nullable`。
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

/// Record 结构化子类型：`sub_fields` 覆盖 `sup_fields` 全部字段且类型相容。
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

/// ADT 同名子类型：直接比较类型名。
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

/// 子类型规则链：按顺序尝试每条规则，首条命中（返回 `Some`）即定夺。
/// 顺序与原 if-let 链一致：自反 → null→nullable → nullable 内层 →
/// record → ADT 同名 → throw。`trait 结构化` 子类型需
/// `sema_result`，由调用方通过 `is_trait_structural_subtype` 判定。
const SUBTYPE_RULES: &[&dyn SubtypeRule] = &[
    &ReflexiveRule,
    &NullToNullableRule,
    &NullableInnerRule,
    &RecordSubtypeRule,
    &AdtNameRule,
    &ThrowSubtypeRule,
];

/// 判断 `sub` 是否为 `sup` 的子类型。
///
/// 通过 `SUBTYPE_RULES` 规则链分派：自反性、null→nullable、nullable 内层、
/// record 结构子类型、ADT 同名、Throw 子类型。任一规则命中即返回其结果；
/// 全部未命中返回 `false`。`trait 结构化` 子类型需
/// `sema_result`，由调用方通过 `is_trait_structural_subtype` 判定。
pub fn is_subtype(arena: &TypeArena, sub: TypeHandle, sup: TypeHandle) -> bool {
    for rule in SUBTYPE_RULES.iter() {
        if let Some(ok) = rule.check(arena, sub, sup) {
            return ok;
        }
    }
    false
}

/// 记录字段匹配核心：`provided` 字段集合是否覆盖 `required` 全部字段。
/// 按字段名匹配（位置字段 `name == None` 视为同名），匹配后调用 `check`
/// 校验字段类型兼容性。`check` 签名为 `|arena, provided_ty, required_ty|`。
///
/// 消除 `is_record_subtype` 与 `record_arg_satisfies` 的字段遍历重复。
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

/// 记录类型结构化子类型判定：`sub_fields` 是否覆盖 `sup_fields` 全部字段。
/// 按字段名匹配，递归校验字段类型满足子类型关系（宽度+深度子类型）。
pub fn is_record_subtype(
    arena: &TypeArena,
    sub_fields: &[FieldType],
    sup_fields: &[FieldType],
) -> bool {
    // sub_fields 提供，sup_fields 要求；sub 字段类型须为 sup 字段类型的子类型。
    match_record_fields(arena, sup_fields, sub_fields, is_subtype)
}

/// Throw 子类型判定：值类型与错误类型需同时满足子类型关系。
pub fn is_throw_subtype(
    arena: &TypeArena,
    sub_val: TypeHandle,
    sub_err: TypeHandle,
    sup_val: TypeHandle,
    sup_err: TypeHandle,
) -> bool {
    is_subtype(arena, sub_val, sup_val) && is_subtype(arena, sub_err, sup_err)
}

/// trait 结构化子类型判定：`sub_name` 的方法集合需覆盖 `super_name` 的全部方法名。
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

/// 判断实参记录 `arg` 是否满足形参记录 `param` 的字段要求。
/// 非记录类型视为满足；记录类型则递归校验每个形参字段都存在且类型满足。
pub fn record_arg_satisfies(arena: &TypeArena, param: TypeHandle, arg: TypeHandle) -> bool {
    let rp = arena.resolve(param);
    let ra = arena.resolve(arg);
    let param_ct = arena.get(rp);
    let arg_ct = arena.get(ra);
    match (param_ct, arg_ct) {
        (Ty::Record(_), Ty::Record(_)) => {
            let pf = arena.record_fields(rp);
            let af = arena.record_fields(ra);
            // pf 要求，af 提供；递归校验。
            match_record_fields(arena, pf, af, |a, prov, req| record_arg_satisfies(a, req, prov))
        }
        _ => true,
    }
}

// =========================================================================
// phase5: throw_check 辅助 — 数值宽化与返回类型统一
//
// 对 `src/sema/throw_check.zig` 中数值宽化辅助函数的 Rust 移植。
// InferContext 方法（unify_return_type / try_widen_unify / check_propagate /
// check_throw_stmt）在下方 InferContext impl 块中实现。
// =========================================================================

/// 返回整型的秩（用于宽化比较），同宽有符号/无符号共享同一秩，非整型返回 0。
#[inline]
pub fn int_type_rank(ty: &Ty) -> u8 {
    ty.int_rank().unwrap_or(0)
}

/// 返回浮点类型的秩（用于宽化比较），非浮点返回 0。
#[inline]
pub fn float_type_rank(ty: &Ty) -> u8 {
    ty.float_bit_width().map(|b| b as u8).unwrap_or(0)
}

/// 判断是否为有符号整型（自由函数版，与 Ty::is_signed_int 等价）。
#[inline]
pub fn is_signed_int_ct(ty: &Ty) -> bool {
    ty.is_signed_int()
}

/// 判断数值类型 `from` 是否可被隐式宽化为数值类型 `to`。
/// 覆盖整型之间、浮点之间以及整型到浮点的宽化规则。
pub fn can_coerce_numeric(arena: &TypeArena, to: TypeHandle, from: TypeHandle) -> bool {
    let to_ct = arena.get(arena.resolve(to));
    let from_ct = arena.get(arena.resolve(from));
    let to_int = int_type_rank(&to_ct);
    let from_int = int_type_rank(&from_ct);
    let to_float = float_type_rank(&to_ct);
    let from_float = float_type_rank(&from_ct);

    // 整型之间：同秩或目标秩更大时允许宽化
    if to_int > 0 && from_int > 0 {
        let to_signed = is_signed_int_ct(&to_ct);
        let from_signed = is_signed_int_ct(&from_ct);
        if to_int == from_int && to_signed == from_signed {
            return true;
        }
        if to_signed == from_signed {
            return to_int >= from_int;
        }
        // 有符号 → 无符号需目标秩严格更大以容纳符号位
        return to_int > from_int;
    }
    // 浮点之间：同秩或目标秩更大时允许宽化（禁止窄化）
    if to_float > 0 && from_float > 0 {
        return to_float >= from_float;
    }
    // 整型 → 浮点：允许
    if to_float > 0 && from_int > 0 {
        return true;
    }
    false
}

// =========================================================================
// phase5: kind_check — 类型种类检查
//
// 对 `src/sema/kind_check.zig` 的 Rust 移植。
// 校验类型注解中类型构造器的使用是否与其种类（arity）一致。
// =========================================================================

/// 返回类型名为 `name` 的类型构造器所期望的类型参数个数（种类 arity）。
/// 内置高阶类型使用固定 arity，自定义 ADT 取其声明的类型参数个数，其余裸类型名 arity 为 0。
pub fn arity_of_type_name(sema_result: &SemaResult, name: &str) -> usize {
    if let Some(arity) = generic_type_arity(name) {
        return arity as usize;
    }
    match sema_result.get_type_def(name) {
        Some(def) => def.type_params.len(),
        None => 0,
    }
}

/// 判断 `name` 是否为当前作用域内的类型参数。
fn is_type_param(name: &str, type_param_names: &[&str]) -> bool {
    type_param_names.contains(&name)
}

/// 递归检查类型节点树中每个类型构造器的使用是否符合其种类 arity。
/// `type_param_names` 给出当前作用域内合法的类型参数名（可作具体类型使用）。
/// 发现不匹配时将错误追加到 `errors`。
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

/// 计算类型节点剩余的种类 arity：即还差多少个类型参数才能成为具体类型。
/// 裸类型名返回其 arity；部分应用的高阶类型返回 (arity - 已提供参数数)。
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
// phase5: module_check — 模块结构检查
//
// 对 `src/sema/module_check.zig` 的 Rust 移植。
// 提供模块成员方法签名摘要，以及模块是否结构化满足某 trait 所需方法集合的判定。
// =========================================================================

/// 模块成员方法的签名摘要：方法名与参数个数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodSig {
    pub name: Box<str>,
    pub arity: usize,
}

/// 模块结构化匹配 trait 的失败原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchReason {
    /// 匹配成功
    Ok,
    /// 缺少方法
    Missing,
    /// 参数个数不符
    ArityMismatch,
}

/// 模块结构化匹配 trait 的结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    pub ok: bool,
    pub missing_method: Option<Box<str>>,
    pub arity_expected: usize,
    pub arity_got: usize,
    pub reason: MatchReason,
}

impl MatchResult {
    /// 匹配成功。
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

/// 模块检查器：判断一组提供的方法签名是否结构化满足一组必需的方法签名。
#[derive(Debug, Default)]
pub struct ModuleChecker;

impl ModuleChecker {
    pub fn new() -> Self {
        ModuleChecker
    }

    /// 逐个校验 `required` 中的方法是否在 `provided` 中存在且参数个数一致。
    /// 任一方法缺失或参数个数不符即返回带原因的失败结果。
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

    /// 从 trait 方法声明中收集无方法体（即需要被实现）的方法签名。
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
// sema v2: Peer Type Resolution — 统一类型统一入口
//
// 设计理念：
// 统一 literal_promotion + try_widen_unify + if 分支统一为单一 peer_type 入口。
// 给定 N 个类型，求最兼容的共同类型（join / least upper bound）。
//
// 规则（按优先级）：
// 1. 空列表 → Unknown
// 2. 单元素 → 该类型
// 3. 全相同 → 该类型
// 4. 含 Never → 过滤 Never 后递归（发散类型不贡献值）
// 5. 数值类型 → 取最宽（int→int 取宽、float 优先、int→float）
// 6. nullable 传播 → Nullable<peer(inner types)>
// 7. throw 传播 → Throw<peer(value types), peer(error types)>
// 8. ADT → 查 error_newtype 子类型关系
// 9. 无公共类型 → Unknown（记录错误）
// =========================================================================

/// 求多个类型的共同类型（join / least upper bound）。
///
/// 统一入口：替代分散的 literal_promotion / try_widen_unify / if 分支统一。
/// 返回能容纳所有输入类型的最兼容类型。
///
/// **规则**（按优先级）：
/// 1. 空列表 → `Unknown`
/// 2. 单元素 → 该类型
/// 3. 含 `Never` → 过滤后递归（发散路径不贡献值）
/// 4. 全相同（结构相等）→ 该类型
/// 5. 全为数值 → 取最宽（int→int 取位宽最大、float 优先于 int、int→float 宽化）
/// 6. 全为 nullable → `Nullable<peer(inners)>`
/// 7. 全为 throw → `Throw<peer(values), peer(errors)>`
/// 8. 含 nullable + 非 nullable → `Nullable<peer(all inners)>`
/// 9. ADT error 子类型 → 取超类型
/// 10. 无公共类型 → `Unknown`
pub fn peer_type(arena: &mut TypeArena, types: &[TypeHandle]) -> TypeHandle {
    if types.is_empty() {
        return arena.make(Ty::Unknown);
    }
    if types.len() == 1 {
        return types[0];
    }

    // 过滤 Never 和 Void（发散/无值类型不贡献值）
    // Never：diverging（return/throw/break/continue）
    // Void：无有意义的值（如 if-then 无 else 且 then 为语句）
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
        // 全是 Never/Void → 返回第一个原始类型（保留 Never 优先）
        if types
            .iter()
            .any(|&t| matches!(arena.get(arena.resolve(t)), Ty::Never))
        {
            return arena.make(Ty::Never);
        }
        return arena.make(Ty::Void);
    }
    if non_trivial.len() == 1 {
        return non_trivial[0];
    }

    // 全相同（结构相等）→ 返回第一个
    let first = non_trivial[0];
    if non_trivial[1..].iter().all(|&t| types_equal(arena, first, t)) {
        return first;
    }

    // 全为数值 → 取最宽
    let all_numeric = non_trivial.iter().all(|&t| {
        let ct = arena.get(arena.resolve(t));
        ct.is_int() || ct.is_float()
    });
    if all_numeric {
        return peer_numeric(arena, &non_trivial);
    }

    // 全为 nullable → Nullable<peer(inners)>
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

    // 含 nullable + 非 nullable → Nullable<peer(all inners)>
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

    // 全为 throw → Throw<peer(values), peer(errors)>
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

    // 不兼容 → Unknown
    arena.make(Ty::Unknown)
}

/// 二元运算的 peer type resolution（内化字面量提升规则）。
///
/// 统一替代 `literal_promotion`：字面量提升规则内化为此函数的一部分，
/// 消除 literal_promotion 与 peer_type 的双轨制。
///
/// **规则**（按优先级）：
/// 1. 一侧字面量、另一侧变量 → 返回变量类型（字面量提升到变量类型）
/// 2. 两侧都是字面量 → `peer_numeric` 取最宽
/// 3. 两侧都是变量 → `peer_type` 求共同类型（数值取最宽、nullable 传播等）
pub fn peer_type_binary(
    arena: &mut TypeArena,
    left: TypeHandle,
    right: TypeHandle,
    left_is_literal: bool,
    right_is_literal: bool,
) -> TypeHandle {
    // 规则 1：字面量提升到变量类型
    if left_is_literal && !right_is_literal {
        return arena.resolve(right);
    }
    if !left_is_literal && right_is_literal {
        return arena.resolve(left);
    }

    // 规则 2 & 3：两侧都是字面量或都是变量 → peer_type
    peer_type(arena, &[left, right])
}

/// 数值类型的 peer resolution：取最宽类型。
///
/// 规则：
/// 1. 含 float → 取最宽 float（位宽最大）
/// 2. 全 int → 取最宽 int（同符号取位宽最大；跨符号取无符号位宽最大）
fn peer_numeric(arena: &mut TypeArena, types: &[TypeHandle]) -> TypeHandle {
    let resolved: Vec<Ty> = types
        .iter()
        .map(|&t| arena.get(arena.resolve(t)))
        .collect();

    // 含 float → 取最宽 float
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

    // 全 int → 取最宽
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
