#![allow(dead_code)]

use crate::sema::Sema::*;
use crate::ast::Ast::{
    AstArena, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module,
    Param, SelectArm, Spanned, Stmt, StmtId,
    TypeNode, TypeParam, TypeRef as AstTypeRef,
};
use rustc_hash::FxHashMap;

/// 单态化递归深度上限：防止极深泛型调用链导致栈溢出。
/// in_progress.len() 即当前递归深度，达到上限时停止递归。
const MAX_MONOMORPH_DEPTH: usize = 256;

// ====== 以下是从 Inference.rs（原 SemaInfer.rs）740-2408 行原样迁移的代码 ======
// =========================================================================
// monomorph — 单态化实例化
//
// v3 spec §5.2: 迁自 src/sema/monomorph.zig。
// 职责：识别所有泛型调用点 → 推导 type_args → 去重 → 确定实例集合。
//
// 适配 Rust：
// - 表达式 key 由裸指针 `@intFromPtr` 改为 `ExprId.0 as u64`（AstArena 索引）
// - 类型解析委托 `resolve_type_node_resolved`（接收 `Option<AstTypeRef>` 而非 `*TypeNode`）
// - 借用分离：`WalkCtx` 不持有 `sema_result`，通过独立字段参数传递，避免
//   `&mut SemaResult` 与 `&mut WalkCtx` 循环借用；`instance` 作为栈上局部变量
//   在 `push` 前完成体解析，与 `sema_result` 无别名
// - `field_access` 元信息按 `TypeDefKind` 区分 Record（field_id 从 0）与
//   ADT/Newtype（field_id 从 1，`__tag=0`），修正 Zig 版 Record 索引偏移
// =========================================================================

/// Compute a stable identity hash for a TypeHandle (for monomorph cache keys).
/// Builtins hash by canonical type_id; user types hash by name; composites by family name.
fn type_identity_hash(arena: &TypeArena, h: TypeHandle) -> u64 {
    let resolved = arena.resolve(h);
    let ty = arena.get(resolved);
    // Builtins: use canonical type_id
    if let Some(tid) = ty.type_id() {
        return tid as u64;
    }
    // User types: use name as identity
    let name: &str = match &ty {
        Ty::Adt(_) => arena.adt_parts(resolved).0,
        Ty::Generic(_) => arena.generic_parts(resolved).0,
        Ty::Trait(_) => arena.trait_parts(resolved).0,
        Ty::TraitObject(_) => arena.trait_object_parts(resolved).0,
        _ => ty.name(),
    };
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a 64-bit 哈希（迁自 monomorph.zig:hashTypeArgs）。
/// 输入为 `TypeHandle` 列表，通过 `type_identity_hash` 派生稳定标识。
pub fn hash_type_args(arena: &TypeArena, type_args: &[TypeHandle]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &ta in type_args {
        h ^= type_identity_hash(arena, ta);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 构造单态化缓存键：func_name 与 type_args 的 FNV-1a 组合 u64 哈希（无 String 分配）。
pub fn build_cache_key(func_name: &str, arena: &TypeArena, type_args: &[TypeHandle]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in func_name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= hash_type_args(arena, type_args);
    h.wrapping_mul(0x100000001b3)
}

/// 查找已有单态化实例（仅查询缓存，不创建）。
pub fn find_instance(
    arena: &TypeArena,
    sema_result: &SemaResult,
    func_name: &str,
    type_args: &[TypeHandle],
) -> Option<u32> {
    let cache_key = build_cache_key(func_name, arena, type_args);
    sema_result.monomorph_index.get(&cache_key).copied()
}

/// Compute whether a TypeHandle corresponds to a reference type (str or user-defined).
/// Mirrors `Ty::is_ref()` semantics: str or dynamic type, excluding nullable.
fn is_ref_type(arena: &TypeArena, h: TypeHandle) -> bool {
    let ty = arena.get(h);
    !matches!(ty, Ty::Nullable(_))
        && matches!(
            ty,
            Ty::Str
                | Ty::Adt(_)
                | Ty::Generic(_)
                | Ty::Trait(_)
                | Ty::TraitObject(_)
                | Ty::Record(_)
                | Ty::ModuleRef(_)
        )
}

// ── AST 遍历上下文 ──

/// AST 遍历上下文：携带函数名 → 声明映射与循环检测表。
///
/// 刻意不持有 `sema_result`：所有需要 `&mut SemaResult` 的函数将其作为独立参数
/// 接收，使 `&mut ctx.in_progress` 与 `&mut sema_result` 可同时存活（split borrow）。
struct WalkCtx<'a> {
    ast: &'a AstArena<'a>,
    /// 函数名 → FunDecl 引用，用于推导 type_args 时查询参数类型注解与返回类型
    func_decls: FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    /// 函数名 → 所在模块 arena（跨模块单态化时，get_or_create_instance 需用被调函数
    /// 所在模块的 arena 解引用 body ExprId / 参数类型注解，而非调用点模块 arena）
    func_arenas: FxHashMap<&'a str, &'a AstArena<'a>>,
    /// 函数名 → 所在模块名（跨模块单态化时，expr_types 的 key 必须用被调函数所在模块名，
    /// 而非调用点模块名，确保 IR Builder 查找时 key 一致）
    func_module_names: FxHashMap<&'a str, &'a str>,
    /// 循环检测：正在实例化的 cache_key → instance_id（前向引用支持）
    in_progress: FxHashMap<u64, u32>,
    /// 当前模块名（用于实参 expr_types 查询，实参属于调用点模块）
    module_name: &'a str,
}

/// 推导泛型调用的 type_args
///
/// 优先级：
/// 1. 显式类型实参（call expr 的 type_args 字段，如 `foo<i32>(x)`）
/// 2. 隐式推断：
///    a. `.named` 类型注解（如 `init: A`）→ 实参 `ExprInfo` 的 `TypeHandle`
///    b. `.function` 类型注解（如 `f: (A, T) -> A`）→ lambda 实参的参数类型注解
///    c. `.function` 返回类型注解 → lambda 实参的返回类型（注解或 body 推断）
///
/// 未匹配的类型参数用 `arena.make_adt` 创建占位 Adt（name = 参数名）。
fn infer_type_args<'a>(
    func_name: &str,
    arguments: &[ExprId],
    type_args_hint: Option<&[AstTypeRef]>,
    sig: &FuncSigInfo,
    ast: &'a AstArena<'a>,
    func_decls: &FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &FxHashMap<&'a str, &'a AstArena<'a>>,
    module_name: &str,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) -> Vec<TypeHandle> {
    // 1. 显式类型实参：直接解析每个 TypeNode
    if let Some(hints) = type_args_hint {
        if !hints.is_empty() {
            let mut args = Vec::with_capacity(hints.len());
            for &tn in hints {
                let h = resolve_type_node_resolved(arena, Some(tn), &[], ast, sema_result)
                    .unwrap_or_else(|| {
                        sema_result.add_error(SemaError::new(
                            &format!("failed to resolve type argument in {}", func_name),
                            0, 0,
                        ));
                        arena.make(crate::types::Ty::Unknown)
                    });
                args.push(h);
            }
            return args;
        }
    }

    // 2. 隐式推断
    let fd_decl = match func_decls.get(func_name).copied() {
        Some(d) => d,
        None => {
            // AST 不可达（可能是方法或内建函数）：为每个类型参数创建具名 Adt 占位
            return sig
                .type_params
                .iter()
                .map(|tp| arena.make_adt((*tp).clone(), Box::new([])))
                .collect();
        }
    };
    // 被调函数所在模块的 arena：跨模块时类型注解 TypeId 属于被调模块 arena，
    // 必须用 fd_ast（而非 ctx.ast）访问，否则越界。
    let fd_ast = func_arenas.get(func_name).copied().unwrap_or(ast);
    let fd = match &fd_decl.node {
        Decl::FunDecl {
            type_params,
            params,
            return_type,
            body,
            is_async,
            ..
        } => FunDeclView {
            type_params,
            params,
            return_type: *return_type,
            body: *body,
            is_async: *is_async,
        },
        _ => unreachable!("func_decls only stores FunDecl"),
    };

    let mut name_to_handle: FxHashMap<&str, TypeHandle> = FxHashMap::default();

    let is_type_param = |name: &str| sig.type_params.iter().any(|tp| tp.as_ref() == name);

    let param_count = fd.params.len().min(arguments.len());

    // Pass 1: 匹配 .named 类型注解（如 `init: A` → 实参类型）
    for (i, arg) in arguments.iter().enumerate().take(param_count) {
        let param_type = match fd.params[i].type_annotation {
            Some(t) => t,
            None => continue,
        };
        let pname = match &fd_ast.ty(param_type).node {
            TypeNode::Named { name } => *name,
            _ => continue,
        };
        if !is_type_param(pname) || name_to_handle.contains_key(pname) {
            continue;
        }
        let arg_key = module_expr_key(module_name, arg.0 as u64);
        if let Some(info) = sema_result.get_expr(arg_key) {
            name_to_handle.insert(pname, info.ty);
        }
    }

    // Pass 2: 匹配 .function 类型注解（如 `f: (A, T) -> A`）against lambda 实参
    for (i, arg) in arguments.iter().enumerate().take(param_count) {
        let param_type = match fd.params[i].type_annotation {
            Some(t) => t,
            None => continue,
        };
        let fn_type = match &fd_ast.ty(param_type).node {
            TypeNode::Function {
                params: fn_params,
                return_type: fn_ret,
            } => (fn_params.as_slice(), *fn_ret),
            _ => continue,
        };
        let lambda = match &ast.expr(*arg).node {
            Expr::Lambda {
                params: lambda_params,
                return_type: lambda_rt,
                body,
                ..
            } => (lambda_params.as_slice(), *lambda_rt, body),
            _ => continue,
        };

        // 匹配函数类型参数与 lambda 参数
        let (fn_params, fn_ret) = fn_type;
        let (lambda_params, lambda_rt, _lambda_body) = lambda;
        let match_count = fn_params.len().min(lambda_params.len());
        for j in 0..match_count {
            let fp_name = match &fd_ast.ty(fn_params[j]).node {
                TypeNode::Named { name } => *name,
                _ => continue,
            };
            if !is_type_param(fp_name) || name_to_handle.contains_key(fp_name) {
                continue;
            }
            if let Some(lt) = lambda_params[j].type_annotation {
                if let Some(h) =
                    resolve_type_node_resolved(arena, Some(lt), &[], ast, sema_result)
                {
                    name_to_handle.insert(fp_name, h);
                }
            }
        }

        // 匹配函数返回类型注解 → lambda 返回类型
        let ret_name = match &fd_ast.ty(fn_ret).node {
            TypeNode::Named { name } => Some(*name),
            _ => None,
        };
        if let Some(ret_name) = ret_name {
            if is_type_param(ret_name) && !name_to_handle.contains_key(ret_name) {
                if let Some(lrt) = lambda_rt {
                    if let Some(h) =
                        resolve_type_node_resolved(arena, Some(lrt), &[], ast, sema_result)
                    {
                        name_to_handle.insert(ret_name, h);
                    }
                } else if let Some(h) = infer_lambda_return_type(lambda, ast, module_name, sema_result, arena) {
                    name_to_handle.insert(ret_name, h);
                }
            }
        }
    }

    // Pass 3: .generic 类型注解（如 `l: Lst<T>`）— 目前无法从 ref 通道提取元素类型，
    // 仅记录未绑定的类型参数名，依赖 Pass 1/2 已绑定的类型参数（跳过未绑定）

    // 输出 type_args：以类型参数名构造占位 Adt，使 resolve_type_node_resolved 按名匹配
    let mut args = Vec::with_capacity(sig.type_params.len());
    for tp_name in sig.type_params.iter() {
        let h = if let Some(&h) = name_to_handle.get(tp_name.as_ref()) {
            h
        } else {
            arena.make_adt((*tp_name).clone(), Box::new([]))
        };
        args.push(h);
    }
    args
}

/// 从 lambda body 推断返回类型。
/// 优先：显式返回类型注解 → body expression 的 ExprInfo → block trailing_expr 的 ExprInfo。
fn infer_lambda_return_type<'a>(
    lambda: (&'a [Param<'a>], Option<AstTypeRef>, &'a LambdaBody),
    ast: &'a AstArena<'a>,
    module_name: &str,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) -> Option<TypeHandle> {
    let (_, lambda_rt, body) = lambda;
    if let Some(rt) = lambda_rt {
        return resolve_type_node_resolved(arena, Some(rt), &[], ast, sema_result);
    }
    match body {
        LambdaBody::Expression(body_expr) => {
            let key = module_expr_key(module_name, body_expr.0 as u64);
            sema_result.get_expr(key).map(|info| info.ty)
        }
        LambdaBody::Block(block_expr) => {
            if let Expr::Block { trailing: Some(trailing), .. } = &ast.expr(*block_expr).node {
                let key = module_expr_key(module_name, trailing.0 as u64);
                return sema_result.get_expr(key).map(|info| info.ty);
            }
            None
        }
    }
}

/// 查找或创建 `MonomorphInstance`
///
/// 1. 查 `monomorph_index` 缓存命中 → 返回 instance_id
/// 2. 未命中：创建栈上局部实例、注册 `in_progress`（前向引用支持）
/// 3. 用具体 type_args 解析函数体内所有表达式类型（可能触发前向引用）
/// 4. 解析完成后 `push` 到 `monomorph_instances`、写入缓存
fn get_or_create_instance<'a>(
    func_name: &str,
    type_args: &[TypeHandle],
    fd_decl: &'a Spanned<Decl<'a>>,
    ast: &'a AstArena<'a>,
    func_decls: &FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &FxHashMap<&'a str, &'a AstArena<'a>>,
    func_module_names: &FxHashMap<&'a str, &'a str>,
    in_progress: &mut FxHashMap<u64, u32>,
    sema_result: &mut SemaResult,
    module_name: &'a str,
    arena: &mut TypeArena,
) -> u32 {
    let cache_key = build_cache_key(func_name, arena, type_args);

    // 1. 查缓存
    if let Some(&idx) = sema_result.monomorph_index.get(&cache_key) {
        return idx;
    }

    // 2. 循环检测：前向引用支持
    if let Some(&existing_id) = in_progress.get(&cache_key) {
        return existing_id;
    }
    // 递归深度上限：in_progress.len() 即当前递归深度，超限停止递归防止栈溢出
    if in_progress.len() >= MAX_MONOMORPH_DEPTH {
        panic!("monomorph recursion depth exceeded {} for function {}", MAX_MONOMORPH_DEPTH, func_name);
    }

    // 3. 新建栈上实例
    let fd = match &fd_decl.node {
        Decl::FunDecl {
            type_params,
            params,
            return_type,
            body,
            is_async,
            ..
        } => FunDeclView {
            type_params,
            params,
            return_type: *return_type,
            body: *body,
            is_async: *is_async,
        },
        _ => unreachable!("func_decls only stores FunDecl"),
    };

    let instance_id = sema_result.monomorph_instances.len() as u32;
    let return_handle =
        resolve_type_node_resolved(arena, fd.return_type, type_args, ast, sema_result)
            .unwrap_or_else(|| {
                sema_result.add_error(SemaError::new(
                    &format!("failed to resolve return type of {}", func_name),
                    0, 0,
                ));
                arena.make(crate::types::Ty::Unknown)
            });

    let mut instance = MonomorphInstance {
        instance_id,
        func_name: func_name.into(),
        module_name: module_name.into(),
        type_args: type_args.to_vec().into_boxed_slice(),
        chan_layout: ChanLayout::empty(),
        return_type: return_handle,
        is_async: fd.is_async,
        expr_types: FxHashMap::default(),
        field_accesses: FxHashMap::default(),
    };

    // 4. 标记为正在实例化（前向引用支持）
    in_progress.insert(cache_key, instance_id);

    // 5. 递归解析函数体类型（instance 是栈上局部，与 sema_result 无别名）
    // module_name 是被调函数所在模块名，确保 expr_types key 与 IR Builder 查找一致
    resolve_instance_body_types(
        &mut instance,
        &fd,
        ast,
        func_decls,
        func_arenas,
        func_module_names,
        in_progress,
        sema_result,
        type_args,
        module_name,
        arena,
    );

    // 6. 写入实例表与缓存
    sema_result.monomorph_instances.push(instance);
    sema_result.monomorph_index.insert(cache_key, instance_id);
    instance_id
}

/// FunDecl 字段视图（从 `Decl::FunDecl` 提取，便于跨函数传递）。
struct FunDeclView<'a> {
    type_params: &'a [TypeParam<'a>],
    params: &'a [Param<'a>],
    return_type: Option<AstTypeRef>,
    body: ExprId,
    is_async: bool,
}

// ── AST 递归遍历：收集泛型调用点 ──

/// 处理直接调用表达式（callee 为标识符）。
///
/// 仅处理 callee 是 identifier 的直接函数调用。方法调用、闭包调用等由
/// `process_method_call` 处理或跳过（递归遍历仍会进入 recv/arguments）。
#[allow(clippy::too_many_arguments)]
fn process_call<'a>(
    callee: ExprId,
    arguments: &[ExprId],
    type_args_hint: Option<&[AstTypeRef]>,
    call_expr: ExprId,
    ast: &'a AstArena<'a>,
    func_decls: &FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &FxHashMap<&'a str, &'a AstArena<'a>>,
    func_module_names: &FxHashMap<&'a str, &'a str>,
    in_progress: &mut FxHashMap<u64, u32>,
    sema_result: &mut SemaResult,
    module_name: &'a str,
    arena: &mut TypeArena,
) {
    // 仅处理直接标识符调用：foo(args) 或 foo<T>(args)
    let func_name = match &ast.expr(callee).node {
        Expr::Ident(name) => *name,
        _ => return,
    };
    // 查函数签名：跳过未注册函数与非泛型函数
    let sig_owned: Option<FuncSigInfo> = sema_result
        .get_func_sig(func_name).cloned();
    let sig = match sig_owned {
        Some(s) if !s.type_params.is_empty() => s,
        _ => return,
    };

    // 查函数 AST（用于参数类型注解与返回类型）
    let fd_decl = match func_decls.get(func_name).copied() {
        Some(d) => d,
        None => return,
    };

    // 推导 type_args（显式或隐式）
    // module_name 必须用调用点所在模块：实参表达式的类型信息以
    // module_expr_key(调用点模块, expr_id) 为 key 存入 expr_types，
    // 若用空串将导致 infer_type_args 查不到实参类型，T 无法绑定。
    let type_args = infer_type_args(
        func_name,
        arguments,
        type_args_hint,
        &sig,
        ast,
        func_decls,
        func_arenas,
        module_name,
        sema_result,
        arena,
    );

    // 查找或创建实例
    // ast 用被调函数所在模块 arena（跨模块时 body ExprId 属于被调模块 arena），
    // 回退调用点 arena（同模块调用场景）
    let callee_ast = func_arenas.get(func_name).copied().unwrap_or(ast);
    // module_name 用被调函数所在模块名（跨模块时 expr_types key 必须与 IR Builder 查找一致）
    let callee_module_name = func_module_names.get(func_name).copied().unwrap_or(module_name);
    let instance_id = get_or_create_instance(
        func_name,
        &type_args,
        fd_decl,
        callee_ast,
        func_decls,
        func_arenas,
        func_module_names,
        in_progress,
        sema_result,
        callee_module_name,
        arena,
    );

    // 记录调用点 → 实例映射（用 module_expr_key 避免跨模块 ExprId 碰撞）
    let call_key = crate::sema::Sema::module_expr_key(module_name, call_expr.0 as u64);
    sema_result
        .call_instantiations
        .insert(call_key, instance_id);
}

/// 处理方法调用表达式。
///
/// 方法调用通过 trait 分派，完整解析需要对象类型构造 mangled 名。此处采用最佳努力
/// 策略：直接以方法名查 `func_sig`，命中则处理；未命中则跳过。递归遍历仍会进入
/// recv/arguments，保证嵌套调用被收集。
#[allow(clippy::too_many_arguments)]
fn process_method_call<'a>(
    method: &str,
    arguments: &[ExprId],
    type_args_hint: Option<&[AstTypeRef]>,
    call_expr: ExprId,
    ast: &'a AstArena<'a>,
    func_decls: &FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &FxHashMap<&'a str, &'a AstArena<'a>>,
    func_module_names: &FxHashMap<&'a str, &'a str>,
    in_progress: &mut FxHashMap<u64, u32>,
    sema_result: &mut SemaResult,
    module_name: &'a str,
    arena: &mut TypeArena,
) {
    // 直接以方法名查 func_sig（覆盖同名顶层函数的罕见场景）
    let sig_owned: Option<FuncSigInfo> = sema_result.get_func_sig(method).cloned();
    let sig = match sig_owned {
        Some(s) if !s.type_params.is_empty() => s,
        Some(_) => return,
        None => return,
    };

    let fd_decl = match func_decls.get(method).copied() {
        Some(d) => d,
        None => return,
    };

    let type_args = infer_type_args(
        method,
        arguments,
        type_args_hint,
        &sig,
        ast,
        func_decls,
        func_arenas,
        module_name,
        sema_result,
        arena,
    );

    // ast 用被调函数所在模块 arena（跨模块 Module.fun() 调用时 body 属于被调模块）
    let callee_ast = func_arenas.get(method).copied().unwrap_or(ast);
    // module_name 用被调函数所在模块名（跨模块时 expr_types key 必须与 IR Builder 查找一致）
    let callee_module_name = func_module_names.get(method).copied().unwrap_or(module_name);
    let instance_id = get_or_create_instance(
        method,
        &type_args,
        fd_decl,
        callee_ast,
        func_decls,
        func_arenas,
        func_module_names,
        in_progress,
        sema_result,
        callee_module_name,
        arena,
    );
    let call_key = crate::sema::Sema::module_expr_key(module_name, call_expr.0 as u64);
    sema_result
        .call_instantiations
        .insert(call_key, instance_id);

    // v3 阶段 1：记录方法分派元信息（最佳努力匹配，完整 trait 解析留待后续阶段）
    // 键使用 module_expr_key（与 call_instantiations 一致），供 IR 查询 intrinsic
    let dispatch_key = crate::sema::Sema::module_expr_key(module_name, call_expr.0 as u64);
    sema_result.method_dispatches.insert(
        dispatch_key,
        DispatchInfo {
            trait_id: 0,
            method_idx: 0,
            impl_fn_idx: 0,
            instance_id,
            intrinsic: None,
        },
    );
}

/// 递归遍历 Stmt，收集所有嵌套的泛型调用点。
fn walk_stmt<'a>(
    stmt: StmtId,
    ctx: &mut WalkCtx<'a>,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    let node = &ctx.ast.stmt(stmt).node;
    match node {
        Stmt::ValDecl { value, .. } => walk_expr(*value, ctx, sema_result, arena),
        Stmt::VarDecl { value, .. } => walk_expr(*value, ctx, sema_result, arena),
        Stmt::Assignment { target, value } => {
            walk_expr(*target, ctx, sema_result, arena);
            walk_expr(*value, ctx, sema_result, arena);
        }
        Stmt::FieldAssignment { object, value, .. } => {
            walk_expr(*object, ctx, sema_result, arena);
            walk_expr(*value, ctx, sema_result, arena);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            walk_expr(*target, ctx, sema_result, arena);
            walk_expr(*value, ctx, sema_result, arena);
        }
        Stmt::Expression { expr } => walk_expr(*expr, ctx, sema_result, arena),
        Stmt::Return { value } => {
            if let Some(v) = value { walk_expr(*v, ctx, sema_result, arena); }
        }
        Stmt::Defer { expr } => walk_expr(*expr, ctx, sema_result, arena),
        Stmt::Throw { expr } => walk_expr(*expr, ctx, sema_result, arena),
        Stmt::Break | Stmt::Continue => {}
        Stmt::For { iterable, body, .. } => {
            walk_expr(*iterable, ctx, sema_result, arena);
            walk_expr(*body, ctx, sema_result, arena);
        }
        Stmt::While { condition, body } => {
            walk_expr(*condition, ctx, sema_result, arena);
            walk_expr(*body, ctx, sema_result, arena);
        }
        Stmt::Loop { body } => walk_expr(*body, ctx, sema_result, arena),
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            crate::ast::Ast::Decl::FunDecl { body, .. } => walk_expr(*body, ctx, sema_result, arena),
            crate::ast::Ast::Decl::TypeDecl { methods, .. }
            | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                for m in methods.iter() {
                    if let Some(body) = m.body { walk_expr(body, ctx, sema_result, arena); }
                }
            }
            _ => {}
        },
    }
}

/// 递归遍历 Expr，收集所有嵌套的泛型调用点。
///
/// 对 `call`/`method_call`/`safe_method_call` 三种调用表达式，提取调用元信息并
/// 推导 type_args。同时递归进入所有子表达式，确保嵌套调用被完整收集。
fn walk_expr<'a>(
    expr: ExprId,
    ctx: &mut WalkCtx<'a>,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    // 先复制不可变引用字段，再用 &mut ctx.in_progress（split borrow）
    let ast = ctx.ast;
    let func_decls = &ctx.func_decls;
    let func_arenas = &ctx.func_arenas;
    let func_module_names = &ctx.func_module_names;
    let node = &ast.expr(expr).node;
    match node {
        // ── 调用表达式：核心收集目标 ──
        Expr::Call {
            callee,
            args,
            type_args,
        } => {
            let hint = type_args.as_deref();
            process_call(
                *callee,
                args.as_slice(),
                hint,
                expr,
                ast,
                func_decls,
                func_arenas,
                func_module_names,
                &mut ctx.in_progress,
                sema_result,
                ctx.module_name,
                arena,
            );
            walk_expr(*callee, ctx, sema_result, arena);
            for &arg in args {
                walk_expr(arg, ctx, sema_result, arena);
            }
        }
        Expr::MethodCall {
            recv,
            method,
            args,
            type_args,
        } => {
            let hint = type_args.as_deref();
            process_method_call(
                method,
                args.as_slice(),
                hint,
                expr,
                ast,
                func_decls,
                func_arenas,
                func_module_names,
                &mut ctx.in_progress,
                sema_result,
                ctx.module_name,
                arena,
            );
            walk_expr(*recv, ctx, sema_result, arena);
            for &arg in args {
                walk_expr(arg, ctx, sema_result, arena);
            }
        }
        Expr::SafeMethodCall {
            recv,
            method,
            args,
            type_args,
        } => {
            let hint = type_args.as_deref();
            process_method_call(
                method,
                args.as_slice(),
                hint,
                expr,
                ast,
                func_decls,
                func_arenas,
                func_module_names,
                &mut ctx.in_progress,
                sema_result,
                ctx.module_name,
                arena,
            );
            walk_expr(*recv, ctx, sema_result, arena);
            for &arg in args {
                walk_expr(arg, ctx, sema_result, arena);
            }
        }

        // ── 一元/二元/赋值 ──
        Expr::Binary { op: _, lhs, rhs } => {
            walk_expr(*lhs, ctx, sema_result, arena);
            walk_expr(*rhs, ctx, sema_result, arena);
        }
        Expr::Unary { operand, .. } => walk_expr(*operand, ctx, sema_result, arena),
        Expr::RefOf(operand) => walk_expr(*operand, ctx, sema_result, arena),
        Expr::Deref(operand) => walk_expr(*operand, ctx, sema_result, arena),
        Expr::Assign { target, value } => {
            walk_expr(*target, ctx, sema_result, arena);
            walk_expr(*value, ctx, sema_result, arena);
        }
        Expr::CompoundAssign { target, value, .. } => {
            walk_expr(*target, ctx, sema_result, arena);
            walk_expr(*value, ctx, sema_result, arena);
        }
        Expr::NonNullAssert(e) => walk_expr(*e, ctx, sema_result, arena),
        Expr::Propagate(e) => walk_expr(*e, ctx, sema_result, arena),
        Expr::Elvis { lhs, rhs } => {
            walk_expr(*lhs, ctx, sema_result, arena);
            walk_expr(*rhs, ctx, sema_result, arena);
        }

        // ── 字段访问与索引 ──
        Expr::FieldAccess { recv, .. } => walk_expr(*recv, ctx, sema_result, arena),
        Expr::SafeAccess { recv, .. } => walk_expr(*recv, ctx, sema_result, arena),
        Expr::Index { recv, index } => {
            walk_expr(*recv, ctx, sema_result, arena);
            walk_expr(*index, ctx, sema_result, arena);
        }
        Expr::Slice { recv, start, end, .. } => {
            walk_expr(*recv, ctx, sema_result, arena);
            walk_expr(*start, ctx, sema_result, arena);
            walk_expr(*end, ctx, sema_result, arena);
        }

        // ── 容器字面量 ──
        Expr::ArrayLit { elements, fill } => {
            for &e in elements {
                walk_expr(e, ctx, sema_result, arena);
            }
            if let Some((fv, fc)) = fill {
                walk_expr(*fv, ctx, sema_result, arena);
                walk_expr(*fc, ctx, sema_result, arena);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                walk_expr(f.value, ctx, sema_result, arena);
            }
        }
        Expr::RecordExtend { base, updates } => {
            walk_expr(*base, ctx, sema_result, arena);
            for f in updates {
                walk_expr(f.value, ctx, sema_result, arena);
            }
        }

        // ── 控制流 ──
        Expr::Lambda { body, .. } => match body {
            LambdaBody::Block(b) => walk_expr(*b, ctx, sema_result, arena),
            LambdaBody::Expression(e) => walk_expr(*e, ctx, sema_result, arena),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr(*cond, ctx, sema_result, arena);
            walk_expr(*then_branch, ctx, sema_result, arena);
            if let Some(eb) = else_branch {
                walk_expr(*eb, ctx, sema_result, arena);
            }
        }
        Expr::Block { stmts, trailing } => {
            for &s in stmts {
                walk_stmt(s, ctx, sema_result, arena);
            }
            if let Some(te) = trailing {
                walk_expr(*te, ctx, sema_result, arena);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_expr(*scrutinee, ctx, sema_result, arena);
            for arm in arms {
                if let Some(g) = arm.guard {
                    walk_expr(g, ctx, sema_result, arena);
                }
                walk_expr(arm.body, ctx, sema_result, arena);
            }
        }

        // ── 并发/异步 ──
        Expr::Atomic(e) => walk_expr(*e, ctx, sema_result, arena),
        Expr::Lazy(e) => walk_expr(*e, ctx, sema_result, arena),
        Expr::Select(arms) => {
            for arm in arms {
                match arm {
                    SelectArm::Receive {
                        channel_expr, body, ..
                    } => {
                        walk_expr(*channel_expr, ctx, sema_result, arena);
                        walk_expr(*body, ctx, sema_result, arena);
                    }
                    SelectArm::Timeout { duration, body } => {
                        walk_expr(*duration, ctx, sema_result, arena);
                        walk_expr(*body, ctx, sema_result, arena);
                    }
                }
            }
        }

        // ── 字符串插值 ──
        Expr::StrInterp(parts) => {
            for part in parts {
                if let InterpolationPart::Expression(e) = part {
                    walk_expr(*e, ctx, sema_result, arena);
                }
            }
        }

        // ── inline trait value：方法体可能含泛型调用 ──
        Expr::InlineTrait(methods) => {
            for method in methods {
                if let Some(body) = method.body {
                    walk_expr(body, ctx, sema_result, arena);
                }
            }
        }

        // ── 终端节点：无需递归 ──
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit
        | Expr::Ident(_) => {}
    }
}

// ── 主入口：collect_monomorph_instances ──

/// 收集模块中所有泛型调用点，产出单态化实例集合
///
/// v3 spec §5.2 `collectMonomorphInstances` 算法：
/// 1. 构建 `func_name → fun_decl` 映射，供推导 type_args 时查询参数类型注解
/// 2. 遍历所有顶层声明：
///    a. 非泛型 `fun_decl` → 创建空 type_args 实例
///    b. 所有 `fun_decl` 体 / `type_decl` 方法体 / `expr_decl` → 递归遍历
/// 3. 对每个泛型调用点：推导 type_args → 去重 → 创建实例 → 记录调用点映射
///
/// 泛型函数本身不创建空实例（其具体实例由调用点驱动生成）。
/// 方法调用的完整 trait 分派解析留待后续阶段，当前仅做最佳努力匹配。
pub fn collect_monomorph_instances<'a>(
    module: &'a Module<'a>,
    all_modules: &[&'a Module<'a>],
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    // 注册内置类型方法签名（合成 TypeDefInfo），使方法查找走统一 (type_id, method_idx) 路径
    register_builtin_method_sigs(sema_result);

    let mut ctx = WalkCtx {
        ast: &module.arena,
        func_decls: FxHashMap::default(),
        func_arenas: FxHashMap::default(),
        func_module_names: FxHashMap::default(),
        in_progress: FxHashMap::default(),
        module_name: module.name,
    };

    // 1. 构建 func_name → &Spanned<Decl> + 所在模块 arena + 所在模块名 映射（跨模块收集顶层 fun_decl）
    // 跨模块单态化：调用 std.math.Math.abs<T>(x) 时，func_decls 需能命中 abs（定义在 Math 模块），
    // func_arenas 提供 abs 所在模块 arena，供 get_or_create_instance 解引用 body ExprId，
    // func_module_names 提供 abs 所在模块名，确保 expr_types 的 key 与 IR Builder 查找一致。
    for m in all_modules {
        for decl in &m.declarations {
            if let Decl::FunDecl { name, .. } = &decl.node {
                ctx.func_decls.insert(name, decl);
                ctx.func_arenas.insert(name, &m.arena);
                ctx.func_module_names.insert(name, m.name);
            }
        }
    }

    // 2. 遍历所有顶层声明
    let declarations: Vec<&Spanned<Decl<'a>>> = module.declarations.iter().collect();
    for decl in declarations {
        match &decl.node {
            Decl::FunDecl {
                name,
                type_params,
                params,
                return_type,
                body,
                is_async,
                ..
            } => {
                // 非泛型函数：创建空 type_args 实例
                // （泛型函数不创建空实例，由调用点驱动）
                if type_params.is_empty() {
                    let empty_type_args: Vec<TypeHandle> = Vec::new();
                    let return_handle = resolve_type_node_resolved(
                        arena,
                        *return_type,
                        &empty_type_args,
                        &module.arena,
                        sema_result,
                    )
                    .unwrap_or_else(|| {
                        sema_result.add_error(SemaError::new(
                            &format!("failed to resolve return type of {}", name),
                            0, 0,
                        ));
                        arena.make(crate::types::Ty::Unknown)
                    });

                    let instance = MonomorphInstance {
                        instance_id: sema_result.monomorph_instances.len() as u32,
                        func_name: (*name).into(),
                        module_name: module.name.into(),
                        type_args: Vec::new().into_boxed_slice(),
                        chan_layout: ChanLayout::empty(),
                        return_type: return_handle,
                        is_async: *is_async,
                        expr_types: FxHashMap::default(),
                        field_accesses: FxHashMap::default(),
                    };
                    sema_result.monomorph_instances.push(instance);

                    // 非泛型函数：遍历函数体收集泛型调用点
                    // （泛型函数体的调用点在 resolveInstanceBodyTypes 中发现，
                    //  因为它们依赖当前实例的 type_args 上下文才能正确推断类型实参）
                    walk_expr(*body, &mut ctx, sema_result, arena);
                }
                let _ = params; // params 未在非泛型分支使用
            }

            // 类型声明：遍历非泛型方法体（泛型方法体在实例化时发现）
            Decl::TypeDecl { methods, .. } => {
                // 收集需遍历的方法体，避免在借用 methods 时 &mut sema_result
                let bodies: Vec<ExprId> = methods
                    .iter()
                    .filter(|m| m.type_params.is_empty())
                    .filter_map(|m| m.body)
                    .collect();
                for body in bodies {
                    walk_expr(body, &mut ctx, sema_result, arena);
                }
            }

            // 顶层表达式声明：遍历表达式
            Decl::ExprDecl { expr, stmt } => {
                walk_expr(*expr, &mut ctx, sema_result, arena);
                if let Some(s) = stmt {
                    walk_stmt(*s, &mut ctx, sema_result, arena);
                }
            }

            // import / pack：无需遍历
            _ => {}
        }
    }
}

// ── 实例体类型解析 ──

/// 用具体 type_args 解析函数体内所有表达式类型
///
/// 递归遍历函数体 AST，对每个表达式计算其类型并存入 `instance.expr_types`。
/// 对 `field_access` 表达式额外存入 `instance.field_accesses`。
/// 替代 IRBuilder 的 `infer*` 系列函数。
fn resolve_instance_body_types<'a>(
    instance: &mut MonomorphInstance,
    fd: &FunDeclView<'a>,
    ast: &'a AstArena<'a>,
    func_decls: &'a FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &'a FxHashMap<&'a str, &'a AstArena<'a>>,
    func_module_names: &'a FxHashMap<&'a str, &'a str>,
    in_progress: &mut FxHashMap<u64, u32>,
    sema_result: &mut SemaResult,
    type_args: &[TypeHandle],
    module_name: &'a str,
    arena: &mut TypeArena,
) {
    use crate::sema::Inference::InferContext;

    // ── 步骤 1：用 InferContext 实例化模式解析函数体类型 ──
    // 创建临时 InferContext，用具体 type_args 替换 rigid TypeVar，
    // 跑一遍 infer_expr 将所有表达式类型写入 local_expr_types。
    let local_expr_types: FxHashMap<u64, ExprInfo>;
    {
        let mut infer_ctx = InferContext::new(arena, sema_result);
        infer_ctx.current_module_name = module_name.to_string();

        // push type bindings（先放 rigid var，enter_instantiation_mode 会替换为具体 type_args）
        let type_param_names: Vec<&str> = fd.type_params.iter().map(|tp| tp.name).collect();
        infer_ctx.push_type_bindings(
            &type_param_names.iter().map(|&name| (name, None)).collect::<Vec<_>>(),
        );

        // 进入实例化模式：rigid var → 具体 type_args
        infer_ctx.enter_instantiation_mode(
            instance.func_name.clone(),
            type_args.to_vec().into_boxed_slice(),
            &type_param_names,
            module_name.to_string(),
            in_progress.clone(),
        );

        // 创建环境并注册函数参数
        let fn_env = infer_ctx.env.root();
        for param in fd.params {
            let h = if let Some(ta) = param.type_annotation {
                infer_ctx.type_from_ast(ta, ast)
            } else {
                infer_ctx.arena.fresh_type_var()
            };
            infer_ctx.env.define(fn_env, param.name, h);
        }

        // 设置返回类型
        let ret_ty = if let Some(rt) = fd.return_type {
            infer_ctx.type_from_ast(rt, ast)
        } else {
            infer_ctx.arena.fresh_type_var()
        };
        infer_ctx.expected_return = Some(ret_ty);

        // 推断函数体（实例化模式下 unify_or_constrain 跳过，store_expr_info 写入 local_expr_types）
        let _ = infer_ctx.infer_expr(fd.body, ast, fn_env, infer_ctx.expected_return);

        // 离开实例化模式，取出暂存类型
        if let Some((let_local, _field_accesses, let_in_progress)) =
            infer_ctx.leave_instantiation_mode()
        {
            local_expr_types = let_local;
            *in_progress = let_in_progress;
        } else {
            local_expr_types = FxHashMap::default();
        }

        infer_ctx.pop_type_bindings();
    } // infer_ctx dropped，释放 &mut arena 和 &mut sema_result

    // 合并 local_expr_types 到 instance.expr_types + sema_result.expr_types
    // instance.expr_types：实例本地查询（IR Builder 用）
    // sema_result.expr_types：全局查询（process_call → infer_type_args 用）
    for (key, info) in &local_expr_types {
        instance.expr_types.insert(*key, info.clone());
    }
    for (key, info) in local_expr_types {
        sema_result.put_expr(key, info);
    }

    // ── 步骤 2：用 walk_expr + process_call 触发嵌套调用的单态化 ──
    // 复用顶层路径：walk_expr 遍历函数体，遇到 Call/MethodCall 时 process_call
    // 从 sema_result.expr_types 查询实参类型（步骤 1 已写入），推断 type_args 并创建嵌套实例
    let mut walk_ctx = WalkCtx {
        ast,
        func_decls: func_decls.clone(),
        func_arenas: func_arenas.clone(),
        func_module_names: func_module_names.clone(),
        in_progress: in_progress.clone(),
        module_name,
    };
    walk_expr(fd.body, &mut walk_ctx, sema_result, arena);
    *in_progress = walk_ctx.in_progress;
}

// ====== 以下是新增的 trait 默认方法单态化逻辑 ======

/// 判断类型是否在 AST 层显式实现了某方法（有 body）。
///
/// 用于跳过 trait 默认方法特化：类型已显式覆写该方法时不需要生成特化子图。
/// 与 Ir.rs 步骤 0 注册 method_subgraphs 的条件一致（method.body.is_some()）。
fn type_has_explicit_method<'a>(module: &'a Module<'a>, type_name: &str, method_name: &str) -> bool {
    for decl in &module.declarations {
        if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &decl.node {
            if *name == type_name {
                return methods.iter().any(|m| &*m.name == method_name && m.body.is_some());
            }
        }
    }
    false
}

/// 收集 trait 默认方法单态化实例。
///
/// 为每个实现 trait 但未显式覆写该方法的类型生成一个 TraitDefaultInstance 条目。
/// IR 层（IrBuilder）消费此表预注册并编译特化子图。
///
/// 算法：
/// 1. 遍历模块中的 TraitDecl，获取 trait_idx
/// 2. 从 witness_table 收集所有实现该 trait 的类型 (type_id, type_name)
/// 3. 对每个有 body 的默认方法，为每个实现类型生成特化实例
/// 4. 跳过类型已显式覆写该方法的情况（AST 层判断 type_has_explicit_method）
pub fn collect_trait_default_instances<'a>(
    module: &'a Module<'a>,
    sema_result: &mut SemaResult,
) {
    for decl in &module.declarations {
        if let crate::ast::Ast::Decl::TraitDecl { name, methods, .. } = &decl.node {
            let trait_idx = match sema_result.trait_def_index.get(*name).copied() {
                Some(idx) => idx,
                None => continue,
            };
            // 收集所有实现该 trait 的类型（从 witness_table）
            let impl_entries: Vec<(u16, String)> = sema_result
                .witness_table
                .entries()
                .iter()
                .filter(|e| e.trait_name.as_ref() == *name)
                .filter_map(|e| {
                    // type_id → type_name（反查 type_defs）
                    sema_result.type_defs.iter().enumerate()
                        .find(|(i, _)| crate::types::dynamic_type_id(*i as u16) == e.type_id)
                        .map(|(_, td)| (e.type_id, td.name.to_string()))
                })
                .collect();

            for (method_idx, method) in methods.iter().enumerate() {
                if method.body.is_none() {
                    continue;
                }
                let method_name: &str = method.name.as_ref();
                for (type_id, type_name) in &impl_entries {
                    // 跳过类型已显式覆写该方法的情况
                    if type_has_explicit_method(module, type_name, method_name) {
                        continue;
                    }
                    sema_result.trait_default_instances.push(TraitDefaultInstance {
                        type_id: *type_id,
                        type_name: type_name.as_str().into(),
                        trait_idx,
                        trait_name: (*name).into(),
                        method_idx: method_idx as u16,
                    });
                }
            }
        }
    }
}

/// Sema 后单态化统一入口。
///
/// 在 Sema 阶段完成后调用，执行两类单态化实例收集：
/// 1. `collect_monomorph_instances`：泛型函数调用点驱动的单态化
/// 2. `collect_trait_default_instances`：trait 默认方法按实现类型特化
///
/// IR 层（IrBuilder）消费 SemaResult 中的 monomorph_instances 和 trait_default_instances
/// 生成对应的特化子图。
pub fn run_monomorphization<'a>(
    module: &'a Module<'a>,
    all_modules: &[&'a Module<'a>],
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    collect_monomorph_instances(module, all_modules, sema_result, arena);
    collect_trait_default_instances(module, sema_result);
}
