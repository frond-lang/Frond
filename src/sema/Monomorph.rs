use crate::sema::Sema::*;
use crate::ast::Ast::{
    AstArena, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module,
    Param, SelectArm, Spanned, Stmt, StmtId,
    TypeNode, TypeParam, TypeRef as AstTypeRef,
};
use rustc_hash::FxHashMap;

/// Maximum recursion depth for monomorphization: prevents stack overflow from
/// extremely deep generic call chains.
/// `in_progress.len()` is the current recursion depth; recursion stops when the
/// limit is reached.
const MAX_MONOMORPH_DEPTH: usize = 256;

// ====== The code below was migrated verbatim from Inference.rs (formerly
//         SemaInfer.rs), lines 740-2408. ======
// =========================================================================
// monomorph — monomorphization instantiation.
//
// v3 spec §5.2: migrated from src/sema/monomorph.zig.
// Responsibility: identify all generic call sites → infer type_args →
// deduplicate → determine the instance set.
//
// Rust adaptations:
// - Expression keys changed from raw-pointer `@intFromPtr` to
//   `ExprId.0 as u64` (AstArena index).
// - Type resolution delegates to `resolve_type_node_resolved` (which takes
//   `Option<AstTypeRef>` rather than `*TypeNode`).
// - Borrow separation: `WalkCtx` does not hold `sema_result`; it is passed as a
//   separate parameter to avoid a circular borrow between `&mut SemaResult` and
//   `&mut WalkCtx`. `instance` is a stack-local that finishes body resolution
//   before `push`, so it has no alias with `sema_result`.
// - `field_access` metadata distinguishes Record (field_id starts at 0) from
//   ADT/Newtype (field_id starts at 1, `__tag=0`) per `TypeDefKind`, fixing the
//   Record index offset bug in the Zig version.
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
        Type::Adt(_) => arena.adt_parts(resolved).0,
        Type::Generic(_) => arena.generic_parts(resolved).0,
        Type::Trait(_) => arena.trait_parts(resolved).0,
        Type::TraitObject(_) => arena.trait_object_parts(resolved).0,
        _ => ty.name(),
    };
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a 64-bit hash (migrated from monomorph.zig:hashTypeArgs).
/// Takes a list of `TypeHandle`s and derives a stable identity via
/// `type_identity_hash`.
pub fn hash_type_args(arena: &TypeArena, type_args: &[TypeHandle]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &ta in type_args {
        h ^= type_identity_hash(arena, ta);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Build a monomorphization cache key: an FNV-1a combined u64 hash of
/// `func_name` and `type_args` (no String allocation).
pub fn build_cache_key(func_name: &str, arena: &TypeArena, type_args: &[TypeHandle]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in func_name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= hash_type_args(arena, type_args);
    h.wrapping_mul(0x100000001b3)
}

/// Find an existing monomorphization instance (cache lookup only; does not
/// create).
pub fn find_instance(
    arena: &TypeArena,
    sema_result: &SemaResult,
    func_name: &str,
    type_args: &[TypeHandle],
) -> Option<u32> {
    let cache_key = build_cache_key(func_name, arena, type_args);
    sema_result.monomorph_index.get(&cache_key).copied()
}

// ── AST traversal context ──

/// AST traversal context: carries the function-name → declaration mapping and
/// the cycle-detection table.
///
/// Deliberately does not hold `sema_result`: all functions needing
/// `&mut SemaResult` take it as a separate parameter, so `&mut ctx.in_progress`
/// and `&mut sema_result` can coexist (split borrow).
struct WalkCtx<'a> {
    ast: &'a AstArena<'a>,
    /// Function name → `FunDecl` reference, used to look up parameter type
    /// annotations and return types while inferring type_args.
    func_decls: FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    /// Function name → owning module arena (for cross-module monomorphization,
    /// `get_or_create_instance` must use the callee's module arena to
    /// dereference body ExprIds / parameter type annotations, not the call-site
    /// module arena).
    func_arenas: FxHashMap<&'a str, &'a AstArena<'a>>,
    /// Function name → owning module name (for cross-module monomorphization,
    /// the `expr_types` key must use the callee's module name, not the call-site
    /// module name, so the IR Builder lookup keys match).
    func_module_names: FxHashMap<&'a str, &'a str>,
    /// Cycle detection: cache_key currently being instantiated → instance_id
    /// (forward-reference support).
    in_progress: FxHashMap<u64, u32>,
    /// Current module name (used for argument `expr_types` lookup; arguments
    /// belong to the call-site module).
    module_name: &'a str,
}

/// Infer the type_args of a generic call.
///
/// Priority:
/// 1. Explicit type arguments (the `type_args` field of the call expr, e.g.
///    `foo<i32>(x)`).
/// 2. Implicit inference:
///    a. `.named` type annotation (e.g. `init: A`) → the argument's `ExprInfo`
///       `TypeHandle`.
///    b. `.function` type annotation (e.g. `f: (A, T) -> A`) → the lambda
///       argument's parameter type annotations.
///    c. `.function` return-type annotation → the lambda argument's return type
///       (annotation or body inference).
///
/// Unmatched type parameters get a placeholder Adt via `arena.make_adt` (name =
/// parameter name).
fn infer_type_args<'a>(
    func_name: &str,
    arguments: &[ExprId],
    type_args_hint: Option<&[AstTypeRef]>,
    type_params: &[Box<str>],
    ast: &'a AstArena<'a>,
    func_decls: &FxHashMap<&'a str, &'a Spanned<Decl<'a>>>,
    func_arenas: &FxHashMap<&'a str, &'a AstArena<'a>>,
    module_name: &str,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) -> Vec<TypeHandle> {
    // 1. Explicit type arguments: resolve each TypeNode directly.
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
                        arena.make(crate::types::Type::Unknown)
                    });
                args.push(h);
            }
            return args;
        }
    }

    // 2. Implicit inference.
    let fd_decl = match func_decls.get(func_name).copied() {
        Some(d) => d,
        None => {
            // AST-unreachable (likely a method or built-in): create a named Adt
            // placeholder for each type parameter.
            return type_params
                .iter()
                .map(|tp| arena.make_adt((*tp).clone(), Box::new([])))
                .collect();
        }
    };
    // The callee's owning module arena: for cross-module calls the type
    // annotation TypeIds belong to the callee's module arena, so they must be
    // accessed via `fd_ast` (not `ctx.ast`), otherwise indexing goes out of
    // bounds.
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

    let is_type_param = |name: &str| type_params.iter().any(|tp| tp.as_ref() == name);

    let param_count = fd.params.len().min(arguments.len());

    // Pass 1: match `.named` type annotations (e.g. `init: A` → argument type).
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

    // Pass 2: match `.function` type annotations (e.g. `f: (A, T) -> A`)
    // against lambda arguments.
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

        // Match function-type parameters against lambda parameters.
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

        // Match function return-type annotation → lambda return type.
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

    // Pass 3: `.generic` type annotations (e.g. `l: Lst<T>`) — currently cannot
    // extract the element type from the ref channel; only records unbound type
    // parameter names and relies on type parameters already bound in Pass 1/2
    // (skipping unbound ones).

    // Emit type_args: build a placeholder Adt from the type parameter name so
    // `resolve_type_node_resolved` can match by name.
    let mut args = Vec::with_capacity(type_params.len());
    for tp_name in type_params.iter() {
        let h = if let Some(&h) = name_to_handle.get(tp_name.as_ref()) {
            h
        } else {
            arena.make_adt((*tp_name).clone(), Box::new([]))
        };
        args.push(h);
    }
    args
}

/// Infer the return type from a lambda body.
/// Priority: explicit return-type annotation → body expression's `ExprInfo` →
/// block `trailing_expr`'s `ExprInfo`.
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

/// Find or create a `MonomorphInstance`.
///
/// 1. Check `monomorph_index` cache; on hit, return the instance_id.
/// 2. On miss: create a stack-local instance, register it in `in_progress`
///    (forward-reference support).
/// 3. Resolve all expression types in the function body using the concrete
///    type_args (may trigger forward references).
/// 4. After resolution completes, `push` to `monomorph_instances` and write the
///    cache.
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

    // 1. Check the cache.
    if let Some(&idx) = sema_result.monomorph_index.get(&cache_key) {
        return idx;
    }

    // 2. Cycle detection: forward-reference support.
    if let Some(&existing_id) = in_progress.get(&cache_key) {
        return existing_id;
    }
    // Recursion-depth limit: `in_progress.len()` is the current depth; stop
    // recursing past the limit to prevent stack overflow.
    if in_progress.len() >= MAX_MONOMORPH_DEPTH {
        panic!("monomorph recursion depth exceeded {} for function {}", MAX_MONOMORPH_DEPTH, func_name);
    }

    // 3. Create a stack-local instance.
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
                arena.make(crate::types::Type::Unknown)
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

    // 4. Mark as in-progress (forward-reference support).
    in_progress.insert(cache_key, instance_id);

    // 5. Recursively resolve body types (`instance` is stack-local, with no
    //    alias with `sema_result`).
    // `module_name` is the callee's module name, ensuring the `expr_types` key
    // matches the IR Builder lookup.
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

    // 6. Write to the instance table and cache.
    sema_result.monomorph_instances.push(instance);
    sema_result.monomorph_index.insert(cache_key, instance_id);
    // Record module ownership for incremental purge (monomorph index).
    sema_result.module_ownership.monomorph_indices
        .entry(module_name.to_string())
        .or_default()
        .insert(instance_id);
    instance_id
}

/// Field view of `FunDecl` (extracted from `Decl::FunDecl` for easy cross-
/// function passing).
struct FunDeclView<'a> {
    type_params: &'a [TypeParam<'a>],
    params: &'a [Param<'a>],
    return_type: Option<AstTypeRef>,
    body: ExprId,
    is_async: bool,
}

// ── AST recursive traversal: collect generic call sites ──

/// Process a direct call expression (callee is an identifier).
///
/// Only handles direct function calls where the callee is an identifier. Method
/// calls, closure calls, etc. are handled by `process_method_call` or skipped
/// (recursive traversal still descends into recv/arguments).
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
    // Only handle direct identifier calls: `foo(args)` or `foo<T>(args)`.
    let func_name = match &ast.expr(callee).node {
        Expr::Ident(name) => *name,
        _ => return,
    };
    // Look up the function signature: skip unregistered and non-generic
    // functions. Clone only the lightweight type_params (short names like "T")
    // instead of the entire FuncSigInfo, to release the sema_result borrow
    // before the mutable borrow in infer_type_args.
    let type_params: Box<[Box<str>]> = match sema_result.get_func_sig(func_name) {
        Some(s) if !s.type_params.is_empty() => s.type_params.clone(),
        _ => return,
    };

    // Look up the function AST (for parameter type annotations and return type).
    let fd_decl = match func_decls.get(func_name).copied() {
        Some(d) => d,
        None => return,
    };

    // Infer type_args (explicit or implicit).
    // `module_name` must be the call-site module: argument expression type info
    // is stored in `expr_types` under the key
    // `module_expr_key(call_site_module, expr_id)`; using an empty string would
    // cause `infer_type_args` to miss the argument types and leave T unbound.
    let type_args = infer_type_args(
        func_name,
        arguments,
        type_args_hint,
        &type_params,
        ast,
        func_decls,
        func_arenas,
        module_name,
        sema_result,
        arena,
    );

    // Find or create the instance.
    // `ast` uses the callee's module arena (for cross-module calls the body
    // ExprIds belong to the callee's arena), falling back to the call-site arena
    // (same-module call case).
    let callee_ast = func_arenas.get(func_name).copied().unwrap_or(ast);
    // `module_name` uses the callee's module name (for cross-module calls the
    // `expr_types` key must match the IR Builder lookup).
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

    // Record the call-site → instance mapping (uses `module_expr_key` to avoid
    // cross-module ExprId collisions).
    let call_key = crate::sema::Sema::module_expr_key(module_name, call_expr.0 as u64);
    sema_result
        .call_instantiations
        .insert(call_key, instance_id);
}

/// Process a method-call expression.
///
/// Method calls go through trait dispatch, and full resolution requires the
/// object type to construct the mangled name. A best-effort strategy is used
/// here: look up `func_sig` directly by method name and process on hit, skip
/// otherwise. Recursive traversal still descends into recv/arguments, ensuring
/// nested calls are collected.
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
    // Look up `func_sig` directly by method name (covers the rare case of a
    // same-named top-level function). Clone only the lightweight type_params
    // (short names like "T") instead of the entire FuncSigInfo, to release the
    // sema_result borrow before the mutable borrow in infer_type_args.
    let type_params: Box<[Box<str>]> = match sema_result.get_func_sig(method) {
        Some(s) if !s.type_params.is_empty() => s.type_params.clone(),
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
        &type_params,
        ast,
        func_decls,
        func_arenas,
        module_name,
        sema_result,
        arena,
    );

    // `ast` uses the callee's module arena (for cross-module `Module.fun()`
    // calls the body belongs to the callee's module).
    let callee_ast = func_arenas.get(method).copied().unwrap_or(ast);
    // `module_name` uses the callee's module name (for cross-module calls the
    // `expr_types` key must match the IR Builder lookup).
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

    // v3 phase 1: record method-dispatch metadata (best-effort match; full
    // trait resolution is deferred to a later phase).
    // The key uses `module_expr_key` (consistent with `call_instantiations`) so
    // the IR can query the intrinsic.
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

/// Recursively walk a `Stmt`, collecting all nested generic call sites.
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

/// Recursively walk an `Expr`, collecting all nested generic call sites.
///
/// For the three call-expression kinds (`call`/`method_call`/`safe_method_call`),
/// extracts the call metadata and infers type_args. Also recurses into all
/// sub-expressions to ensure nested calls are fully collected.
fn walk_expr<'a>(
    expr: ExprId,
    ctx: &mut WalkCtx<'a>,
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    // Copy the immutable-reference fields first, then use
    // `&mut ctx.in_progress` (split borrow).
    let ast = ctx.ast;
    let func_decls = &ctx.func_decls;
    let func_arenas = &ctx.func_arenas;
    let func_module_names = &ctx.func_module_names;
    let node = &ast.expr(expr).node;
    match node {
        // ── Call expressions: primary collection target ──
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

        // ── Unary/binary/assignment ──
        Expr::Binary { op: _, lhs, rhs } => {
            walk_expr(*lhs, ctx, sema_result, arena);
            walk_expr(*rhs, ctx, sema_result, arena);
        }
        Expr::Unary { operand, .. } => walk_expr(*operand, ctx, sema_result, arena),
        Expr::As { expr, .. } => walk_expr(*expr, ctx, sema_result, arena),
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

        // ── Field access and indexing ──
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

        // ── Container literals ──
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

        // ── Control flow ──
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

        // ── Concurrency/async ──
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

        // ── String interpolation ──
        Expr::StrInterp(parts) => {
            for part in parts {
                if let InterpolationPart::Expression(e) = part {
                    walk_expr(*e, ctx, sema_result, arena);
                }
            }
        }

        // ── Inline trait value: method bodies may contain generic calls ──
        Expr::InlineTrait(methods) => {
            for method in methods {
                if let Some(body) = method.body {
                    walk_expr(body, ctx, sema_result, arena);
                }
            }
        }

        // ── Terminal nodes: no recursion needed ──
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

// ── Main entry point: collect_monomorph_instances ──

/// Collect all generic call sites in a module, producing the monomorphization
/// instance set.
///
/// v3 spec §5.2 `collectMonomorphInstances` algorithm:
/// 1. Build the `func_name → fun_decl` mapping, used to look up parameter type
///    annotations while inferring type_args.
/// 2. Walk all top-level declarations:
///    a. Non-generic `fun_decl` → create an empty-type_args instance.
///    b. All `fun_decl` bodies / `type_decl` method bodies / `expr_decl` →
///       recursive traversal.
/// 3. For each generic call site: infer type_args → deduplicate → create the
///    instance → record the call-site mapping.
///
/// Generic functions themselves do not create empty instances (their concrete
/// instances are driven by call sites). Full trait-dispatch resolution for
/// method calls is deferred to a later phase; only best-effort matching is
/// done here.
pub fn collect_monomorph_instances<'a>(
    module: &'a Module<'a>,
    all_modules: &[&'a Module<'a>],
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    // Register built-in type method signatures (synthetic TypeDefInfo) so
    // method lookup goes through the unified (type_id, method_idx) path.
    register_builtin_method_sigs(sema_result);

    let mut ctx = WalkCtx {
        ast: &module.arena,
        func_decls: FxHashMap::default(),
        func_arenas: FxHashMap::default(),
        func_module_names: FxHashMap::default(),
        in_progress: FxHashMap::default(),
        module_name: module.name,
    };

    // 1. Build func_name → &Spanned<Decl> + owning module arena + owning module
    //    name mappings (collect top-level fun_decls across modules).
    // Cross-module monomorphization: when calling `std.math.Math.abs<T>(x)`,
    // `func_decls` must find `abs` (defined in the Math module), `func_arenas`
    // provides the arena of abs's module so `get_or_create_instance` can
    // dereference body ExprIds, and `func_module_names` provides abs's module
    // name so the `expr_types` key matches the IR Builder lookup.
    for m in all_modules {
        for decl in &m.declarations {
            if let Decl::FunDecl { name, .. } = &decl.node {
                ctx.func_decls.insert(name, decl);
                ctx.func_arenas.insert(name, &m.arena);
                ctx.func_module_names.insert(name, m.name);
            }
        }
    }

    // 2. Walk all top-level declarations.
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
                // Non-generic function: create an empty-type_args instance.
                // (Generic functions do not create empty instances; they are
                // driven by call sites.)
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
                        arena.make(crate::types::Type::Unknown)
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
                    let mono_idx = instance.instance_id;
                    sema_result.monomorph_instances.push(instance);
                    // Record module ownership for incremental purge (monomorph index).
                    sema_result.module_ownership.monomorph_indices
                        .entry(module.name.to_string())
                        .or_default()
                        .insert(mono_idx);

                    // Non-generic function: walk the function body to collect
                    // generic call sites.
                    // (Call sites inside a generic function body are discovered
                    // in `resolveInstanceBodyTypes`, since they depend on the
                    // current instance's type_args context to correctly infer
                    // type arguments.)
                    walk_expr(*body, &mut ctx, sema_result, arena);
                }
                let _ = params; // params unused in the non-generic branch
            }

            // Type declaration: walk non-generic method bodies (generic method
            // bodies are discovered during instantiation).
            Decl::TypeDecl { methods, .. } => {
                // Collect the method bodies to walk, avoiding `&mut sema_result`
                // while borrowing `methods`.
                let bodies: Vec<ExprId> = methods
                    .iter()
                    .filter(|m| m.type_params.is_empty())
                    .filter_map(|m| m.body)
                    .collect();
                for body in bodies {
                    walk_expr(body, &mut ctx, sema_result, arena);
                }
            }

            // Top-level expression declaration: walk the expression.
            Decl::ExprDecl { expr, stmt } => {
                walk_expr(*expr, &mut ctx, sema_result, arena);
                if let Some(s) = stmt {
                    walk_stmt(*s, &mut ctx, sema_result, arena);
                }
            }

            // import / pack: no traversal needed.
            _ => {}
        }
    }
}

// ── Instance body type resolution ──

/// Resolve all expression types in the function body using the concrete
/// type_args.
///
/// Recursively walks the function body AST, computing the type of each
/// expression and storing it in `instance.expr_types`. For `field_access`
/// expressions, additionally stores info in `instance.field_accesses`.
/// Replaces the IRBuilder `infer*` family of functions.
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

    // ── Step 1: resolve body types using InferContext instantiation mode ──
    // Create a temporary InferContext, replace the rigid TypeVars with the
    // concrete type_args, and run `infer_expr` once to write all expression
    // types into `local_expr_types`.
    let local_expr_types: FxHashMap<u64, ExprInfo>;
    {
        let mut infer_ctx = InferContext::new(arena, sema_result);
        infer_ctx.current_module_name = module_name.to_string();

        // Push type bindings (place rigid vars first; `enter_instantiation_mode`
        // will replace them with the concrete type_args).
        let type_param_names: Vec<&str> = fd.type_params.iter().map(|tp| tp.name).collect();
        infer_ctx.push_type_bindings(
            &type_param_names.iter().map(|&name| (name, None)).collect::<Vec<_>>(),
        );

        // Enter instantiation mode: rigid var → concrete type_args.
        infer_ctx.enter_instantiation_mode(
            instance.func_name.clone(),
            type_args.to_vec().into_boxed_slice(),
            &type_param_names,
            module_name.to_string(),
            in_progress.clone(),
        );

        // Create the environment and register the function parameters.
        let fn_env = infer_ctx.env.root();
        for param in fd.params {
            let h = if let Some(ta) = param.type_annotation {
                infer_ctx.type_from_ast(ta, ast)
            } else {
                infer_ctx.arena.fresh_type_var()
            };
            infer_ctx.env.define(fn_env, param.name, h);
        }

        // Set the return type.
        let ret_ty = if let Some(rt) = fd.return_type {
            infer_ctx.type_from_ast(rt, ast)
        } else {
            infer_ctx.arena.fresh_type_var()
        };
        infer_ctx.expected_return = Some(ret_ty);

        // Infer the function body (in instantiation mode `unify_or_constrain`
        // is skipped; `store_expr_info` writes to `local_expr_types`).
        let _ = infer_ctx.infer_expr(fd.body, ast, fn_env, infer_ctx.expected_return);

        // Leave instantiation mode and retrieve the stashed types.
        if let Some((let_local, _field_accesses, let_in_progress)) =
            infer_ctx.leave_instantiation_mode()
        {
            local_expr_types = let_local;
            *in_progress = let_in_progress;
        } else {
            local_expr_types = FxHashMap::default();
        }

        infer_ctx.pop_type_bindings();
    } // infer_ctx dropped, releasing &mut arena and &mut sema_result

    // Merge `local_expr_types` into `instance.expr_types` +
    // `sema_result.expr_types`.
    // `instance.expr_types`: instance-local lookup (used by the IR Builder).
    // `sema_result.expr_types`: global lookup (used by `process_call` →
    // `infer_type_args`).
    for (key, info) in &local_expr_types {
        instance.expr_types.insert(*key, info.clone());
    }
    for (key, info) in local_expr_types {
        sema_result.put_expr(key, info);
    }

    // ── Step 2: use `walk_expr` + `process_call` to trigger monomorphization of
    //    nested calls ──
    // Reuse the top-level path: `walk_expr` walks the function body; on
    // Call/MethodCall it calls `process_call`, which looks up argument types
    // from `sema_result.expr_types` (written in step 1), infers type_args, and
    // creates nested instances.
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

// ====== New trait-default-method monomorphization logic below ======

/// Reflect auto-impl method names: every receiver type gets these regardless of
/// implemented traits, so `override fun type_name()` etc. may legitimately have
/// no trait default to override.
fn is_reflect_method_name(name: &str) -> bool {
    matches!(
        name,
        "repr" | "type_name" | "kind" | "constructor" | "size" | "alignment" | "field_count" | "field_name"
    )
}

/// Validate the method ↔ trait-default bindings of every type declaration in
/// the module. Enforces the override/delegate system:
///
/// - R1: a declared method with a body that shadows an implemented trait's
///   default must carry `override` (shadowing by accident is an error).
/// - R2: `override` must target an implemented trait's *default* (a method with
///   a body); implementing an abstract trait method is not overriding. Reflect
///   auto-impl names are exempt (they override an implicit builtin, not a trait).
/// - R3: a method name inherited from multiple implemented traits' defaults
///   without any declaration is ambiguous — it must be resolved by an explicit
///   override or a delegate (`fun m(...): R = A.m`).
/// - R4: a delegate `= A.m` must name an implemented trait, use the same method
///   name, target a defaulted method, and a pure delegation (no body) must not
///   carry `override`.
///
/// Runs after all declarations of the module are checked (trait definitions are
/// registered regardless of declaration order), before
/// `collect_trait_default_instances`.
pub fn validate_trait_method_bindings<'a>(
    module: &'a Module<'a>,
    sema_result: &mut SemaResult,
) {
    struct Finding(Box<str>, u32, u32);
    let mut findings: Vec<Finding> = Vec::new();

    for decl in &module.declarations {
        let crate::ast::Ast::Decl::TypeDecl { name, implemented_traits, methods, .. } = &decl.node
        else {
            continue;
        };
        // Implemented trait names, deduped in declaration order.
        let mut traits: Vec<Box<str>> = Vec::new();
        for it in implemented_traits.iter() {
            let t: Box<str> = it.trait_name.into();
            if !traits.iter().any(|x| x.as_ref() == t.as_ref()) {
                traits.push(t);
            }
        }
        // Union of method names across the implemented traits (owned, to release
        // the sema_result borrow before reporting).
        let method_names: Vec<String> = {
            let mut ns: Vec<String> = Vec::new();
            for t in &traits {
                if let Some(td) = sema_result.get_trait_def(t) {
                    for m in td.methods.iter() {
                        if !ns.iter().any(|x| x == m.name.as_ref()) {
                            ns.push(m.name.to_string());
                        }
                    }
                }
            }
            ns
        };

        for m_name in &method_names {
            let declared = methods.iter().find(|m| m.name == m_name.as_str());
            // Implemented traits providing a default (bodied) method of this name.
            let providers: Vec<String> = traits
                .iter()
                .filter(|t| {
                    sema_result
                        .get_trait_def(t)
                        .map(|td| {
                            td.methods
                                .iter()
                                .any(|m| m.name.as_ref() == m_name.as_str() && m.has_body)
                        })
                        .unwrap_or(false)
                })
                .map(|t| t.to_string())
                .collect();
            let (line, column) = (decl.span.line, decl.span.column);

            if let Some(m) = declared {
                let (has_body, is_override) = (m.body.is_some(), m.is_override);

                // R4: delegate target validation.
                if let Some(d) = m.delegate.as_ref() {
                    if !traits.iter().any(|t| t.as_ref() == d.trait_name) {
                        findings.push(Finding(
                            format!(
                                "type '{}' does not implement trait '{}'; cannot delegate '{}'",
                                name, d.trait_name, m_name
                            )
                            .into(),
                            line,
                            column,
                        ));
                    } else if d.method_name != m_name.as_str() {
                        findings.push(Finding(
                            format!(
                                "delegate target must use the same method name: '{}.{}' on '{}.{}'",
                                d.trait_name, d.method_name, name, m_name
                            )
                            .into(),
                            line,
                            column,
                        ));
                    } else {
                        let has_default = sema_result
                            .get_trait_def(d.trait_name)
                            .map(|td| {
                                td.methods
                                    .iter()
                                    .any(|m| m.name.as_ref() == m_name.as_str() && m.has_body)
                            })
                            .unwrap_or(false);
                        if !has_default {
                            findings.push(Finding(
                                format!(
                                    "trait '{}' has no default implementation of '{}'; a delegate must target a trait default",
                                    d.trait_name, m_name
                                )
                                .into(),
                                line,
                                column,
                            ));
                        }
                    }
                    if is_override && !has_body {
                        findings.push(Finding(
                            format!(
                                "pure delegation of '{}.{}' inherits the default; remove 'override'",
                                name, m_name
                            )
                            .into(),
                            line,
                            column,
                        ));
                    }
                }

                // R2: `override` must target an implemented trait default.
                if is_override && providers.is_empty() && !is_reflect_method_name(m_name) {
                    findings.push(Finding(
                        format!(
                            "'override' on '{}.{}' but no implemented trait provides a default '{}'",
                            name, m_name, m_name
                        )
                        .into(),
                        line,
                        column,
                    ));
                }

                // R1: shadowing a default with a body requires `override`.
                if has_body && !is_override && !providers.is_empty() {
                    findings.push(Finding(
                        format!(
                            "'{}.{}' overrides a trait default (from [{}]); mark it 'override'",
                            name,
                            m_name,
                            providers.join(", ")
                        )
                        .into(),
                        line,
                        column,
                    ));
                }
            } else if providers.len() > 1 {
                // R3: ambiguous inherited default.
                findings.push(Finding(
                    format!(
                        "ambiguous trait default '{}': type '{}' inherits conflicting defaults from [{}]; resolve with an explicit override or a delegate ('fun {}(...): ... = {}.{}')",
                        m_name,
                        name,
                        providers.join(", "),
                        m_name,
                        providers[0],
                        m_name
                    )
                    .into(),
                    line,
                    column,
                ));
            }
        }
    }

    for f in findings {
        sema_result.add_error(crate::sema::Sema::SemaError {
            message: f.0,
            line: f.1,
            column: f.2,
            file_path: None,
        });
    }
}

/// Collect trait-default-method monomorphization instances.
///
/// Generates a `TraitDefaultInstance` entry for each type whose *bound* trait
/// default needs a specialized subgraph. The binding comes from
/// `SemaResult::resolve_method_binding` (explicit delegate `= A.m` or unique
/// provider); instances are emitted only for the bound trait, which makes
/// multi-trait same-name conflicts deterministic instead of resolved by
/// trait-table iteration order.
///
/// An overriding type normally gets no instance (its own method_subgraphs entry
/// wins dispatch); the exception is `super`: when sema recorded the
/// (type, trait, method) triple in `super_targets`, the default subgraph must
/// exist even though the type overrides it.
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
            // Collect all types implementing this trait (from the
            // witness_table).
            let impl_entries: Vec<(u16, String)> = sema_result
                .witness_table
                .entries()
                .filter(|e| e.trait_name.as_ref() == *name)
                .filter_map(|e| {
                    // type_id → type_name (O(1) reverse-lookup via
                    // type_id_to_name index, replacing the former O(n) linear
                    // scan over type_defs).
                    sema_result.type_id_to_name.get(&e.type_id)
                        .map(|name| (e.type_id, name.to_string()))
                })
                .collect();

            for (method_idx, method) in methods.iter().enumerate() {
                if method.body.is_none() {
                    continue;
                }
                let method_name: &str = method.name.as_ref();
                for (type_id, type_name) in &impl_entries {
                    // The type's implemented trait names (witness table).
                    let traits: Vec<Box<str>> = sema_result
                        .witness_table
                        .entries()
                        .filter(|e| e.type_id == *type_id)
                        .map(|e| e.trait_name.clone())
                        .collect();
                    match sema_result.resolve_method_binding(&traits, type_name, method_name) {
                        crate::sema::Sema::MethodBinding::Bound { trait_name, overridden }
                            if trait_name.as_ref() == *name =>
                        {
                            if overridden
                                && !sema_result.super_targets.contains(&(
                                    type_name.as_str().into(),
                                    trait_name.clone(),
                                    method_name.into(),
                                ))
                            {
                                // The type overrides the default and never calls
                                // super into it — no specialized default needed.
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
                        _ => continue,
                    }
                }
            }
        }
    }
}

/// Unified post-sema monomorphization entry point.
///
/// Called after the Sema phase completes; runs two kinds of monomorphization
/// instance collection:
/// 1. `collect_monomorph_instances`: call-site-driven monomorphization of
///    generic functions.
/// 2. `collect_trait_default_instances`: specialization of trait default
///    methods per implementing type.
///
/// The IR layer (IrBuilder) consumes `monomorph_instances` and
/// `trait_default_instances` in `SemaResult` to generate the corresponding
/// specialized subgraphs.
pub fn run_monomorphization<'a>(
    module: &'a Module<'a>,
    all_modules: &[&'a Module<'a>],
    sema_result: &mut SemaResult,
    arena: &mut TypeArena,
) {
    collect_monomorph_instances(module, all_modules, sema_result, arena);
    collect_trait_default_instances(module, sema_result);
}
