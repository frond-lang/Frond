//! CallInfer — Call / method-call inference and member type lookup. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Infer an `Expr::Call` expression (extracted from `infer_expr_inner`).
    pub(super) fn infer_call_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::Call { callee, args, .. } => {
                // Scalar type name used as a constructor call (e.g. `u64(x)`): scalar types
                // have no constructors — the correct spelling is a cast. This must run
                // FIRST: a scalar type name otherwise infers as a builtin conversion
                // function type (Fn(scalar) -> scalar), which silently compiles to a
                // target-less Call node evaluating to void (the `__read_u64_le` bug class).
                if let Expr::Ident(name) = &ast.expr(*callee).node {
                    if let Some(tag) = crate::value::ValueTag::from_name(name) {
                        if tag.is_int() || tag.is_float()
                            || matches!(tag, crate::value::ValueTag::Bool | crate::value::ValueTag::Char)
                        {
                            let span = ast.expr(expr).span;
                            self.add_error_at(
                                &format!(
                                    "'{}' is a scalar type and has no constructor; use 'x as {}' for conversion",
                                    name, name
                                ),
                                span.line,
                                span.column,
                            );
                            for &a in args.iter() {
                                let _ = self.infer_expr(a, ast, env, None);
                            }
                            return self.arena.make(Type::Unknown);
                        }
                    }
                }
                // ── Constructor multi-mapping disambiguation ──
                // When callee is an Ident that maps to multiple same-named constructors, disambiguate by priority:
                //   1. Type-oriented: when expected_ty is an Adt, select by type_name
                //   2. Arity: when type-oriented disambiguation fails (expected is a TypeVar or not provided),
                //      select the unique constructor matching by arity
                let callee_ty = if let Expr::Ident(name) = &ast.expr(*callee).node {
                    let ctors = self.sema_result.get_ctor_defs(name);
                    // Privacy gate for BARE constructor calls (same-module calls
                    // pass naturally inside ctor_privacy_error): cross-module
                    // construction through a constructor with any private field
                    // is rejected. Single mapping is decided here; ambiguous
                    // mappings are gated after disambiguation below.
                    if ctors.len() == 1 {
                        let c = &ctors[0];
                        let gate = self.ctor_privacy_error(&c.type_name, &c.name);
                        if let Some(msg) = gate {
                            let span = ast.expr(expr).span;
                            self.add_error_at(&msg, span.line, span.column);
                            return self.arena.fresh_type_var();
                        }
                    }
                    if ctors.len() > 1 {
                        let selected: Option<(Box<str>, Box<[TypeRepr]>)> = {
                            let mut found: Option<&CtorDefInfo> = None;
                            // 1. Type-oriented disambiguation
                            if let Some(exp) = expected {
                                let exp_resolved = self.arena.resolve(exp);
                                if let Type::Adt(_) = self.arena.get(exp_resolved) {
                                    let (exp_type_name, _) = self.arena.adt_parts(exp_resolved);
                                    let matches: Vec<_> = ctors.iter()
                                        .filter(|c| c.type_name.as_ref() == exp_type_name)
                                        .collect();
                                    if matches.len() == 1 {
                                        found = Some(matches[0]);
                                    }
                                }
                            }
                            // 2. Arity disambiguation (fallback when type-oriented fails)
                            if found.is_none() {
                                let arity_matches: Vec<_> = ctors.iter()
                                    .filter(|c| c.field_type_reprs.len() == args.len())
                                    .collect();
                                if arity_matches.len() == 1 {
                                    found = Some(arity_matches[0]);
                                }
                            }
                            found.map(|c| (c.type_name.clone(), c.field_type_reprs.clone()))
                        };
                        match selected {
                            Some((type_name, field_type_reprs)) => {
                                // Privacy gate: ambiguous mapping resolved to a
                                // cross-module constructor with private fields.
                                if let Some(msg) = self.ctor_privacy_error(&type_name, name) {
                                    let span = ast.expr(expr).span;
                                    self.add_error_at(&msg, span.line, span.column);
                                    return self.arena.fresh_type_var();
                                }
                                let param_types: Vec<TypeHandle> = field_type_reprs
                                    .iter()
                                    .map(|r| self.type_repr_to_handle(r))
                                    .collect();
                                let ret_ty = self.arena.make_adt(type_name, Box::new([]));
                                if param_types.is_empty() {
                                    ret_ty
                                } else {
                                    self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
                                }
                            }
                            None => {
                                let span = ast.expr(expr).span;
                                let type_names: Vec<&str> = ctors.iter()
                                    .map(|c| c.type_name.as_ref())
                                    .collect();
                                self.add_error_at(
                                    &format!(
                                        "ambiguous constructor '{}': defined by types [{}]; use Type.{} to disambiguate or provide a type context",
                                        name,
                                        type_names.join(", "),
                                        name,
                                    ),
                                    span.line,
                                    span.column,
                                );
                                self.arena.fresh_type_var()
                            }
                        }
                    } else if ctors.len() == 1
                        && args.is_empty()
                        && ctors[0].field_type_reprs.is_empty()
                    {
                        // Bug #69: Zero-arg constructor called with `()` syntax.
                        // Zero-arg constructors are registered as values (ADT type), not
                        // function types, so `Unit()` is equivalent to the bare value `Unit`.
                        let ret_ty = self.arena.make_adt(
                            ctors[0].type_name.clone(),
                            Box::new([]),
                        );
                        if let Some(exp) = expected {
                            self.unify_or_constrain(ret_ty, exp);
                        }
                        ret_ty
                    } else {
                        // [Implicit this] Try resolving as this.method(args) before
                        // falling through to infer_expr (which would report undefined).
                        if let Some(this_ty) = self.current_this_type() {
                            let call_span = ast.expr(expr).span;
                            if let Some(fn_ty) = self.lookup_method_type(this_ty, name, call_span.line, call_span.column) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    // Skip params[0] (this), match args with params[1..].
                                    let n = params.len().min(args.len() + 1);
                                    for i in 1..n {
                                        let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                                        let sp = ast.expr(args[i - 1]).span;
                                        self.unify_call_arg(params[i], arg_ty, sp.line, sp.column);
                                    }
                                    // Store callee's ExprInfo so that pending_implicit_this
                                    // (flushed in infer_expr) can attach the implicit_this marker.
                                    // Without this, the marker is lost because we bypass
                                    // infer_expr(callee) on this fast path.
                                    self.store_expr_info(*callee, fn_ty);
                                    self.pending_implicit_this = Some((
                                        *callee,
                                        crate::sema::Sema::ImplicitThisAccess::Method((*name).to_string().into_boxed_str()),
                                    ));
                                    return return_type;
                                }
                            }
                        }
                        self.infer_expr(*callee, ast, env, None)
                    }
                } else {
                    self.infer_expr(*callee, ast, env, None)
                };
                let resolved_callee = self.arena.resolve(callee_ty);

                // Instantiation mode: skip HM unify (types were already checked in the sema HM stage);
                // only infer argument types and return the return type. Monomorphization triggers are orchestrated externally.
                if self.instantiation_ctx.is_some() {
                    // ModuleRef call: look up the function signature from the module env.
                    if let Type::ModuleRef(_) = self.arena.get(resolved_callee) {
                        let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                        if let Some(func_name) = path.rsplit('.').next() {
                            if let Some(fn_ty) = self.sema_result.env.lookup_local(module_env, func_name) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                    // Ordinary function call: infer argument types and return the return type.
                    let inst_callee = self.instantiate_fn_type(resolved_callee);
                    if let Type::Fn(_) = self.arena.get(inst_callee) {
                        let (params, return_type) = self.arena.fn_parts(inst_callee);
                        let params: Vec<TypeHandle> = params.to_vec();
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                        }
                        return return_type;
                    }
                    // Non-Fn callee: report an error and return Unknown.
                    let span = ast.expr(expr).span;
                    let callee_name = self
                        .arena
                        .type_name(resolved_callee)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("{:?}", self.arena.get(resolved_callee)));
                    self.add_error_at(
                        &format!("cannot call non-function value of type '{}'", callee_name),
                        span.line,
                        span.column,
                    );
                    for &a in args.iter() {
                        let _ = self.infer_expr(a, ast, env, None);
                    }
                    return self.arena.make(Type::Unknown);
                }

                // ModuleRef call: callee is a module path reference (e.g. "std.reflect.Reflect.format");
                // look up the function signature by its trailing bare name directly in the module env carried by the ModuleRef (no parent-env traversal).
                if let Type::ModuleRef(_) = self.arena.get(resolved_callee) {
                    let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                    // The trailing segment is the function name (e.g. "std.reflect.Reflect.format" → "format").
                    if let Some(func_name) = path.rsplit('.').next() {
                        if let Some(fn_ty) = self.sema_result.env.lookup_local(module_env, func_name) {
                            // Instantiate the polymorphic function type to avoid type-constraint clashes across calls.
                            let inst_fn = self.instantiate_fn_type(fn_ty);
                            if let Type::Fn(_) = self.arena.get(inst_fn) {
                                let (params, return_type) = self.arena.fn_parts(inst_fn);
                                let params: Vec<TypeHandle> = params.to_vec();
                                if params.len() == args.len() {
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                                        if let Err(e) = self.try_widen_unify(param_ty, arg_ty) {
                                            self.add_error(&format!("argument type incompatible with parameter type: {}", e));
                                        }
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // Instantiate the polymorphic function type (replace rigid vars / unbound TypeVars with fresh non-rigid vars)
                // so each call has its own type variables, avoiding type-constraint clashes across calls.
                let inst_callee = self.instantiate_fn_type(resolved_callee);
                if let Type::Fn(_) = self.arena.get(inst_callee) {
                    let (params, return_type) = self.arena.fn_parts(inst_callee);
                    let params: Vec<TypeHandle> = params.to_vec();
                    if params.len() == args.len() {
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                            // Hard-concrete mismatch fails now; TypeVars keep the
                            // constraint path (see unify_call_arg).
                            let sp = ast.expr(arg).span;
                            self.unify_call_arg(param_ty, arg_ty, sp.line, sp.column);
                        }
                    }
                    // Always return the declared return type, to avoid cascading type loss from argument mismatches.
                    // If there is an expected type, unify the return type with it to solve pending TypeVars in the return type
                    // (e.g. Ok(void) returns Throw<void, '_E>; expected=Throw<void, IOError> solves E=IOError).
                    if let Some(exp) = expected {
                        self.unify_or_constrain(return_type, exp);
                    }
                    return return_type;
                }
                // Fallback: infer all arguments and unify the callee with (args -> ret).
                // (Scalar-name constructors are rejected at the top of this arm.)
                let ret_ty = self.arena.fresh_type_var();
                let arg_types: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.infer_expr(a, ast, env, None))
                    .collect();
                let expected_fn = self.arena.make_fn(
                    arg_types.into_boxed_slice(),
                    ret_ty,
                );
                self.unify_or_constrain(callee_ty, expected_fn);
                ret_ty
            }
            _ => unreachable!("infer_call_expr called on non-Call expression"),
        }
    }

    /// Infer an `Expr::MethodCall` / `Expr::SafeMethodCall` expression (extracted from `infer_expr_inner`).
    pub(super) fn infer_method_call_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        match &ast.expr(expr).node {
            Expr::MethodCall { recv, method, args, .. }
            | Expr::SafeMethodCall { recv, method, args, .. } => {
                // super.method(args): static dispatch to the bound trait-default
                // layer of the enclosing type. Must be handled before anything
                // that infers the receiver (`super` is not a value and resolves
                // in no environment).
                if let Expr::Ident("super") = &ast.expr(*recv).node {
                    return self.infer_super_method_call(expr, ast, env, expected, method, args);
                }
                // Lib.open(path) / Lib.embed(path): builtin native-library constructors.
                // The receiver `Lib` is a type name, not a value, so this must run
                // BEFORE receiver inference. A local binding named `Lib` shadows it.
                if let Expr::Ident("Lib") = &ast.expr(*recv).node {
                    let shadowed = self
                        .sema_result.env
                        .lookup_with_pred(env, "Lib", |_| true)
                        .is_some();
                    if !shadowed && (*method == "open" || *method == "embed") {
                        let span = ast.expr(expr).span;
                        if args.len() != 1 {
                            self.add_error_at(
                                &format!("Lib.{} takes exactly 1 argument (path: str), got {}", method, args.len()),
                                span.line,
                                span.column,
                            );
                            return self.arena.fresh_type_var();
                        }
                        let str_ty = self.make_builtin(Type::Str);
                        let arg_ty = self.infer_expr(args[0], ast, env, Some(str_ty));
                        self.unify_or_constrain(str_ty, arg_ty);
                        let lib_ty = self.make_builtin(Type::Lib);
                        let err_ty = self.ffi_error_ty();
                        let ret = self.arena.make_throw(lib_ty, err_ty);
                        // Mark recv as module-func-recv (IR compilation skips the recv node).
                        let recv_key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            recv.0 as u64,
                        );
                        self.sema_result.module_func_recv_exprs.insert(recv_key);
                        return ret;
                    }
                }

                // Qualified-name syntax: Type.Ctor(args) (qualified call of a constructor with arguments)
                if let Expr::Ident(type_name) = &ast.expr(*recv).node {
                    if let Some((ctor_type_name, field_type_reprs)) =
                        self.check_qualified_ctor(type_name, method)
                    {
                        // Privacy gate: cross-module construction through a
                        // constructor with any private field is rejected.
                        if let Some(msg) = self.ctor_privacy_error(type_name, method) {
                            let span = ast.expr(expr).span;
                            self.add_error_at(&msg, span.line, span.column);
                            return self.arena.fresh_type_var();
                        }
                        if !field_type_reprs.is_empty() {
                            // Constructor with arguments: build a function type and go through call inference
                            let param_types: Vec<TypeHandle> = field_type_reprs
                                .iter()
                                .map(|r| self.type_repr_to_handle(r))
                                .collect();
                            let ret_ty = self.arena.make_adt(ctor_type_name, Box::new([]));
                            let fn_ty = self.arena.make_fn(
                                param_types.into_boxed_slice(),
                                ret_ty,
                            );
                            let (params, return_type) = self.arena.fn_parts(fn_ty);
                            let params: Vec<TypeHandle> = params.to_vec();
                            if params.len() == args.len() {
                                for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                    let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                                    self.unify_or_constrain(param_ty, arg_ty);
                                }
                            }
                            if let Some(exp) = expected {
                                self.unify_or_constrain(return_type, exp);
                            }
                            // Mark recv as module-func-recv (skip recv during IR compilation)
                            let recv_key = crate::sema::Sema::module_expr_key(
                                &self.current_module_name,
                                recv.0 as u64,
                            );
                            self.sema_result.module_func_recv_exprs.insert(recv_key);
                            return return_type;
                        }
                        // Zero-argument constructor in MethodCall: report an error
                        let span = ast.expr(expr).span;
                        self.add_error_at(
                            &format!(
                                "constructor '{}' of type '{}' takes no arguments; use {}.{} syntax",
                                method, type_name, type_name, method
                            ),
                            span.line,
                            span.column,
                        );
                        return self.arena.fresh_type_var();
                    }
                }

                let recv_ty = self.infer_expr(*recv, ast, env, None);

                // Path 0a: ModuleRef recv → module-path function call.
                // When recv is a ModuleRef (e.g. std.net.UdpSocket), method is a top-level function in that module;
                // look it up by its bare name directly in the module env carried by the ModuleRef (no parent-env traversal).
                let recv_resolved_0a = self.arena.resolve(recv_ty);
                if let Type::ModuleRef(_) = self.arena.get(recv_resolved_0a) {
                    let (mod_path, module_env) = self.arena.module_ref_parts(recv_resolved_0a);
                    let found = self.sema_result.env.lookup_local(module_env, method);
                    // Directory-module semantics: when lookup_local misses in the current module env,
                    // search sibling modules in the same directory (e.g. Math.sqrt where sqrt lives in Power.frond,
                    // with Math and Power both under the std.math directory).
                    let found = found.or_else(|| {
                        self.lookup_sibling_module_fn(mod_path, module_env, method)
                    });
                    if let Some(fn_ty) = found {
                        let inst_fn = self.instantiate_fn_type(fn_ty);
                        if let Type::Fn(_) = self.arena.get(inst_fn) {
                            let (params, return_type) = self.arena.fn_parts(inst_fn);
                            let params: Vec<TypeHandle> = params.to_vec();
                            let n = params.len().min(args.len());
                            for i in 0..n {
                                let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                let sp = ast.expr(args[i]).span;
                                self.unify_call_arg(params[i], arg_ty, sp.line, sp.column);
                            }
                            // Mark recv as a module-function-call receiver so IR compilation does not pass recv.
                            // (Consistent with path 0b: ModuleRef recv has Module.fun(args) semantics.)
                            let recv_key = module_expr_key(
                                &self.current_module_name,
                                recv.0 as u64,
                            );
                            self.sema_result.module_func_recv_exprs.insert(recv_key);
                            return return_type;
                        }
                    }
                }

                // Path 0b: constructor recv (type name == module name) → module function call (Zig-style @This semantics).
                // When recv is a type constructor (Fn, with return_type Adt) and the type name matches a module name,
                // look up free functions by the method's bare name in that module's env.
                // Typical scenario: after `import std.time.Duration`, Duration.from_millis(100),
                // where Duration is both a type and a module (file with the same name; predefine redefine overwrote the ModuleRef).
                if let Type::Fn(_) = self.arena.get(recv_resolved_0a) {
                    let (_, ret_ty) = self.arena.fn_parts(recv_resolved_0a);
                    let ret_resolved = self.arena.resolve(ret_ty);
                    if let Type::Adt(_) = self.arena.get(ret_resolved) {
                        let (type_name, _) = self.arena.adt_parts(ret_resolved);
                        if let Some(&mod_env) = self.ctor_module_envs.get(type_name) {
                            if let Some(fn_ty) = self.sema_result.env.lookup_local(mod_env, method) {
                                // Guard: `TypeName.method(args)` must not dispatch to a TYPE
                                // method. Type methods carry an implicit `this` at param slot 0
                                // while this call site passes no receiver, so the IR would land
                                // every argument one slot left of its parameter (silent garbage).
                                // Type methods are callable via an instance `x.m(args)` or the
                                // bare form `m(recv, args)`; factories belong at module level
                                // (e.g. Instant.now()).
                                let is_type_method = self.sema_result.get_type_def(type_name)
                                    .map(|def| def.methods.iter().any(|m| m.name.as_ref() == *method))
                                    .unwrap_or(false);                                if is_type_method {
                                    let span = ast.expr(expr).span;
                                    self.add_error_at(
                                        &format!(
                                            "method '{}.{}' cannot be called through the type name; call it on an instance (x.{}(..)) or via the bare form {}(instance, ..), or move the factory to module level",
                                            type_name, method, method, method
                                        ),
                                        span.line,
                                        span.column,
                                    );
                                    return self.arena.fresh_type_var();
                                }
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Type::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    let n = params.len().min(args.len());
                                    for i in 0..n {
                                        let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                        let sp = ast.expr(args[i]).span;
                                        self.unify_call_arg(params[i], arg_ty, sp.line, sp.column);
                                    }
                                    // Mark recv as a module-function-call receiver so IR compilation does not pass recv.
                                    let recv_key = module_expr_key(
                                        &self.current_module_name,
                                        recv.0 as u64,
                                    );
                                    self.sema_result.module_func_recv_exprs.insert(recv_key);
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // Language-level intrinsic tagging: await/recv are recognized uniformly by sema
                // and registered into method_dispatches for IR consumption (eliminates IR-side string guards).
                // await is a general suspend semantic (for all types); recv is tagged only for Channel/Receiver types.
                {
                    let intrinsic = if *method == "await" && args.is_empty() {
                        Some(crate::sema::Sema::IntrinsicKind::Await)
                    } else if *method == "recv" && args.is_empty() {
                        match self.arena.get(recv_resolved_0a) {
                            Type::Channel(_) | Type::Receiver(_) => {
                                Some(crate::sema::Sema::IntrinsicKind::ChannelAwait)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if intrinsic.is_some() {
                        let key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            expr.0 as u64,
                        );
                        self.sema_result.method_dispatches.insert(
                            key,
                            crate::sema::Sema::DispatchInfo {
                                trait_id: 0,
                                method_idx: 0,
                                impl_fn_idx: 0,
                                instance_id: 0,
                                intrinsic,
                            },
                        );
                    }
                }

                // Path 1 (preferred): type-aware method lookup.
                // lookup_method_type looks up the receiver's type against witness_table / func_sigs / builtin methods,
                // ensuring same-named methods (e.g. Instant.add_duration vs DateTime.add_duration) dispatch to the correct signature.
                let method_fn_ty = {
                    let call_span = ast.expr(expr).span;
                    self.lookup_method_type(recv_ty, method, call_span.line, call_span.column)
                };
                if let Some(fn_ty) = method_fn_ty {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Type::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // The first parameter is self; skip it.
                        let n = params.len().min(args.len() + 1);
                        for i in 1..n {
                            let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                            let sp = ast.expr(args[i - 1]).span;
                            self.unify_call_arg(params[i], arg_ty, sp.line, sp.column);
                        }
                        return return_type;
                    }
                }

                // Path 0 (fallback): look up a binding named after the method as an Fn type in env (free function with a self parameter).
                // Use lookup_with_pred to skip same-named non-function bindings (e.g. a local variable shadowing a free function).
                // In Frond `recv.method(args)` is sugar for `method(recv, args)`.
                if let Some(fn_ty) = self.sema_result.env.lookup_with_pred(env, method, |ty| {
                    let r = self.arena.resolve(ty);
                    matches!(self.arena.get(r), Type::Fn(_))
                }) {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Type::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // Candidacy gate (Bug #103): this fallback used to accept
                        // ANY same-named binding — receiver type and arity were
                        // never checked, so `x.format()` on i32 dispatched to
                        // std Format.format (stdlib is globally env-visible by
                        // design) and panicked at runtime. A binding is a
                        // candidate only if its arity matches AND its first
                        // parameter can accept the receiver: unify is the test
                        // (it succeeds for str ↔ T[] — byte semantics — so
                        // `"abc".iter()` keeps resolving here). A HARD failure
                        // on two concrete types rejects the candidate and lets
                        // resolution continue to the "no method" fallback;
                        // TypeVar/Unknown receivers keep the lenient
                        // constraint path (the solver may still bind them).
                        let arity_ok = params.len() == args.len() + 1;
                        let recv_ok = if params.is_empty() {
                            true
                        } else {
                            match self.arena.unify(params[0], recv_ty) {
                                Ok(_) => true,
                                Err(_) => {
                                    // Lenient only when a TypeVar survives ANYWHERE in
                                    // either type (e.g. iter<T>'s `T[]`: head is a concrete
                                    // Array but the element is fresh — the solver binds it
                                    // later, which is how `"abc".iter()` resolves). Two
                                    // fully-concrete types that cannot unify = hard reject.
                                    let pending = |t: TypeHandle| type_contains_typevar(&self.arena, t);
                                    if pending(params[0]) || pending(recv_ty) {
                                        self.unify_or_constrain(params[0], recv_ty);
                                        true
                                    } else {
                                        false
                                    }
                                }
                            }
                        };
                        if arity_ok && recv_ok {
                            // The first parameter is self/receiver: unify recv with params[0].
                            // This lets the free function's generic parameters be inferred from the receiver's type (e.g. iter<T> infers T from arr: T[]).
                            if !params.is_empty() {
                                self.unify_or_constrain(params[0], recv_ty);
                            }
                            // The remaining parameters are inferred from args.
                            let n = params.len().min(args.len() + 1);
                            for i in 1..n {
                                let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                                let sp = ast.expr(args[i - 1]).span;
                                self.unify_call_arg(params[i], arg_ty, sp.line, sp.column);
                            }
                            return return_type;
                        }
                        // Not a candidate: fall through to the paths below.
                    }
                }

                // await is a general suspend semantic: it produces no value; it only suspends the frame waiting for an event.
                // The IR layer uses infer_event_source_kind to decide the event-source kind based on the recv type
                // (AsyncJoin/Channel/Timer); the Sema layer uniformly returns void.
                if *method == "await" && args.is_empty() {
                    return self.make_builtin(Type::Void);
                }

                // reflect trait methods (auto-impl): any receiver type gets reflect methods
                // (format/type_name/kind/field_count/...) without explicit trait declaration.
                // This is the Sema-side recognition that pairs with Builder::lookup_intrinsic +
                // reflect_method_intrinsic — the method call type-checks here, and lowers to a
                // CF_REFLECT_* compute_fn at IR build time.
                if let Some(ret_ty) = self.reflect_method_return_type(*method, args.len(), recv_ty) {
                    // Infer args (for type-checking side effects) but discard their constraints.
                    for &a in args.iter() {
                        let _ = self.infer_expr(a, ast, env, None);
                    }
                    return ret_ty;
                }

                // Lib / ForeignFn builtin methods (structural, reflect-style):
                // lookup/has_symbol/close on `Lib`; call (any arity) on `ForeignFn[R]`.
                // Pairs with Builder::lib_method_intrinsic — the call type-checks here
                // and lowers to a CF_LIB_* / CF_FFN_CALL compute_fn at IR build time.
                {
                    let recv_resolved_l = self.arena.resolve(recv_ty);
                    let recv_ct = self.arena.get(recv_resolved_l);
                    let needs_lib_dispatch = matches!(recv_ct, Type::Lib)
                        || (matches!(recv_ct, Type::ForeignFn(_)) && *method == "call");
                    if needs_lib_dispatch {
                        for &a in args.iter() {
                            let _ = self.infer_expr(a, ast, env, None);
                        }
                        let ret_ty = self.lib_method_return_type(
                            recv_ct,
                            recv_resolved_l,
                            method,
                            args.len(),
                            expected,
                        );
                        return match ret_ty {
                            Some(t) => t,
                            None => {
                                // Wrong arity for the recognized Lib method.
                                let span = ast.expr(expr).span;
                                self.add_error_at(
                                    &format!("bad Lib method call '{}': lookup takes (name: str, args: str); has_symbol takes (name: str); close takes no args", method),
                                    span.line,
                                    span.column,
                                );
                                self.arena.fresh_type_var()
                            }
                        };
                    }
                }

                // Fallback: infer arguments and return a fresh var.
                // For a receiver whose type is already determined (not TypeVar/Unknown/Never), report "method does not exist"
                // to help the user locate the problem; for a TypeVar receiver, silently return a fresh var (inference pending, deferred to the solver).
                let span = ast.expr(expr).span;
                let recv_resolved = self.arena.resolve(recv_ty);
                match self.arena.get(recv_resolved) {
                    Type::TypeVar(_) | Type::Unknown | Type::Never => {
                        // Receiver type pending; silently return a fresh var.
                    }
                    Type::Void => {
                        // void receiver: handled by the IR layer (void method call).
                    }
                    ct => {
                        // Receiver type is determined but method lookup failed: report an error.
                        let recv_name = self.arena.type_name(recv_resolved)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{:?}", ct));
                        self.add_error_at(
                            &format!("no method '{}' on type '{}'", method, recv_name),
                            span.line,
                            span.column,
                        );
                    }
                }
                for &a in args.iter() {
                    let _ = self.infer_expr(a, ast, env, None);
                }
                self.arena.fresh_type_var()
            }
            _ => unreachable!("infer_method_call_expr called on non-MethodCall expression"),
        }
    }

    /// Builds a Type::Fn type from the owned data of a MethodSigInfo.
    /// Both parameters and the return type are fully resolved from TypeRepr via type_repr_to_handle,
    /// correctly handling nested generics (e.g. Async<Throw<T, E>>), arrays, Nullable, and other compound types,
    /// overcoming the limitation that type_name only stores the top-level name.
    pub(super) fn build_fn_type_from_sig(
        &mut self,
        param_type_reprs: Vec<TypeRepr>,
        return_type_repr: Option<TypeRepr>,
        _recv_ty: TypeHandle,
    ) -> TypeHandle {
        // ThisType is resolved by type_repr_to_handle via current_this_type();
        // the caller (lookup_method_type) has already pushed recv_ty as self_type.
        let params: Vec<TypeHandle> = param_type_reprs
            .iter()
            .map(|repr| self.type_repr_to_handle(repr))
            .collect();
        let return_type = match return_type_repr {
            Some(repr) => self.type_repr_to_handle(&repr),
            None => self.arena.fresh_type_var(),
        };
        self.arena.make_fn(params.into_boxed_slice(), return_type)
    }

    /// Constructs a TypeHandle from a self-contained TypeRepr (does not depend on AstArena references).
    /// Mirrors the logic of type_from_ast_with_params, but reads from TypeRepr instead of AST TypeNode.
    /// Used to restore cross-module method return types (MethodSigInfo.return_type_repr).
    pub(super) fn type_repr_to_handle(&mut self, repr: &TypeRepr) -> TypeHandle {
        match repr {
            TypeRepr::Named(name) => {
                let empty_map: FxHashMap<String, TypeHandle> = FxHashMap::default();
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name.as_ref(), &empty_map, &mut visiting)
            }
            TypeRepr::ThisType => match self.current_this_type() {
                Some(ty) => ty,
                None => self.arena.fresh_type_var(),
            },
            TypeRepr::Generic(name, args) => {
                let new_args: Vec<TypeHandle> =
                    args.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // Builtin generic types (Throw/Atomic/Async/Channel, etc.) construct dedicated Type variants.
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic(name.clone(), args_box);
                }
                // trait definition → Trait type.
                if self.sema_result.get_trait_def(name).is_some() {
                    return self.arena.make_trait(name.clone(), args_box);
                }
                // User-defined generic ADT.
                let has_type_params = self
                    .sema_result
                    .get_type_def(name)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt(name.clone(), args_box);
                }
                // Fallback: construct a Generic (may be undefined or a forward reference; reported on later use).
                self.arena.make_generic(name.clone(), args_box)
            }
            TypeRepr::Nullable(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_nullable(inner_ty)
            }
            TypeRepr::Ref(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, false)
            }
            TypeRepr::RawPtr(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, true)
            }
            TypeRepr::Function(params, return_type) => {
                let p: Vec<TypeHandle> =
                    params.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let r = self.type_repr_to_handle(return_type);
                self.arena.make_fn(p.into_boxed_slice(), r)
            }
            TypeRepr::Array(elem, _) => {
                let elem_ty = self.type_repr_to_handle(elem);
                self.arena.make_array(elem_ty, None)
            }
        }
    }

    /// Looks up the method signature for an object type (returns a function type whose first parameter is self).
    /// Infer `super.method(args)` — static dispatch to the bound trait-default
    /// implementation of the enclosing type.
    ///
    /// `super` is a layer view of `this`, not a value: the call resolves to the
    /// trait default that `method` is bound to on the enclosing type (explicit
    /// delegate `= A.m` or unique provider), bypassing the type's own override.
    /// The resolved `(trait_idx, method_idx)` is recorded into
    /// `sema_result.super_dispatches` for the IR builder, and the
    /// (type, trait, method) triple into `super_targets` so the monomorph phase
    /// generates the specialized default subgraph even for overriding types.
    fn infer_super_method_call(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
        method: &str,
        args: &[ExprId],
    ) -> TypeHandle {
        use crate::sema::Sema::MethodBinding;
        let span = ast.expr(expr).span;

        // Inside a trait default method, `super` would mean the parent trait's
        // default — trait parents are not implemented yet.
        if self.current_trait_name.is_some() {
            self.add_error_at(
                "super is not allowed inside a trait default method (trait parents are not implemented yet)",
                span.line,
                span.column,
            );
            return self.arena.fresh_type_var();
        }
        let traits: Vec<Box<str>> = match self.current_type_decl_traits.clone() {
            Some(t) => t,
            None => {
                self.add_error_at(
                    "super is only valid inside a type method",
                    span.line,
                    span.column,
                );
                return self.arena.fresh_type_var();
            }
        };
        // The enclosing type's name (from the pushed `this` type) — needed to
        // consult the declared methods' delegate annotations.
        let type_name: Box<str> = self
            .current_this_type()
            .and_then(|t| {
                let r = self.arena.resolve(t);
                match self.arena.get(r) {
                    Type::Adt(_) => {
                        let (n, _) = self.arena.adt_parts(r);
                        Some(n.into())
                    }
                    _ => None,
                }
            })
            .unwrap_or_default();
        if type_name.is_empty() {
            self.add_error_at(
                "super: the enclosing type could not be determined",
                span.line,
                span.column,
            );
            return self.arena.fresh_type_var();
        }

        let binding = self
            .sema_result
            .resolve_method_binding(&traits, &type_name, method);
        let trait_name: Box<str> = match binding {
            MethodBinding::Bound { trait_name, .. } => trait_name,
            MethodBinding::Ambiguous(list) => {
                self.add_error_at(
                    &format!(
                        "ambiguous super: '{}' has default implementations from [{}]; bind it explicitly on the method ('... = Trait.{}')",
                        method,
                        list.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(", "),
                        method,
                    ),
                    span.line,
                    span.column,
                );
                return self.arena.fresh_type_var();
            }
            MethodBinding::Unbound => {
                self.add_error_at(
                    &format!(
                        "no trait default '{}' is available via super on type '{}'",
                        method, type_name
                    ),
                    span.line,
                    span.column,
                );
                return self.arena.fresh_type_var();
            }
        };

        // Trait method signature (position + shape) for type-checking the call.
        let trait_idx = self.sema_result.trait_def_index.get(trait_name.as_ref()).copied();
        let sig = self.sema_result.get_trait_def(trait_name.as_ref()).and_then(|td| {
            td.methods
                .iter()
                .enumerate()
                .find(|(_, m)| m.name.as_ref() == method)
                .map(|(i, m)| (i as u16, m.param_count, m.return_type, m.has_body))
        });
        let (trait_idx, method_idx, param_count, return_type, has_body) =
            match (trait_idx, sig) {
                (Some(t), Some((mi, pc, rt, hb))) => (t, mi, pc, rt, hb),
                _ => {
                    self.add_error_at(
                        &format!(
                            "super: trait '{}' or its method '{}' could not be resolved",
                            trait_name, method
                        ),
                        span.line,
                        span.column,
                    );
                    return self.arena.fresh_type_var();
                }
            };
        if !has_body {
            self.add_error_at(
                &format!(
                    "super: trait '{}' method '{}' has no default body to call",
                    trait_name, method
                ),
                span.line,
                span.column,
            );
            return self.arena.fresh_type_var();
        }

        // Build the callable type (mirror of the trait-typed receiver path):
        // params[0] is `this`; the remaining parameters are inferred from args.
        let this_ty = self.current_this_type().unwrap_or_else(|| self.arena.fresh_type_var());
        let params: Vec<TypeHandle> = (0..param_count)
            .map(|i| if i == 0 { this_ty } else { self.arena.fresh_type_var() })
            .collect();
        let fn_ty = self.arena.make_fn(params.into_boxed_slice(), return_type);
        let inst_fn = self.instantiate_fn_type(fn_ty);
        if let Type::Fn(_) = self.arena.get(inst_fn) {
            let (param_types, ret_ty) = self.arena.fn_parts(inst_fn);
            let param_types: Vec<TypeHandle> = param_types.to_vec();
            let n = param_types.len().min(args.len() + 1);
            for i in 1..n {
                let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(param_types[i]));
                let sp = ast.expr(args[i - 1]).span;
                self.unify_call_arg(param_types[i], arg_ty, sp.line, sp.column);
            }
            if let Some(exp) = expected {
                self.unify_or_constrain(ret_ty, exp);
            }
            // Record the static dispatch target for the IR builder and mark the
            // (type, trait, method) triple as needing a default subgraph.
            let key = module_expr_key(&self.current_module_name, expr.0 as u64);
            self.sema_result.super_dispatches.insert(key, (trait_idx, method_idx));
            self.sema_result.super_targets.insert((
                type_name.clone(),
                trait_name.clone(),
                method.into(),
            ));
            return ret_ty;
        }
        self.arena.fresh_type_var()
    }

    pub(super) fn lookup_method_type(
        &mut self,
        recv_ty: TypeHandle,
        method: &str,
        line: u32,
        column: u32,
    ) -> Option<TypeHandle> {
        let resolved = self.arena.resolve(recv_ty);

        // ── Receiver normalization ──
        // Wrapper types (Nullable/Ref) recursively forward method lookup to the inner type,
        // so calls like s?.len() / (&arr).len() auto-unwrap to the correct method table.
        // Nullable's own methods (is_null) are handled via the unified TypeDefInfo path and are not forwarded.
        match self.arena.get(resolved) {
            Type::Nullable(_) => {
                // Nullable's own method (is_null) goes through the TypeDefInfo("nullable") path;
                // other methods are recursively forwarded to the inner type.
                if method != "is_null" {
                    let inner = self.arena.nullable_inner(resolved);
                    return self.lookup_method_type(inner, method, line, column);
                }
            }
            Type::Ref(_) => {
                // Ref auto-deref: method lookup on &T forwards to T.
                let inner = self.arena.ref_parts(resolved).0;
                return self.lookup_method_type(inner, method, line, column);
            }
            _ => {}
        }

        // Privacy gate: type methods are module-scoped unless `pub`. Report and
        // still resolve the signature — returning None sends the caller into a
        // lookup-retry fallback loop that can wedge the fixpoint solver.
        if let Some(tn) = self.arena.type_name(resolved).map(|s| s.to_string()) {
            if let Some(msg) = self.method_privacy_error(&tn, method) {
                self.add_error_at(&msg, line, column);
            }
        }
        // Push recv_ty as the Self type so that, inside build_fn_type_from_sig,
        // type_repr_to_handle(ThisType) resolves to the receiver type correctly,
        // without special-casing the first parameter by position.
        self.push_this_type(resolved);

        // Generic type-parameter binding: bind the type definition's type-parameter names (e.g. "T") to the concrete
        // type arguments in the receiver type, so that T in a method signature (e.g. `pub fun next(&self): T?`)
        // is resolved via type_binding_stack to the corresponding type argument in the receiver,
        // rather than producing an orphan fresh_type_var.
        //
        // Handles Adt (user-defined generics) and builtin types (Array/Nullable/Throw/Generic) uniformly,
        // so that generic parameters in builtin-type method signatures also bind correctly.
        let mut pushed_bindings = false;
        let builtin_bindings: Option<(Box<str>, Vec<TypeHandle>)> = match self.arena.get(resolved) {
            Type::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                Some((name.into(), type_args.to_vec()))
            }
            Type::Array(_) => {
                let (element_type, _) = self.arena.array_parts(resolved);
                Some(("array".into(), vec![element_type]))
            }
            Type::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                Some(("nullable".into(), vec![inner]))
            }
            Type::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                Some(("Throw".into(), vec![value_type, error_type]))
            }
            // Builtin generic dedicated variants: extract element/value types as type_args bindings.
            Type::Channel(_) => Some(("Channel".into(), vec![self.arena.channel_elem(resolved)])),
            Type::Async(_) => Some(("Async".into(), vec![self.arena.async_value(resolved)])),
            Type::Lazy(_) => Some(("Lazy".into(), vec![self.arena.lazy_value(resolved)])),
            Type::Atomic(_) => Some(("Atomic".into(), vec![self.arena.atomic_elem(resolved)])),
            Type::Sender(_) => Some(("Sender".into(), vec![self.arena.sender_elem(resolved)])),
            Type::Receiver(_) => Some(("Receiver".into(), vec![self.arena.receiver_elem(resolved)])),
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                Some((name.into(), args.to_vec()))
            }
            _ => None,
        };
        if let Some((type_name, actual_args)) = builtin_bindings {
            if let Some(def) = self.sema_result.get_type_def(type_name.as_ref()) {
                if !def.type_params.is_empty() && def.type_params.len() == actual_args.len() {
                    self.type_binding_stack.push();
                    for (pname, &arg) in def.type_params.iter().zip(actual_args.iter()) {
                        self.type_binding_stack.insert_top(pname.as_ref(), arg);
                    }
                    pushed_bindings = true;
                }
            }
        }

        let result = self.lookup_method_type_inner(resolved, method);
        if pushed_bindings {
            self.pop_type_bindings();
        }
        self.pop_this_type();
        result
    }

    pub(super) fn lookup_method_type_inner(
        &mut self,
        resolved: TypeHandle,
        method: &str,
    ) -> Option<TypeHandle> {
        match self.arena.get(resolved) {
            Type::Trait(_) => {
                let (name, _) = self.arena.trait_parts(resolved);
                // For a trait type (e.g. l: Logger), look up trait_def.methods directly to restore the method signature;
                // parameters use fresh_type_var (a trait method's exact parameter types are determined by the implementing type).
                if let Some(td) = self.sema_result.get_trait_def(name) {
                    if let Some(sig) = td.methods.iter().find(|m| m.name.as_ref() == method) {
                        // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
                        // the remaining parameters still use fresh_type_var (exact types are determined by the implementing type).
                        let params: Vec<TypeHandle> = (0..sig.param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        let return_type = sig.return_type;
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                }
            }
            Type::TypeVar(idx) => {
                // Inside a trait default method, current_this_type() is a rigid TypeVar
                // representing the (unknown) implementing type. Method lookup must consult
                // the current trait's method signatures rather than the receiver's (nonexistent)
                // method table. This enables bare `method()` calls inside trait default bodies.
                if self.arena.type_vars[idx as usize].is_rigid {
                    if let Some(ref trait_name) = self.current_trait_name {
                        if let Some(td) = self.sema_result.get_trait_def(trait_name) {
                            if let Some(sig) = td.methods.iter().find(|m| m.name.as_ref() == method) {
                                let params: Vec<TypeHandle> = (0..sig.param_count)
                                    .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                                    .collect();
                                let return_type = sig.return_type;
                                return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // Builtin type-name mapping: Array/Nullable/Throw are structural variants where arena.type_name returns None
        // or recurses to the inner type. Method lookup needs a unified type name to query type_def_index;
        // map them here to the synthetic TypeDefInfo names used at registration.
        let type_name: Option<String> = match self.arena.get(resolved) {
            Type::Array(_) => Some("array".to_string()),
            Type::Str => Some("str".to_string()),
            Type::Nullable(_) => Some("nullable".to_string()),
            Type::Throw(_) => Some("Throw".to_string()),
            _ => self.arena.type_name(resolved).map(|s| s.to_string()),
        };

        // v2 convergence: path 1 — query witness_table (trait method dispatch, indexed by type_id).
        if let Some(ref name) = type_name {
            let type_id = self
                .sema_result
                .type_def_index
                .get(name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                for entry in self.witness_table.entries() {
                    if entry.type_id != tid {
                        continue;
                    }
                    // Get the signature from TypeDefInfo.methods (looked up by method_name).
                    // Extract owned data to release the sema_result borrow.
                    let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                        if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                            self.sema_result.type_defs[&type_idx]
                                .methods
                                .iter()
                                .find(|m| m.name.as_ref() == method)
                                .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                        } else {
                            None
                        };
                    if let Some((param_type_reprs, return_type_repr)) = sig_data {
                        return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
                    }
                    // TypeDefInfo.methods miss → query trait_def.methods (trait default methods).
                    // When a type implements a trait via `type T: Trait = ...` without overriding a method,
                    // method_slots is empty and the method signature is obtained from trait_def.
                    let trait_sig_data: Option<(u8, TypeHandle)> =
                        self.sema_result
                            .get_trait_def(entry.trait_name.as_ref())
                            .and_then(|td| {
                                td.methods
                                    .iter()
                                    .find(|m| m.name.as_ref() == method)
                                    .map(|m| (m.param_count, m.return_type))
                            });
                    if let Some((param_count, return_type)) = trait_sig_data {
                        // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
                        // the remaining parameters use fresh_type_var (exact types are determined by the implementing type).
                        let params: Vec<TypeHandle> = (0..param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                    // The current trait does not have this method; continue checking other trait implementations.
                }
            }
        }

        // v2: path 1.5 — TraitObject receiver; restore the real signature from method_sigs.
        // First extract the sig data (param_count + return_type) into owned variables,
        // releasing the arena.types borrow before constructing the Fn type.
        let trait_sig_data: Option<(u8, TypeHandle)> =
            if let Type::TraitObject(_) = self.arena.get(resolved) {
                let (_, method_sigs) = self.arena.trait_object_parts(resolved);
                method_sigs
                    .iter()
                    .find(|m| m.name.as_ref() == method)
                    .map(|sig| (sig.param_count, sig.return_type))
            } else {
                None
            };
        if let Some((param_count, return_type)) = trait_sig_data {
            // params[0] is self, bound to the receiver type (resolved) to avoid producing an orphan TypeVar;
            // the remaining parameters use fresh_type_var (exact types are determined by the implementing type).
            let params: Vec<TypeHandle> = (0..param_count)
                .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                .collect();
            return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
        }

        // v2 convergence: path 2 — query TypeDefInfo.methods (the type's own methods, indexed by method_idx).
        if let Some(ref name) = type_name {
            let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                    self.sema_result.type_defs[&type_idx]
                        .methods
                        .iter()
                        .find(|m| m.name.as_ref() == method)
                        .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                } else {
                    None
                };
            if let Some((param_type_reprs, return_type_repr)) = sig_data {
                return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
            }
        }

        None
    }

    /// Try resolving `name` as an instance field of `this_ty` (implicit this).
    /// Returns the field type on success, None on failure (no error reported).
    /// Used by the Ident fallback when lexical lookup fails inside a method body.
    pub(super) fn try_implicit_this_field(
        &mut self,
        this_ty: TypeHandle,
        name: &str,
    ) -> Option<TypeHandle> {
        let resolved = self.arena.resolve(this_ty);
        // Ref auto-deref: field access on &T forwards to T.
        let inner = match self.arena.get(resolved) {
            Type::Ref(_) => self.arena.ref_parts(resolved).0,
            _ => resolved,
        };
        // Inside a trait default method, this_ty is a rigid TypeVar.
        // We can't verify field existence at trait definition time (the implementing
        // type provides the fields). Be permissive: treat the bare identifier as an
        // implicit this field access and return a fresh_type_var, matching the old
        // behavior where self.field on a TypeVar silently returned a fresh_type_var.
        // Field resolution is deferred to monomorphization specialization.
        if let Type::TypeVar(_) = self.arena.get(inner) {
            return Some(self.arena.fresh_type_var());
        }
        let type_name = self.arena.type_name(inner)?.to_string();
        let field_id = self.sema_result.lookup_field_id(&type_name, name)?;
        // Look up the constructor from the TYPE definition (not ctor_def_index,
        // which can return a wrong constructor when multiple types share the same
        // constructor name, e.g. FileKind.File vs type File = File(...)).
        let def = self.sema_result.get_type_def(&type_name)?;
        let kind = def.kind;
        let repr = {
            let ctor = def.constructors.iter()
                .find(|c| c.field_names.iter().any(|fname| fname.as_deref() == Some(name)))?;
            let idx = match kind {
                TypeDefKind::Record => field_id as usize,
                _ => (field_id as usize).saturating_sub(1),
            };
            ctor.field_type_reprs.get(idx).cloned()?
        };
        Some(self.type_repr_to_handle(&repr))
    }

    /// Looks up the field type for an object type.
    /// line/column are used to locate errors when the field does not exist (passed in by the caller from the AST span).
    pub(super) fn lookup_field_type(&mut self, recv_ty: TypeHandle, field: &str, line: u32, column: u32) -> TypeHandle {
        let resolved = self.arena.resolve(recv_ty);

        // Ref auto-deref: field access on &T forwards to T.
        // For reference types like &Record / &Adt, strip the Ref first and then take the normal field-lookup path,
        // to avoid the type_name indirection returning None (and silently failing) when the inner is a TypeVar.
        if let Type::Ref(_) = self.arena.get(resolved) {
            let inner = self.arena.ref_parts(resolved).0;
            return self.lookup_field_type(inner, field, line, column);
        }

        // ModuleRef field access: look up the field by bare name in the module env carried by the ModuleRef.
        //
        // Use lookup_local (which does not traverse the parent env chain) to handle uniformly:
        // - submodules: ensure_module_env registers the submodule's short name in the parent env when creating the hierarchical env.
        // - in-module symbols: predeclare_declarations has registered functions/constructors into module_env.
        // On miss, report an error; no string concatenation or prefix check is needed.
        if let Type::ModuleRef(_) = self.arena.get(resolved) {
            let (path, module_env) = self.arena.module_ref_parts(resolved);
            if let Some(sym_ty) = self.sema_result.env.lookup_local(module_env, field) {
                return sym_ty;
            }
            self.add_error_at(
                &format!("no module or symbol '{}.{}'", path, field),
                line,
                column,
            );
            return self.arena.make(Type::Unknown);
        }

        let type_name = self.arena.type_name(resolved).map(|s| s.to_string());
        if let Some(name) = type_name {
            if let Some(field_id) = self.sema_result.lookup_field_id(&name, field) {
                // Privacy gate: fields are module-scoped unless `pub`. Record the
                // error but keep resolving the REAL field type — returning Unknown
                // here sends the fixpoint solver into a unify loop.
                if let Some(msg) = self.field_privacy_error(&name, field) {
                    self.add_error_at(&msg, line, column);
                }
                // Look up the constructor from the TYPE definition (not ctor_def_index,
                // which can return a wrong constructor when multiple types share the same
                // constructor name, e.g. FileKind.File vs type File = File(...)).
                if let Some(def) = self.sema_result.get_type_def(&name) {
                    let kind = def.kind;
                    let idx = match kind {
                        TypeDefKind::Record => field_id as usize,
                        _ => (field_id as usize).saturating_sub(1),
                    };
                    // Find the constructor that actually has this field.
                    if let Some(repr) = def.constructors.iter()
                        .find(|c| c.field_names.iter().any(|fname| fname.as_deref() == Some(field)))
                        .and_then(|ctor| ctor.field_type_reprs.get(idx).cloned())
                    {
                        return self.type_repr_to_handle(&repr);
                    }
                    return self.arena.fresh_type_var();
                }
            }
        }
        // Record structural fields.
        let ct = self.arena.get(resolved);
        if let Type::Record(_) = ct {
            let fields = self.arena.record_fields(resolved);
            for f in fields.iter() {
                if f.name.as_deref() == Some(field) {
                    return f.ty;
                }
            }
        }
        // Channel<T>.sender / .receiver fields: return Sender<T> / Receiver<T>.
        // (Already supported at runtime in Value.rs; the Sema layer fills in the type signature.)
        if let Type::Channel(_) = ct {
            let elem = self.arena.channel_elem(resolved);
            match field {
                "sender" => return self.arena.make_sender(elem),
                "receiver" => return self.arena.make_receiver(elem),
                _ => {}
            }
        }
        // Field not found: for a determined type, report a "no such field" error (consistent with the method-call fallback);
        // for pending types (TypeVar/Unknown/Never/Void), silently return a fresh var, deferring to the solver's global diagnostics.
        match ct {
            Type::Record(_) => {
                self.add_error_at(&format!("no such field '{}' on this type", field), line, column);
                self.arena.fresh_type_var()
            }
            Type::Adt(_) => {
                let (name, _) = self.arena.adt_parts(resolved);
                // For a registered Adt type, report a "no such field" error; for unregistered ones, permissively allow.
                if self.sema_result.get_type_def(name).is_some() {
                    self.add_error_at(
                        &format!("no such field '{}' on type '{}'", field, name),
                        line,
                        column,
                    );
                }
                self.arena.fresh_type_var()
            }
            // Pending types: silently return a fresh var (inference pending, deferred to the solver's global diagnostics).
            Type::TypeVar(_) | Type::Unknown
            | Type::Never | Type::Void => {
                self.arena.fresh_type_var()
            }
            // Determined type but field lookup failed: report an error.
            ct_other => {
                let recv_name = self.arena.type_name(resolved)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:?}", ct_other));
                self.add_error_at(
                    &format!("no such field '{}' on type '{}'", field, recv_name),
                    line,
                    column,
                );
                self.arena.fresh_type_var()
            }
        }
    }

    // ── infer_stmt ──

}

/// Deep TypeVar scan (Bug #103 candidacy gate): true when a TypeVar survives
/// anywhere in the type — top-level OR nested (e.g. `T[]`'s element). Such
/// types stay on the lenient constraint path; only fully-concrete types can
/// hard-reject a Path 0 free-function candidate.
pub(super) fn type_contains_typevar(arena: &crate::types::Arena::TypeArena, h: TypeHandle) -> bool {
    let resolved = arena.resolve(h);
    if matches!(arena.get(resolved), Type::TypeVar(_) | Type::Unknown) {
        return true;
    }
    let mut found = false;
    arena.for_each_child(resolved, |child| {
        if !found && type_contains_typevar(arena, child) {
            found = true;
        }
    });
    found
}
