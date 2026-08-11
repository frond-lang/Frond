use bumpalo::Bump;
use kuzo::sema::Inference::InferContext;
use kuzo::sema::Sema::{SemaResult, TypeArena};
use kuzo::tooling::common::Pipeline;

#[test]
fn test_purge_module_removes_expr_types() {
    let arena = Bump::new();
    let source = "fun foo(): void { val x = 42 }\n";
    let parse_result = Pipeline::parse_entry_module_lsp(&arena, source, "test.kz");

    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);
    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    // Run sema on the module
    let all_modules = vec![&parse_result.module];
    ctx.check_module_with_env(&parse_result.module, root_env, &all_modules);
    drop(ctx);

    // Verify expr_types has entries
    assert!(!sema_result.expr_types.is_empty(), "expr_types should have entries after sema");

    // Get the module name - it might be "test.kz" or something else
    let module_name = parse_result.module.name.to_string();

    // Purge the module
    sema_result.purge_module(&module_name);

    // Verify expr_types for that module are removed
    // After purge, the module's expr_type_keys should be gone
    assert!(
        !sema_result.module_ownership.expr_type_keys.contains_key(&module_name),
        "expr_type_keys for {} should be removed after purge", module_name
    );
}

#[test]
fn test_purge_module_removes_func_sig_index() {
    let arena = Bump::new();
    let source = "fun foo(): void { val x = 42 }\n";
    let parse_result = Pipeline::parse_entry_module_lsp(&arena, source, "test.kz");

    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);
    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    let all_modules = vec![&parse_result.module];
    ctx.check_module_with_env(&parse_result.module, root_env, &all_modules);
    drop(ctx);

    // Verify func_sig_index has "foo"
    assert!(sema_result.func_sig_index.contains_key("foo"), "foo should be in func_sig_index");

    let module_name = parse_result.module.name.to_string();
    sema_result.purge_module(&module_name);

    // After purge, "foo" should be removed from func_sig_index
    assert!(
        !sema_result.func_sig_index.contains_key("foo"),
        "foo should be removed from func_sig_index after purge"
    );
}

#[test]
fn test_purge_then_repopulate_succeeds() {
    let arena = Bump::new();
    let source = "fun foo(): void { val x = 42 }\n";
    let parse_result = Pipeline::parse_entry_module_lsp(&arena, source, "test.kz");

    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);
    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    let all_modules = vec![&parse_result.module];
    ctx.check_module_with_env(&parse_result.module, root_env, &all_modules);
    drop(ctx);

    let module_name = parse_result.module.name.to_string();

    // Purge
    sema_result.purge_module(&module_name);

    // Re-populate should succeed (put_func_sig returns true after purge)
    let mut ctx2 = InferContext::new(&mut type_arena, &mut sema_result);
    ctx2.reset_state();
    let root_env2 = ctx2.env.root();
    ctx2.register_builtins(root_env2);
    ctx2.check_module_with_env(&parse_result.module, root_env2, &all_modules);
    drop(ctx2);

    // Verify foo is back in func_sig_index
    assert!(
        sema_result.func_sig_index.contains_key("foo"),
        "foo should be back in func_sig_index after re-populate"
    );
}
