//! Task 13: 增量 sema 等价性测试
//!
//! 验证 `run_sema_incremental` 与 `run_sema_pipeline_lsp`（全量）的结果等价性，
//! 以及 `replace_module` + `dirty_closure` + `run_sema_incremental` 的组合不崩溃。

use bumpalo::Bump;
use kuzo::module::ModuleLoader;
use kuzo::tooling::common::Diagnostic::Severity;
use kuzo::tooling::common::Pipeline::{self, SemaIncrementalOutcome, SemaOutcome};
use rustc_hash::FxHashSet;

/// 辅助：对给定 source 跑一次全量 sema，返回 (type_arena, sema_result, diagnostics)。
fn run_full_sema(source: &str) -> (kuzo::sema::Sema::TypeArena, kuzo::sema::Sema::SemaResult, Vec<kuzo::tooling::common::Diagnostic::Diagnostic>) {
    let arena = Bump::new();
    let parse = Pipeline::parse_entry_module_lsp(&arena, source, "test.kz");
    assert!(
        parse.diagnostics.is_empty(),
        "parse should produce no diagnostics, got: {:?}",
        parse.diagnostics
    );

    let mut loader = ModuleLoader::new();
    let dep_keys: Vec<String> = loader.load_transitive_imports(&parse.module);
    let std_keys: Vec<String> = Vec::new();

    let outcome = Pipeline::run_sema_pipeline_lsp(
        &loader,
        &std_keys,
        &dep_keys,
        &parse.module,
        "test.kz",
    );

    match outcome {
        SemaOutcome::Ok { type_arena, sema_result, diagnostics } => (type_arena, sema_result, diagnostics),
        SemaOutcome::Err(diagnostics) => panic!("full sema returned Err: {:?}", diagnostics),
    }
}

/// 测试 1：全量 sema 基本工作——合法源码应无错误，且 func_sig_index / expr_types 有条目。
#[test]
fn test_full_sema_works() {
    let source = "fun foo(): void { val x = 42 }\nfun main(): void { foo() }\n";
    let (_type_arena, sema_result, diagnostics) = run_full_sema(source);

    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "expected no sema errors, got: {:?}",
        errors
    );

    assert!(
        sema_result.func_sig_index.contains_key("foo"),
        "foo should be in func_sig_index"
    );
    assert!(
        sema_result.func_sig_index.contains_key("main"),
        "main should be in func_sig_index"
    );
    assert!(
        !sema_result.expr_types.is_empty(),
        "expr_types should have entries after sema"
    );
}

/// 测试 2：增量 sema 不崩溃——replace_module 后跑增量 recheck，验证不 panic 且产出合理结果。
#[test]
fn test_incremental_sema_does_not_crash() {
    let source1 = "fun foo(): void { val x = 42 }\n";
    let source2 = "fun foo(): void { val x = 99 }\n"; // 函数体变化，非 API 变更

    // 1. 全量 sema source1
    let (mut type_arena, mut sema_result, _diag) = run_full_sema(source1);
    assert!(
        sema_result.func_sig_index.contains_key("foo"),
        "baseline: foo should be in func_sig_index"
    );

    // 2. replace_module 为 source2（重新建一个 loader 以模拟 LSP 场景下的同一 loader）
    let arena2 = Bump::new();
    let parse2 = Pipeline::parse_entry_module_lsp(&arena2, source2, "test.kz");
    let mut loader = ModuleLoader::new();
    let _dep_keys = loader.load_transitive_imports(&parse2.module);
    assert!(
        loader.replace_module("test.kz", source2),
        "replace_module should succeed for valid source2"
    );

    // 3. 计算 dirty closure
    let dirty: FxHashSet<String> = loader.dirty_closure("test.kz");
    assert!(
        dirty.contains("test.kz"),
        "dirty closure should contain the changed module"
    );

    // 4. 运行增量 sema——不应崩溃
    let incr_outcome = Pipeline::run_sema_incremental(
        &loader,
        &dirty,
        &mut type_arena,
        &mut sema_result,
    );

    match incr_outcome {
        SemaIncrementalOutcome::Ok { sema_result, diagnostics, rechecked, .. } => {
            assert!(
                rechecked.contains("test.kz"),
                "test.kz should be in the rechecked set"
            );
            let errors: Vec<_> = diagnostics
                .iter()
                .filter(|d| d.severity == Severity::Error)
                .collect();
            assert!(
                errors.is_empty(),
                "expected no sema errors after incremental, got: {:?}",
                errors
            );
            assert!(
                sema_result.func_sig_index.contains_key("foo"),
                "foo should still be in func_sig_index after incremental recheck"
            );
            assert!(
                !sema_result.expr_types.is_empty(),
                "expr_types should have entries after incremental recheck"
            );
        }
        SemaIncrementalOutcome::NeedsFull => {
            // 单模块 dirty closure 不应触发 NeedsFull（非 builtin，且占比远小于 50%）
            panic!("incremental sema returned NeedsFull for a single-module change; expected Ok");
        }
        SemaIncrementalOutcome::Err(diagnostics) => {
            panic!("incremental sema returned Err: {:?}", diagnostics);
        }
    }
}

/// 测试 3：增量 sema 与全量 sema 的等价性——
/// 对同一最终源码 source2，比较「全量直接跑」与「先跑 source1 再增量到 source2」的结果。
#[test]
fn test_incremental_sema_equivalence() {
    let source1 = "fun foo(): void { val x = 42 }\n";
    let source2 = "fun foo(): void { val x = 99 }\nfun bar(): void { foo() }\n";

    // ── 参考结果：对 source2 直接全量 sema ──
    let (_ta_full, sr_full, diag_full) = run_full_sema(source2);
    let full_error_count = sr_full.errors.len();
    let full_has_foo = sr_full.func_sig_index.contains_key("foo");
    let full_has_bar = sr_full.func_sig_index.contains_key("bar");
    let full_diag_error_count = diag_full
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .count();

    // ── 增量路径：先全量 source1，再 replace + 增量到 source2 ──
    let (mut type_arena, mut sema_result, _diag1) = run_full_sema(source1);

    // 用 source2 的 import 图构建 loader，再把 test.kz replace 进去
    let arena2 = Bump::new();
    let parse2 = Pipeline::parse_entry_module_lsp(&arena2, source2, "test.kz");
    let mut loader = ModuleLoader::new();
    let _dep_keys = loader.load_transitive_imports(&parse2.module);
    assert!(
        loader.replace_module("test.kz", source2),
        "replace_module should succeed"
    );

    let dirty: FxHashSet<String> = loader.dirty_closure("test.kz");
    let incr_outcome = Pipeline::run_sema_incremental(
        &loader,
        &dirty,
        &mut type_arena,
        &mut sema_result,
    );

    let sr_incr = match incr_outcome {
        SemaIncrementalOutcome::Ok { sema_result, diagnostics, .. } => {
            let incr_diag_error_count = diagnostics
                .iter()
                .filter(|d| d.severity == Severity::Error)
                .count();
            assert_eq!(
                incr_diag_error_count, full_diag_error_count,
                "diagnostic error count should match between full and incremental"
            );
            sema_result
        }
        SemaIncrementalOutcome::NeedsFull => panic!("incremental returned NeedsFull; expected Ok"),
        SemaIncrementalOutcome::Err(d) => panic!("incremental returned Err: {:?}", d),
    };

    // ── 等价性断言 ──
    assert_eq!(
        sr_incr.errors.len(),
        full_error_count,
        "sema error count should match: full={} incr={}",
        full_error_count,
        sr_incr.errors.len()
    );
    assert_eq!(
        sr_incr.func_sig_index.contains_key("foo"),
        full_has_foo,
        "foo presence in func_sig_index should match"
    );
    assert_eq!(
        sr_incr.func_sig_index.contains_key("bar"),
        full_has_bar,
        "bar presence in func_sig_index should match (bar only exists in source2)"
    );
    assert!(
        !sr_incr.expr_types.is_empty(),
        "expr_types should have entries after incremental recheck"
    );
}
