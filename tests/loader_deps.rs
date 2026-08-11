use kuzo::module::ModuleLoader;
use std::fs;
use std::path::PathBuf;

/// Creates a unique temp directory for test fixtures, cleaning up any prior contents.
fn make_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("kuzo_loader_deps_{}_{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("failed to create test dir");
    dir
}

#[test]
fn test_dirty_closure_single_module() {
    // A module with no importers: dirty closure is just itself
    let loader = ModuleLoader::new();
    let dirty = loader.dirty_closure("nonexistent.kz");
    assert_eq!(dirty.len(), 1);
    assert!(dirty.contains("nonexistent.kz"));
}

#[test]
fn test_dirty_closure_with_reverse_deps() {
    // Build a real import graph on disk:
    //   entry.kz imports a   (entry is NOT cached, so not in the graph)
    //   a.kz      imports b
    //   b.kz      (leaf)
    //
    // After load_transitive_imports:
    //   forward_deps["a.kz"] = {"b.kz"}
    //   reverse_deps["b.kz"] = {"a.kz"}
    //   dirty_closure("b.kz") = {"b.kz", "a.kz"}
    //   dirty_closure("a.kz") = {"a.kz"}  (nothing imports a.kz among cached modules)
    let dir = make_test_dir("reverse_deps");
    fs::write(dir.join("b.kz"), "pub fun world(): void { }\n").unwrap();
    fs::write(dir.join("a.kz"), "import b\n\npub fun hello(): void { }\n").unwrap();

    let arena = bumpalo::Bump::new();
    let entry = kuzo::tooling::common::Pipeline::parse_entry_module_or_exit(
        &arena,
        "import a\n\nfun main(): void { }\n",
        "entry.kz",
    );
    let mut loader = ModuleLoader::new();
    loader.add_search_path(&dir);
    let _ = loader.load_transitive_imports(&entry);

    // dirty_closure of the leaf (b.kz) should include its importer a.kz
    let dirty_b = loader.dirty_closure("b.kz");
    assert!(dirty_b.contains("b.kz"), "dirty_closure should contain the changed module");
    assert!(dirty_b.contains("a.kz"), "dirty_closure should contain the direct importer");

    // dirty_closure of a.kz (which nothing imports) should be just itself
    let dirty_a = loader.dirty_closure("a.kz");
    assert_eq!(dirty_a.len(), 1, "nothing imports a.kz, closure should be just a.kz");
    assert!(dirty_a.contains("a.kz"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_dirty_closure_transitive_chain() {
    // Three-level chain: c.kz <- b.kz <- a.kz
    //   a.kz imports b, b.kz imports c
    //   dirty_closure("c.kz") should include {c.kz, b.kz, a.kz}
    let dir = make_test_dir("transitive_chain");
    fs::write(dir.join("c.kz"), "pub fun c(): void { }\n").unwrap();
    fs::write(dir.join("b.kz"), "import c\n\npub fun b(): void { }\n").unwrap();
    fs::write(dir.join("a.kz"), "import b\n\npub fun a(): void { }\n").unwrap();

    let arena = bumpalo::Bump::new();
    let entry = kuzo::tooling::common::Pipeline::parse_entry_module_or_exit(
        &arena,
        "import a\n\nfun main(): void { }\n",
        "entry.kz",
    );
    let mut loader = ModuleLoader::new();
    loader.add_search_path(&dir);
    let _ = loader.load_transitive_imports(&entry);

    let dirty_c = loader.dirty_closure("c.kz");
    assert!(dirty_c.contains("c.kz"), "should contain changed module");
    assert!(dirty_c.contains("b.kz"), "should contain direct importer");
    assert!(dirty_c.contains("a.kz"), "should contain transitive importer");
    assert_eq!(dirty_c.len(), 3, "dirty closure of c.kz should have 3 modules");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_forward_deps_populated_after_load() {
    // After loading, forward_deps for user modules should be populated.
    // The entry module itself is NOT cached, so we check the loaded dep modules.
    let dir = make_test_dir("forward_deps");
    fs::write(dir.join("b.kz"), "pub fun world(): void { }\n").unwrap();
    fs::write(dir.join("a.kz"), "import b\n\npub fun hello(): void { }\n").unwrap();

    let arena = bumpalo::Bump::new();
    let entry = kuzo::tooling::common::Pipeline::parse_entry_module_or_exit(
        &arena,
        "import a\n\nfun main(): void { }\n",
        "entry.kz",
    );
    let mut loader = ModuleLoader::new();
    loader.add_search_path(&dir);
    let _ = loader.load_transitive_imports(&entry);

    // a.kz imports b.kz
    let fwd_a = loader.get_forward_deps("a.kz");
    assert!(fwd_a.is_some(), "forward_deps for a.kz should exist");
    let fwd_a = fwd_a.unwrap();
    assert!(!fwd_a.is_empty(), "forward_deps for a.kz should be non-empty");
    assert!(fwd_a.contains("b.kz"), "forward_deps for a.kz should contain b.kz");

    // b.kz has no imports
    let fwd_b = loader.get_forward_deps("b.kz");
    assert!(fwd_b.is_some(), "forward_deps for b.kz should exist");
    assert!(fwd_b.unwrap().is_empty(), "forward_deps for b.kz should be empty");

    let _ = fs::remove_dir_all(&dir);
}
