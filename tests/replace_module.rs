use kuzo::module::ModuleLoader;

#[test]
fn test_replace_module_updates_cache() {
    let mut loader = ModuleLoader::new();
    let source1 = "fun foo(): void { }\n";
    let source2 = "fun bar(): void { }\n";

    assert!(loader.replace_module("test_replace.kz", source1));
    assert!(loader.get_module_by_key("test_replace.kz").is_some());

    assert!(loader.replace_module("test_replace.kz", source2));
    let m = loader
        .get_module_by_key("test_replace.kz")
        .expect("module should exist");
    // Verify the module was re-parsed with new content
    assert!(!m.declarations.is_empty());
}

#[test]
fn test_replace_module_updates_dep_graph() {
    let mut loader = ModuleLoader::new();
    // Module with no imports
    let source1 = "fun foo(): void { }\n";
    loader.replace_module("test_dep.kz", source1);

    let fwd = loader.get_forward_deps("test_dep.kz");
    assert!(fwd.is_some());
    assert!(fwd.unwrap().is_empty(), "no imports = empty forward_deps");

    // Replace with module that imports std.io.File
    let source2 = "import std.io.File\n\nfun foo(): void { }\n";
    loader.replace_module("test_dep.kz", source2);

    let fwd = loader.get_forward_deps("test_dep.kz");
    assert!(fwd.is_some());
    assert!(
        !fwd.unwrap().is_empty(),
        "should have forward_deps after adding import"
    );
}

#[test]
fn test_replace_module_invalid_syntax() {
    let mut loader = ModuleLoader::new();
    // Invalid syntax should return false
    let result = loader.replace_module("bad.kz", "val x = ;;;;");
    assert!(!result, "invalid syntax should return false");
    assert!(
        loader.get_module_by_key("bad.kz").is_none(),
        "module should not be cached on parse failure"
    );
}
