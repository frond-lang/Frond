//! Symbols — stdlib C symbol address cache (replaces the deleted `bindings_addr`
//! hardcoded table).
//!
//! stdlib `@extern("C") #{ }#` functions are compiled and linked into the kuzo
//! binary by build.rs. At runtime, [`platform::ResolveSelfSymbol`] resolves their
//! addresses by name via dlsym/GetProcAddress. This module layers a lazy cache on
//! top to avoid hitting the system API on every FFI call.

use std::sync::OnceLock;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

/// Global symbol address cache: symbol name → address (stored as `usize` since
/// raw pointers are not `Send`/`Sync`).
static CACHE: OnceLock<Mutex<FxHashMap<String, usize>>> = OnceLock::new();

/// Resolve a stdlib C symbol address by name.
///
/// The first lookup goes through
/// [`platform::ResolveSelfSymbol::resolve_self_symbol`] (dlsym self-lookup) and
/// caches the result; subsequent lookups read the cache directly. Returns `None`
/// when the symbol is not found (not exported or name mismatch).
pub fn resolve(name: &str) -> Option<*mut core::ffi::c_void> {
    let cache = CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
    // Check the cache first (within read-lock granularity).
    {
        let map = cache.lock();
        if let Some(&addr) = map.get(name) {
            return Some(addr as *mut core::ffi::c_void);
        }
    }
    // Cache miss: fall back to dlsym self-lookup.
    let addr = crate::platform::ResolveSelfSymbol::resolve_self_symbol(name)?;
    let addr_usize = addr as usize;
    cache.lock().insert(name.to_string(), addr_usize);
    Some(addr)
}
