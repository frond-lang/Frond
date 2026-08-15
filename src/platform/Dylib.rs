//! Dylib — load external dynamic libraries by path (dlopen / LoadLibraryW),
//! resolve symbols in them, and release the handle.
//!
//! This is the path-based counterpart of `ResolveSelfSymbol` (which only looks
//! up symbols in the current process). It backs the builtin `Lib` type:
//! `Lib.open(path)` loads a system/user library; `Lib.embed(path)` extracts a
//! build-time resource to a cache file and loads it through here as well.
//!
//! Platform mechanism:
//! - **Linux/macOS**: `dlopen(path, RTLD_NOW)` + `dlsym` + `dlclose`.
//!   RTLD_NOW resolves all symbols eagerly so that a missing dependency fails
//!   at open time with a diagnosable error instead of at call time.
//! - **Windows**: `LoadLibraryW` (UTF-16 path; Kuzo `str` is UTF-8) +
//!   `GetProcAddress` + `FreeLibrary`.
//!
//! Handles are raw pointers owned by `value::LibShared` (Drop → `close`).

/// Load a dynamic library by path. `Err` carries a human-readable reason
/// (missing file, bad arch, unresolved dependency).
pub fn open(path: &str) -> Result<*mut core::ffi::c_void, String> {
    #[cfg(unix)]
    {
        // SAFETY: dlopen on a caller-supplied path returns a refcounted handle
        // (or NULL + dlerror text). The handle is released exactly once by
        // LibShared::drop → close().
        unsafe {
            extern "C" {
                fn dlopen(filename: *const core::ffi::c_char, flag: i32) -> *mut core::ffi::c_void;
                fn dlerror() -> *const core::ffi::c_char;
            }
            const RTLD_NOW: i32 = 2;
            // Clear any stale error state before the call.
            dlerror();
            let mut buf: Vec<u8> = Vec::with_capacity(path.len() + 1);
            buf.extend_from_slice(path.as_bytes());
            buf.push(0);
            let handle = dlopen(buf.as_ptr() as *const core::ffi::c_char, RTLD_NOW);
            if handle.is_null() {
                let err = dlerror();
                let msg = if err.is_null() {
                    "unknown dlopen failure".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned()
                };
                Err(format!("dlopen('{}') failed: {}", path, msg))
            } else {
                Ok(handle)
            }
        }
    }

    #[cfg(windows)]
    {
        // SAFETY: LoadLibraryW returns a refcounted HMODULE (or NULL with
        // GetLastError). Released exactly once by LibShared::drop → close().
        unsafe {
            extern "system" {
                fn LoadLibraryW(lpwLibFileName: *const u16) -> *mut core::ffi::c_void;
            }
            let mut wide: Vec<u16> = path.encode_utf16().collect();
            wide.push(0);
            let handle = LoadLibraryW(wide.as_ptr());
            if handle.is_null() {
                Err(format!("LoadLibraryW('{}') failed (file missing, wrong arch, or dependency error)", path))
            } else {
                Ok(handle)
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err("dynamic library loading is not supported on this platform".to_string())
    }
}

/// Look up a symbol address by name inside a loaded library handle.
/// `None` = symbol not found.
pub fn symbol(handle: *mut core::ffi::c_void, name: &str) -> Option<*mut core::ffi::c_void> {
    if handle.is_null() {
        return None;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(name.len() + 1);
    buf.extend_from_slice(name.as_bytes());
    buf.push(0);

    #[cfg(unix)]
    {
        // SAFETY: dlsym on a valid handle; symbol type correctness is the
        // caller's responsibility (the sig string in Lib.lookup).
        unsafe {
            extern "C" {
                fn dlsym(handle: *mut core::ffi::c_void, symbol: *const core::ffi::c_char) -> *mut core::ffi::c_void;
            }
            let addr = dlsym(handle, buf.as_ptr() as *const core::ffi::c_char);
            if addr.is_null() { None } else { Some(addr) }
        }
    }

    #[cfg(windows)]
    {
        // SAFETY: GetProcAddress on a valid HMODULE; NULL on a miss.
        unsafe {
            extern "system" {
                fn GetProcAddress(hModule: *mut core::ffi::c_void, lpProcName: *const u8) -> *mut core::ffi::c_void;
            }
            let addr = GetProcAddress(handle, buf.as_ptr());
            if addr.is_null() { None } else { Some(addr) }
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = buf;
        None
    }
}

/// Release a library handle (refcount decrement). Idempotent per handle owner
/// (LibShared guards with its `closed` flag).
pub fn close(handle: *mut core::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    #[cfg(unix)]
    {
        // SAFETY: handle came from open() and is released exactly once.
        unsafe {
            extern "C" {
                fn dlclose(handle: *mut core::ffi::c_void) -> i32;
            }
            dlclose(handle);
        }
    }
    #[cfg(windows)]
    {
        // SAFETY: handle came from LoadLibraryW and is released exactly once.
        unsafe {
            extern "system" {
                fn FreeLibrary(hLibModule: *mut core::ffi::c_void) -> i32;
            }
            FreeLibrary(handle);
        }
    }
}
