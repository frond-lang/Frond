//! Stdlib source embed table and lookups.
//!
//! Uses `include_str!` to embed `.frond` source files into the binary at compile time,
//! for `Loader` to parse.
//!
//! ## Directory layout
//!
//! ```text
//! src/stdlib/
//! ├── builtin/          # builtin modules (visible by default, no import needed)
//! │   ├── {io,net,time}/Raw.frond  # @extern("C") primitives (split by domain)
//! │   ├── error/        # Err/Error/IOError/TimeError
//! │   ├── reflect/      # runtime reflection formatting (Reflect.format)
//! │   ├── io/           # Reader/Writer trait + Console(print/println/scan...)
//! │   └── iter/         # Iterator<T> iterator
//! └── std/              # standard library (requires `import std.xxx`)
//!     ├── io/           # File/Path/Buffered/Dir/Fs
//!     ├── time/         # Duration/Instant/SystemTime/DateTime/Calendar/Timer
//!     └── net/          # Addr/TcpListener/TcpStream/UdpSocket/Dns
//! ```

/// A stdlib file entry: (relative path, source content).
pub type StdlibFile = (&'static str, &'static str);

/// Builtin module file manifest (visible by default, no import needed).
///
/// Ordered by dependency:
///   Raw(@extern primitives) → error → reflect → io(Reader/Writer/Console) → net → time → iter
/// @extern("C") primitives load first: globally visible, available to builtin/std wrappers
/// {io,net,time}/Raw.frond are loaded together at the primitive layer
pub const BUILTIN_FILES: &[StdlibFile] = &[
    // @extern("C") primitive modules (globally visible, split by domain into Raw.frond)
    ("builtin/io/Raw.frond", include_str!("../../../std/builtin/io/Raw.frond")),
    ("builtin/net/Raw.frond", include_str!("../../../std/builtin/net/Raw.frond")),
    ("builtin/time/Raw.frond", include_str!("../../../std/builtin/time/Raw.frond")),
    // error module
    ("builtin/error/pack.frond", include_str!("../../../std/builtin/error/pack.frond")),
    ("builtin/error/Err.frond", include_str!("../../../std/builtin/error/Err.frond")),
    ("builtin/error/Error.frond", include_str!("../../../std/builtin/error/Error.frond")),
    ("builtin/error/IOError.frond", include_str!("../../../std/builtin/error/IOError.frond")),
    ("builtin/error/TimeError.frond", include_str!("../../../std/builtin/error/TimeError.frond")),
    ("builtin/error/OsError.frond", include_str!("../../../std/builtin/error/OsError.frond")),
    ("builtin/error/FfiError.frond", include_str!("../../../std/builtin/error/FfiError.frond")),
    ("builtin/error/JsonError.frond", include_str!("../../../std/builtin/error/JsonError.frond")),
    ("builtin/error/FmtError.frond", include_str!("../../../std/builtin/error/FmtError.frond")),
    ("builtin/error/ParseError.frond", include_str!("../../../std/builtin/error/ParseError.frond")),
    // reflect module (runtime reflection, Raw.frond primitives + Reflect.frond wrapper)
    ("builtin/reflect/pack.frond", include_str!("../../../std/builtin/reflect/pack.frond")),
    ("builtin/reflect/Raw.frond", include_str!("../../../std/builtin/reflect/Raw.frond")),
    ("builtin/reflect/Reflect.frond", include_str!("../../../std/builtin/reflect/Reflect.frond")),
    // io module (Reader/Writer trait + Console standard IO)
    ("builtin/io/pack.frond", include_str!("../../../std/builtin/io/pack.frond")),
    ("builtin/io/Reader.frond", include_str!("../../../std/builtin/io/Reader.frond")),
    ("builtin/io/Writer.frond", include_str!("../../../std/builtin/io/Writer.frond")),
    ("builtin/io/Console.frond", include_str!("../../../std/builtin/io/Console.frond")),
    // net module (pack declaration; Raw already loaded at the primitive layer)
    ("builtin/net/pack.frond", include_str!("../../../std/builtin/net/pack.frond")),
    // time module (pack declaration; Raw already loaded at the primitive layer)
    ("builtin/time/pack.frond", include_str!("../../../std/builtin/time/pack.frond")),
    // str module (UTF-8 decoding primitives; depended on by the iter module)
    ("builtin/str/pack.frond", include_str!("../../../std/builtin/str/pack.frond")),
    ("builtin/str/Raw.frond", include_str!("../../../std/builtin/str/Raw.frond")),
    // iter module
    ("builtin/iter/pack.frond", include_str!("../../../std/builtin/iter/pack.frond")),
    ("builtin/iter/Iterator.frond", include_str!("../../../std/builtin/iter/Iterator.frond")),
    // os module (process-environment domain primitives)
    ("builtin/os/pack.frond", include_str!("../../../std/builtin/os/pack.frond")),
    ("builtin/os/Raw.frond", include_str!("../../../std/builtin/os/Raw.frond")),
    // rand module (PRNG step primitive)
    ("builtin/rand/pack.frond", include_str!("../../../std/builtin/rand/pack.frond")),
    ("builtin/rand/Raw.frond", include_str!("../../../std/builtin/rand/Raw.frond")),
    ("builtin/crypto/Raw.frond", include_str!("../../../std/builtin/crypto/Raw.frond")),
    // mem module (u8[] buffer primitives over libc mem*)
    ("builtin/mem/pack.frond", include_str!("../../../std/builtin/mem/pack.frond")),
    ("builtin/mem/Raw.frond", include_str!("../../../std/builtin/mem/Raw.frond")),
    // sort module (array sort/sort_by + sorted-array search, method form)
    ("builtin/sort/pack.frond", include_str!("../../../std/builtin/sort/pack.frond")),
    ("builtin/sort/Raw.frond", include_str!("../../../std/builtin/sort/Raw.frond")),
    ("builtin/encoding/pack.frond", include_str!("../../../std/builtin/encoding/pack.frond")),
    ("builtin/encoding/Raw.frond", include_str!("../../../std/builtin/encoding/Raw.frond")),
    ("builtin/sort/Sort.frond", include_str!("../../../std/builtin/sort/Sort.frond")),
];

/// Standard library module file manifest (requires `import std.xxx` to load).
///
/// Ordered by dependency: io → time → net
/// reflect has moved to builtin/reflect (visible by default); Console has moved to builtin/io (visible by default)
pub const STD_FILES: &[StdlibFile] = &[
    // io module (Console has moved to builtin/io)
    ("std/os/pack.frond", include_str!("../../../std/os/pack.frond")),
    ("std/os/Env.frond", include_str!("../../../std/os/Env.frond")),
    ("std/os/Tty.frond", include_str!("../../../std/os/Tty.frond")),
    ("std/os/Proc.frond", include_str!("../../../std/os/Proc.frond")),
    ("std/os/Info.frond", include_str!("../../../std/os/Info.frond")),
    ("std/os/Os.frond", include_str!("../../../std/os/Os.frond")),
    ("std/io/pack.frond", include_str!("../../../std/io/pack.frond")),
    ("std/io/Path.frond", include_str!("../../../std/io/Path.frond")),
    ("std/io/File.frond", include_str!("../../../std/io/File.frond")),
    ("std/io/Buffered.frond", include_str!("../../../std/io/Buffered.frond")),
    ("std/io/Dir.frond", include_str!("../../../std/io/Dir.frond")),
    ("std/io/Fs.frond", include_str!("../../../std/io/Fs.frond")),
    // time module
    ("std/time/pack.frond", include_str!("../../../std/time/pack.frond")),
    ("std/time/Duration.frond", include_str!("../../../std/time/Duration.frond")),
    ("std/time/Instant.frond", include_str!("../../../std/time/Instant.frond")),
    ("std/time/SystemTime.frond", include_str!("../../../std/time/SystemTime.frond")),
    ("std/time/DateTime.frond", include_str!("../../../std/time/DateTime.frond")),
    ("std/time/Calendar.frond", include_str!("../../../std/time/Calendar.frond")),
    ("std/time/Timer.frond", include_str!("../../../std/time/Timer.frond")),
    // net module (TcpStream before TcpListener: TcpListener depends on __net_tcp_close defined in TcpStream)
    ("std/net/pack.frond", include_str!("../../../std/net/pack.frond")),
    ("std/net/Addr.frond", include_str!("../../../std/net/Addr.frond")),
    ("std/net/Dns.frond", include_str!("../../../std/net/Dns.frond")),
    ("std/net/TcpStream.frond", include_str!("../../../std/net/TcpStream.frond")),
    ("std/net/TcpListener.frond", include_str!("../../../std/net/TcpListener.frond")),
    ("std/net/UdpSocket.frond", include_str!("../../../std/net/UdpSocket.frond")),
    // math module
    ("std/math/pack.frond",  include_str!("../../../std/math/pack.frond")),
    ("std/math/Math.frond",  include_str!("../../../std/math/Math.frond")),
    ("std/math/Power.frond", include_str!("../../../std/math/Power.frond")),
    ("std/math/Trig.frond",  include_str!("../../../std/math/Trig.frond")),
    ("std/math/Round.frond", include_str!("../../../std/math/Round.frond")),
    // str module → migrated into std/core (type modules share one library)
    // core module: std/core/pack.frond makes `import std.core` load every
    // sub-library; type namespaces live under types/.
    ("std/core/pack.frond", include_str!("../../../std/core/pack.frond")),
    ("std/core/types/pack.frond", include_str!("../../../std/core/types/pack.frond")),
    ("std/core/types/Str.frond", include_str!("../../../std/core/types/Str.frond")),
    ("std/core/types/Bool.frond", include_str!("../../../std/core/types/Bool.frond")),
    ("std/core/types/F64.frond", include_str!("../../../std/core/types/F64.frond")),
    ("std/core/types/F32.frond", include_str!("../../../std/core/types/F32.frond")),
    ("std/core/types/F16.frond", include_str!("../../../std/core/types/F16.frond")),
    ("std/core/types/F128.frond", include_str!("../../../std/core/types/F128.frond")),
    ("std/core/types/I8.frond", include_str!("../../../std/core/types/I8.frond")),
    ("std/core/types/I16.frond", include_str!("../../../std/core/types/I16.frond")),
    ("std/core/types/I32.frond", include_str!("../../../std/core/types/I32.frond")),
    ("std/core/types/I64.frond", include_str!("../../../std/core/types/I64.frond")),
    ("std/core/types/I128.frond", include_str!("../../../std/core/types/I128.frond")),
    ("std/core/types/U8.frond", include_str!("../../../std/core/types/U8.frond")),
    ("std/core/types/U16.frond", include_str!("../../../std/core/types/U16.frond")),
    ("std/core/types/U32.frond", include_str!("../../../std/core/types/U32.frond")),
    ("std/core/types/U64.frond", include_str!("../../../std/core/types/U64.frond")),
    ("std/core/types/U128.frond", include_str!("../../../std/core/types/U128.frond")),
    ("std/core/types/Usize.frond", include_str!("../../../std/core/types/Usize.frond")),
    ("std/core/types/Isize.frond", include_str!("../../../std/core/types/Isize.frond")),
    // fmt sub-library of std/core: number formatting (radix 2..36, padding)
    ("std/core/fmt/pack.frond", include_str!("../../../std/core/fmt/pack.frond")),
    ("std/core/fmt/Fmt.frond", include_str!("../../../std/core/fmt/Fmt.frond")),
    // hash sub-library of std/core: algorithm collection (FNV/CRC32/Adler32/xxHash64)
    ("std/core/hash/pack.frond", include_str!("../../../std/core/hash/pack.frond")),
    ("std/core/hash/Hash.frond", include_str!("../../../std/core/hash/Hash.frond")),
    ("std/core/hash/Crc32.frond", include_str!("../../../std/core/hash/Crc32.frond")),
    ("std/core/hash/Adler32.frond", include_str!("../../../std/core/hash/Adler32.frond")),
    ("std/core/hash/Xxh64.frond", include_str!("../../../std/core/hash/Xxh64.frond")),
    ("std/core/hash/Sha256.frond", include_str!("../../../std/core/hash/Sha256.frond")),
    ("std/core/hash/Sha512.frond", include_str!("../../../std/core/hash/Sha512.frond")),
    // crypto module (key agreement / AEAD wrappers over builtin/crypto)
    ("std/crypto/pack.frond", include_str!("../../../std/crypto/pack.frond")),
    ("std/crypto/X25519.frond", include_str!("../../../std/crypto/X25519.frond")),
    ("std/crypto/AesGcm.frond", include_str!("../../../std/crypto/AesGcm.frond")),
    ("std/crypto/EcdsaP256.frond", include_str!("../../../std/crypto/EcdsaP256.frond")),
    ("std/crypto/EcdsaP384.frond", include_str!("../../../std/crypto/EcdsaP384.frond")),
    ("std/tls/Cert.frond", include_str!("../../../std/tls/Cert.frond")),
    ("std/tls/Chain.frond", include_str!("../../../std/tls/Chain.frond")),
    ("std/tls/Roots.frond", include_str!("../../../std/tls/Roots.frond")),
    // tls module (pure-Frond TLS 1.3 client over std.crypto; depends on net/crypto/hash)
    ("std/tls/pack.frond", include_str!("../../../std/tls/pack.frond")),
    ("std/tls/Tls13.frond", include_str!("../../../std/tls/Tls13.frond")),
    // mem sub-library of std/core: generic T[] container primitives
    ("std/core/mem/pack.frond", include_str!("../../../std/core/mem/pack.frond")),
    ("std/core/mem/Mem.frond", include_str!("../../../std/core/mem/Mem.frond")),
    // rand module (PRNG wrappers over builtin/rand)
    ("std/rand/pack.frond", include_str!("../../../std/rand/pack.frond")),
    ("std/rand/Rand.frond", include_str!("../../../std/rand/Rand.frond")),
    // json module (pure Frond; layered: value ADT / parser / serializer)
    ("std/json/pack.frond", include_str!("../../../std/json/pack.frond")),
    ("std/json/Json.frond", include_str!("../../../std/json/Json.frond")),
    ("std/json/Parse.frond", include_str!("../../../std/json/Parse.frond")),
    ("std/json/Format.frond", include_str!("../../../std/json/Format.frond")),
    // collections module (pure Frond containers: List + str/i64 keyed maps & sets)
    ("std/collections/pack.frond", include_str!("../../../std/collections/pack.frond")),
    ("std/collections/List.frond", include_str!("../../../std/collections/List.frond")),
    ("std/collections/ArrayList.frond", include_str!("../../../std/collections/ArrayList.frond")),
    ("std/collections/LinkedList.frond", include_str!("../../../std/collections/LinkedList.frond")),
    ("std/collections/Map.frond", include_str!("../../../std/collections/Map.frond")),
    ("std/collections/Set.frond", include_str!("../../../std/collections/Set.frond")),
    ("std/collections/HashMap.frond", include_str!("../../../std/collections/HashMap.frond")),
    ("std/collections/IntMap.frond", include_str!("../../../std/collections/IntMap.frond")),
    ("std/collections/HashSet.frond", include_str!("../../../std/collections/HashSet.frond")),
    ("std/collections/IntSet.frond", include_str!("../../../std/collections/IntSet.frond")),
    ("std/encoding/pack.frond", include_str!("../../../std/encoding/pack.frond")),
    ("std/encoding/Hex.frond", include_str!("../../../std/encoding/Hex.frond")),
    ("std/encoding/Base64.frond", include_str!("../../../std/encoding/Base64.frond")),
];

/// Looks up a stdlib file by path.
pub fn find(path: &str) -> Option<&'static str> {
    BUILTIN_FILES
        .iter()
        .chain(STD_FILES.iter())
        .find(|(p, _)| *p == path)
        .map(|(_, src)| *src)
}
