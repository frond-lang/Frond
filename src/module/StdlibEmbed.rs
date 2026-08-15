//! Stdlib source embed table and lookups.
//!
//! Uses `include_str!` to embed `.kz` source files into the binary at compile time,
//! for `Loader` to parse.
//!
//! ## Directory layout
//!
//! ```text
//! src/stdlib/
//! ├── builtin/          # builtin modules (visible by default, no import needed)
//! │   ├── {io,net,time}/Raw.kz  # @extern("C") primitives (split by domain)
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
/// {io,net,time}/Raw.kz are loaded together at the primitive layer
pub const BUILTIN_FILES: &[StdlibFile] = &[
    // @extern("C") primitive modules (globally visible, split by domain into Raw.kz)
    ("builtin/io/Raw.kz", include_str!("../stdlib/builtin/io/Raw.kz")),
    ("builtin/net/Raw.kz", include_str!("../stdlib/builtin/net/Raw.kz")),
    ("builtin/time/Raw.kz", include_str!("../stdlib/builtin/time/Raw.kz")),
    ("builtin/terminal/Raw.kz", include_str!("../stdlib/builtin/terminal/Raw.kz")),
    // error module
    ("builtin/error/pack.kz", include_str!("../stdlib/builtin/error/pack.kz")),
    ("builtin/error/Err.kz", include_str!("../stdlib/builtin/error/Err.kz")),
    ("builtin/error/Error.kz", include_str!("../stdlib/builtin/error/Error.kz")),
    ("builtin/error/IOError.kz", include_str!("../stdlib/builtin/error/IOError.kz")),
    ("builtin/error/TimeError.kz", include_str!("../stdlib/builtin/error/TimeError.kz")),
    ("builtin/error/TerminalError.kz", include_str!("../stdlib/builtin/error/TerminalError.kz")),
    ("builtin/error/OsError.kz", include_str!("../stdlib/builtin/error/OsError.kz")),
    // reflect module (runtime reflection, Raw.kz primitives + Reflect.kz wrapper)
    ("builtin/reflect/pack.kz", include_str!("../stdlib/builtin/reflect/pack.kz")),
    ("builtin/reflect/Raw.kz", include_str!("../stdlib/builtin/reflect/Raw.kz")),
    ("builtin/reflect/Reflect.kz", include_str!("../stdlib/builtin/reflect/Reflect.kz")),
    // io module (Reader/Writer trait + Console standard IO)
    ("builtin/io/pack.kz", include_str!("../stdlib/builtin/io/pack.kz")),
    ("builtin/io/Reader.kz", include_str!("../stdlib/builtin/io/Reader.kz")),
    ("builtin/io/Writer.kz", include_str!("../stdlib/builtin/io/Writer.kz")),
    ("builtin/io/Console.kz", include_str!("../stdlib/builtin/io/Console.kz")),
    // net module (pack declaration; Raw already loaded at the primitive layer)
    ("builtin/net/pack.kz", include_str!("../stdlib/builtin/net/pack.kz")),
    // time module (pack declaration; Raw already loaded at the primitive layer)
    ("builtin/time/pack.kz", include_str!("../stdlib/builtin/time/pack.kz")),
    // str module (UTF-8 decoding primitives; depended on by the iter module)
    ("builtin/str/pack.kz", include_str!("../stdlib/builtin/str/pack.kz")),
    ("builtin/str/Raw.kz", include_str!("../stdlib/builtin/str/Raw.kz")),
    // iter module
    ("builtin/iter/pack.kz", include_str!("../stdlib/builtin/iter/pack.kz")),
    ("builtin/iter/Iterator.kz", include_str!("../stdlib/builtin/iter/Iterator.kz")),
    // terminal module (pack declaration; Raw already loaded at the primitive layer)
    ("builtin/terminal/pack.kz", include_str!("../stdlib/builtin/terminal/pack.kz")),
    ("builtin/os/pack.kz", include_str!("../stdlib/builtin/os/pack.kz")),
    ("builtin/os/Raw.kz", include_str!("../stdlib/builtin/os/Raw.kz")),
];

/// Standard library module file manifest (requires `import std.xxx` to load).
///
/// Ordered by dependency: io → time → net
/// reflect has moved to builtin/reflect (visible by default); Console has moved to builtin/io (visible by default)
pub const STD_FILES: &[StdlibFile] = &[
    // io module (Console has moved to builtin/io)
    ("std/os/pack.kz", include_str!("../stdlib/std/os/pack.kz")),
    ("std/os/Env.kz", include_str!("../stdlib/std/os/Env.kz")),
    ("std/os/Tty.kz", include_str!("../stdlib/std/os/Tty.kz")),
    ("std/os/Proc.kz", include_str!("../stdlib/std/os/Proc.kz")),
    ("std/os/Info.kz", include_str!("../stdlib/std/os/Info.kz")),
    ("std/os/Os.kz", include_str!("../stdlib/std/os/Os.kz")),
    ("std/io/pack.kz", include_str!("../stdlib/std/io/pack.kz")),
    ("std/io/Path.kz", include_str!("../stdlib/std/io/Path.kz")),
    ("std/io/File.kz", include_str!("../stdlib/std/io/File.kz")),
    ("std/io/Buffered.kz", include_str!("../stdlib/std/io/Buffered.kz")),
    ("std/io/Dir.kz", include_str!("../stdlib/std/io/Dir.kz")),
    ("std/io/Fs.kz", include_str!("../stdlib/std/io/Fs.kz")),
    // time module
    ("std/time/pack.kz", include_str!("../stdlib/std/time/pack.kz")),
    ("std/time/Duration.kz", include_str!("../stdlib/std/time/Duration.kz")),
    ("std/time/Instant.kz", include_str!("../stdlib/std/time/Instant.kz")),
    ("std/time/SystemTime.kz", include_str!("../stdlib/std/time/SystemTime.kz")),
    ("std/time/DateTime.kz", include_str!("../stdlib/std/time/DateTime.kz")),
    ("std/time/Calendar.kz", include_str!("../stdlib/std/time/Calendar.kz")),
    ("std/time/Timer.kz", include_str!("../stdlib/std/time/Timer.kz")),
    // net module (TcpStream before TcpListener: TcpListener depends on __net_tcp_close defined in TcpStream)
    ("std/net/pack.kz", include_str!("../stdlib/std/net/pack.kz")),
    ("std/net/Addr.kz", include_str!("../stdlib/std/net/Addr.kz")),
    ("std/net/Dns.kz", include_str!("../stdlib/std/net/Dns.kz")),
    ("std/net/TcpStream.kz", include_str!("../stdlib/std/net/TcpStream.kz")),
    ("std/net/TcpListener.kz", include_str!("../stdlib/std/net/TcpListener.kz")),
    ("std/net/UdpSocket.kz", include_str!("../stdlib/std/net/UdpSocket.kz")),
    // math module
    ("std/math/pack.kz",  include_str!("../stdlib/std/math/pack.kz")),
    ("std/math/Math.kz",  include_str!("../stdlib/std/math/Math.kz")),
    ("std/math/Power.kz", include_str!("../stdlib/std/math/Power.kz")),
    ("std/math/Trig.kz",  include_str!("../stdlib/std/math/Trig.kz")),
    ("std/math/Round.kz", include_str!("../stdlib/std/math/Round.kz")),
    // terminal module (high-level terminal abstraction)
    ("std/terminal/pack.kz", include_str!("../stdlib/std/terminal/pack.kz")),
    ("std/terminal/Session.kz", include_str!("../stdlib/std/terminal/Session.kz")),
    ("std/terminal/Ansi.kz", include_str!("../stdlib/std/terminal/Ansi.kz")),
    ("std/terminal/Key.kz", include_str!("../stdlib/std/terminal/Key.kz")),
];

/// Looks up a stdlib file by path.
pub fn find(path: &str) -> Option<&'static str> {
    BUILTIN_FILES
        .iter()
        .chain(STD_FILES.iter())
        .find(|(p, _)| *p == path)
        .map(|(_, src)| *src)
}
