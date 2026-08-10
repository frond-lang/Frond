//! 标准库源码嵌入表与查询。
//!
//! 使用 `include_str!` 在编译期将 `.kz` 源文件嵌入二进制，供 `Loader` parse。
//!
//! ## 目录结构
//!
//! ```text
//! src/stdlib/
//! ├── builtin/          # 内置模块（默认可见，无需 import）
//! │   ├── {io,net,time}/Raw.kz  # @extern("C") 原语（按领域拆分）
//! │   ├── error/        # Err/Error/CastError/IOError/TimeError
//! │   ├── cast/         # Raw.kz(@extern 原语) + Cast.kz(kuzo wrapper)
//! │   ├── reflect/      # 运行时反射格式化（Reflect.format）
//! │   ├── io/           # Reader/Writer trait + Console(print/println/scan...)
//! │   └── iter/         # Iterator<T> 迭代器
//! └── std/              # 标准库（需 import std.xxx）
//!     ├── io/           # File/Path/Buffered/Dir/Fs
//!     ├── time/         # Duration/Instant/SystemTime/DateTime/Calendar/Timer
//!     └── net/          # Addr/TcpListener/TcpStream/UdpSocket/Dns
//! ```

/// 标准库文件条目：(相对路径, 源码内容)
pub type StdlibFile = (&'static str, &'static str);

/// builtin 模块文件清单（默认可见，无需 import）
///
/// 顺序按依赖关系排列：
///   Raw(@extern 原语) → error → cast(Raw+Cast) → reflect → io(Reader/Writer/Console) → net → time → iter
/// @extern("C") 原语最先加载：全局可见，供 builtin/std wrapper 调用
/// cast/Raw.kz 与 {io,net,time}/Raw.kz 一起在原语层加载
pub const BUILTIN_FILES: &[StdlibFile] = &[
    // @extern("C") 原语模块（全局可见，按领域拆分到 Raw.kz）
    ("builtin/io/Raw.kz", include_str!("../stdlib/builtin/io/Raw.kz")),
    ("builtin/net/Raw.kz", include_str!("../stdlib/builtin/net/Raw.kz")),
    ("builtin/time/Raw.kz", include_str!("../stdlib/builtin/time/Raw.kz")),
    // error 模块
    ("builtin/error/pack.kz", include_str!("../stdlib/builtin/error/pack.kz")),
    ("builtin/error/Err.kz", include_str!("../stdlib/builtin/error/Err.kz")),
    ("builtin/error/Error.kz", include_str!("../stdlib/builtin/error/Error.kz")),
    ("builtin/error/CastError.kz", include_str!("../stdlib/builtin/error/CastError.kz")),
    ("builtin/error/IOError.kz", include_str!("../stdlib/builtin/error/IOError.kz")),
    ("builtin/error/TimeError.kz", include_str!("../stdlib/builtin/error/TimeError.kz")),
    // cast 模块（类型转换原语 + kuzo wrapper）
    ("builtin/cast/pack.kz", include_str!("../stdlib/builtin/cast/pack.kz")),
    ("builtin/cast/Raw.kz", include_str!("../stdlib/builtin/cast/Raw.kz")),
    ("builtin/cast/Cast.kz", include_str!("../stdlib/builtin/cast/Cast.kz")),
    // reflect 模块（运行时反射，Raw.kz 原语 + Reflect.kz wrapper）
    ("builtin/reflect/pack.kz", include_str!("../stdlib/builtin/reflect/pack.kz")),
    ("builtin/reflect/Raw.kz", include_str!("../stdlib/builtin/reflect/Raw.kz")),
    ("builtin/reflect/Reflect.kz", include_str!("../stdlib/builtin/reflect/Reflect.kz")),
    // io 模块（Reader/Writer trait + Console 标准IO）
    ("builtin/io/pack.kz", include_str!("../stdlib/builtin/io/pack.kz")),
    ("builtin/io/Reader.kz", include_str!("../stdlib/builtin/io/Reader.kz")),
    ("builtin/io/Writer.kz", include_str!("../stdlib/builtin/io/Writer.kz")),
    ("builtin/io/Console.kz", include_str!("../stdlib/builtin/io/Console.kz")),
    // net 模块（pack 声明，Raw 已在原语层加载）
    ("builtin/net/pack.kz", include_str!("../stdlib/builtin/net/pack.kz")),
    // time 模块（pack 声明，Raw 已在原语层加载）
    ("builtin/time/pack.kz", include_str!("../stdlib/builtin/time/pack.kz")),
    // str 模块（UTF-8 解码原语，iter 模块依赖）
    ("builtin/str/pack.kz", include_str!("../stdlib/builtin/str/pack.kz")),
    ("builtin/str/Raw.kz", include_str!("../stdlib/builtin/str/Raw.kz")),
    // iter 模块
    ("builtin/iter/pack.kz", include_str!("../stdlib/builtin/iter/pack.kz")),
    ("builtin/iter/Iterator.kz", include_str!("../stdlib/builtin/iter/Iterator.kz")),
];

/// std 模块文件清单（需 import std.xxx 加载）
///
/// 顺序按依赖关系排列：io → time → net
/// reflect 已移至 builtin/reflect（默认可见），Console 已移至 builtin/io（默认可见）
pub const STD_FILES: &[StdlibFile] = &[
    // io 模块（Console 已移至 builtin/io）
    ("std/io/pack.kz", include_str!("../stdlib/std/io/pack.kz")),
    ("std/io/Path.kz", include_str!("../stdlib/std/io/Path.kz")),
    ("std/io/File.kz", include_str!("../stdlib/std/io/File.kz")),
    ("std/io/Buffered.kz", include_str!("../stdlib/std/io/Buffered.kz")),
    ("std/io/Dir.kz", include_str!("../stdlib/std/io/Dir.kz")),
    ("std/io/Fs.kz", include_str!("../stdlib/std/io/Fs.kz")),
    // time 模块
    ("std/time/pack.kz", include_str!("../stdlib/std/time/pack.kz")),
    ("std/time/Duration.kz", include_str!("../stdlib/std/time/Duration.kz")),
    ("std/time/Instant.kz", include_str!("../stdlib/std/time/Instant.kz")),
    ("std/time/SystemTime.kz", include_str!("../stdlib/std/time/SystemTime.kz")),
    ("std/time/DateTime.kz", include_str!("../stdlib/std/time/DateTime.kz")),
    ("std/time/Calendar.kz", include_str!("../stdlib/std/time/Calendar.kz")),
    ("std/time/Timer.kz", include_str!("../stdlib/std/time/Timer.kz")),
    // net 模块（TcpStream 在 TcpListener 之前：TcpListener 依赖 __net_tcp_close 定义于 TcpStream）
    ("std/net/pack.kz", include_str!("../stdlib/std/net/pack.kz")),
    ("std/net/Addr.kz", include_str!("../stdlib/std/net/Addr.kz")),
    ("std/net/Dns.kz", include_str!("../stdlib/std/net/Dns.kz")),
    ("std/net/TcpStream.kz", include_str!("../stdlib/std/net/TcpStream.kz")),
    ("std/net/TcpListener.kz", include_str!("../stdlib/std/net/TcpListener.kz")),
    ("std/net/UdpSocket.kz", include_str!("../stdlib/std/net/UdpSocket.kz")),
    // math 模块
    ("std/math/pack.kz",  include_str!("../stdlib/std/math/pack.kz")),
    ("std/math/Math.kz",  include_str!("../stdlib/std/math/Math.kz")),
    ("std/math/Power.kz", include_str!("../stdlib/std/math/Power.kz")),
    ("std/math/Trig.kz",  include_str!("../stdlib/std/math/Trig.kz")),
    ("std/math/Round.kz", include_str!("../stdlib/std/math/Round.kz")),
];

/// 按路径查找标准库文件
pub fn find(path: &str) -> Option<&'static str> {
    BUILTIN_FILES
        .iter()
        .chain(STD_FILES.iter())
        .find(|(p, _)| *p == path)
        .map(|(_, src)| *src)
}

/// 按模块名前缀查找（如 "std/io" 返回所有 std/io/*.kz）
pub fn find_by_prefix(prefix: &str) -> impl Iterator<Item = StdlibFile> + use<'_> {
    BUILTIN_FILES
        .iter()
        .chain(STD_FILES.iter())
        .copied()
        .filter(move |(p, _)| p.starts_with(prefix))
}
