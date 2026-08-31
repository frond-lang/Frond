//! Symbols — name interner(NAME_RESOLUTION_PLAN S1)。
//!
//! 编译器里的"名字"从此有了唯一身份:`Sym(u32)`。同一字符串只驻留一次,
//! 后续所有比较/查表都是整数等值。规则:
//! - **插入路径走 intern**(put_type_def / 构造器与 trait 登记 / func_sig
//!   登记 / field_id_map 写入)——键在进表前先驻留;
//! - **读取路径走 find**(不插入)——IR 期/诊断期的查找与今天的字符串
//!   探测语义一致:键没被登记过就是 miss;
//! - `resolve(Sym) -> &str` 借用 `&self` 的存储,无锁无克隆(编译单线程
//!   分阶段;Analyzer 的 rayon 并行不触本表)。
//!
//! 行为中立法则:今天两个字符串键相等 ⟺ 换键后两个 Sym 相等——驻留表
//! 不改变任何解析裁决,只消灭重复哈希与拼写/拼接类静默错配。

use rustc_hash::FxHashMap;

/// Interned name handle。Copy + Eq + Hash,全编译器流通的名称身份。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(pub u32);

/// 驻留表:字符串 ↔ Sym 双向。增长型(只加不删),随 SemaResult 生命周期。
#[derive(Debug, Default)]
pub struct Symbols {
    map: FxHashMap<String, u32>,
    strings: Vec<Box<str>>,
}

impl Symbols {
    pub fn new() -> Self {
        Self::default()
    }

    /// 驻留(插入或取已有)。只在登记键的写入路径调用。
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.map.get(s) {
            return Sym(id);
        }
        let id = self.strings.len() as u32;
        self.strings.push(s.into());
        self.map.insert(s.to_string(), id);
        Sym(id)
    }

    /// 只读查找(不插入)。读取路径用:未登记的名字就是不存在,
    /// 与今天对 String 键的 miss 语义一致。
    pub fn find(&self, s: &str) -> Option<Sym> {
        self.map.get(s).map(|&id| Sym(id))
    }

    /// Sym → 原字符串。借用 &self,零克隆。
    pub fn resolve(&self, sym: Sym) -> &str {
        self.strings
            .get(sym.0 as usize)
            .map(|s| s.as_ref())
            .unwrap_or("")
    }

    /// 已驻留名字总数(诊断/统计)。
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// 驻留表中全部名字(诊断遍历;顺序 = 驻留序,确定性)。
    pub fn iter(&self) -> impl Iterator<Item = (Sym, &str)> {
        self.strings
            .iter()
            .enumerate()
            .map(|(i, s)| (Sym(i as u32), s.as_ref()))
    }
}
