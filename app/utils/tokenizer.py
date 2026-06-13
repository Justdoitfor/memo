"""中文分词适配 — jieba HMM + 单字过滤 + LRU 缓存

用途:
  1. FTS5 BM25 索引/查询: cut_for_fts() 返回空格连接的 token 串
  2. Trivial episode 过滤 / Pattern Miner: extract_keywords() TF-IDF 抽词

设计取舍:
  - 不写 SQLite C 扩展 tokenizer: 维护成本高 + 跨平台二进制麻烦
    改在 Python 侧切好后用 FTS5 'simple' tokenizer 按空格切, 等价但简单
  - 单字中文过滤: 中文单字 IDF 极低, 召回噪声大于信号 (e.g. "花" 命中
    花费/花园/花絮一堆无关文档). 英文/数字单字符保留 (IDF 高)
  - 数字/英文标点保持原样, 不动 (jieba 默认就保留)
  - 懒加载词典 + LRU: 启动 0 cost, 高频文本切分 <1µs

降级:
  - import jieba 失败时, cut_for_fts 退化为按 CJK/word 简单切分,
    不引入 hard requirement (理论上 jieba 装不上时项目仍可运行)
"""

from __future__ import annotations

import re
from functools import lru_cache
from threading import Lock

from loguru import logger

_DICT_LOADED = False
_DICT_LOCK = Lock()
_CN_RE = re.compile(r"[一-鿿]+")
# 降级路径: 没有 jieba 时按 CJK 字符串 / 英数字单元切
_FALLBACK_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z0-9_]+")

# 标记 jieba 是否可用 (动态探测, 不抛 ImportError)
try:
    import jieba  # type: ignore
    import jieba.analyse  # type: ignore

    _JIEBA_OK = True
except Exception as e:  # pragma: no cover - 依赖缺失时降级
    jieba = None  # type: ignore
    _JIEBA_OK = False
    logger.warning(f"jieba 未安装, 中文分词降级为 CJK 简单切分: {e}")


# 项目领域高频词 — 来自 _PRED_REGISTRY 中文模板的固定词
# (避免 jieba 把这些词误切, e.g. "过敏原" → "过敏" + "原")
_DOMAIN_WORDS = (
    "过敏原", "工作流", "代码评审", "知识图谱", "记忆中间件", "向量召回",
    "时间窗口", "行为信号", "用户画像", "隐式偏好",
)


def _ensure_dict() -> None:
    """懒加载 jieba 词典 + 领域词 + 用户词典. 仅运行一次."""
    global _DICT_LOADED
    if _DICT_LOADED or not _JIEBA_OK:
        return
    with _DICT_LOCK:
        if _DICT_LOADED:
            return
        # 关闭 jieba 启动日志 (默认会打 "Building prefix dict...")
        jieba.setLogLevel(60)
        for w in _DOMAIN_WORDS:
            jieba.add_word(w, freq=1000)
        # 加载用户自定义词典 (config 配置项)
        try:
            from app.config import config

            user_dict = getattr(config, "jieba_user_dict_path", None)
            if user_dict and user_dict.exists():
                jieba.load_userdict(str(user_dict))
                logger.info(f"jieba 加载用户词典: {user_dict}")
        except Exception as e:
            logger.debug(f"jieba 加载用户词典失败 (跳过): {e}")
        _DICT_LOADED = True


@lru_cache(maxsize=10_000)
def cut_for_fts(text: str) -> str:
    """切分中文 → 空格连接的 token 串, 供 FTS5 simple tokenizer 索引/查询.

    规则:
      - 中文按 jieba HMM 切, 单字中文过滤 (IDF 太低)
      - 英文/数字单字符保留 (IDF 高, 例如 'C', '5')
      - 标点不进 token
    """
    if not text:
        return ""

    # 全局 enable_jieba 关闭 → 走降级路径
    try:
        from app.config import config

        if not getattr(config, "enable_jieba", True):
            return _fallback_cut(text)
    except Exception:
        pass

    if not _JIEBA_OK:
        return _fallback_cut(text)

    _ensure_dict()
    tokens: list[str] = []
    for seg in jieba.cut(text, HMM=True):  # type: ignore[union-attr]
        seg = seg.strip()
        if not seg:
            continue
        # 单字中文过滤 (英文/数字单字符保留)
        if len(seg) == 1 and _CN_RE.match(seg):
            continue
        # 跳过纯标点/空白
        if not _FALLBACK_TOKEN_RE.fullmatch(seg):
            # jieba 可能切出 "你好," 这种带标点 token, 提取核心
            inner_tokens = _FALLBACK_TOKEN_RE.findall(seg)
            for t in inner_tokens:
                if len(t) == 1 and _CN_RE.match(t):
                    continue
                tokens.append(t)
            continue
        tokens.append(seg)
    return " ".join(tokens)


def _fallback_cut(text: str) -> str:
    """降级切分: 按 CJK 字符串 + 英数字单元拆, 不分词. 比 unicode61 略好.

    仍然会把"花生过敏"作为一整个 CJK 串处理, 不会按单字爆开.
    召回精度比 jieba 切词差, 但好于 unicode61 单字模式.
    """
    if not text:
        return ""
    return " ".join(_FALLBACK_TOKEN_RE.findall(text))


@lru_cache(maxsize=2_000)
def extract_keywords(text: str, top_k: int = 5) -> tuple[str, ...]:
    """TF-IDF 关键词抽取, 供 trivial 路径判定 / query expansion 用.

    返回 tuple 而非 list, 让 lru_cache 能直接缓存 (list 不可哈希).
    降级路径: 无 jieba 时按 cut_for_fts 取前 top_k 个 token.
    """
    if not text:
        return ()
    if not _JIEBA_OK:
        return tuple(cut_for_fts(text).split()[:top_k])
    _ensure_dict()
    try:
        return tuple(jieba.analyse.extract_tags(text, topK=top_k, withWeight=False))  # type: ignore[union-attr]
    except Exception as e:
        logger.debug(f"extract_keywords 失败, 降级: {e}")
        return tuple(cut_for_fts(text).split()[:top_k])
