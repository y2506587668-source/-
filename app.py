"""
app.py — 新闻爬虫 → GPT结构化提取 → 板块映射 → 板块成分股候选 → 信号落盘 → 回测雏形

✅ 功能清单（对齐需求）：
1) 多源网页爬取：东方财富多个栏目 + 可自定义 URL / RSS
2) 对每条新闻用 OpenAI GPT 提取：sentiment(-2..2)、impact(0..3)、tags、summary（Structured Outputs / JSON Schema）
3) 稳健板块映射：关键词字典 + GPT tags 双通道；支持可编辑 sector_bk_map.json（自定义/申万板块 → 东方财富 BK 代码 + 关键词）
4) 从东方财富 push2 API 抓取 BK 成分股；按可配置打分规则排序输出候选
5) 聚合板块得分并生成信号（关注/中性/谨慎）
6) 每次运行自动落盘 snapshot（CSV/JSON + 每板块 stocks.csv），便于复盘/回测
7) Streamlit UI：参数、运行按钮、表格、图表、sector_bk_map 编辑器、回测雏形（读取 snapshots 计算简单收益指标）
8) 常见异常与反爬：UA、限速、缓存、重试；关键步骤输出可读日志与错误提示

⚠️ 免责声明：
- 本应用输出为“候选池/关注信号”，仅用于研究与教育演示，不构成投资建议。
- 东方财富相关数据接口属于非官方公开调用方式，字段与可用性可能随时变化；应用内提供降级方案与日志提示。

环境要求：
- Python 3.10+
- 依赖见本文档底部（requirements 片段）
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# 基础常量与默认配置
# =========================

APP_NAME = "🧠 新闻 → 板块 → 个股：情绪信号与回测雏形"
APP_VERSION = "1.0.0"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_NEWS_PER_SOURCE = 20
DEFAULT_TOTAL_NEWS_LIMIT = 60

DEFAULT_SLEEP_SEC = 1.0
DEFAULT_RANDOM_JITTER = 0.4

EASTMONEY_SECTIONS: Dict[str, str] = {
    "国内经济": "https://finance.eastmoney.com/a/cgnjj.html",
    "国际经济": "https://finance.eastmoney.com/a/cgjjj.html",
    "宏观经济": "https://finance.eastmoney.com/a/cmacro.html",
    "财经首页": "https://finance.eastmoney.com/",
}

EASTMONEY_PUSH2_BASE = "https://push2.eastmoney.com"
EASTMONEY_PUSH2HIS_BASE = "https://push2his.eastmoney.com"

DEFAULT_UT = "bd1d9ddb04089700cf9c27f6f7426281"

DATA_DIR = Path("data")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = Path("sector_bk_map.json")

FALLBACK_SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "AI/算力": ["人工智能", "大模型", "算力", "服务器", "GPU", "AIGC", "数据中心", "IDC"],
    "半导体": ["半导体", "芯片", "晶圆", "封测", "EDA", "光刻"],
    "新能源": ["新能源", "光伏", "风电", "储能", "电池", "锂电", "充电桩"],
    "汽车/智能驾驶": ["智能驾驶", "自动驾驶", "车企", "电动车", "智驾", "汽车零部件"],
    "金融/券商": ["券商", "银行", "保险", "融资", "利率", "降准", "降息"],
    "消费": ["消费", "白酒", "零售", "餐饮", "旅游", "免税", "家电"],
    "地产/基建": ["地产", "房地产", "基建", "城投", "地方债", "水泥", "钢铁"],
    "医药": ["医药", "创新药", "医疗", "疫苗", "医保", "器械"],
    "军工": ["军工", "航天", "航空", "导弹", "舰船"],
    "黄金/大宗": ["黄金", "原油", "有色", "铜", "铝", "煤炭", "大宗商品"],
}

DEFAULT_MAP_TEMPLATE: Dict[str, Any] = {
    "meta": {
        "schema_version": 1,
        "last_updated": dt.date.today().isoformat(),
        "notes": "示例模板：请按你的“申万/自定义”板块体系补全 BK 代码与关键词。"
    },
    "sectors": [
        {
            "enabled": True,
            "sector": "半导体",
            "bk_codes": ["BK0475"],
            "keywords": ["半导体", "芯片", "晶圆", "封测", "EDA", "光刻"],
            "priority": 80,
            "notes": "示例：用你自己的 BK 代码覆盖"
        },
        {
            "enabled": True,
            "sector": "AI/算力",
            "bk_codes": ["BK0737"],
            "keywords": ["人工智能", "大模型", "算力", "GPU", "AIGC", "数据中心", "IDC", "服务器"],
            "priority": 70,
            "notes": "示例：AI 相关"
        },
        {
            "enabled": True,
            "sector": "金融/券商",
            "bk_codes": ["BK0473"],
            "keywords": ["券商", "证券", "银行", "保险", "降准", "降息", "资本市场"],
            "priority": 60,
            "notes": "示例：证券/银行/保险等"
        }
    ]
}


# =========================
# UI 日志（可读日志输出）
# =========================

def ui_log(msg: str, level: str = "INFO") -> None:
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    now = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{now}] [{level}] {msg}")


def ui_show_logs() -> None:
    with st.expander("📜 运行日志（点击展开）", expanded=False):
        logs = st.session_state.get("logs", [])
        if not logs:
            st.caption("暂无日志。点击“运行”后这里会显示关键步骤信息。")
            return
        st.code("\n".join(logs[-400:]))
        if st.button("清空日志", key="clear_logs"):
            st.session_state["logs"] = []
            st.rerun()


# =========================
# HTTP 与反爬：Session + 重试 + 限速
# =========================

@dataclass
class CrawlPolicy:
    sleep_sec: float = DEFAULT_SLEEP_SEC
    jitter_sec: float = DEFAULT_RANDOM_JITTER
    timeout_sec: int = 15
    max_retries: int = 2


def _sleep(policy: CrawlPolicy) -> None:
    delay = max(0.0, policy.sleep_sec + random.random() * policy.jitter_sec)
    time.sleep(delay)


@st.cache_resource
def get_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (NewsSentimentResearchBot/1.0; +https://example.com/bot)",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def parse_maybe_jsonp(text: str) -> Any:
    t = (text or "").strip()
    if not t:
        raise ValueError("空响应")
    if t[0] != "{" and t[0] != "[":
        m = re.search(r"[\(\[]\s*({.*})\s*[\)\]]\s*;?\s*$", t, flags=re.S)
        if m:
            t = m.group(1)
        else:
            m2 = re.search(r"\(\s*({.*})\s*\)\s*;?\s*$", t, flags=re.S)
            if m2:
                t = m2.group(1)
    return json.loads(t)


def http_get_text(url: str, params: Optional[Dict[str, Any]], policy: CrawlPolicy) -> str:
    s = get_http_session()
    try:
        r = s.get(url, params=params, timeout=policy.timeout_sec)
        r.raise_for_status()
        return r.text
    except Exception as e:
        raise RuntimeError(f"请求失败：{url} | {e}") from e
    finally:
        _sleep(policy)


def http_get_json(url: str, params: Optional[Dict[str, Any]], policy: CrawlPolicy) -> Dict[str, Any]:
    txt = http_get_text(url, params=params, policy=policy)
    try:
        return parse_maybe_jsonp(txt)
    except Exception as e:
        raise RuntimeError(f"JSON/JSONP 解析失败：{url} | {e}") from e


# =========================
# 读取/编辑/保存 sector_bk_map.json
# =========================

def ensure_map_file() -> None:
    if MAP_PATH.exists():
        return
    ui_log("未发现 sector_bk_map.json，已自动生成模板文件。", "WARN")
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_MAP_TEMPLATE, f, ensure_ascii=False, indent=2)


def load_sector_map() -> Dict[str, Any]:
    ensure_map_file()
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "sectors" not in data or not isinstance(data["sectors"], list):
            raise ValueError("sector_bk_map.json 缺少 sectors 列表")
        return data
    except Exception as e:
        ui_log(f"加载 sector_bk_map.json 失败：{e}", "ERROR")
        return DEFAULT_MAP_TEMPLATE


def sector_map_to_df(m: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for s in m.get("sectors", []):
        rows.append({
            "enabled": bool(s.get("enabled", True)),
            "sector": str(s.get("sector", "")).strip(),
            "bk_codes": s.get("bk_codes", []) or [],
            "keywords": s.get("keywords", []) or [],
            "priority": int(s.get("priority", 50)),
            "notes": str(s.get("notes", "")).strip(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["enabled", "sector", "bk_codes", "keywords", "priority", "notes"])
    df["sector"] = df["sector"].fillna("").astype(str)
    return df


def _coerce_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        parts = re.split(r"[,\s]+", x.strip())
        return [p for p in (p.strip() for p in parts) if p]
    return [str(x).strip()]


def df_to_sector_map(df: pd.DataFrame) -> Dict[str, Any]:
    sectors = []
    for _, r in df.iterrows():
        sector = str(r.get("sector", "")).strip()
        if not sector:
            continue
        sectors.append({
            "enabled": bool(r.get("enabled", True)),
            "sector": sector,
            "bk_codes": _coerce_list(r.get("bk_codes")),
            "keywords": _coerce_list(r.get("keywords")),
            "priority": int(r.get("priority", 50) or 50),
            "notes": str(r.get("notes", "")).strip(),
        })
    out = {
        "meta": {
            "schema_version": 1,
            "last_updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "由 Streamlit 编辑器自动保存。"
        },
        "sectors": sectors
    }
    return out


def save_sector_map(m: Dict[str, Any]) -> None:
    tmp_path = MAP_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    tmp_path.replace(MAP_PATH)


# =========================
# 新闻抓取：东方财富多栏目 + RSS + 自定义 URL
# =========================

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def normalize_url(base_url: str, href: str) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    try:
        from urllib.parse import urljoin
        return urljoin(base_url, href)
    except Exception:
        return href


@st.cache_data(ttl=180)
def fetch_eastmoney_news_list(section_url: str, limit: int, policy: CrawlPolicy) -> pd.DataFrame:
    html = http_get_text(section_url, params=None, policy=policy)
    soup = BeautifulSoup(html, "html.parser")

    items = soup.select("div.newsList li a") or soup.select("div.newsList a")
    data = []
    for a in items:
        title = normalize_text(a.get_text())
        href = normalize_url(section_url, a.get("href", ""))
        if not title or not href:
            continue
        if len(title) < 8:
            continue
        data.append({"title": title, "url": href})
        if len(data) >= limit:
            break
    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def fetch_generic_page_titles(page_url: str, limit: int, policy: CrawlPolicy) -> pd.DataFrame:
    html = http_get_text(page_url, params=None, policy=policy)
    soup = BeautifulSoup(html, "html.parser")
    data = []

    for a in soup.select("a")[:400]:
        title = normalize_text(a.get_text())
        href = normalize_url(page_url, a.get("href", ""))
        if not title or not href:
            continue
        if len(title) < 10:
            continue
        if any(x in title for x in ["登录", "注册", "免责声明", "更多", "首页", "关于我们"]):
            continue
        data.append({"title": title, "url": href})
        if len(data) >= limit:
            break

    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def fetch_rss_feed(feed_url: str, limit: int, policy: CrawlPolicy) -> pd.DataFrame:
    xml_text = http_get_text(feed_url, params=None, policy=policy)
    feed = feedparser.parse(xml_text)

    rows = []
    for e in (feed.entries or [])[:limit]:
        title = normalize_text(getattr(e, "title", ""))
        link = normalize_url(feed_url, getattr(e, "link", ""))
        published = normalize_text(getattr(e, "published", "") or getattr(e, "updated", ""))
        summary = normalize_text(getattr(e, "summary", "") or getattr(e, "description", ""))
        if title and link:
            rows.append({"title": title, "url": link, "published": published, "summary_hint": summary[:500]})
    return pd.DataFrame(rows)


def is_probably_rss(url: str) -> bool:
    u = (url or "").lower()
    return any(u.endswith(x) for x in [".xml", ".rss", ".atom"]) or ("rss" in u and "?" in u) or ("feed" in u)


@st.cache_data(ttl=3600)
def fetch_article_body(url: str, max_chars: int, policy: CrawlPolicy) -> str:
    html = http_get_text(url, params=None, policy=policy)
    soup = BeautifulSoup(html, "html.parser")

    candidates = [
        soup.select_one("div#ContentBody"),
        soup.select_one("div.article-body"),
        soup.select_one("div.newsContent"),
        soup.select_one("div.txtinfos"),
        soup.select_one("article"),
    ]
    text = ""
    for c in candidates:
        if c:
            text = normalize_text(c.get_text(" ", strip=True))
            if len(text) >= 120:
                break
    if not text:
        text = normalize_text(soup.get_text(" ", strip=True))

    return text[:max_chars]


def dedupe_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["title_norm"] = tmp["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    tmp["url_norm"] = tmp["url"].astype(str).str.strip()
    tmp = tmp.drop_duplicates(subset=["title_norm", "url_norm"])
    tmp = tmp.drop(columns=["title_norm", "url_norm"])
    return tmp.reset_index(drop=True)


# =========================
# OpenAI GPT：Structured Outputs 结构化提取
# =========================

def sha_user_identifier(s: str) -> str:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return h[:32]


NEWS_SIGNAL_JSON_SCHEMA: Dict[str, Any] = {
    "name": "news_signal",
    "description": "从财经新闻中提取市场情绪与影响强度，并输出主题标签与摘要。",
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "integer", "minimum": -2, "maximum": 2},
            "impact": {"type": "integer", "minimum": 0, "maximum": 3},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "summary": {"type": "string", "maxLength": 80},
        },
        "required": ["sentiment", "impact", "tags", "summary"],
        "additionalProperties": False,
    },
    "strict": True,
}


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_data(ttl=24 * 3600)
def gpt_extract_one(
    title: str,
    body: str,
    model: str,
    user_id_hash: str,
) -> Dict[str, Any]:
    client = get_openai_client(os.getenv("OPENAI_API_KEY", ""))
    system_msg = (
        "你是量化研究助理。请基于输入新闻进行情绪与影响评估。"
        "输出必须符合给定 JSON schema。"
    )
    user_msg = f"""
请分析以下财经新闻，对A股整体情绪做判断，并提取结构化字段：

- sentiment：-2(强利空)、-1(利空)、0(中性)、1(利好)、2(强利好)
- impact：0(几乎无影响)~3(强影响)
- tags：3~8 个中文主题关键词
- summary：一句话原因（<=80字）

新闻标题：{title}

新闻正文（可能截断）：{body[:1800] if body else "（无正文，仅标题）"}
""".strip()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": NEWS_SIGNAL_JSON_SCHEMA},
            safety_identifier=user_id_hash,
        )
        msg = completion.choices[0].message

        refusal = getattr(msg, "refusal", None)
        if refusal:
            return {"sentiment": 0, "impact": 0, "tags": [], "summary": "模型拒答/无法评估"}

        content = (msg.content or "").strip()
        data = json.loads(content)

        data["sentiment"] = int(max(-2, min(2, int(data.get("sentiment", 0)))))
        data["impact"] = int(max(0, min(3, int(data.get("impact", 1)))))
        data["tags"] = [normalize_text(x) for x in (data.get("tags", []) or []) if normalize_text(x)][:8]
        data["summary"] = normalize_text(data.get("summary", ""))[:80]
        return data
    except Exception:
        return {"sentiment": 0, "impact": 1, "tags": [], "summary": "提取失败（降级为中性）"}


# =========================
# 板块映射：关键词 + GPT tags 双通道
# =========================

def build_sector_keyword_index(m: Dict[str, Any]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for s in m.get("sectors", []):
        sector = str(s.get("sector", "")).strip()
        if not sector:
            continue
        if not bool(s.get("enabled", True)):
            continue
        kws = s.get("keywords", None)
        if kws:
            idx[sector] = _coerce_list(kws)
        elif sector in FALLBACK_SECTOR_KEYWORDS:
            idx[sector] = FALLBACK_SECTOR_KEYWORDS[sector]
        else:
            idx[sector] = []
    for sec, kws in FALLBACK_SECTOR_KEYWORDS.items():
        idx.setdefault(sec, kws)
    return idx


def match_sectors_for_news(
    title: str,
    gpt_tags: List[str],
    gpt_summary: str,
    body: str,
    keyword_index: Dict[str, List[str]],
    max_hits: int = 3,
) -> List[str]:
    text = " ".join([title, gpt_summary, " ".join(gpt_tags or []), body or ""])
    text = normalize_text(text)

    hits: List[Tuple[str, int]] = []
    for sector, kws in keyword_index.items():
        score = 0
        for kw in kws:
            if kw and kw in text:
                score += 1
        if score > 0:
            hits.append((sector, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in hits[:max_hits]] or ["其他"]


# =========================
# 东方财富 push2：BK 成分股抓取 + 个股打分
# =========================

def fetch_bk_constituents(
    bk_code: str,
    limit: int,
    ut: str,
    policy: CrawlPolicy,
) -> pd.DataFrame:
    url = f"{EASTMONEY_PUSH2_BASE}/api/qt/clist/get"
    params = {
        "pn": 1,
        "pz": int(limit),
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": f"b:{bk_code}",
        "ut": ut,
        "fields": ",".join([
            "f12", "f14", "f2", "f3",
            "f5", "f6", "f7", "f8",
            "f62", "f66", "f72",
        ]),
    }
    j = http_get_json(url, params=params, policy=policy)
    diff = ((j.get("data") or {}).get("diff")) or []
    rows = []
    for d in diff:
        rows.append({
            "code": d.get("f12"),
            "name": d.get("f14"),
            "last": d.get("f2"),
            "pct": d.get("f3"),
            "vol": d.get("f5"),
            "amount": d.get("f6"),
            "amp": d.get("f7"),
            "turnover": d.get("f8"),
            "main_inflow": d.get("f62"),
            "super_inflow": d.get("f66"),
            "big_inflow": d.get("f72"),
        })
    df = pd.DataFrame(rows)
    return df


@dataclass
class StockScoreWeights:
    w_pct: float = 0.6
    w_main_inflow: float = 0.4
    w_turnover: float = 0.0
    inflow_scale: float = 1e8


SCORE_PRESETS: Dict[str, StockScoreWeights] = {
    "动量为主（仅涨跌幅）": StockScoreWeights(w_pct=1.0, w_main_inflow=0.0, w_turnover=0.0),
    "动量+资金流（默认）": StockScoreWeights(w_pct=0.6, w_main_inflow=0.4, w_turnover=0.0),
    "资金流为主": StockScoreWeights(w_pct=0.3, w_main_inflow=0.7, w_turnover=0.0),
    "动量+资金流+换手（更均衡）": StockScoreWeights(w_pct=0.5, w_main_inflow=0.35, w_turnover=0.15),
}


def score_stocks(df: pd.DataFrame, w: StockScoreWeights) -> pd.DataFrame:
    out = df.copy()
    for c in ["pct", "main_inflow", "turnover", "last"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")

    out["score"] = (
        w.w_pct * out["pct"].fillna(0)
        + w.w_main_inflow * (out["main_inflow"].fillna(0) / float(w.inflow_scale))
        + w.w_turnover * out["turnover"].fillna(0)
    )
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# =========================
# 板块打分与信号生成
# =========================

@dataclass
class SectorSignalPolicy:
    pos_threshold: float = 0.9
    neg_threshold: float = -0.9


def aggregate_sector_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    tmp = news_df.copy()
    tmp["article_score"] = tmp["sentiment"] * (1.0 + tmp["impact"] / 3.0)

    rows = []
    for _, r in tmp.iterrows():
        for sec in r["sectors"]:
            rows.append({"sector": sec, "article_score": r["article_score"]})
    s = pd.DataFrame(rows)
    if s.empty:
        return pd.DataFrame(columns=["sector", "count", "avg_score", "score", "signal"])

    agg = s.groupby("sector").agg(
        count=("article_score", "count"),
        avg_score=("article_score", "mean"),
    ).reset_index()
    agg["score"] = agg.apply(lambda x: x["avg_score"] * math.log1p(x["count"]), axis=1)
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    return agg


def add_sector_signal(agg: pd.DataFrame, p: SectorSignalPolicy) -> pd.DataFrame:
    out = agg.copy()
    def to_signal(v: float) -> str:
        if v >= p.pos_threshold:
            return "关注 📈"
        if v <= p.neg_threshold:
            return "谨慎 📉"
        return "中性 ⚖️"
    out["signal"] = out["score"].apply(to_signal)
    return out


# =========================
# Snapshot 落盘（用于回测/复盘）
# =========================

def snapshot_save(
    params: Dict[str, Any],
    news_enriched: pd.DataFrame,
    sector_signals: pd.DataFrame,
    sector_stock_candidates: Dict[str, pd.DataFrame],
) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SNAPSHOT_DIR / f"snapshot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": ts,
        "params": params,
        "sector_signals": sector_signals.to_dict(orient="records"),
        "news_count": int(len(news_enriched)),
        "sectors_with_stocks": list(sector_stock_candidates.keys()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    news_enriched.to_csv(out_dir / "news_enriched.csv", index=False, encoding="utf-8-sig")
    sector_signals.to_csv(out_dir / "sector_signals.csv", index=False, encoding="utf-8-sig")

    for sec, df in sector_stock_candidates.items():
        safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", sec)
        df.to_csv(out_dir / f"stocks_{safe}.csv", index=False, encoding="utf-8-sig")

    logs = st.session_state.get("logs", [])
    (out_dir / "run.log").write_text("\n".join(logs), encoding="utf-8")

    return out_dir


# =========================
# 回测雏形：读取 snapshots → 拉取 K 线 → 计算简单收益
# =========================

def infer_secid_from_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return ""
    if c.startswith("6"):
        return f"1.{c}"
    return f"0.{c}"


@st.cache_data(ttl=3600)
def fetch_daily_kline(
    secid: str,
    beg: str,
    end: str,
    ut: str,
    policy: CrawlPolicy,
    fqt: int = 1,
    klt: int = 101,
    lmt: int = 200,
) -> pd.DataFrame:
    url = f"{EASTMONEY_PUSH2HIS_BASE}/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "beg": beg,
        "end": end,
        "klt": klt,
        "fqt": fqt,
        "lmt": lmt,
        "ut": ut,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    j = http_get_json(url, params=params, policy=policy)
    data = j.get("data") or {}
    klines = data.get("klines") or []
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        rows.append({
            "date": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "vol": float(parts[5]) if len(parts) > 5 else None,
            "amount": float(parts[6]) if len(parts) > 6 else None,
            "amp": float(parts[7]) if len(parts) > 7 else None,
            "pct": float(parts[8]) if len(parts) > 8 else None,
            "chg": float(parts[9]) if len(parts) > 9 else None,
            "turnover": float(parts[10]) if len(parts) > 10 else None,
        })
    df = pd.DataFrame(rows)
    return df


def compute_forward_return(kline: pd.DataFrame, anchor_date: str, horizon: int) -> Optional[float]:
    if kline.empty:
        return None
    kl = kline.copy()
    kl["date"] = pd.to_datetime(kl["date"], errors="coerce")
    kl = kl.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    anchor = pd.to_datetime(anchor_date, errors="coerce")
    if pd.isna(anchor):
        return None
    candidates = kl[kl["date"] <= anchor]
    if candidates.empty:
        t0_idx = 0
    else:
        t0_idx = int(candidates.index.max())

    t1_idx = t0_idx + int(horizon)
    if t1_idx >= len(kl):
        return None

    c0 = float(kl.loc[t0_idx, "close"])
    c1 = float(kl.loc[t1_idx, "close"])
    if c0 <= 0:
        return None
    return c1 / c0 - 1.0


def list_snapshots() -> List[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted([p for p in SNAPSHOT_DIR.iterdir() if p.is_dir() and p.name.startswith("snapshot_")], reverse=True)


def load_snapshot_sector_signals(snapshot: Path) -> pd.DataFrame:
    p = snapshot / "sector_signals.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_snapshot_stocks(snapshot: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for f in snapshot.glob("stocks_*.csv"):
        try:
            df = pd.read_csv(f)
            sector = f.stem.replace("stocks_", "")
            out[sector] = df
        except Exception:
            continue
    return out


# =========================
# Streamlit UI
# =========================

def main() -> None:
    st.set_page_config(page_title="新闻情绪板块个股信号", layout="wide")
    st.title(APP_NAME)
    st.caption(f"版本：{APP_VERSION} ｜ 研究用途 Demo（不构成投资建议）")

    with st.sidebar:
        st.header("⚙️ 参数设置")

        st.subheader("新闻源")
        chosen_sections = st.multiselect(
            "选择东方财富栏目（至少 1 个）",
            options=list(EASTMONEY_SECTIONS.keys()),
            default=["国内经济", "国际经济"],
        )
        per_source = st.slider("每个新闻源抓取条数", 5, 60, DEFAULT_NEWS_PER_SOURCE, 5)
        total_limit = st.slider("总新闻上限（去重后）", 10, 200, DEFAULT_TOTAL_NEWS_LIMIT, 10)

        custom_urls = st.text_area(
            "自定义 URL / RSS（每行一个，可混合）",
            value="",
            height=100,
            placeholder="例如：某RSS地址或新闻列表页URL（每行一个）",
        )

        st.subheader("反爬与性能")
        sleep_sec = st.slider("请求间隔(秒)", 0.0, 3.0, DEFAULT_SLEEP_SEC, 0.1)
        jitter = st.slider("随机抖动(秒)", 0.0, 1.0, DEFAULT_RANDOM_JITTER, 0.1)
        fetch_body = st.checkbox("抓取正文（更准但更慢/更易触发反爬）", value=True)
        max_body_chars = st.slider("正文最大截断长度", 300, 3000, 1600, 100)

        st.subheader("OpenAI")
        model = st.text_input("模型", value=DEFAULT_OPENAI_MODEL)
        cache_gpt = st.checkbox("缓存 GPT 结果（省钱/更快）", value=True)
        st.caption("若你想强制重新抽取，可在右上角菜单清除缓存或取消勾选缓存。")

        st.subheader("板块信号阈值")
        pos_th = st.slider("关注阈值（>=）", 0.2, 2.0, 0.9, 0.1)
        neg_th = st.slider("谨慎阈值（<=）", -2.0, -0.2, -0.9, 0.1)

        st.subheader("个股候选")
        ut = st.text_input("东方财富 ut 参数（可留默认）", value=DEFAULT_UT)
        bk_stock_limit = st.slider("每个 BK 拉取成分股条数", 20, 200, 80, 10)
        top_stocks = st.slider("每个板块输出候选股数", 5, 50, 15, 5)
        top_sectors = st.slider("输出 Top 板块数", 3, 20, 8, 1)

        preset_name = st.selectbox("个股打分预设", options=list(SCORE_PRESETS.keys()), index=1)
        use_custom_weights = st.checkbox("自定义打分权重", value=False)
        preset = SCORE_PRESETS[preset_name]
        if use_custom_weights:
            w_pct = st.number_input("w_pct（涨跌幅权重）", value=float(preset.w_pct), step=0.05)
            w_inflow = st.number_input("w_main_inflow（主力净流入权重）", value=float(preset.w_main_inflow), step=0.05)
            w_turn = st.number_input("w_turnover（换手率权重）", value=float(preset.w_turnover), step=0.05)
            inflow_scale = st.number_input("inflow_scale（净流入缩放，默认 1e8=亿元）", value=float(preset.inflow_scale))
            weights = StockScoreWeights(w_pct=w_pct, w_main_inflow=w_inflow, w_turnover=w_turn, inflow_scale=inflow_scale)
        else:
            weights = preset

        st.subheader("保存与回测")
        save_snapshot_flag = st.checkbox("运行后保存 snapshot", value=True)

    tab_run, tab_map, tab_backtest = st.tabs(["🚀 运行", "🗺️ 板块映射编辑器", "📈 回测雏形"])

    policy = CrawlPolicy(
        sleep_sec=float(sleep_sec),
        jitter_sec=float(jitter),
        timeout_sec=15,
        max_retries=2,
    )

    with tab_run:
        st.subheader("运行：抓取新闻 → GPT抽取 → 板块信号 → 个股候选 → (可选)保存 snapshot")
        run_btn = st.button("开始运行", type="primary")

        if run_btn:
            st.session_state["logs"] = []
            ui_log("开始运行。")

            if not chosen_sections and not custom_urls.strip():
                st.error("请至少选择一个东方财富栏目或提供自定义 URL/RSS。")
                ui_log("未选择新闻源，终止。", "ERROR")
                ui_show_logs()
                st.stop()

            if not os.getenv("OPENAI_API_KEY"):
                st.error("未检测到 OPENAI_API_KEY。请按文档配置环境变量后再运行。")
                ui_log("未检测到 OPENAI_API_KEY，终止。", "ERROR")
                ui_show_logs()
                st.stop()

            ui_log("步骤1：抓取新闻列表（多源）。")
            frames = []

            for sec in chosen_sections:
                url = EASTMONEY_SECTIONS.get(sec)
                if not url:
                    continue
                try:
                    df = fetch_eastmoney_news_list(url, limit=per_source, policy=policy)
                    if not df.empty:
                        df["source"] = f"东方财富-{sec}"
                        frames.append(df)
                        ui_log(f"抓取 {sec} 成功：{len(df)} 条。")
                    else:
                        ui_log(f"抓取 {sec} 为空（可能结构变动/被拦）。", "WARN")
                except Exception as e:
                    ui_log(f"抓取 {sec} 失败：{e}", "WARN")

            custom_list = [x.strip() for x in custom_urls.splitlines() if x.strip()]
            for u in custom_list:
                try:
                    if is_probably_rss(u):
                        df = fetch_rss_feed(u, limit=per_source, policy=policy)
                        src_name = "自定义RSS"
                    else:
                        df = fetch_generic_page_titles(u, limit=per_source, policy=policy)
                        src_name = "自定义网页"
                    if not df.empty:
                        df["source"] = src_name
                        frames.append(df)
                        ui_log(f"抓取自定义源成功：{u} | {len(df)} 条。")
                    else:
                        ui_log(f"抓取自定义源为空：{u}", "WARN")
                except Exception as e:
                    ui_log(f"抓取自定义源失败：{u} | {e}", "WARN")

            if not frames:
                st.error("没有抓到任何新闻。可能被反爬拦截或页面结构变更。")
                ui_show_logs()
                st.stop()

            news = pd.concat(frames, ignore_index=True)
            news = dedupe_news(news).head(int(total_limit)).reset_index(drop=True)

            st.markdown("### 1) 新闻列表（去重后）")
            st.dataframe(
                news[["source", "title", "url"]],
                use_container_width=True,
                column_config={"url": st.column_config.LinkColumn("url")},
            )
            ui_log(f"新闻去重后数量：{len(news)}")

            ui_log("步骤2：GPT 结构化提取（sentiment/impact/tags/summary）。")
            user_id_hash = sha_user_identifier("streamlit_user")

            enriched_rows = []
            prog = st.progress(0)

            for i, r in news.iterrows():
                title = str(r["title"])
                url = str(r["url"])
                source = str(r["source"])

                body = ""
                if fetch_body:
                    try:
                        body = fetch_article_body(url, max_chars=int(max_body_chars), policy=policy)
                    except Exception:
                        body = ""

                if cache_gpt:
                    g = gpt_extract_one(title, body, model, user_id_hash)
                else:
                    g = gpt_extract_one(title, body + f"\nNOCACHE={random.random()}", model, user_id_hash)

                enriched_rows.append({
                    "source": source,
                    "title": title,
                    "url": url,
                    "body": body[:400] + ("..." if len(body) > 400 else ""),
                    "sentiment": g.get("sentiment", 0),
                    "impact": g.get("impact", 1),
                    "tags": g.get("tags", []),
                    "summary": g.get("summary", ""),
                })

                prog.progress((i + 1) / len(news))

            df = pd.DataFrame(enriched_rows)

            st.markdown("### 2) 单条新闻：GPT提取结果")
            show_df = df.copy()
            show_df["tags"] = show_df["tags"].apply(lambda x: " / ".join(x) if isinstance(x, list) else str(x))
            st.dataframe(
                show_df[["source", "title", "sentiment", "impact", "tags", "summary", "url"]],
                use_container_width=True,
                column_config={"url": st.column_config.LinkColumn("url")},
            )

            ui_log("步骤3：板块映射（关键词 + GPT tags 双通道）。")
            sector_map = load_sector_map()
            keyword_index = build_sector_keyword_index(sector_map)

            df["sectors"] = df.apply(
                lambda row: match_sectors_for_news(
                    title=row["title"],
                    gpt_tags=row["tags"],
                    gpt_summary=row["summary"],
                    body=row.get("body", ""),
                    keyword_index=keyword_index,
                    max_hits=3,
                ),
                axis=1
            )

            st.markdown("### 3) 新闻 → 板块归因")
            tmp = df.copy()
            tmp["sectors"] = tmp["sectors"].apply(lambda x: " / ".join(x) if isinstance(x, list) else str(x))
            st.dataframe(
                tmp[["title", "sentiment", "impact", "sectors", "summary"]],
                use_container_width=True,
            )

            ui_log("步骤4：聚合板块得分与信号生成。")
            agg = aggregate_sector_scores(df)
            sector_policy = SectorSignalPolicy(pos_threshold=float(pos_th), neg_threshold=float(neg_th))
            sector_signals = add_sector_signal(agg, sector_policy)

            st.markdown("### 4) 板块信号（聚合）")
            st.dataframe(sector_signals, use_container_width=True)
            st.bar_chart(sector_signals.set_index("sector")[["score"]])

            ui_log("步骤5：板块 → BK 成分股 → 个股候选。")
            enabled_sectors_df = sector_map_to_df(sector_map)
            enabled_sectors_df = enabled_sectors_df[enabled_sectors_df["enabled"] == True]

            top_df = sector_signals.head(int(top_sectors))
            sector_stock_candidates: Dict[str, pd.DataFrame] = {}

            for _, row in top_df.iterrows():
                sector = row["sector"]
                sig = row.get("signal", "")
                score = float(row.get("score", 0))

                if "关注" not in str(sig):
                    continue

                m_row = enabled_sectors_df[enabled_sectors_df["sector"].astype(str) == str(sector)]
                bk_codes: List[str] = []
                if not m_row.empty:
                    bk_codes = _coerce_list(m_row.iloc[0]["bk_codes"])

                if not bk_codes:
                    ui_log(f"板块 {sector} 未配置 BK 代码，跳过个股候选。", "WARN")
                    continue

                all_constituents = []
                for bk in bk_codes:
                    try:
                        cons = fetch_bk_constituents(bk, limit=int(bk_stock_limit), ut=ut, policy=policy)
                        if cons.empty:
                            ui_log(f"BK {bk} 成分股为空或抓取失败。", "WARN")
                            continue
                        cons["bk"] = bk
                        all_constituents.append(cons)
                        ui_log(f"拉取 BK {bk} 成分股：{len(cons)} 条。")
                    except Exception as e:
                        ui_log(f"拉取 BK {bk} 失败：{e}", "WARN")

                if not all_constituents:
                    continue

                stocks = pd.concat(all_constituents, ignore_index=True)
                stocks = stocks.dropna(subset=["code"]).drop_duplicates(subset=["code"]).reset_index(drop=True)

                ranked = score_stocks(stocks, weights).head(int(top_stocks)).copy()
                ranked.insert(0, "sector_score", score)
                ranked.insert(0, "sector", sector)
                ranked.insert(0, "sector_signal", sig)

                sector_stock_candidates[sector] = ranked

            st.markdown("### 5) 个股候选（仅对“关注”板块）")
            if not sector_stock_candidates:
                st.info("没有可输出的个股候选：可能没有“关注”板块，或 BK 接口抓取失败/未配置映射。")
            else:
                combined = pd.concat(sector_stock_candidates.values(), ignore_index=True)
                st.dataframe(combined, use_container_width=True)

            if save_snapshot_flag:
                ui_log("步骤6：保存 snapshot。")
                params = {
                    "sections": chosen_sections,
                    "per_source": per_source,
                    "total_limit": total_limit,
                    "custom_urls": custom_list,
                    "fetch_body": fetch_body,
                    "max_body_chars": max_body_chars,
                    "openai_model": model,
                    "cache_gpt": cache_gpt,
                    "sector_thresholds": {"pos": pos_th, "neg": neg_th},
                    "ut": ut,
                    "bk_stock_limit": bk_stock_limit,
                    "top_sectors": top_sectors,
                    "top_stocks": top_stocks,
                    "stock_score_weights": weights.__dict__,
                }

                out_dir = snapshot_save(params, df, sector_signals, sector_stock_candidates)
                st.success(f"✅ 已保存 snapshot：{out_dir}")
                ui_log(f"snapshot 已保存：{out_dir}")

                summary_bytes = (out_dir / "summary.json").read_bytes()
                st.download_button(
                    "下载 summary.json",
                    data=summary_bytes,
                    file_name="summary.json",
                    mime="application/json",
                )

            ui_show_logs()

    with tab_map:
        st.subheader("板块映射编辑器：sector_bk_map.json（可编辑/保存/下载）")
        st.caption("字段说明：sector=你的板块名；bk_codes=对应东方财富 BK 代码列表；keywords=用于新闻归因关键词。")

        m = load_sector_map()
        df_map = sector_map_to_df(m)

        edited = st.data_editor(
            df_map,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "enabled": st.column_config.CheckboxColumn("enabled"),
                "sector": st.column_config.TextColumn("sector", help="你的板块名（申万/自定义）"),
                "bk_codes": st.column_config.ListColumn("bk_codes", help="例如：BK0473"),
                "keywords": st.column_config.ListColumn("keywords", help="用于新闻->板块匹配"),
                "priority": st.column_config.NumberColumn("priority", help="优先级（可选）"),
                "notes": st.column_config.TextColumn("notes", help="备注"),
            },
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("保存到 sector_bk_map.json"):
                new_map = df_to_sector_map(edited)
                save_sector_map(new_map)
                st.success("已保存。")
                ui_log("保存 sector_bk_map.json 成功。")

        with col2:
            if st.button("重置为模板"):
                save_sector_map(DEFAULT_MAP_TEMPLATE)
                st.warning("已重置。")
                ui_log("已重置 sector_bk_map.json 为模板。", "WARN")
                st.rerun()

        with col3:
            cur = df_to_sector_map(edited)
            st.download_button(
                "下载当前映射 JSON",
                data=json.dumps(cur, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="sector_bk_map.json",
                mime="application/json",
            )

        ui_show_logs()

    with tab_backtest:
        st.subheader("回测雏形：读取 snapshots → 简单收益评估（等权）")
        st.caption("说明：这里只是雏形，用于验证“信号/候选”是否有基础统计优势；真实回测需处理交易日、滑点、成本、风控等。")

        snaps = list_snapshots()
        if not snaps:
            st.info("暂无 snapshots。请先在“运行”页勾选保存 snapshot 并执行一次。")
            return

        snap_names = [p.name for p in snaps]
        chosen = st.multiselect("选择 snapshot（可多选）", options=snap_names, default=snap_names[:1])
        horizon = st.slider("收益计算窗口（交易日）", 1, 20, 5, 1)
        max_stocks_per_sector_bt = st.slider("每板块用于回测的股票数（取候选 TopN）", 1, 30, 10, 1)

        run_bt = st.button("运行回测（雏形）", type="primary", key="bt_run")

        if run_bt:
            st.session_state["logs"] = []
            ui_log("开始回测。")

            rows = []
            policy_bt = CrawlPolicy(sleep_sec=0.5, jitter_sec=0.2, timeout_sec=15, max_retries=2)

            for name in chosen:
                snap = SNAPSHOT_DIR / name
                m = re.search(r"snapshot_(\d{8})_(\d{6})", name)
                if m:
                    d_str = m.group(1)
                    anchor_date = dt.datetime.strptime(d_str, "%Y%m%d").date().isoformat()
                else:
                    anchor_date = dt.date.today().isoformat()

                signals = load_snapshot_sector_signals(snap)
                stocks_map = load_snapshot_stocks(snap)
                if signals.empty or not stocks_map:
                    ui_log(f"{name} 缺少 sector_signals 或 stocks 文件。", "WARN")
                    continue

                all_returns = []
                n_used = 0

                for _, sdf in stocks_map.items():
                    if sdf.empty:
                        continue
                    sdf = sdf.head(int(max_stocks_per_sector_bt))
                    for _, r in sdf.iterrows():
                        code = str(r.get("code", "")).strip()
                        if not code:
                            continue
                        secid = infer_secid_from_code(code)
                        if not secid:
                            continue
                        try:
                            kl = fetch_daily_kline(
                                secid=secid,
                                beg="0",
                                end="20500101",
                                ut=DEFAULT_UT,
                                policy=policy_bt,
                                lmt=200,
                            )
                            ret = compute_forward_return(kl, anchor_date=anchor_date, horizon=int(horizon))
                            if ret is None:
                                continue
                            all_returns.append(ret)
                            n_used += 1
                        except Exception:
                            continue

                if not all_returns:
                    ui_log(f"{name} 无法计算收益（可能 K 线接口失败或样本不足）。", "WARN")
                    continue

                s = pd.Series(all_returns)
                rows.append({
                    "snapshot": name,
                    "anchor_date": anchor_date,
                    "horizon_td": horizon,
                    "n_stocks": n_used,
                    "mean_ret": float(s.mean()),
                    "median_ret": float(s.median()),
                    "win_rate": float((s > 0).mean()),
                    "best": float(s.max()),
                    "worst": float(s.min()),
                })

                ui_log(f"{name} 回测完成：n={n_used}，mean={s.mean():.4f}")

            res = pd.DataFrame(rows)
            if res.empty:
                st.error("回测结果为空：可能网络被拦截、K线接口失效，或 snapshot 缺少 stocks 文件。")
                ui_show_logs()
                st.stop()

            st.markdown("### 回测结果汇总")
            st.dataframe(res, use_container_width=True)

            st.markdown("### 平均收益（按 snapshot）")
            st.bar_chart(res.set_index("snapshot")[["mean_ret"]])

            st.download_button(
                "下载回测结果 CSV",
                data=res.to_csv(index=False).encode("utf-8-sig"),
                file_name="backtest_result.csv",
                mime="text/csv",
            )

            ui_show_logs()


if __name__ == "__main__":
    main()
