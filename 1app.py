# app.py  (NO feedparser version)
# 多源网页新闻(东方财富+自定义网页URL) → GPT结构化情绪 → 板块映射(可编辑JSON) → BK成分股候选 → snapshot落盘 → 回测雏形
# 运行：pip install -U streamlit pandas requests beautifulsoup4 openai urllib3
#      set OPENAI_API_KEY=xxx  (Windows) / export OPENAI_API_KEY=xxx (mac/linux)
#      streamlit run app.py

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
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 基础常量
# =========================
APP_NAME = "🧠 新闻 → 板块 → 个股：情绪信号 & 回测雏形（无RSS依赖版）"
APP_VERSION = "1.0.1"

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
    "金融/券商": ["券商", "证券", "银行", "保险", "降准", "降息", "资本市场"],
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
        "notes": "示例模板：请按你的板块体系补全 BK 代码与关键词。",
    },
    "sectors": [
        {
            "enabled": True,
            "sector": "半导体",
            "bk_codes": ["BK0475"],
            "keywords": ["半导体", "芯片", "晶圆", "封测", "EDA", "光刻"],
            "priority": 80,
            "notes": "示例：请用你自己的 BK 覆盖",
        },
        {
            "enabled": True,
            "sector": "AI/算力",
            "bk_codes": ["BK0737"],
            "keywords": ["人工智能", "大模型", "算力", "GPU", "AIGC", "数据中心", "IDC", "服务器"],
            "priority": 70,
            "notes": "示例：AI相关",
        },
        {
            "enabled": True,
            "sector": "金融/券商",
            "bk_codes": ["BK0473"],
            "keywords": ["券商", "证券", "银行", "保险", "降准", "降息", "资本市场"],
            "priority": 60,
            "notes": "示例：金融相关",
        },
    ],
}

# =========================
# UI 日志
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
            st.caption("暂无日志。点击“开始运行”后这里会显示关键步骤。")
            return
        st.code("\n".join(logs[-400:]))
        if st.button("清空日志", key="clear_logs"):
            st.session_state["logs"] = []
            st.rerun()

# =========================
# HTTP / 反爬：Session + 重试 + 限速
# =========================
@dataclass
class CrawlPolicy:
    sleep_sec: float = DEFAULT_SLEEP_SEC
    jitter_sec: float = DEFAULT_RANDOM_JITTER
    timeout_sec: int = 15

def _sleep(policy: CrawlPolicy) -> None:
    delay = max(0.0, policy.sleep_sec + random.random() * policy.jitter_sec)
    time.sleep(delay)

@st.cache_resource
def get_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (NewsSentimentResearchBot/1.0)",
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
    if t[0] not in "{[":
        m = re.search(r"\(\s*({.*})\s*\)\s*;?\s*$", t, flags=re.S)
        if m:
            t = m.group(1)
    return json.loads(t)

def http_get_text(url: str, params: Optional[Dict[str, Any]], policy: CrawlPolicy) -> str:
    s = get_http_session()
    try:
        r = s.get(url, params=params, timeout=policy.timeout_sec)
        r.raise_for_status()
        return r.text
    finally:
        _sleep(policy)

def http_get_json(url: str, params: Optional[Dict[str, Any]], policy: CrawlPolicy) -> Dict[str, Any]:
    txt = http_get_text(url, params=params, policy=policy)
    return parse_maybe_jsonp(txt)

# =========================
# sector_bk_map.json：读/写/编辑
# =========================
def ensure_map_file() -> None:
    if MAP_PATH.exists():
        return
    ui_log("未发现 sector_bk_map.json，已自动生成模板文件。", "WARN")
    MAP_PATH.write_text(json.dumps(DEFAULT_MAP_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")

def load_sector_map() -> Dict[str, Any]:
    ensure_map_file()
    try:
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        ui_log(f"加载 sector_bk_map.json 失败：{e}", "ERROR")
        return DEFAULT_MAP_TEMPLATE

def _coerce_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        parts = re.split(r"[,\s]+", x.strip())
        return [p for p in (p.strip() for p in parts) if p]
    return [str(x).strip()]

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
    return df

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
    return {
        "meta": {
            "schema_version": 1,
            "last_updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "由 Streamlit 编辑器自动保存。",
        },
        "sectors": sectors,
    }

def save_sector_map(m: Dict[str, Any]) -> None:
    tmp = MAP_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MAP_PATH)

# =========================
# 新闻抓取：东方财富 + 自定义网页列表URL
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
    from urllib.parse import urljoin
    return urljoin(base_url, href)

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
    for a in soup.select("a")[:500]:
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
    tmp = tmp.drop_duplicates(subset=["title_norm", "url_norm"]).drop(columns=["title_norm", "url_norm"])
    return tmp.reset_index(drop=True)

# =========================
# OpenAI GPT：结构化提取（JSON Schema）
# =========================
def sha_user_identifier(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:32]

NEWS_SIGNAL_JSON_SCHEMA: Dict[str, Any] = {
    "name": "news_signal",
    "description": "从财经新闻中提取情绪与影响，并给出主题标签与一句话原因。",
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
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

@st.cache_data(ttl=24 * 3600)
def gpt_extract_one(title: str, body: str, model: str, user_id_hash: str) -> Dict[str, Any]:
    client = get_openai_client()
    system_msg = "你是量化研究助理。请基于输入新闻评估对A股整体情绪与影响。输出必须符合JSON schema。"
    user_msg = f"""
请分析以下财经新闻，对A股整体情绪做判断，并输出字段：
- sentiment：-2强利空,-1利空,0中性,1利好,2强利好
- impact：0~3
- tags：3~8个中文关键词
- summary：<=80字一句话原因

标题：{title}
正文（可能截断）：{body[:1800] if body else "（无正文，仅标题）"}
""".strip()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": NEWS_SIGNAL_JSON_SCHEMA},
            safety_identifier=user_id_hash,
        )
        msg = completion.choices[0].message
        refusal = getattr(msg, "refusal", None)
        if refusal:
            return {"sentiment": 0, "impact": 0, "tags": [], "summary": "模型拒答/无法评估"}

        data = json.loads((msg.content or "").strip())
        data["sentiment"] = int(max(-2, min(2, int(data.get("sentiment", 0)))))
        data["impact"] = int(max(0, min(3, int(data.get("impact", 1)))))
        data["tags"] = [normalize_text(x) for x in (data.get("tags", []) or []) if normalize_text(x)][:8]
        data["summary"] = normalize_text(data.get("summary", ""))[:80]
        return data
    except Exception:
        return {"sentiment": 0, "impact": 1, "tags": [], "summary": "提取失败（降级为中性）"}

# =========================
# 板块映射：关键词 + GPT tags
# =========================
def build_sector_keyword_index(m: Dict[str, Any]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for s in m.get("sectors", []):
        if not bool(s.get("enabled", True)):
            continue
        sector = str(s.get("sector", "")).strip()
        if not sector:
            continue
        kws = s.get("keywords", None)
        idx[sector] = _coerce_list(kws) if kws else FALLBACK_SECTOR_KEYWORDS.get(sector, [])
    for sec, kws in FALLBACK_SECTOR_KEYWORDS.items():
        idx.setdefault(sec, kws)
    return idx

def match_sectors_for_news(title: str, tags: List[str], summary: str, body: str,
                           keyword_index: Dict[str, List[str]], max_hits: int = 3) -> List[str]:
    text = normalize_text(" ".join([title, summary, " ".join(tags or []), body or ""]))
    hits: List[Tuple[str, int]] = []
    for sector, kws in keyword_index.items():
        score = sum(1 for kw in kws if kw and kw in text)
        if score > 0:
            hits.append((sector, score))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in hits[:max_hits]] or ["其他"]

# =========================
# 东方财富 push2：BK 成分股 + 个股打分
# =========================
def fetch_bk_constituents(bk_code: str, limit: int, ut: str, policy: CrawlPolicy) -> pd.DataFrame:
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
        "fields": ",".join(["f12", "f14", "f2", "f3", "f8", "f62"]),
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
            "turnover": d.get("f8"),
            "main_inflow": d.get("f62"),
        })
    return pd.DataFrame(rows)

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
# 板块聚合 → 信号
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
    return agg.sort_values("score", ascending=False).reset_index(drop=True)

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
# Snapshot 落盘
# =========================
def snapshot_save(params: Dict[str, Any],
                  news_enriched: pd.DataFrame,
                  sector_signals: pd.DataFrame,
                  sector_stock_candidates: Dict[str, pd.DataFrame]) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SNAPSHOT_DIR / f"snapshot_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": ts,
        "params": params,
        "news_count": int(len(news_enriched)),
        "sector_signals": sector_signals.to_dict(orient="records"),
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
# 回测雏形：K线抓取 + 前向收益
# =========================
def infer_secid_from_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return ""
    # 简化：6开头认为沪市(1)，否则深市(0)
    if c.startswith("6"):
        return f"1.{c}"
    return f"0.{c}"

@st.cache_data(ttl=3600)
def fetch_daily_kline(secid: str, ut: str, policy: CrawlPolicy,
                      beg: str = "0", end: str = "20500101", lmt: int = 200) -> pd.DataFrame:
    url = f"{EASTMONEY_PUSH2HIS_BASE}/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "beg": beg,
        "end": end,
        "klt": 101,
        "fqt": 1,
        "lmt": lmt,
        "ut": ut,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    j = http_get_json(url, params=params, policy=policy)
    klines = ((j.get("data") or {}).get("klines")) or []
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        rows.append({"date": parts[0], "close": float(parts[2])})
    return pd.DataFrame(rows)

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
    t0 = int(candidates.index.max()) if not candidates.empty else 0
    t1 = t0 + int(horizon)
    if t1 >= len(kl):
        return None
    c0 = float(kl.loc[t0, "close"])
    c1 = float(kl.loc[t1, "close"])
    if c0 <= 0:
        return None
    return c1 / c0 - 1.0

def list_snapshots() -> List[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted([p for p in SNAPSHOT_DIR.iterdir() if p.is_dir() and p.name.startswith("snapshot_")], reverse=True)

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
# Streamlit 主程序
# =========================
def main() -> None:
    st.title(APP_NAME)
    st.caption(f"版本：{APP_VERSION} ｜ 研究用途 Demo（不构成投资建议）")

    with st.sidebar:
        st.header("⚙️ 参数设置")

        st.subheader("新闻源")
        chosen_sections = st.multiselect(
            "选择东方财富栏目（至少1个）",
            options=list(EASTMONEY_SECTIONS.keys()),
            default=["国内经济", "国际经济"],
        )
        per_source = st.slider("每个新闻源抓取条数", 5, 60, DEFAULT_NEWS_PER_SOURCE, 5)
        total_limit = st.slider("总新闻上限（去重后）", 10, 200, DEFAULT_TOTAL_NEWS_LIMIT, 10)

        custom_urls_text = st.text_area(
            "自定义网页列表URL（每行一个，非RSS）",
            value="",
            height=90,
            placeholder="例如：某财经站新闻列表页URL（每行一个）",
        )

        st.subheader("反爬与性能")
        sleep_sec = st.slider("请求间隔(秒)", 0.0, 3.0, DEFAULT_SLEEP_SEC, 0.1)
        jitter = st.slider("随机抖动(秒)", 0.0, 1.0, DEFAULT_RANDOM_JITTER, 0.1)
        fetch_body = st.checkbox("抓取正文（更准但更慢）", value=True)
        max_body_chars = st.slider("正文最大截断长度", 300, 3000, 1600, 100)

        st.subheader("OpenAI")
        model = st.text_input("模型", value=DEFAULT_OPENAI_MODEL)
        cache_gpt = st.checkbox("缓存 GPT 结果（更快省钱）", value=True)

        st.subheader("板块信号阈值")
        pos_th = st.slider("关注阈值(>=)", 0.2, 2.0, 0.9, 0.1)
        neg_th = st.slider("谨慎阈值(<=)", -2.0, -0.2, -0.9, 0.1)

        st.subheader("板块→个股")
        ut = st.text_input("东方财富 ut（可留默认）", value=DEFAULT_UT)
        bk_stock_limit = st.slider("每个BK拉取成分股条数", 20, 200, 80, 10)
        top_sectors = st.slider("输出Top板块数", 3, 20, 8, 1)
        top_stocks = st.slider("每板块输出候选股数", 5, 50, 15, 5)

        preset_name = st.selectbox("个股打分预设", options=list(SCORE_PRESETS.keys()), index=1)
        use_custom = st.checkbox("自定义打分权重", value=False)
        preset = SCORE_PRESETS[preset_name]
        if use_custom:
            w_pct = st.number_input("w_pct(涨跌幅权重)", value=float(preset.w_pct), step=0.05)
            w_inflow = st.number_input("w_main_inflow(主力净流入权重)", value=float(preset.w_main_inflow), step=0.05)
            w_turn = st.number_input("w_turnover(换手权重)", value=float(preset.w_turnover), step=0.05)
            inflow_scale = st.number_input("inflow_scale(净流入缩放，1e8=亿元)", value=float(preset.inflow_scale))
            weights = StockScoreWeights(w_pct=w_pct, w_main_inflow=w_inflow, w_turnover=w_turn, inflow_scale=inflow_scale)
        else:
            weights = preset

        st.subheader("保存与回测")
        save_snapshot_flag = st.checkbox("运行后保存 snapshot", value=True)

    policy = CrawlPolicy(sleep_sec=float(sleep_sec), jitter_sec=float(jitter), timeout_sec=15)

    tab_run, tab_map, tab_bt = st.tabs(["🚀 运行", "🗺️ 板块映射编辑器", "📈 回测雏形"])

    # -------- 运行 --------
    with tab_run:
        run_btn = st.button("开始运行", type="primary")
        if run_btn:
            st.session_state["logs"] = []
            ui_log("开始运行。")

            if not os.getenv("OPENAI_API_KEY"):
                st.error("未检测到 OPENAI_API_KEY，请先配置环境变量。")
                ui_log("未检测到 OPENAI_API_KEY，终止。", "ERROR")
                ui_show_logs()
                st.stop()

            if not chosen_sections and not custom_urls_text.strip():
                st.error("请至少选择一个东方财富栏目或提供自定义网页URL。")
                ui_log("没有新闻源，终止。", "ERROR")
                ui_show_logs()
                st.stop()

            ui_log("步骤1：抓取新闻列表（多源网页）。")
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
                        ui_log(f"抓取 {sec} 成功：{len(df)}条。")
                    else:
                        ui_log(f"抓取 {sec} 为空（结构变化/被拦截）。", "WARN")
                except Exception as e:
                    ui_log(f"抓取 {sec} 失败：{e}", "WARN")

            custom_urls = [x.strip() for x in custom_urls_text.splitlines() if x.strip()]
            for u in custom_urls:
                try:
                    df = fetch_generic_page_titles(u, limit=per_source, policy=policy)
                    if not df.empty:
                        df["source"] = "自定义网页"
                        frames.append(df)
                        ui_log(f"抓取自定义源成功：{u} | {len(df)}条。")
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
            ui_log(f"去重后新闻数量：{len(news)}")

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

            show_df = df.copy()
            show_df["tags"] = show_df["tags"].apply(lambda x: " / ".join(x) if isinstance(x, list) else str(x))

            st.markdown("### 2) 单条新闻：GPT 提取结果")
            st.dataframe(
                show_df[["source", "title", "sentiment", "impact", "tags", "summary", "url"]],
                use_container_width=True,
                column_config={"url": st.column_config.LinkColumn("url")},
            )

            ui_log("步骤3：新闻→板块映射（可编辑 sector_bk_map.json + fallback关键词）。")
            sector_map = load_sector_map()
            keyword_index = build_sector_keyword_index(sector_map)

            df["sectors"] = df.apply(
                lambda row: match_sectors_for_news(
                    title=row["title"],
                    tags=row["tags"],
                    summary=row["summary"],
                    body=row.get("body", ""),
                    keyword_index=keyword_index,
                    max_hits=3,
                ),
                axis=1
            )

            tmp = df.copy()
            tmp["sectors"] = tmp["sectors"].apply(lambda x: " / ".join(x) if isinstance(x, list) else str(x))

            st.markdown("### 3) 新闻 → 板块归因")
            st.dataframe(tmp[["title", "sentiment", "impact", "sectors", "summary"]], use_container_width=True)

            ui_log("步骤4：聚合板块得分与信号。")
            agg = aggregate_sector_scores(df)
            sector_policy = SectorSignalPolicy(pos_threshold=float(pos_th), neg_threshold=float(neg_th))
            sector_signals = add_sector_signal(agg, sector_policy)

            st.markdown("### 4) 板块信号（聚合）")
            st.dataframe(sector_signals, use_container_width=True)
            if not sector_signals.empty:
                st.bar_chart(sector_signals.set_index("sector")[["score"]])

            ui_log("步骤5：板块→BK→成分股→候选股（仅‘关注’板块）。")
            map_df = sector_map_to_df(sector_map)
            map_df = map_df[map_df["enabled"] == True]

            top_df = sector_signals.head(int(top_sectors))
            sector_stock_candidates: Dict[str, pd.DataFrame] = {}

            for _, row in top_df.iterrows():
                sector = row["sector"]
                sig = row.get("signal", "")
                score = float(row.get("score", 0))

                if "关注" not in str(sig):
                    continue

                m_row = map_df[map_df["sector"].astype(str) == str(sector)]
                bk_codes: List[str] = []
                if not m_row.empty:
                    bk_codes = _coerce_list(m_row.iloc[0]["bk_codes"])

                if not bk_codes:
                    ui_log(f"板块 {sector} 未配置 BK 代码，跳过选股。", "WARN")
                    continue

                all_constituents = []
                for bk in bk_codes:
                    try:
                        cons = fetch_bk_constituents(bk, limit=int(bk_stock_limit), ut=ut, policy=policy)
                        if cons.empty:
                            ui_log(f"BK {bk} 成分股为空/抓取失败。", "WARN")
                            continue
                        cons["bk"] = bk
                        all_constituents.append(cons)
                        ui_log(f"拉取 BK {bk} 成分股：{len(cons)}条。")
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

            st.markdown("### 5) 个股候选（仅关注板块）")
            if not sector_stock_candidates:
                st.info("没有可输出的个股候选：可能没有‘关注’板块，或 BK 未配置/抓取失败。")
            else:
                combined = pd.concat(sector_stock_candidates.values(), ignore_index=True)
                st.dataframe(combined, use_container_width=True)

            if save_snapshot_flag:
                ui_log("步骤6：保存 snapshot。")
                params = {
                    "sections": chosen_sections,
                    "per_source": per_source,
                    "total_limit": total_limit,
                    "custom_urls": custom_urls,
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
                summary_bytes = (out_dir / "summary.json").read_bytes()
                st.download_button("下载 summary.json", data=summary_bytes, file_name="summary.json", mime="application/json")

            ui_show_logs()

    # -------- 映射编辑器 --------
    with tab_map:
        st.subheader("sector_bk_map.json 编辑器（可编辑/保存/下载）")
        st.caption("字段：sector=你的板块名；bk_codes=对应东方财富BK代码列表；keywords=新闻归因关键词。")
        m = load_sector_map()
        df_map = sector_map_to_df(m)

        edited = st.data_editor(
            df_map,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "enabled": st.column_config.CheckboxColumn("enabled"),
                "sector": st.column_config.TextColumn("sector"),
                "bk_codes": st.column_config.ListColumn("bk_codes"),
                "keywords": st.column_config.ListColumn("keywords"),
                "priority": st.column_config.NumberColumn("priority"),
                "notes": st.column_config.TextColumn("notes"),
            },
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("保存到 sector_bk_map.json"):
                save_sector_map(df_to_sector_map(edited))
                st.success("已保存。")
        with c2:
            if st.button("重置为模板"):
                save_sector_map(DEFAULT_MAP_TEMPLATE)
                st.warning("已重置。")
                st.rerun()
        with c3:
            cur = df_to_sector_map(edited)
            st.download_button("下载当前映射 JSON",
                               data=json.dumps(cur, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="sector_bk_map.json",
                               mime="application/json")

        ui_show_logs()

    # -------- 回测雏形 --------
    with tab_bt:
        st.subheader("回测雏形：读取 snapshots → 简单前向收益（等权）")
        snaps = list_snapshots()
        if not snaps:
            st.info("暂无 snapshots。请先在“运行”页勾选保存 snapshot 并执行一次。")
        else:
            snap_names = [p.name for p in snaps]
            chosen = st.multiselect("选择 snapshot（可多选）", options=snap_names, default=snap_names[:1])
            horizon = st.slider("收益窗口（交易日）", 1, 20, 5, 1)
            max_stocks_per_sector_bt = st.slider("每板块用于回测的TopN股票", 1, 30, 10, 1)

            run_bt = st.button("运行回测（雏形）", type="primary", key="bt_run")
            if run_bt:
                st.session_state["logs"] = []
                ui_log("开始回测。")
                rows = []
                policy_bt = CrawlPolicy(sleep_sec=0.5, jitter_sec=0.2, timeout_sec=15)

                for name in chosen:
                    snap = SNAPSHOT_DIR / name
                    m = re.search(r"snapshot_(\d{8})_(\d{6})", name)
                    anchor_date = dt.datetime.strptime(m.group(1), "%Y%m%d").date().isoformat() if m else dt.date.today().isoformat()

                    stocks_map = load_snapshot_stocks(snap)
                    if not stocks_map:
                        ui_log(f"{name} 缺少 stocks 文件。", "WARN")
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
                            try:
                                kl = fetch_daily_kline(secid=secid, ut=DEFAULT_UT, policy=policy_bt, lmt=200)
                                ret = compute_forward_return(kl, anchor_date=anchor_date, horizon=int(horizon))
                                if ret is None:
                                    continue
                                all_returns.append(ret)
                                n_used += 1
                            except Exception:
                                continue

                    if not all_returns:
                        ui_log(f"{name} 无法计算收益（K线失败或样本不足）。", "WARN")
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
                    ui_log(f"{name} 回测完成：n={n_used}, mean={s.mean():.4f}")

                res = pd.DataFrame(rows)
                if res.empty:
                    st.error("回测结果为空：可能K线接口失败/样本不足/被限流。")
                    ui_show_logs()
                else:
                    st.dataframe(res, use_container_width=True)
                    st.bar_chart(res.set_index("snapshot")[["mean_ret"]])
                    st.download_button("下载回测结果 CSV",
                                       data=res.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="backtest_result.csv",
                                       mime="text/csv")
                    ui_show_logs()

if __name__ == "__main__":
    main()
