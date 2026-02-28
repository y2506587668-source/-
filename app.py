import os
import re
import json
import math
import time
import datetime as dt
from typing import List, Dict, Tuple

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

# ----------------------------
# 基础配置
# ----------------------------
st.set_page_config(page_title="新闻 → 板块 → 个股 信号仪表盘", layout="wide")
st.title("📰 新闻爬虫 → GPT情绪 → 板块匹配 → 板块/个股候选 → 信号落盘")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("请先设置环境变量 OPENAI_API_KEY（例如：export OPENAI_API_KEY='xxx'）")

client = OpenAI(api_key=OPENAI_API_KEY)

HEADERS = {"User-Agent": "Mozilla/5.0 (NewsSentimentBot/1.0)"}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# 1) 多源新闻：东方财富栏目（可扩展）
# ----------------------------
EASTMONEY_SECTIONS = {
    "国内经济": "https://finance.eastmoney.com/a/cgnjj.html",
    "国际经济": "https://finance.eastmoney.com/a/cgjjj.html",
    "宏观经济": "https://finance.eastmoney.com/a/cmacro.html",
    "财经要闻": "https://finance.eastmoney.com/news/cywjh.html",
}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

@st.cache_data(ttl=180)
def fetch_eastmoney_list(url: str, limit=20, source_name="东方财富") -> pd.DataFrame:
    """
    抓东方财富列表页标题+链接（HTML 结构可能变，必要时改 selector）
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    # 常见：div.newsList li a
    items = soup.select("div.newsList li a") or soup.select("div.newsList a")
    data = []
    for a in items[:limit]:
        title = normalize_text(a.get_text())
        link = a.get("href", "")
        if link.startswith("//"):
            link = "https:" + link
        elif link.startswith("/"):
            link = "https://finance.eastmoney.com" + link
        if title and link:
            data.append({"title": title, "url": link, "source": source_name})
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def fetch_article_text(url: str, max_chars=2200) -> str:
    """
    尝试抽取正文（更准但更慢）
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")

        candidates = [
            soup.select_one("div#ContentBody"),
            soup.select_one("div.article-body"),
            soup.select_one("div.newsContent"),
            soup.select_one("div.txtinfos"),
        ]
        text = ""
        for c in candidates:
            if c:
                text = normalize_text(c.get_text(" ", strip=True))
                if len(text) >= 80:
                    break
        if not text:
            text = normalize_text(soup.get_text(" ", strip=True))
        return text[:max_chars]
    except Exception:
        return ""

# （可选）用户自定义 RSS/网页：你可以在侧边栏输入 URL，仍然走 HTML 抓取（RSS 也算网页源）
@st.cache_data(ttl=300)
def fetch_generic_titles(url: str, limit=20, source_name="自定义源") -> pd.DataFrame:
    """
    非严格 RSS 解析：尽量抓页面里出现的 a 标签文本（适合新闻列表页、部分 RSS 展示页）
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.select("a")[:300]

        data = []
        for a in links:
            title = normalize_text(a.get_text())
            href = a.get("href", "")
            if not title or len(title) < 8:
                continue
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                # 简单拼接（不保证所有网站都正确）
                from urllib.parse import urljoin
                href = urljoin(url, href)

            if href.startswith("http"):
                data.append({"title": title, "url": href, "source": source_name})
            if len(data) >= limit:
                break
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# ----------------------------
# 2) GPT：新闻情绪/影响/标签（JSON）
# ----------------------------
def gpt_analyze_one(title: str, content: str = "") -> dict:
    prompt = f"""
你是量化研究助理。请基于【新闻标题】和【新闻正文摘要】判断对A股的情绪与影响强度。
只输出严格JSON（不要代码块、不要多余文字）。

规则：
- sentiment：-2(强利空)、-1(利空)、0(中性)、1(利好)、2(强利好)
- impact：0(几乎无影响)~3(强影响)
- tags：提炼3~6个中文关键词/主题
- summary：一句话原因（<=30字）

新闻标题：{title}
新闻正文摘要：{content[:1500] if content else "（无正文，仅标题）"}
""".strip()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        data["sentiment"] = int(max(-2, min(2, data.get("sentiment", 0))))
        data["impact"] = int(max(0, min(3, data.get("impact", 1))))
        data["tags"] = (data.get("tags", []) or [])[:8]
        data["summary"] = normalize_text(data.get("summary", ""))
        return data
    except Exception:
        return {"sentiment": 0, "impact": 1, "tags": [], "summary": ""}

# ----------------------------
# 3) 更稳板块映射：关键词 + GPT tags
# ----------------------------
SECTOR_KEYWORDS = {
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

def match_sectors(title: str, tags: List[str]) -> List[str]:
    text = f"{title} " + " ".join(tags or [])
    hits = []
    for sector, kws in SECTOR_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                hits.append(sector)
                break
    return hits or ["其他"]

# ----------------------------
# 4) 板块 → BK行业代码：从东方财富“行业资金流向页”抓 BK 列表
#    该页能看到行业名称入口（如 证券Ⅱ 等），链接含 BKxxxx 代码。:contentReference[oaicite:2]{index=2}
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_bk_industry_list() -> pd.DataFrame:
    """
    从 data.eastmoney.com/bkzj/ 的某个行业页抓“更多板块资金流向”弹层里出现的 BK 链接
    这里用一个行业页作为入口（例如 BK0473 证券Ⅱ），再解析所有 BKxxxx 的 href。
    """
    entry = "https://data.eastmoney.com/bkzj/BK0473.html"
    resp = requests.get(entry, headers=HEADERS, timeout=15)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    a_tags = soup.select("a[href*='BK']")
    seen = {}
    for a in a_tags:
        href = a.get("href", "")
        name = normalize_text(a.get_text())
        m = re.search(r"(BK\d{4})", href)
        if m and name:
            bk = m.group(1)
            # 拼成完整链接
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = "https://data.eastmoney.com" + href
            elif href.startswith("BK"):
                href = "https://data.eastmoney.com/bkzj/" + href
            if bk not in seen:
                seen[bk] = {"bk": bk, "industry_name": name, "url": href}
    df = pd.DataFrame(list(seen.values()))
    # 只保留“行业”相关（弹层里也会混概念/地域入口时，可能要再过滤）
    return df.sort_values("bk").reset_index(drop=True)

# ----------------------------
# 5) 成分股抓取：东方财富 clist/get + fs=b:BKxxxx
#    clist/get 常用于抓列表行情数据（示例/说明见公开抓取案例）。:contentReference[oaicite:3]{index=3}
# ----------------------------
@st.cache_data(ttl=120)
def fetch_bk_constituents(bk: str, limit=50) -> pd.DataFrame:
    """
    成分股列表（字段尽量取常用：代码/名称/涨跌幅/最新价/主力净流入等）
    注：fields 的可用集合会变，拿不到就空列
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1,
        "pz": limit,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",              # 按涨跌幅排序（常用）
        "fs": f"b:{bk}",          # 关键：按板块过滤
        "fields": ",".join([
            "f12",  # code
            "f14",  # name
            "f2",   # last price
            "f3",   # pct
            "f62",  # 主力净流入（常见字段，可能为空）
            "f66",  # 超大单净流入（可能为空）
            "f72",  # 大单净流入（可能为空）
        ])
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    data = r.json()
    diff = (data.get("data") or {}).get("diff") or []
    out = []
    for d in diff:
        out.append({
            "code": d.get("f12"),
            "name": d.get("f14"),
            "last": d.get("f2"),
            "pct": d.get("f3"),
            "main_inflow": d.get("f62"),
            "super_inflow": d.get("f66"),
            "big_inflow": d.get("f72"),
        })
    return pd.DataFrame(out)

# ----------------------------
# 6) 信号聚合：新闻→板块分数；板块→个股候选
# ----------------------------
def aggregate_sector_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    # 文章得分：sentiment * (1 + impact/3)
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

    # 数量权重（避免单条新闻极端影响）
    agg["score"] = agg.apply(lambda x: x["avg_score"] * math.log1p(x["count"]), axis=1)
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

    def to_signal(v: float) -> str:
        if v >= 0.9:
            return "关注 📈"
        if v <= -0.9:
            return "谨慎 📉"
        return "中性 ⚖️"

    agg["signal"] = agg["score"].apply(to_signal)
    return agg

def pick_top_stocks_for_bk(bk: str, n=10) -> pd.DataFrame:
    df = fetch_bk_constituents(bk, limit=max(50, n))
    if df.empty:
        return df
    # 简单打分：涨跌幅 + 主力净流入（若有）
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    df["main_inflow"] = pd.to_numeric(df["main_inflow"], errors="coerce")
    df["score"] = df["pct"].fillna(0) * 0.7 + (df["main_inflow"].fillna(0) / 1e8) * 0.3
    return df.sort_values("score", ascending=False).head(n).reset_index(drop=True)

# ----------------------------
# 7) 落盘：用于回测/复盘
# ----------------------------
def save_run_snapshot(sector_df: pd.DataFrame,
                      top_sector_stocks: Dict[str, pd.DataFrame],
                      raw_news_df: pd.DataFrame) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(DATA_DIR, f"snapshot_{ts}")
    os.makedirs(base, exist_ok=True)

    sector_df.to_csv(os.path.join(base, "sector_signals.csv"), index=False, encoding="utf-8-sig")
    raw_news_df.to_csv(os.path.join(base, "news_enriched.csv"), index=False, encoding="utf-8-sig")

    # 个股候选
    for sec, sdf in top_sector_stocks.items():
        safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", sec)
        sdf.to_csv(os.path.join(base, f"stocks_{safe}.csv"), index=False, encoding="utf-8-sig")

    # 也写一份 json 汇总，方便程序化回测读取
    summary = {
        "timestamp": ts,
        "sector_signals": sector_df.to_dict(orient="records"),
        "top_stocks": {k: v.to_dict(orient="records") for k, v in top_sector_stocks.items()},
        "news_count": int(len(raw_news_df)),
    }
    with open(os.path.join(base, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return base

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("参数")
    per_section = st.slider("每个栏目抓取条数", 10, 60, 20, 5)
    use_body = st.checkbox("抓正文（更准但更慢）", value=True)
    sleep_sec = st.slider("抓正文间隔(秒)", 0, 3, 1, 1)

    st.divider()
    st.subheader("多源增强")
    selected_sections = st.multiselect(
        "选择东方财富栏目",
        list(EASTMONEY_SECTIONS.keys()),
        default=list(EASTMONEY_SECTIONS.keys())[:2]
    )
    custom_url = st.text_input("可选：自定义新闻列表页/RSS展示页URL（留空则不抓）", "")

    st.divider()
    st.subheader("板块→个股")
    top_k_sectors = st.slider("输出 Top 板块数", 3, 10, 5, 1)
    top_k_stocks = st.slider("每个板块输出 Top 个股候选数", 5, 30, 10, 5)

run = st.button("🚀 抓取新闻 → 生成板块/个股信号", type="primary")

if run:
    # 1) 聚合多源新闻
    all_news = []
    for sec in selected_sections:
        url = EASTMONEY_SECTIONS[sec]
        df = fetch_eastmoney_list(url, limit=per_section, source_name=f"东方财富-{sec}")
        all_news.append(df)

    if custom_url.strip():
        df2 = fetch_generic_titles(custom_url.strip(), limit=per_section, source_name="自定义源")
        if not df2.empty:
            all_news.append(df2)

    news_df = pd.concat(all_news, ignore_index=True).drop_duplicates(subset=["title"]).reset_index(drop=True)
    if news_df.empty:
        st.error("未抓到新闻，可能被拦截或页面结构变更。")
        st.stop()

    st.subheader("1) 原始新闻列表（去重后）")
    st.dataframe(news_df, use_container_width=True)

    # 2) GPT enrich
    st.subheader("2) GPT 情绪/影响/主题 + 板块匹配")
    enriched = []
    prog = st.progress(0)
    for i, row in news_df.iterrows():
        title, url = row["title"], row["url"]
        body = fetch_article_text(url) if use_body else ""
        g = gpt_analyze_one(title, body)
        sectors = match_sectors(title, g.get("tags", []))

        enriched.append({
            **row,
            "sentiment": g["sentiment"],
            "impact": g["impact"],
            "tags": " / ".join(g.get("tags", [])),
            "summary": g.get("summary", ""),
            "sectors": sectors,
        })

        prog.progress((i + 1) / len(news_df))
        if use_body and sleep_sec > 0:
            time.sleep(sleep_sec)

    df = pd.DataFrame(enriched)
    st.dataframe(df[["source", "title", "sentiment", "impact", "tags", "summary", "sectors", "url"]],
                 use_container_width=True)

    # 3) 板块聚合信号
    sector_df = aggregate_sector_scores(df)
    st.subheader("3) 板块信号（聚合）")
    st.dataframe(sector_df, use_container_width=True)

    # 4) 板块→个股：用 BK 行业代码映射
    st.subheader("4) 板块 → 行业BK → 成分股候选（先板块后个股）")
    bk_df = fetch_bk_industry_list()
    if bk_df.empty:
        st.warning("未抓到 BK 行业列表（可能页面结构变更/被拦截），将跳过行业BK→成分股。")
        st.stop()

    # 做一个“我们的板块名”→“东方财富行业名/BK” 的粗匹配（你后续可改成更精确映射）
    # 逻辑：如果行业名里包含关键词，就认为对应
    def map_sector_to_bk(sector_name: str) -> List[Tuple[str, str]]:
        # 返回 [(bk, industry_name), ...]
        key_map = {
            "半导体": ["半导体"],
            "AI/算力": ["IT服务", "计算机设备", "通信设备", "软件开发", "元件", "消费电子"],
            "新能源": ["光伏设备", "风电设备", "电池", "能源金属"],
            "汽车/智能驾驶": ["乘用车", "汽车零部件", "商用车"],
            "金融/券商": ["证券", "银行", "保险"],
            "消费": ["白酒", "一般零售", "旅游及景区", "家电", "饮料乳品"],
            "地产/基建": ["房地产开发", "基础建设", "铁路公路", "房屋建设"],
            "医药": ["化学制药", "生物制品", "医疗器械", "医疗服务", "中药"],
            "军工": ["航天装备", "航空装备", "军工电子"],
            "黄金/大宗": ["贵金属", "工业金属", "小金属", "煤炭开采", "炼化及贸易"],
        }
        patterns = key_map.get(sector_name, [])
        hits = []
        for p in patterns:
            sub = bk_df[bk_df["industry_name"].str.contains(p, na=False)]
            for _, r in sub.head(3).iterrows():
                hits.append((r["bk"], r["industry_name"]))
        return hits

    # 只对 TopK “关注/偏正面”的板块做个股候选
    top_sectors = sector_df.head(top_k_sectors)
    top_sector_stocks = {}

    for _, r in top_sectors.iterrows():
        sec = r["sector"]
        sig = r["signal"]
        if "关注" not in sig:
            continue

        bk_hits = map_sector_to_bk(sec)
        if not bk_hits:
            continue

        st.markdown(f"### ✅ {sec}（{sig}）")
        for bk, ind_name in bk_hits:
            st.caption(f"行业映射：{ind_name} / {bk}")
            stocks = pick_top_stocks_for_bk(bk, n=top_k_stocks)
            if stocks.empty:
                st.write("成分股抓取为空")
                continue
            st.dataframe(stocks[["code", "name", "last", "pct", "main_inflow", "score"]], use_container_width=True)
            # 合并到汇总（按板块名汇总，多个 BK 的候选会堆叠）
            top_sector_stocks.setdefault(sec, pd.DataFrame())
            top_sector_stocks[sec] = pd.concat([top_sector_stocks[sec], stocks], ignore_index=True)

    # 5) 落盘：为回测/复盘准备
    st.subheader("5) 信号落盘（用于回测/复盘）")
    out_dir = save_run_snapshot(sector_df, top_sector_stocks, df)
    st.success(f"已保存到：{out_dir}（包含 sector_signals.csv / news_enriched.csv / stocks_*.csv / summary.json）")

    # 图表
    st.subheader("6) 板块得分图")
    st.bar_chart(sector_df.set_index("sector")[["score"]])

    st.info("⚠️ 以上为基于新闻文本与网站公开数据的自动化信号，不构成投资建议。")
