import streamlit as st
import akshare as ak
import pandas as pd
from openai import OpenAI

# =============================
# 页面设置
# =============================
st.set_page_config(page_title="全球新闻 + 板块量化系统", layout="wide")
st.title("🌍 全球新闻驱动板块评分系统")

# =============================
# OpenAI 初始化
# =============================
client = OpenAI(api_key="你的APIKEY")  # ← 填入你的 OpenAI Key

# =============================
# 自动抓取全球新闻
# =============================
@st.cache_data(ttl=300)
def get_global_news():
    try:
        news_df = ak.stock_news_em()  # 最新新闻
        if news_df is None or news_df.empty:
            return None
        return news_df.head(10)
    except Exception as e:
        st.error("新闻接口获取失败")
        st.text(str(e))
        return None

# =============================
# 新闻情绪分析
# =============================
def analyze_news_batch(news_list):
    news_text = "\n".join(news_list)
    prompt = f"""
    以下是最近全球财经新闻，请判断整体对A股情绪：
    只回答：利好 / 利空 / 中性

    新闻：
    {news_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        if "利好" in result:
            return 1
        elif "利空" in result:
            return -1
        else:
            return 0
    except Exception as e:
        st.error("新闻情绪分析失败")
        st.text(str(e))
        return 0

# =============================
# 板块评分
# =============================
def calculate_score(sector_df, news_score):
    # 涨跌幅列自动识别
    if "涨跌幅" in sector_df.columns:
        sector_df["涨跌幅"] = sector_df["涨跌幅"].str.replace("%","").astype(float)
    elif "changeRate" in sector_df.columns:
        sector_df["涨跌幅"] = sector_df["changeRate"].astype(float)
    else:
        sector_df["涨跌幅"] = 0.0

    max_val = sector_df["涨跌幅"].max()
    min_val = sector_df["涨跌幅"].min()

    if max_val == min_val:
        sector_df["板块强度"] = 0.5
    else:
        sector_df["板块强度"] = (sector_df["涨跌幅"] - min_val) / (max_val - min_val)

    sector_df["综合评分"] = 0.6 * sector_df["板块强度"] + 0.4 * news_score
    return sector_df.sort_values("综合评分", ascending=False)

# =============================
# 主程序
# =============================
if st.button("🚀 自动分析全球新闻"):

    # 1️⃣ 获取新闻
    with st.spinner("抓取全球新闻中..."):
        news_df = get_global_news()

    if news_df is None or news_df.empty:
        st.error("新闻获取失败或为空")
        st.stop()

    # 2️⃣ 自动适配新闻标题列
    st.write("新闻列名:", news_df.columns.tolist())
    possible_title_cols = ["标题", "title", "news_title"]
    title_col = next((c for c in possible_title_cols if c in news_df.columns), news_df.columns[0])

    st.subheader("📰 最新全球新闻")
    st.dataframe(news_df[[title_col]])

    # 3️⃣ 新闻情绪分析
    with st.spinner("分析新闻情绪中..."):
        news_score = analyze_news_batch(news_df[title_col].tolist())

    if news_score == 1:
        st.success("整体新闻情绪：利好 📈")
    elif news_score == -1:
        st.error("整体新闻情绪：利空 📉")
    else:
        st.info("整体新闻情绪：中性 ⚖️")

    # 4️⃣ 获取板块数据
    with st.spinner("获取板块数据中..."):
        try:
            sector_df = ak.stock_board_industry_name_em()
            if sector_df is None or sector_df.empty:
                st.error("板块数据暂时无法获取")
                st.stop()
        except Exception as e:
            st.error("板块接口异常")
            st.text(str(e))
            st.stop()

    # 5️⃣ 综合评分
    result_df = calculate_score(sector_df, news_score)
    st.subheader("🔥 板块综合评分 Top10")
    st.dataframe(result_df.head(10), use_container_width=True)
