import streamlit as st
import akshare as ak
import pandas as pd
from openai import OpenAI
import re

st.set_page_config(page_title="全球新闻+板块量化系统 v2.0", layout="wide")
st.title("🌍 全球新闻驱动板块评分系统 v2.0")

# ------------------------
# OpenAI 初始化
# ------------------------
client = OpenAI(api_key="你的APIKEY")  # 填你的 OpenAI Key

# ------------------------
# 自动抓取新闻
# ------------------------
@st.cache_data(ttl=300)
def get_global_news():
    try:
        news_df = ak.stock_news_em()  # 最新10条新闻
        if news_df is None or news_df.empty:
            return None
        return news_df.head(10)
    except Exception as e:
        st.warning("新闻接口获取失败")
        st.text(str(e))
        return None

# ------------------------
# 每条新闻情绪分析
# ------------------------
def analyze_single_news(news_title):
    # UTF-8 处理
    text = str(news_title).encode("utf-8", errors="ignore").decode("utf-8")
    prompt = f"""
    请判断以下新闻对A股板块情绪：
    只回答：利好 / 利空 / 中性
    新闻：
    {text}
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
    except:
        return 0  # 默认中性

# ------------------------
# 新闻匹配板块
# ------------------------
def map_news_to_sector(news_title, sector_df):
    matched_sectors = []
    for sector_name in sector_df["板块名称"] if "板块名称" in sector_df.columns else sector_df.iloc[:,0]:
        if sector_name in news_title:
            matched_sectors.append(sector_name)
    return matched_sectors

# ------------------------
# 综合评分
# ------------------------
def calculate_score(sector_df, news_df):
    # 涨跌幅列识别
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
        sector_df["板块强度"] = (sector_df["涨跌幅"] - min_val)/(max_val - min_val)

    # 板块新闻评分
    sector_df["新闻情绪"] = 0.0
    for idx, row in sector_df.iterrows():
        matched_scores = []
        for title in news_df["新闻标题"]:
            score = analyze_single_news(title)
            # 检查是否匹配板块
            if map_news_to_sector(title, sector_df) and row[0] in map_news_to_sector(title, sector_df):
                matched_scores.append(score)
        if matched_scores:
            sector_df.at[idx,"新闻情绪"] = sum(matched_scores)/len(matched_scores)
    # 综合评分
    sector_df["综合评分"] = 0.6*sector_df["板块强度"] + 0.4*sector_df["新闻情绪"]
    return sector_df.sort_values("综合评分", ascending=False)

# ------------------------
# 主程序
# ------------------------
if st.button("🚀 自动分析全球新闻"):
    with st.spinner("抓取全球新闻中..."):
        news_df = get_global_news()
    if news_df is None or news_df.empty:
        st.error("新闻获取失败")
        st.stop()

    # 自动适配新闻标题列
    title_col = "新闻标题" if "新闻标题" in news_df.columns else news_df.columns[0]
    st.subheader("📰 最新全球新闻")
    st.dataframe(news_df[[title_col]])

    # 获取板块
    with st.spinner("获取板块数据中..."):
        try:
            sector_df = ak.stock_board_industry_name_em()
            if sector_df is None or sector_df.empty:
                st.warning("板块数据暂时无法获取")
                st.stop()
        except Exception as e:
            st.warning("板块接口异常")
            st.text(str(e))
            st.stop()

    # 综合评分
    with st.spinner("计算板块综合评分..."):
        result_df = calculate_score(sector_df, news_df)
    st.subheader("🔥 板块综合评分 Top10")
    st.dataframe(result_df.head(10), use_container_width=True)
