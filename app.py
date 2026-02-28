import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("📊 超短辅助判断系统")

# =============================
# 1️⃣ 市场情绪
# =============================

st.header("🔥 市场情绪")

today = datetime.now().strftime("%Y%m%d")

try:
    limit_df = ak.stock_zt_pool_em(date=today)
    limit_count = len(limit_df)
except:
    limit_count = 0

if limit_count > 80:
    emotion = "主升期 🔥"
elif limit_count < 30:
    emotion = "冰点 ❄"
else:
    emotion = "震荡期 ⚖"

col1, col2 = st.columns(2)
col1.metric("涨停数量", limit_count)
col2.metric("情绪阶段", emotion)

# =============================
# 2️⃣ 板块强度
# =============================

st.header("📈 板块强度排行")

try:
    sector_df = ak.stock_board_industry_name_em()
except:
    st.error("板块数据暂时无法获取（可能被接口限制）")
    sector_df = None
sector_df = sector_df.sort_values("涨跌幅", ascending=False)

st.dataframe(sector_df.head(10)[["板块名称", "涨跌幅"]])

top_sector = sector_df.iloc[0]["板块名称"]

# =============================
# 3️⃣ 龙头监控
# =============================

st.header("🚀 当前强势龙头")

sector_stocks = ak.stock_board_industry_cons_em(symbol=top_sector)
sector_stocks = sector_stocks.sort_values("涨跌幅", ascending=False)

leader = sector_stocks.iloc[0]

st.write(f"🔥 当前最强板块：{top_sector}")
st.write(f"龙头：{leader['名称']}  涨跌幅：{leader['涨跌幅']}%")

# 分钟K线
try:
    minute_df = ak.stock_zh_a_minute(symbol=leader["代码"], period="5")
    minute_df["时间"] = pd.to_datetime(minute_df["时间"])
    minute_df = minute_df.set_index("时间")
    st.line_chart(minute_df["收盘"])
except:
    st.info("暂无分钟数据")

# =============================
# 4️⃣ 全球新闻 + AI判断
# =============================

st.header("🌍 今日全球新闻")

news_df = ak.news_global_cls()
news_titles = news_df["标题"].head(10)

selected_news = st.selectbox("选择新闻", news_titles)

OPENAI_KEY = st.text_input("输入OpenAI API Key（可选）", type="password")

if st.button("分析新闻影响"):

    if OPENAI_KEY != "":
        client = OpenAI(api_key=OPENAI_KEY)

        prompt = f"""
        判断这条新闻对A股是利好还是利空，
        给出影响板块，
        简要说明原因：
        {selected_news}
        """

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(res.choices[0].message.content)

    else:
        st.warning("未输入API Key，仅做参考阅读")

# =============================
# 5️⃣ 辅助判断建议
# =============================

st.header("🧠 辅助决策建议")

if emotion == "主升期 🔥":
    st.success("情绪较好，可优先关注强势板块龙头回踩机会")

elif emotion == "冰点 ❄":
    st.error("情绪较差，建议轻仓或观望")

else:
    st.info("震荡阶段，优选强势板块龙头低吸")

st.write(f"当前最强方向：{top_sector}")
st.write("策略建议：强板块 + 龙头 + 分时回踩 + 成交量配合")
