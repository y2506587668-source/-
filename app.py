import streamlit as st
import akshare as ak
import pandas as pd
from openai import OpenAI

# =============================
# 页面基础设置
# =============================
st.set_page_config(page_title="板块 + 新闻情绪量化系统", layout="wide")
st.title("📊 板块强度 + 新闻情绪评分系统")

# =============================
# OpenAI 初始化
# =============================
client = OpenAI(api_key="你的APIKEY")  # ←←← 填入你的key

# =============================
# 新闻情绪判断函数
# =============================
def get_news_sentiment(news_text):

    prompt = f"""
    请判断以下新闻对A股整体情绪是利好、利空还是中性。
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
# 板块综合评分计算
# =============================
def calculate_total_score(sector_df, news_score):

    # 转换涨跌幅
    sector_df["涨跌幅"] = sector_df["涨跌幅"].str.replace("%", "").astype(float)

    # 标准化
    max_val = sector_df["涨跌幅"].max()
    min_val = sector_df["涨跌幅"].min()

    if max_val == min_val:
        sector_df["板块强度"] = 0.5
    else:
        sector_df["板块强度"] = (
            (sector_df["涨跌幅"] - min_val) / (max_val - min_val)
        )

    # 综合评分
    sector_df["综合评分"] = 0.6 * sector_df["板块强度"] + 0.4 * news_score

    return sector_df.sort_values("综合评分", ascending=False)


# =============================
# 页面输入区域
# =============================
news_text = st.text_area("📰 输入新闻内容")

if st.button("🚀 开始分析"):

    if news_text.strip() == "":
        st.warning("请输入新闻内容")
        st.stop()

    # ========= 新闻情绪 =========
    with st.spinner("正在分析新闻情绪..."):
        news_score = get_news_sentiment(news_text)

    if news_score == 1:
        st.success("新闻情绪判断：利好 📈")
    elif news_score == -1:
        st.error("新闻情绪判断：利空 📉")
    else:
        st.info("新闻情绪判断：中性 ⚖️")

    # ========= 获取板块 =========
    with st.spinner("正在获取板块数据..."):
        try:
            sector_df = ak.stock_board_industry_name_em()

            if sector_df is None or sector_df.empty:
                st.warning("板块数据暂时无法获取（可能被接口限制）")
                st.stop()

        except Exception as e:
            st.error("板块接口异常")
            st.text(str(e))
            st.stop()

    # ========= 计算综合评分 =========
    result_df = calculate_total_score(sector_df, news_score)

    st.subheader("🔥 综合评分排名（前10）")
    st.dataframe(result_df.head(10), use_container_width=True)
