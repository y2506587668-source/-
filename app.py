import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="网页爬虫财经新闻分析", layout="wide")
st.title("🌎 网页爬虫财经新闻 + 板块分析")

OPENAI_API_KEY = "你的OpenAI_Key"
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# 爬取东方财富财经新闻
# ----------------------------
def fetch_eastmoney_news(limit=10):
    url = "http://finance.eastmoney.com/a/cgnjj.html"  # 东方财富-国内财经新闻
    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        news_items = soup.select("div.newsList a")[:limit]
        data = []
        for item in news_items:
            title = item.get_text(strip=True)
            link = item["href"]
            data.append({"title": title, "url": link})
        return pd.DataFrame(data)
    except Exception as e:
        st.warning("新闻抓取失败")
        st.text(str(e))
        return pd.DataFrame()

# ----------------------------
# GPT 新闻情绪分析
# ----------------------------
def analyze_sentiment(news_list):
    combined_text = "\n".join(news_list)
    prompt = f"""
    以下是财经新闻，请判断整体对A股情绪：
    只回答：利好 / 利空 / 中性
    新闻：
    {combined_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        if "利好" in text:
            return 1
        elif "利空" in text:
            return -1
        else:
            return 0
    except:
        return 0

# ----------------------------
# 主程序
# ----------------------------
if st.button("📡 抓取东方财富新闻"):
    news_df = fetch_eastmoney_news(limit=15)
    if news_df.empty:
        st.error("未抓取到新闻")
        st.stop()

    st.subheader("📰 最新财经新闻")
    st.dataframe(news_df)

    sentiment_score = analyze_sentiment(news_df["title"].tolist())
    msg = "中性 ⚖️"
    if sentiment_score == 1: msg = "利好 📈"
    if sentiment_score == -1: msg = "利空 📉"
    st.markdown(f"**整体新闻情绪分析：{msg}**")
