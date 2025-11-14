import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import time
import logging
from ftfy import fix_text
from typing import Optional, Tuple, List

# ============================
#  KEEPING BACKEND SAME
# ============================

API_KEY = st.secrets["FACTCHECK_API_KEY"] if hasattr(st, "secrets") and "FACTCHECK_API_KEY" in st.secrets else None

def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except:
        pass
    return " ".join(s.split()).strip()

def get_fact_check_results(query):
    if not API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    try:
        r = requests.get(url, params=params)
        data = r.json()
        results = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                results.append({
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "title": review.get("title", ""),
                    "rating": review.get("textualRating", "N/A"),
                    "url": review.get("url", "")
                })
        return results
    except:
        return []

logging.basicConfig(level=logging.INFO)

def scrape_data(start, end):
    base_url = "https://www.politifact.com/factchecks/list/"
    url = base_url
    rows = []
    page = 0

    while url and page < 50:
        page += 1
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all("li", class_="o-listicle__item")

        if not items:
            break

        for card in items:
            date_text = card.find("div", class_="m-statement__desc").text.strip()
            date_match = re.search(r"(\w+ \d{1,2}, \d{4})", date_text)

            if not date_match:
                continue

            claim_date = pd.to_datetime(date_match.group(1))

            if claim_date < pd.to_datetime(start):
                return pd.DataFrame(rows)

            if claim_date > pd.to_datetime(end):
                continue

            text = clean(card.find("div", class_="m-statement__quote").text)
            source = clean(card.find("a", class_="m-statement__name").text)
            label = clean(card.find("img").get("alt"))

            rows.append({"Date": str(claim_date.date()), "Statement": text, "Source": source, "Label": label})

        next_btn = soup.find("a", string=re.compile("Next"))
        if next_btn:
            url = urljoin(base_url, next_btn["href"])
        else:
            break

    return pd.DataFrame(rows)


# ============================
#  MACHINE LEARNING SAME LOGIC
# ============================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


# ============================
#  Modern UI
# ============================

st.set_page_config(page_title="FactCheck AI System", layout="wide")

st.markdown("""
<style>
.stApp {background:#0E1117;color:#EAF0F6;}
.sidebar .sidebar-content {background:#161B22;}
h1,h2,h3,h4 {color:#FFFFFF;}
.box {
    padding:15px;
    background:#161B22;
    border-radius:10px;
    margin-top:10px;
    border:1px solid #2E3542;
}
button {border-radius:8px !important;}
</style>
""", unsafe_allow_html=True)

st.title("📰 AI-Based Fake News Detector & Fact-Checker")

menu = ["🏠 Home", "📌 Scraper", "🤖 Train Model", "🔍 Fact Check"]
choice = st.sidebar.radio("Navigation", menu)

# ============================
#  HOME
# ============================

if choice == "🏠 Home":
    st.markdown("""
    ### Welcome 👋  
    This system allows you to:

    ✔ Scrape live political statements  
    ✔ Train machine learning models  
    ✔ Verify claims using Google Fact Check API  
    """)

# ============================
#  SCRAPER UI
# ============================

elif choice == "📌 Scraper":
    st.subheader("📌 Scrape Political Claims")

    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    if st.button("📥 Start Scraping"):
        with st.spinner("Scraping data..."):
            df = scrape_data(start, end)
        if df.empty:
            st.warning("⚠ No records found in this date range.")
        else:
            st.success(f"Scraped {len(df)} statements 🎯")
            st.dataframe(df)
            st.session_state["dataset"] = df

# ============================
#  TRAINING UI
# ============================

elif choice == "🤖 Train Model":
    st.subheader("🤖 Train Fake News Classifier")

    if "dataset" not in st.session_state:
        st.error("⚠ Please scrape data first.")
    else:
        df = st.session_state["dataset"]

        if st.button("🚀 Run Training"):
            with st.spinner("Training model..."):

                X = df["Statement"]
                y = LabelEncoder().fit_transform(df["Label"])

                X_train, X_test, y_train, y_test = train_test_split(
                    TfidfVectorizer(stop_words='english').fit_transform(X),
                    y,
                    test_size=0.3,
                    random_state=42
                )

                model = MultinomialNB()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

            st.success(f"🎉 Model Trained Successfully — Accuracy: **{acc*100:.2f}%**")

# ============================
#  FACT-CHECK UI
# ============================

elif choice == "🔍 Fact Check":
    st.subheader("🔍 Real-time Claim Verification")

    query = st.text_input("Enter any claim to verify:")

    if st.button("Check Now"):
        with st.spinner("Checking..."):
            results = get_fact_check_results(query)

        if not results:
            st.warning("No verification found.")
        else:
            for r in results[:10]:
                st.markdown(f"""
                <div class='box'>
                <b>Source:</b> {r['publisher']}  
                <br><b>Verdict:</b> {r['rating']}
                <br><b>Title:</b> {r['title']}
                <br><a href="{r['url']}" target="_blank">🔗 Read More</a>
                </div>
                """, unsafe_allow_html=True)

