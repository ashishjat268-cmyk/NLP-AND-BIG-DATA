# app.py (redesigned UI: Modern Dark Tab System)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import csv
from urllib.parse import urljoin
import logging
from typing import Optional, Tuple, List
from ftfy import fix_text

# ------- API KEY FETCH --------
API_KEY = st.secrets.get("FACTCHECK_API_KEY", None)

# ------- UI DARK MODE CSS --------
st.set_page_config(page_title="AI Fact Verification System", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: #E8ECF2;
}
.report-box {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
    border:1px solid rgba(255,255,255,0.07);
}
.header {
    background: linear-gradient(90deg,#16232E,#4A0E5C);
    padding:20px;
    border-radius:10px;
    margin-bottom:20px;
    box-shadow:0px 0px 30px rgba(0,0,0,0.4);
}
h1 {color:white;}
</style>
""", unsafe_allow_html=True)


# ------- CLEAN HELPER -------
def clean(text):
    if text is None:
        return None
    try:
        text = fix_text(text)
    except:
        pass
    return " ".join(text.split()).strip()


# ------- GOOGLE FACT CHECK -------
def google_fact_check(query):
    if not API_KEY:
        return []
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"
    r = requests.get(url)
    data = r.json()

    results = []
    for c in data.get("claims", []):
        for review in c.get("claimReview", []):
            results.append({
                "source": review.get("publisher", {}).get("name", "Unknown"),
                "title": review.get("title", "N/A"),
                "rating": review.get("textualRating", "No Rating"),
                "url": review.get("url", "")
            })
    return results



# ------- SCRAPER --------
def scrape_politifact(start, end):
    base = "https://www.politifact.com/factchecks/list/"
    url = base
    scraped = []
    page = 0

    while url and page < 30:
        page += 1
        resp = requests.get(url)

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.find_all("li", class_="o-listicle__item")

        if not cards:
            break

        for item in cards:
            date_div = item.find("div", class_="m-statement__desc")
            if not date_div:
                continue

            m = re.search(r"(\w+ \d{1,2}, \d{4})", date_div.text)
            if not m:
                continue

            claim_date = pd.to_datetime(m.group(1))

            if claim_date < pd.to_datetime(start):
                return pd.DataFrame(scraped)

            if claim_date > pd.to_datetime(end):
                continue

            text = clean(item.find("div","m-statement__quote").text)
            source = clean(item.find("a","m-statement__name").text)
            label = clean(item.find("img").get("alt"))

            scraped.append({
                "Date": str(claim_date.date()),
                "Statement": text,
                "Source": source,
                "Label": label
            })

        next_btn = soup.find("a", string=re.compile("Next"))
        if next_btn:
            url = urljoin(base, next_btn["href"])
        else:
            break

    return pd.DataFrame(scraped)



# ------- MAIN UI TABS -------
st.markdown("""<div class="header">
<h1>📰 AI Fact Verification System</h1>
<p>Scrape → Train → Verify political facts using machine learning + Google fact-checking.</p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home", "📌 Scraper", "🤖 ML Results", "🔍 Fact-Check"])

# -------- HOME TAB --------
with tabs[0]:
    st.subheader("What this system does:")
    st.write("✔ Scrapes real political claims\n✔ Trains ML models\n✔ Verifies statements via Google Fact Check")

# -------- SCRAPER TAB --------
with tabs[1]:
    st.subheader("📌 Scrape Politifact Data")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    if st.button("Start Scraping"):
        with st.spinner("Fetching data..."):
            df = scrape_politifact(start, end)

        if df.empty:
            st.warning("⚠ No results found. Try different date range.")
        else:
            st.success(f"Collected {len(df)} claims.")
            st.session_state["dataset"] = df
            st.dataframe(df)
            st.download_button("⬇ Download CSV", df.to_csv(index=False), "scraped_data.csv", "text/csv")


# -------- ML RESULTS TAB ******** (Dummy UI - real code same logic) ------
# -------- ML RESULTS TAB --------
with tabs[2]:
    st.subheader("🤖 Train ML Classifiers")

    if "dataset" not in st.session_state:
        st.warning("⚠ Please scrape data first!")
    else:
        df = st.session_state["dataset"]
        st.success(f"Dataset loaded ✔ ({len(df)} records)")

        feature_type = st.selectbox(
            "Select Feature Extraction Method:",
            ["Lexical", "Syntactic", "Semantic", "Pragmatic"]
        )

        if st.button("🚀 Train Models"):
            with st.spinner("Training models... Please wait ⏳"):
                
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB
                from sklearn.metrics import accuracy_score

                X = df["Statement"]
                y = df["Label"]

                # Vectorizer
                vectorizer = TfidfVectorizer(stop_words="english")
                X_vec = vectorizer.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_vec, y, test_size=0.2, random_state=42
                )

                # Model
                model = MultinomialNB()
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                st.subheader("📊 Result")
                st.success(f"Model Accuracy: **{acc*100:.2f}%**")
                st.write("Model successfully trained and tested ✔")

# -------- FACT CHECK TAB --------
with tabs[3]:
    st.subheader("🔍 Verify A Statement")
    q = st.text_input("Enter a claim...")

    if st.button("Check Credibility"):
        res = google_fact_check(q)
        if not res:
            st.warning("No matching verified result found.")
        else:
            for r in res:
                st.markdown(f"""
                <div class="report-box">
                <b>Source:</b> {r['source']}  
                <br><b>Rating:</b> {r['rating']}
                <br><b>Title:</b> {r['title']}
                <br><a href="{r['url']}" target='_blank'>🔗 Read Full Article</a>
                </div>
                """, unsafe_allow_html=True)
