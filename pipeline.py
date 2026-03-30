import os
import time
import json
import requests
import pandas as pd
import anthropic
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

load_dotenv()

API_KEY       = os.getenv("NEWS_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
TOPIC         = '"Federal Reserve" AND ("interest rates" OR inflation)'
CSV_PATH      = "daily_sentiment.csv"

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── fetch ──────────────────────────────────────────────────
def fetch_headlines(topic, api_key, days_back=27):
    headlines = []
    url = "https://newsapi.org/v2/everything"
    now = datetime.now()

    for day_offset in range(days_back, 0, -1):
        day = now - timedelta(days=day_offset)
        day_str = day.strftime("%Y-%m-%d")
        params = {
            "q": topic, "from": day_str, "to": day_str,
            "language": "en", "sortBy": "publishedAt",
            "pageSize": 100, "apiKey": api_key,
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data["status"] != "ok":
            print(f"Error on {day_str}:", data.get("message"))
            continue
        for article in data["articles"]:
            headlines.append({
                "title": article["title"],
                "published": article["publishedAt"][:10],
                "source": article["source"]["name"],
                "url": article["url"],
            })
        print(f"{day_str} — {len(data['articles'])} articles")
        time.sleep(0.5)

    return pd.DataFrame(headlines).drop_duplicates(subset="url")

def fetch_incremental(topic, api_key, csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["published"] = pd.to_datetime(existing["published"]).dt.date.astype(str)
        last_date = existing["published"].max()
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
        days_since = (datetime.now() - last_dt).days - 1
        print(f"Existing data found. Last date: {last_date}")
        print(f"Existing articles: {len(existing)}")
        if days_since <= 0:
            print("Already up to date — skipping fetch")
            return existing
        print(f"Fetching {days_since} new days...")
    else:
        existing = pd.DataFrame()
        days_since = 27
        print("No existing data. Fetching full history...")

    new_df = fetch_headlines(topic, api_key, days_back=days_since)
    if new_df.empty:
        print("No new articles found")
        return existing

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="url")
    combined["published"] = pd.to_datetime(combined["published"]).dt.date.astype(str)
    combined = combined.sort_values("published").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"New articles added: {len(combined) - len(existing)}")
    print(f"Total articles saved: {len(combined)}")
    return combined

# ── score ──────────────────────────────────────────────────
def load_finbert():
    print("Loading FinBERT...")
    return hf_pipeline("text-classification",
                        model="ProsusAI/finbert",
                        device=-1)

def score_finbert(title, finbert):
    try:
        result = finbert(str(title[:512]))[0]
        label = result["label"]
        score = result["score"]
        if label == "positive":   return score
        elif label == "negative": return -score
        else:                     return 0.0
    except:
        return 0.0

# ── aggregate ──────────────────────────────────────────────
def aggregate(df):
    daily = (
        df.groupby("published")
        .agg(
            avg_sentiment=("sentiment", "mean"),
            article_count=("title", "count"),
            positive=("label", lambda x: (x == "positive").sum()),
            negative=("label", lambda x: (x == "negative").sum()),
            neutral=("label", lambda x: (x == "neutral").sum()),
        )
        .reset_index()
    )
    daily["published"] = pd.to_datetime(daily["published"])
    daily = daily.sort_values("published")

    mean_vol = daily["article_count"].mean()
    std_vol  = daily["article_count"].std()
    daily["heat_score"]  = (daily["article_count"] - mean_vol) / std_vol
    daily["high_volume"] = daily["heat_score"] > 1.5

    mean_sentiment = daily["avg_sentiment"].mean()
    std_sentiment  = daily["avg_sentiment"].std()

    anomalies = daily[
        (daily["high_volume"] == True) |
        (daily["avg_sentiment"] < mean_sentiment - 1.5 * std_sentiment)
    ].copy().sort_values("avg_sentiment")

    print(f"Baseline sentiment: {mean_sentiment:.3f}")
    print(f"Anomaly threshold:  {mean_sentiment - 1.5 * std_sentiment:.3f}")
    print(f"Days flagged: {len(anomalies)}")
    return daily, anomalies

# ── investigate ────────────────────────────────────────────
def investigate_with_search(row):
    date_str = row["published"].strftime("%B %d, %Y")
    sentiment = row["avg_sentiment"]
    articles  = row["article_count"]

    messages = [{
        "role": "user",
        "content": f"""You are a senior financial analyst. Search the web for Federal Reserve and inflation news on {date_str}.

Context:
- Date: {date_str}
- Sentiment score: {sentiment:.3f} (-1.0 = very negative, +1.0 = very positive)
- Articles published: {articles}

After searching, respond in exactly this format:

KEYWORDS:
At most three keywords capturing the main themes

SUMMARY:
1-2 sentences explaining what happened and why sentiment hit {sentiment:.3f}

INVESTMENT WATCHOUT:
1-2 sentences on what investors should consider. Be specific about asset classes and sectors.

SOURCES:
List top three most prestige sources found as the name of channel"""
    }]

    max_turns = 6
    turns = 0
    final_report = ""

    while turns < max_turns:
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=700,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages
            )
            turns += 1

            for block in response.content:
                if hasattr(block, "text") and block.text:
                    final_report = block.text.strip()

            cleaned_content = []
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    block.text = block.text.strip()
                cleaned_content.append(block)
            messages.append({"role": "assistant", "content": cleaned_content})

            if response.stop_reason == "end_turn" and len(final_report) > 200:
                return final_report
            if response.stop_reason == "end_turn" and len(final_report) <= 200:
                messages.append({
                    "role": "user",
                    "content": "Please complete your full investigation report."
                })
            time.sleep(20)

        except anthropic.RateLimitError:
            print("Rate limit hit — waiting 90 seconds...")
            time.sleep(90)
            continue

    return final_report if final_report else "Investigation incomplete"

# ── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. fetch
    df = fetch_incremental(TOPIC, API_KEY)

    # 2. score
    finbert = load_finbert()
    print("Scoring headlines...")
    df["sentiment"] = df["title"].apply(lambda t: score_finbert(t, finbert))
    df["label"] = df["sentiment"].apply(
        lambda s: "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
    )
    print(f"Scored {len(df)} articles")

    # 3. aggregate and detect
    daily, anomalies = aggregate(df)

    # 4. investigate anomalies
    if anomalies.empty:
        print("No anomalies detected today — pipeline complete")
    else:
        reports = []
        print("\nRunning agentic investigation...")
        print("=" * 60)

        for _, row in anomalies.iterrows():
            date_str = row["published"].strftime("%B %d, %Y")
            print(f"\nInvestigating {date_str}...")
            print(f"Sentiment: {row['avg_sentiment']:.3f} | Articles: {row['article_count']}")
            print("-" * 40)
            report = investigate_with_search(row)
            print(report)
            print("=" * 60)

            reports.append({
                "date": date_str,
                "sentiment": round(row["avg_sentiment"], 3),
                "articles": int(row["article_count"]),
                "high_volume": bool(row["high_volume"]),
                "report": report,
                "investigated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            print("Waiting 60 seconds...")
            time.sleep(60)

        with open("investigation_reports.json", "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\nSaved {len(reports)} reports to investigation_reports.json")

    print("\nPipeline complete!")