import os
import time
import json
import requests
import pandas as pd
import anthropic
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

    # step 1 — search exactly 3 times, collect raw facts
    search_messages = [{
        "role": "user",
        "content": f"""You are a senior financial analyst.

Search exactly 3 times, no more:
1. Search: "Federal Reserve {date_str}"
2. Search: "inflation news {date_str}"
3. Search: "market reaction Fed {date_str}"

After exactly 3 searches stop. Do not search again.
Return only the key facts you found — no analysis yet."""
    }]

    max_turns = 6
    turns = 0
    search_results = ""

    while turns < max_turns:
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=search_messages
            )

            turns += 1

            for block in response.content:
                if hasattr(block, "text") and block.text and block.text.strip():
                    # accumulate every text block across all turns — don't overwrite
                    # with the last fragment (that left the formatter with ~nothing)
                    search_results += block.text.strip() + "\n\n"

            cleaned_content = []
            for block in response.content:
                if hasattr(block, "text"):
                    if not block.text or not block.text.strip():
                        continue
                    block.text = block.text.strip()
                cleaned_content.append(block)

            if cleaned_content:
                search_messages.append({
                    "role": "assistant",
                    "content": cleaned_content
                })

            if response.stop_reason == "end_turn":
                break

            time.sleep(20)

        except anthropic.RateLimitError:
            print("Rate limit hit — waiting 90 seconds...")
            time.sleep(90)
            continue

    # guard: if the search turns produced no usable findings, skip this day
    # instead of feeding the formatter an empty string (which yields a refusal)
    search_results = search_results.strip()
    if not search_results:
        return None

    # step 2 — format report with no web search (cheap)
    report_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Based on these facts about {date_str}:

{search_results}

Context:
- Sentiment score: {sentiment:.3f} (-1.0 = very negative, +1.0 = very positive)
- Articles published: {articles}

Write a structured report in exactly this format:

KEYWORDS:
3 keywords maximum capturing main themes

SUMMARY:
2-3 sentences explaining what happened and why sentiment hit {sentiment:.3f}

INVESTMENT WATCHOUT:
1-2 sentences on what investors should watch. Be specific about asset classes and sectors.

SOURCES:
Top 3 most prestigious sources from the research"""
        }]
    )

    return report_response.content[0].text.strip()


# ── read reports ───────────────────────────────────────────
def print_reports():
    if not os.path.exists("investigation_reports.json"):
        print("No reports found yet")
        return

    with open("investigation_reports.json", "r") as f:
        reports = json.load(f)

    for report in reports:
        print(f"\n{'='*60}")
        print(f"Date: {report['date']}")
        print(f"Sentiment: {report['sentiment']} | Articles: {report['articles']} | High volume: {report['high_volume']}")
        print(f"Investigated at: {report['investigated_at']}")
        print(f"{'-'*60}")
        print(report['report'])
    print(f"\n{'='*60}")
    print(f"Total reports: {len(reports)}")


# ── main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. fetch
    df = fetch_incremental(TOPIC, API_KEY)

    # 2. score — FinBERT (finance-tuned) plus a VADER (lexical) baseline to compare
    finbert = load_finbert()
    print("Scoring headlines...")
    df["sentiment"] = df["title"].apply(lambda t: score_finbert(t, finbert))
    df["label"] = df["sentiment"].apply(
        lambda s: "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
    )
    vader = SentimentIntensityAnalyzer()
    df["vader"] = df["title"].apply(lambda t: vader.polarity_scores(str(t))["compound"])
    df["vader_label"] = df["vader"].apply(
        lambda s: "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"
    )
    df.to_csv(CSV_PATH, index=False)          # persist scores — don't leave the CSV unscored
    print(f"Scored {len(df)} articles")

    # 3. aggregate and detect
    daily, anomalies = aggregate(df)
    daily.to_csv("daily_aggregate.csv", index=False)   # one row/day — what layer3 reads

    # 4. load already investigated dates
    if os.path.exists("investigation_reports.json"):
        with open("investigation_reports.json", "r") as f:
            existing_reports = json.load(f)
        already_investigated = [r["date"] for r in existing_reports]
    else:
        existing_reports = []
        already_investigated = []

    # 5. filter to only new anomaly days
    anomalies = anomalies[
        ~anomalies["published"].dt.strftime("%B %d, %Y").isin(already_investigated)
    ]

    # 6. investigate new anomalies only
    if anomalies.empty:
        print("No new anomalies to investigate — pipeline complete")
    else:
        new_reports = []
        print("\nRunning agentic investigation...")
        print("=" * 60)

        for _, row in anomalies.iterrows():
            date_str = row["published"].strftime("%B %d, %Y")
            print(f"\nInvestigating {date_str}...")
            print(f"Sentiment: {row['avg_sentiment']:.3f} | Articles: {row['article_count']}")
            print("-" * 40)
            report = investigate_with_search(row)
            if not report:
                print("No usable findings — skipping (not written to reports)")
                print("=" * 60)
                continue
            print(report)
            print("=" * 60)

            new_reports.append({
                "date": date_str,
                "sentiment": round(row["avg_sentiment"], 3),
                "articles": int(row["article_count"]),
                "high_volume": bool(row["high_volume"]),
                "report": report,
                "investigated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            print("Waiting 60 seconds...")
            time.sleep(60)

        # append new reports to existing ones
        all_reports = existing_reports + new_reports
        with open("investigation_reports.json", "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\nSaved {len(new_reports)} new reports. Total: {len(all_reports)}")

    print("\nPipeline complete!")