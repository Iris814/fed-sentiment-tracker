"""
VADER vs FinBERT — do a lexical and a finance-tuned model agree on Fed/inflation headlines?

Reads the scored corpus (daily_sentiment.csv, which carries both a FinBERT `sentiment`
and a VADER `vader` column from pipeline.py), quantifies how much the two disagree, and
saves a scatter + label-agreement figure to vader_vs_finbert.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["negative", "neutral", "positive"]

df = pd.read_csv("daily_sentiment.csv")
df = df.dropna(subset=["sentiment", "vader"])

corr  = float(np.corrcoef(df["sentiment"], df["vader"])[0, 1])
agree = float((df["label"] == df["vader_label"]).mean())
conf  = (pd.crosstab(df["vader_label"], df["label"])
           .reindex(index=LABELS, columns=LABELS, fill_value=0))

# headlines where the two models flip sign hardest
flip = df[((df["sentiment"] > 0.3) & (df["vader"] < -0.3)) |
          ((df["sentiment"] < -0.3) & (df["vader"] > 0.3))]

print(f"Articles: {len(df)}")
print(f"Pearson r (FinBERT vs VADER): {corr:.2f}")
print(f"Label agreement: {agree:.0%}")
print(f"Hard sign-flips: {len(flip)}\n")
print("Example divergences (VADER reacts to tone words, FinBERT to financial framing):")
for _, r in flip.head(5).iterrows():
    print(f"  vader={r['vader']:+.2f}  finbert={r['sentiment']:+.2f}  | {r['title'][:72]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.scatter(df["vader"], df["sentiment"], s=8, alpha=0.25, color="#0d9488")
ax1.axhline(0, color="#94a3b8", lw=.8); ax1.axvline(0, color="#94a3b8", lw=.8)
ax1.set(xlabel="VADER (lexical, compound)", ylabel="FinBERT (finance-tuned)",
        title=f"Per-headline sentiment: VADER vs FinBERT\n"
              f"Pearson r = {corr:.2f}  |  label agreement = {agree:.0%}")
ax2.imshow(conf.values, cmap="GnBu")
ax2.set_xticks(range(3)); ax2.set_xticklabels(LABELS)
ax2.set_yticks(range(3)); ax2.set_yticklabels(LABELS)
ax2.set(xlabel="FinBERT label", ylabel="VADER label", title="Label agreement (counts)")
for i in range(3):
    for j in range(3):
        v = conf.values[i, j]
        ax2.text(j, i, v, ha="center", va="center",
                 color="white" if v > conf.values.max() * 0.5 else "#0f172a", fontsize=9)
fig.tight_layout()
fig.savefig("vader_vs_finbert.png", dpi=130, bbox_inches="tight")
print("\nSaved vader_vs_finbert.png")
