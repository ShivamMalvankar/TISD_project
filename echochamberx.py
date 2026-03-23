"""
EchoChamberX – Bias & Polarization Detection System
====================================================
A modular NLP pipeline for detecting sentiment, bias,
and polarization across multiple text sources.

Dependencies: nltk, re, collections, pandas (optional)
"""

import re
import json
from collections import Counter

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# One-time NLTK downloads (safe to call repeatedly)
# ---------------------------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_sia = SentimentIntensityAnalyzer()
_STOPWORDS = set(stopwords.words("english"))

# Rule-based bias keyword lists
PRO_KEYWORDS = {
    "support", "benefit", "improve", "positive", "success",
    "achievement", "progress", "growth", "advance", "favor",
    "promote", "strengthen", "help", "boost", "gain",
}

ANTI_KEYWORDS = {
    "protest", "oppose", "against", "harm", "violence",
    "threat", "attack", "damage", "fail", "failure",
    "crisis", "danger", "destroy", "corrupt", "abuse",
    "risk", "condemn", "reject", "ban", "block",
}


# ===========================================================================
# 1. PREPROCESSING
# ===========================================================================

def preprocess_text(text: str) -> str:
    """
    Clean raw text:
      - Lowercase
      - Strip URLs
      - Remove special characters / digits
      - Remove stopwords
    Returns a cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove special characters and digits (keep spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    tokens = [w for w in text.split() if w not in _STOPWORDS and len(w) > 1]

    return " ".join(tokens)


# ===========================================================================
# 2. SENTIMENT ANALYSIS  (VADER)
# ===========================================================================

def analyze_sentiment(text: str) -> dict:
    """
    Use VADER SentimentIntensityAnalyzer on the *original* (uncleaned) text
    for best accuracy, then return a structured result.

    Returns:
        {
            "label":    "positive" | "negative" | "neutral",
            "compound": float  (-1.0 … +1.0)
        }
    """
    scores = _sia.polarity_scores(text)
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "compound": compound}


# ===========================================================================
# 3. BIAS CLASSIFICATION
# ===========================================================================

def classify_bias(text: str, sentiment: dict) -> str:
    """
    Rule-based bias classifier.

    Logic:
      - pro_keywords AND positive sentiment  → "pro"
      - anti_keywords AND negative sentiment → "anti"
      - Otherwise                            → "neutral"
    """
    tokens = set(text.lower().split())
    has_pro  = bool(tokens & PRO_KEYWORDS)
    has_anti = bool(tokens & ANTI_KEYWORDS)
    label    = sentiment.get("label", "neutral")

    if has_pro and label == "positive":
        return "pro"
    if has_anti and label == "negative":
        return "anti"
    return "neutral"


# ===========================================================================
# 4. KEYWORD EXTRACTION
# ===========================================================================

def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Return the top-N most frequent words from preprocessed text.

    Returns:
        [{"word": str, "count": int}, ...]
    """
    tokens = text.split()
    if not tokens:
        return []

    most_common = Counter(tokens).most_common(top_n)
    return [{"word": word, "count": count} for word, count in most_common]


# ===========================================================================
# 5. AGGREGATION
# ===========================================================================

def aggregate_results(results_list: list) -> dict:
    """
    Aggregate per-article results into topic-level counts.

    Input:  list of dicts (each produced by the per-article pipeline)
    Output:
        {
            "sentiment": {"positive": int, "negative": int, "neutral": int},
            "bias":      {"pro":      int, "anti":     int, "neutral": int},
            "trend":     [{"index": int, "compound": float, "label": str}, ...]
        }
    """
    sentiment_counts = Counter()
    bias_counts = Counter()
    trend = []

    for i, result in enumerate(results_list):
        sentiment_counts[result["sentiment"]["label"]] += 1
        bias_counts[result["bias"]] += 1
        trend.append({
            "index":    i,
            "compound": result["sentiment"]["compound"],
            "label":    result["sentiment"]["label"],
        })

    return {
        "sentiment": {
            "positive": sentiment_counts["positive"],
            "negative": sentiment_counts["negative"],
            "neutral":  sentiment_counts["neutral"],
        },
        "bias": {
            "pro":     bias_counts["pro"],
            "anti":    bias_counts["anti"],
            "neutral": bias_counts["neutral"],
        },
        "trend": trend,
    }


# ===========================================================================
# 6. POLARIZATION SCORE
# ===========================================================================

def compute_polarization(pro: int, anti: int, neutral: int) -> float:
    """
    Polarization formula:
        polarization = (1 - |pro - anti| / total) * ((pro + anti) / total) * 100

    Ranges from 0 (fully neutral) to 100 (perfectly split, no neutral).
    Returns 0.0 when total == 0.
    """
    total = pro + anti + neutral
    if total == 0:
        return 0.0

    balance   = 1 - abs(pro - anti) / total
    intensity = (pro + anti) / total
    score     = balance * intensity * 100

    return round(score, 2)


# ===========================================================================
# 7. INSIGHT GENERATION
# ===========================================================================

def generate_insight(data: dict) -> str:
    """
    Produce a human-readable insight string based on aggregated data.

    Considers:
      - Polarization level (high / medium / low)
      - Dominant sentiment trend
    """
    score    = data.get("polarization_score", 0)
    sentiment = data.get("sentiment", {})

    pos = sentiment.get("positive", 0)
    neg = sentiment.get("negative", 0)
    neu = sentiment.get("neutral",  0)
    total_s = pos + neg + neu or 1  # avoid division by zero

    # Dominant sentiment
    dominant = max(sentiment, key=sentiment.get) if sentiment else "neutral"
    pct      = round(sentiment.get(dominant, 0) / total_s * 100)

    # Polarization tier
    if score > 70:
        pol_text = "highly polarized with strong opposing views"
    elif score > 40:
        pol_text = "showing moderate division in opinions"
    else:
        pol_text = "showing low polarization with mostly neutral sentiment"

    # Sentiment trend note
    trend_note = f"with {pct}% {dominant} sentiment dominating"

    insight = (
        f"This topic is {pol_text}, "
        f"{trend_note}. "
        f"(Polarization score: {score}/100)"
    )
    return insight


# ===========================================================================
# 8. FINAL PIPELINE
# ===========================================================================

def analyze_topic(text_list: list) -> dict:
    """
    Full EchoChamberX pipeline.

    Args:
        text_list: list of raw text strings (news snippets, tweets, etc.)

    Returns:
        Structured JSON-compatible dict:
        {
            "polarization_score": float,
            "sentiment":  {"positive": int, "negative": int, "neutral": int},
            "bias":       {"pro": int, "anti": int, "neutral": int},
            "trend":      [...],
            "articles":   [...],
            "insight":    str
        }
    """
    if not text_list:
        return {
            "polarization_score": 0.0,
            "sentiment":  {"positive": 0, "negative": 0, "neutral": 0},
            "bias":       {"pro": 0, "anti": 0, "neutral": 0},
            "trend":      [],
            "articles":   [],
            "insight":    "No data provided for analysis.",
        }

    per_article = []

    for idx, raw_text in enumerate(text_list):
        # Stage 1 – Preprocess (cleaned copy for keywords/bias)
        clean = preprocess_text(raw_text)

        # Stage 2 – Sentiment (run on original for VADER accuracy)
        sentiment = analyze_sentiment(raw_text)

        # Stage 3 – Bias
        bias = classify_bias(clean, sentiment)

        # Stage 4 – Keywords
        keywords = extract_keywords(clean, top_n=8)

        per_article.append({
            "index":     idx,
            "raw_text":  raw_text[:200],   # truncate for storage
            "clean_text": clean,
            "sentiment": sentiment,
            "bias":      bias,
            "keywords":  keywords,
        })

    # Stage 5 – Aggregate
    aggregated = aggregate_results(per_article)

    # Stage 6 – Polarization score
    b = aggregated["bias"]
    pol_score = compute_polarization(b["pro"], b["anti"], b["neutral"])

    # Assemble output
    output = {
        "polarization_score": pol_score,
        "sentiment":          aggregated["sentiment"],
        "bias":               aggregated["bias"],
        "trend":              aggregated["trend"],
        "articles":           per_article,
        "insight":            "",          # filled next
    }

    # Stage 7 – Insight
    output["insight"] = generate_insight(output)

    return output


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

SAMPLE_TEXTS = [
    "The new government policy has received widespread support from citizens "
    "who believe it will significantly improve public healthcare and benefit "
    "the most vulnerable communities.",

    "Thousands gathered to protest against the controversial legislation, "
    "condemning it as harmful to civil liberties. Violence broke out near "
    "the parliament as police clashed with demonstrators opposing the bill.",

    "The report presents a balanced overview of the proposed infrastructure "
    "project, outlining both the potential economic gains and the logistical "
    "challenges involved in its implementation.",

    "Activists strongly oppose the new environmental regulations, arguing "
    "they damage local businesses and threaten thousands of jobs across "
    "rural communities already struggling with poverty.",

    "Economists largely agree that the trade agreement will boost exports "
    "and promote growth, though some critics warn of risks to domestic "
    "manufacturing in the short term.",

    "The opposition party condemned the decision, calling it corrupt and "
    "dangerous. They called for an immediate investigation into the abuse "
    "of power by senior officials.",

    "Community leaders expressed support for the urban renewal program, "
    "highlighting its positive impact on housing availability and the "
    "improvement of local schools.",

    "International observers raised concerns about human rights violations, "
    "citing evidence of violence against minorities and the suppression "
    "of free speech under the current administration.",
]


if __name__ == "__main__":
    print("=" * 60)
    print("  EchoChamberX – Bias & Polarization Detection System")
    print("=" * 60)

    result = analyze_topic(SAMPLE_TEXTS)

    # Pretty-print JSON (exclude verbose article clean_text for readability)
    summary = {k: v for k, v in result.items() if k != "articles"}
    print(json.dumps(summary, indent=2))

    print("\n--- Per-Article Summary ---")
    for art in result["articles"]:
        print(
            f"[{art['index']}] Sentiment: {art['sentiment']['label']:8s} "
            f"| Bias: {art['bias']:7s} "
            f"| Compound: {art['sentiment']['compound']:+.3f} "
            f"| Top keywords: {[kw['word'] for kw in art['keywords'][:4]]}"
        )

    print(f"\n{'='*60}")
    print(f"  Polarization Score : {result['polarization_score']}")
    print(f"  Insight            : {result['insight']}")
    print(f"{'='*60}")


def analyze_dataset(data: list[dict]) -> dict:
    """
    Adapter function: Takes the list of dictionaries from the web scraper,
    extracts the text, runs the EchoChamberX NLP pipeline, and formats
    the output for the Flask frontend.
    """
    if not data:
        return {"error": "No data available to analyze."}

    # 1. Extract just the text strings from the scraper's dictionary output
    text_list = [entry.get("text", "") for entry in data]

    # 2. Run your existing brilliant NLP pipeline!
    results = analyze_topic(text_list)

    # 3. Map your polarization score to the UI's Risk Levels
    pol_score = results.get("polarization_score", 0)

    if pol_score > 70:
        risk_level = "High"
        risk_color = "#ff4444"  # Red
    elif pol_score > 40:
        risk_level = "Medium"
        risk_color = "#ffbb33"  # Yellow
    else:
        risk_level = "Low"
        risk_color = "#00C851"  # Green

    # 4. Aggregate the top keywords across all articles to send to the UI
    all_keywords = {}
    for article in results.get("articles", []):
        for kw in article.get("keywords", []):
            word = kw["word"]
            count = kw["count"]
            all_keywords[word] = all_keywords.get(word, 0) + count

    # Sort keywords by highest count and grab the top 8
    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:8]
    top_keywords_formatted = [{"word": w, "count": c} for w, c in sorted_keywords]

    # 5. Return the exact JSON structure the HTML frontend is looking for
    return {
        "summary": results.get("insight", "Analysis complete."),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "top_keywords": top_keywords_formatted,
        "detailed_stats": {
            "sentiment": results.get("sentiment"),
            "bias": results.get("bias"),
            "polarization_score": pol_score
        }
    }