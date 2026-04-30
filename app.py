
import nltk

# Auto-download required NLTK corpora on first run
for _pkg in ['stopwords', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{_pkg}' if 'punkt' in _pkg else f'corpora/{_pkg}')
    except LookupError:
        nltk.download(_pkg, quiet=True)

from flask import Flask, render_template, request
from textblob import TextBlob
from rake_nltk import Rake
import re
import io
import csv
from collections import Counter

app = Flask(__name__)
rake = Rake()

# ─────────────────────────────────────────────
# EXISTING HELPERS (unchanged)
# ─────────────────────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def label_sentiment(text, score):
    text = text.lower()
    neutral_phrases = [
        "not good", "not great", "not bad", "not as good",
        "average", "okay", "fine", "could be better"
    ]
    if any(phrase in text for phrase in neutral_phrases):
        return "Neutral 😐"
    positive_words = ["good", "great", "excellent", "amazing", "nice"]
    negative_words = ["bad", "poor", "worst", "slow", "terrible", "not"]
    has_pos = any(word in text for word in positive_words)
    has_neg = any(word in text for word in negative_words)
    if has_pos and has_neg:
        return "Mixed ⚖️"
    if score > 0.4:
        return "Positive 😊"
    elif score < -0.4:
        return "Negative 😡"
    else:
        return "Neutral 😐"

def predict_rating(score, sentiment):
    if "Negative" in sentiment:
        return 1
    elif "Mixed" in sentiment:
        return 3
    elif "Neutral" in sentiment:
        return 3
    elif score > 0.6:
        return 5
    elif score > 0.3:
        return 4
    else:
        return 3

def get_keywords(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:3]

def is_valid_review(text):
    text = text.strip()
    if text == "" or text.replace(".", "") == "":
        return False
    if len(text) < 5:
        return False
    return True

# ─────────────────────────────────────────────
# NEW FEATURE 1: Confidence Score
# ─────────────────────────────────────────────
def get_confidence(score):
    confidence = int(abs(score) * 100)
    confidence = max(confidence, 10)
    confidence = min(confidence, 99)
    return confidence

# ─────────────────────────────────────────────
# NEW FEATURE 2: Aspect-Based Sentiment Analysis
# ─────────────────────────────────────────────
ASPECTS = {
    "battery":      ["battery", "charge", "charging", "drain", "power"],
    "camera":       ["camera", "photo", "picture", "image", "lens", "shot", "megapixel"],
    "performance":  ["performance", "speed", "fast", "slow", "lag", "freeze", "smooth", "processor", "ram"],
    "display":      ["display", "screen", "brightness", "resolution", "pixel", "hdr"],
    "price":        ["price", "cost", "expensive", "cheap", "value", "worth", "money"],
    "design":       ["design", "build", "quality", "material", "weight", "thin", "sleek"],
    "software":     ["software", "app", "update", "os", "interface", "ui", "bug"],
}

def analyze_aspects(text):
    text_lower = text.lower()
    results = {}
    for aspect, keywords in ASPECTS.items():
        for kw in keywords:
            if kw in text_lower:
                sentences = re.split(r'[.!?]', text_lower)
                for sent in sentences:
                    if kw in sent:
                        score = TextBlob(sent).sentiment.polarity
                        if score > 0.1:
                            label = "Positive 😊"
                        elif score < -0.1:
                            label = "Negative 😡"
                        else:
                            label = "Neutral 😐"
                        results[aspect] = {"label": label, "score": round(score, 2)}
                        break
                break
    return results

# ─────────────────────────────────────────────
# NEW FEATURE 3: Recommendation Engine
# ─────────────────────────────────────────────
NEGATIVE_KEYWORD_MAP = {
    "battery":   "Improve battery life and charging speed",
    "charge":    "Improve charging speed and battery capacity",
    "slow":      "Optimize app performance and reduce lag",
    "lag":       "Enhance processing speed and system responsiveness",
    "freeze":    "Fix software stability and memory management issues",
    "expensive": "Consider offering better pricing or budget variants",
    "price":     "Reconsider pricing to improve value perception",
    "hot":       "Improve thermal management and heat dissipation",
    "heat":      "Improve thermal management and heat dissipation",
    "camera":    "Upgrade camera sensors and improve photo processing",
    "blurry":    "Enhance camera autofocus and image stabilization",
    "dark":      "Improve low-light camera performance",
    "screen":    "Improve display brightness and color accuracy",
    "display":   "Enhance screen resolution and refresh rate",
    "build":     "Improve build quality and use premium materials",
    "software":  "Fix software bugs and improve update stability",
    "bug":       "Address software bugs with a timely patch",
    "update":    "Streamline the update process for smoother experience",
    "storage":   "Offer more built-in storage options",
    "memory":    "Increase RAM or improve memory management",
    "loud":      "Improve audio quality and speaker tuning",
    "sound":     "Enhance audio hardware and software EQ options",
    "wifi":      "Improve Wi-Fi antenna and connection stability",
    "signal":    "Strengthen network signal reception",
    "fragile":   "Reinforce device durability and drop resistance",
    "scratch":   "Use more scratch-resistant materials or coatings",
}

def get_suggestions(text):
    text_lower = text.lower()
    suggestions = []
    seen = set()
    for kw, suggestion in NEGATIVE_KEYWORD_MAP.items():
        if kw in text_lower and suggestion not in seen:
            for sent in re.split(r'[.!?]', text_lower):
                if kw in sent:
                    score = TextBlob(sent).sentiment.polarity
                    if score < 0.05:
                        suggestions.append(suggestion)
                        seen.add(suggestion)
                    break
    return suggestions[:5]

# ─────────────────────────────────────────────
# NEW FEATURE 4: Review Quality Detection
# ─────────────────────────────────────────────
MEANINGFUL_WORDS = {
    "good", "bad", "great", "excellent", "poor", "terrible", "amazing",
    "awful", "fantastic", "horrible", "love", "hate", "recommend",
    "worth", "quality", "performance", "issue", "problem", "feature",
    "best", "worst", "fast", "slow", "easy", "hard", "nice", "weird",
    "broken", "works", "working", "stopped", "disappointed", "satisfied"
}

def get_review_quality(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    meaningful_count = sum(1 for w in words if w in MEANINGFUL_WORDS)
    unique_ratio = len(set(words)) / max(word_count, 1)
    score = 0
    if word_count >= 80:
        score += 2
    elif word_count >= 30:
        score += 1
    if meaningful_count >= 5:
        score += 2
    elif meaningful_count >= 2:
        score += 1
    if unique_ratio > 0.6:
        score += 1
    if score >= 4:
        return "High ⭐⭐⭐"
    elif score >= 2:
        return "Medium ⭐⭐"
    else:
        return "Low ⭐"

# ─────────────────────────────────────────────
# NEW FEATURE 5: Keyword Frequency Insight
# ─────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","is",
    "it","its","this","that","was","are","be","been","with","as","by","he",
    "she","they","we","you","i","my","your","our","their","not","so","if",
    "from","have","has","had","do","did","does","will","would","could","should",
    "than","then","when","which","who","what","how","just","even","also","very",
    "more","about","up","out","can","get","got","one","all","no","into","there"
}

def get_top_words(text, n=3):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    freq = Counter(filtered)
    return freq.most_common(n)

# ─────────────────────────────────────────────
# NEW FEATURE 6/7/8: CSV Analysis helpers
# ─────────────────────────────────────────────
def analyze_csv(file_stream):
    content = file_stream.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(content))
    headers = reader.fieldnames or []

    review_col = None
    for candidate in ["reviews.text", "review", "text", "Review", "Text", "comment", "Comment", "description"]:
        if candidate in headers:
            review_col = candidate
            break
    if review_col is None and headers:
        first_rows = []
        temp_reader = csv.DictReader(io.StringIO(content))
        for i, row in enumerate(temp_reader):
            first_rows.append(row)
            if i >= 4:
                break
        if first_rows:
            avg_lens = {col: sum(len(str(r.get(col, ""))) for r in first_rows) for col in headers}
            review_col = max(avg_lens, key=avg_lens.get)

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    all_words = []
    all_aspects = Counter()
    negative_aspects = Counter()
    total = 0
    MAX_ROWS = 500

    reader2 = csv.DictReader(io.StringIO(content))
    for i, row in enumerate(reader2):
        if i >= MAX_ROWS:
            break
        text = str(row.get(review_col, "")).strip()
        if not text or len(text) < 5:
            continue
        cleaned = clean_text(text)
        score = get_sentiment(cleaned)
        sentiment = label_sentiment(cleaned, score)
        total += 1

        if "Positive" in sentiment:
            counts["Positive"] += 1
        elif "Negative" in sentiment:
            counts["Negative"] += 1
        elif "Mixed" in sentiment:
            counts["Mixed"] += 1
        else:
            counts["Neutral"] += 1

        words = re.findall(r'\b[a-z]{4,}\b', cleaned)
        all_words.extend(w for w in words if w not in STOPWORDS)

        for aspect, keywords in ASPECTS.items():
            for kw in keywords:
                if kw in cleaned:
                    all_aspects[aspect] += 1
                    if score < -0.05:
                        negative_aspects[aspect] += 1
                    break

    top_words = Counter(all_words).most_common(5)
    summary = _build_summary(counts, all_aspects, negative_aspects, total)

    return {
        "total": total,
        "counts": counts,
        "top_words": top_words,
        "top_aspects": all_aspects.most_common(3),
        "negative_aspects": negative_aspects.most_common(3),
        "summary": summary,
        "review_col": review_col,
    }

def _build_summary(counts, all_aspects, negative_aspects, total):
    if total == 0:
        return "No valid reviews found in the file."
    dominant = max(counts, key=counts.get)
    pct = int(counts[dominant] / total * 100)
    neg_aspect_keys = {a for a, _ in negative_aspects.most_common(3)}
    liked = [a for a, _ in all_aspects.most_common(2) if a not in neg_aspect_keys]
    disliked = [a for a, _ in negative_aspects.most_common(2)]
    parts = [f"{pct}% of reviews are {dominant.lower()}."]
    if liked:
        parts.append(f"Users appreciate {' and '.join(liked)}.")
    if disliked:
        parts.append(f"Common complaints focus on {' and '.join(disliked)}.")
    return " ".join(parts)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None
    csv_result = None

    if request.method == 'POST':
        # CSV Upload branch
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            csv_file = request.files['csv_file']
            try:
                csv_result = analyze_csv(csv_file.stream)
            except Exception as e:
                error = f"⚠️ Could not parse CSV: {e}"
            return render_template('index.html', result=None, error=error, csv_result=csv_result)

        # Single review branch
        review = request.form.get('review', '')
        if not is_valid_review(review):
            error = "⚠️ Please enter a valid review (not empty or meaningless text)."
            return render_template('index.html', result=None, error=error)

        cleaned = clean_text(review)
        sentiment_score = get_sentiment(cleaned)
        sentiment = label_sentiment(cleaned, sentiment_score)
        keywords = get_keywords(cleaned)
        rating = predict_rating(sentiment_score, sentiment)

        confidence  = get_confidence(sentiment_score)
        aspects     = analyze_aspects(review)
        suggestions = get_suggestions(review)
        quality     = get_review_quality(review)
        top_words   = get_top_words(review)

        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
        if sentiment.startswith("Positive"):
            sentiment_counts["Positive"] += 1
        elif sentiment.startswith("Negative"):
            sentiment_counts["Negative"] += 1
        elif sentiment.startswith("Mixed"):
            sentiment_counts["Mixed"] += 1
        else:
            sentiment_counts["Neutral"] += 1

        result = {
            'review':      review,
            'sentiment':   sentiment,
            'keywords':    keywords,
            'rating':      rating,
            'counts':      sentiment_counts,
            'confidence':  confidence,
            'aspects':     aspects,
            'suggestions': suggestions,
            'quality':     quality,
            'top_words':   top_words,
        }

    return render_template('index.html', result=result, error=error, csv_result=csv_result)

if __name__ == "__main__":
    app.run(debug=True)
