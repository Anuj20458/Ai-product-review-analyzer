# 🧠 AI Product Review Analyzer

A Flask-based web application that uses NLP to analyze product reviews — one at a time or in bulk — delivering sentiment scores, aspect breakdowns, keyword extraction, improvement suggestions, and more.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Sentiment Analysis** | Classifies reviews as Positive 😊, Negative 😡, Neutral 😐, or Mixed ⚖️ using TextBlob |
| **Confidence Score** | Shows how certain the model is about its sentiment prediction (10–99%) |
| **Aspect-Based Analysis** | Breaks down sentiment by product aspects: battery, camera, performance, display, price, design, and software |
| **Keyword Extraction** | Uses RAKE (Rapid Automatic Keyword Extraction) to surface the top phrases from any review |
| **Star Rating Prediction** | Predicts a 1–5 star rating based on sentiment polarity |
| **Improvement Suggestions** | Generates actionable product recommendations from negative feedback keywords |
| **Review Quality Detection** | Scores reviews as High ⭐⭐⭐, Medium ⭐⭐, or Low ⭐ based on word count, vocabulary richness, and meaningful content |
| **Top Word Frequency** | Highlights the most frequently used meaningful words in a review |
| **Bulk CSV Analysis** | Upload a `.csv` file of reviews and get an aggregated report with sentiment distribution, top aspects, and an AI-generated summary |

---

## 🖥️ Demo

**Single Review Mode** — paste any product review and get a full breakdown instantly.

**Bulk CSV Mode** — upload a CSV with a review text column and analyze up to 500 reviews at once, with sentiment counts, top keywords, and aspect insights.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-review-analyzer.git
cd ai-review-analyzer

# Install dependencies
pip install flask textblob rake-nltk nltk pandas

# NLTK corpora are auto-downloaded on first run
```

### Running the App

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`.

---

## 📁 Project Structure

```
.
├── app.py              # Flask application — all routes and NLP logic
├── project.py          # Standalone script for CSV exploration (dev/analysis)
├── reviews.csv         # Sample dataset of product reviews
└── templates/
    └── index.html      # Frontend UI (single page)
```

---

## 📊 CSV Format

For bulk analysis, your CSV should contain a column with review text. The app auto-detects the review column by checking these names in order:

```
reviews.text · review · text · Review · Text · comment · Comment · description
```

If none match, it picks the column with the longest average content. Up to **500 rows** are processed per upload.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **NLP:** TextBlob, RAKE-NLTK, NLTK
- **Data:** pandas, CSV (stdlib)
- **Frontend:** Vanilla HTML/CSS/JS (single template, no framework)

---

## 📦 Dependencies

```
flask
textblob
rake-nltk
nltk
pandas
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)
