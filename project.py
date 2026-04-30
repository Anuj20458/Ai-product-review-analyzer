import pandas as pd
import re
df = pd.read_csv("reviews.csv")

print(df.head())
reviews = df['reviews.text']

print(reviews.head())



def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

reviews = reviews.apply(clean_text)

from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = reviews.apply(get_sentiment)

def label_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['label'] = df['sentiment'].apply(label_sentiment)
print(df[['reviews.text', 'sentiment', 'label']].head())

from rake_nltk import Rake

rake = Rake()

def get_keywords(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:3]

df['keywords'] = reviews.apply(get_keywords)
print(df[['reviews.text', 'label', 'keywords']].head())

print("\nOverall Sentiment Count:")
print(df['label'].value_counts())
# findind the common issues

from collections import Counter

all_keywords = []

for keywords in df['keywords']:
    all_keywords.extend(keywords)

print("\nTop Issues / Features:")
print(Counter(all_keywords).most_common(10))