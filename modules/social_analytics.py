"""
==============================================================================
Module:      social_analytics.py
Description: Social network text mining: Twitter/X cursor harvesting, regex
             sanitization, TextBlob sentiment, TF-IDF, Gensim LDA, and mention graphs.
==============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional


def clean_tweet_text(text: str) -> str:
    """Bereinigt Rohtexte von Hyperlinks, Mentions, Hashtags und Satzzeichen."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()


def evaluate_sentiment_polarity(text: str) -> str:
    """Klassifiziert die sentimentale Polaritaet (positiv, negativ, neutral)."""
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:
        return "positiv"
    elif score < -0.05:
        return "negativ"
    return "neutral"


def extract_top_tfidf_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Ermittelt die signifikantesten Terme basierend auf TF-IDF-Mittelwerten."""
    vectorizer = TfidfVectorizer(max_features=1000)
    matrix = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
    sorted_keywords = sorted(zip(names, mean_scores), key=lambda x: x[1], reverse=True)
    return [(w, float(s)) for w, s in sorted_keywords[:top_n]]


def extract_lda_topics(texts: List[str], num_topics: int = 3, num_words: int = 5):
    """Fuehrt eine Latent Dirichlet Allocation (LDA) Topic-Modellierung durch."""
    tokenized = [t.split() for t in texts if len(t.split()) > 0]
    if not tokenized:
        return []
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    return lda.print_topics(num_words=num_words)


def build_mention_network(tweet_records: List[Dict[str, Any]]) -> nx.Graph:
    """Konstruiert ein ungerichtetes Interaktionsnetzwerk aus Erwaehnungen."""
    g = nx.Graph()
    for record in tweet_records:
        sender = record.get("user")
        if sender:
            g.add_node(sender)
            for target in record.get("mentions", []):
                g.add_edge(sender, target)
    return g
