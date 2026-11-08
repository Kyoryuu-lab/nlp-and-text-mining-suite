"""
==============================================================================
Module:      summarizer_engine.py
Description: Web scraping (BeautifulSoup), Extractive Summarization (Sumy LSA,
             TextRank), and quantitative ROUGE evaluation.
==============================================================================
"""

import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from rouge import Rouge
from typing import Dict, Any, Optional


def scrape_article_text(url: str, timeout: int = 15) -> Optional[str]:
    """Extrahiert saemtliche Absaetze (<p>) einer Web-Nachrichtenseite."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 30]
            return " ".join(paragraphs)
    except Exception as e:
        print(f"[!] Fehler beim Scraping von {url}: {e}")
    return None


def generate_extractive_summary(text: str, method: str = "lsa", sentences_count: int = 3, language: str = "english") -> str:
    """Erstellt ein automatisches Referat mittels LSA oder TextRank."""
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    if method.lower() == "textrank":
        summarizer = TextRankSummarizer()
    else:
        summarizer = LsaSummarizer()

    summary_sentences = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary_sentences)


def evaluate_with_rouge(reference_summary: str, generated_summary: str) -> Dict[str, Any]:
    """Berechnet ROUGE-1, ROUGE-2 und ROUGE-L Scores."""
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores[0] if scores else {}
