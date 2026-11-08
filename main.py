"""
==============================================================================
Project:     NLP & Text Mining Intelligence Suite
File:        main.py
Author:      Yevhen Biezhyn
Description: Unified CLI orchestrator executing text normalization, syntax
             and POS tagging, sentiment analysis, LDA topic modeling,
             extractive summarization, and ROUGE benchmarking.
==============================================================================
"""

import argparse
from modules.preprocessing import TextNormalizer
from modules.pos_and_syntax import analyze_pos_tags, parse_dependencies_spacy, TransitionNetwork
from modules.social_analytics import (
    clean_tweet_text,
    evaluate_sentiment_polarity,
    extract_top_tfidf_keywords,
    extract_lda_topics,
    build_mention_network
)
from modules.summarizer_engine import generate_extractive_summary, evaluate_with_rouge


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def run_preprocessing_demo():
    print_banner("1. Modul: Morphologische & Lexikalische Normalisierung")
    sample_text = (
        "The technological advances in Natural Language Processing are transforming "
        "how computers understand human languages. Researchers developed powerful neural architectures."
    )
    normalizer = TextNormalizer()
    res = normalizer.process_text(sample_text)

    print(f"[*] Analyseergebnis: {res['sentence_count']} Saetze, {res['word_count']} Tokens")
    print(f"[*] Porter Stemming (Auszug)  : {res['porter_stems'][:6]}")
    print(f"[*] Snowball Stemming (Auszug): {res['snowball_stems'][:6]}")
    print(f"[*] WordNet Lemmatisierung    : {res['nltk_lemmas'][:6]}")
    print(f"[*] Top-Hauefigkeiten (bereinigt): {res['freq_filtered']}")


def run_syntax_demo():
    print_banner("2. Modul: Syntaktische Analyse & Grammatik-Modellierung")
    sample = "Students write comprehensive reports and professors examine them carefully."
    tags, categories, freq = analyze_pos_tags(sample)

    print("[*] Linguistische Kategorienfilterung:")
    for cat_name, items in categories.items():
        print(f"    - {cat_name:<12}: {', '.join(items)}")

    print("\n[*] Initialisiere Augmented Transition Network (ATN)...")
    atn = TransitionNetwork()
    transitions = [
        ("S", "NP1", "Student"), ("NP1", "ADV", "quickly"),
        ("ADV", "V", "wrote"),   ("V", "NP2", "complex"),
        ("NP2", "NP3", "code"),  ("NP3", "PP", "for reviewer")
    ]
    for src, dst, lbl in transitions:
        atn.add_transition(src, dst, lbl)
    atn.render(save_path="atn_network_graph.png")
    print("[+] ATN-Zustandsgraph exportiert nach: atn_network_graph.png")


def run_social_analytics_demo():
    print_banner("3. Modul: Text Mining, Sentiment & Topic Modeling")
    corpus = [
        "Great breakthrough in artificial intelligence and deep neural networks!",
        "Poor performance and critical latency issues observed in production server.",
        "Neutral discussion about modern data processing pipelines and algorithms.",
        "Artificial intelligence will shape future software development ecosystems."
    ]
    cleaned = [clean_tweet_text(t) for t in corpus]

    print("[*] Sentiment-Polaritaetsbewertung:")
    for orig, c in zip(corpus, cleaned):
        print(f"    - \"{orig[:45]}...\" -> {evaluate_sentiment_polarity(c).upper()}")

    print("\n[*] TF-IDF Schluesselbegriffe:")
    keywords = extract_top_tfidf_keywords(cleaned, top_n=5)
    for w, score in keywords:
        print(f"    - {w:<20}: Score = {score:.4f}")

    print("\n[*] LDA Themenextraktion:")
    topics = extract_lda_topics(cleaned, num_topics=2, num_words=3)
    for tid, words in topics:
        print(f"    Thema #{tid}: {words}")


def run_summarization_demo():
    print_banner("4. Modul: Extraktive Zusammenfassung & ROUGE-Validierung")
    article = (
        "Artificial intelligence and automated summarization have seen rapid adoption across media agencies. "
        "Modern systems employ semantic decomposition methods like Latent Semantic Analysis and graph-based models. "
        "These extractive approaches select central sentences directly from source texts without hallucination risk. "
        "Evaluations rely on standard overlap metrics such as ROUGE-1, ROUGE-2, and ROUGE-L to guarantee fidelity. "
        "Consequently, editorial workflows achieve significantly reduced curation latency."
    )
    manual_ref = (
        "Automated summarization systems employ LSA and graph models. "
        "Extractive algorithms select central sentences to avoid hallucination and are verified via ROUGE metrics."
    )

    sum_lsa = generate_extractive_summary(article, method="lsa", sentences_count=2)
    sum_tr = generate_extractive_summary(article, method="textrank", sentences_count=2)

    print(f"[*] Sumy LSA Zusammenfassung:\n    \"{sum_lsa}\"")
    print(f"\n[*] Sumy TextRank Zusammenfassung:\n    \"{sum_tr}\"")

    rouge_scores = evaluate_with_rouge(manual_ref, sum_lsa)
    print("\n[*] ROUGE-Evaluation gegen Referenztext (LSA):")
    for metric, values in rouge_scores.items():
        print(f"    - {metric.upper()}: Recall={values['r']:.4f}, Precision={values['p']:.4f}, F1={values['f']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="NLP & Text Mining Intelligence Suite CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--module",
        choices=["all", "preprocessing", "syntax", "social", "summarize", "web"],
        default="all",
        help=(
            "all           : Fuehrt alle Analyse-Pipelines aus (Standard)\n"
            "preprocessing : Tokenisierung, Stemming & Lemmatisierung\n"
            "syntax        : POS-Tagging & ATN-Uebergangsgraphen\n"
            "social        : Sentiment, TF-IDF & LDA Topic Modeling\n"
            "summarize     : LSA / TextRank Zusammenfassung & ROUGE\n"
            "web           : Startet die Flask-Webanwendung auf Port 5000"
        )
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  NATURAL LANGUAGE PROCESSING & TEXT MINING SUITE")
    print("=" * 78)

    if args.module == "web":
        from web_app.app import app
        print("[*] Starte Flask Webanwendung unter http://127.0.0.1:5000...")
        app.run(debug=True, port=5000)
        return

    dispatch = {
        "preprocessing": run_preprocessing_demo,
        "syntax": run_syntax_demo,
        "social": run_social_analytics_demo,
        "summarize": run_summarization_demo
    }

    if args.module == "all":
        for fn in dispatch.values():
            fn()
    else:
        dispatch[args.module]()

    print("\n" + "=" * 78)
    print("  VERARBEITUNG ERFOLGREICH ABGESCHLOSSEN.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
