"""
==============================================================================
Module:      web_app/app.py
Description: Flask web application providing a browser interface for text
             summarization using Sumy (LSA / TextRank).
==============================================================================
"""

import os
import sys

# Erlaubt Zugriff auf das Hauptmodul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, render_template
from modules.summarizer_engine import generate_extractive_summary

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    original_text = ""
    selected_method = "lsa"

    if request.method == "POST":
        original_text = request.form.get("text", "")
        selected_method = request.form.get("method", "lsa")
        num_sentences = int(request.form.get("sentences", 3))

        if original_text.strip():
            summary = generate_extractive_summary(
                text=original_text,
                method=selected_method,
                sentences_count=num_sentences,
                language="english"
            )

    return render_template("index.html", summary=summary, original_text=original_text, method=selected_method)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
