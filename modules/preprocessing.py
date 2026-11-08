"""
==============================================================================
Module:      preprocessing.py
Description: Text normalization pipeline comparing Porter/Snowball stemming,
             NLTK WordNet vs. SpaCy lemmatization, and frequency distributions.
==============================================================================
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
import spacy
from typing import List, Tuple, Dict, Any


def ensure_nltk_resources():
    """Stellt sicher, dass alle erforderlichen NLTK-Datensätze lokal vorliegen."""
    resources = ['punkt', 'wordnet', 'omw-1.4', 'stopwords']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}' if r == 'punkt' else f'corpora/{r}')
        except LookupError:
            nltk.download(r, quiet=True)


class TextNormalizer:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        ensure_nltk_resources()
        self.porter = PorterStemmer()
        self.snowball = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_en = set(stopwords.words("english"))
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = None

    def process_text(self, text: str) -> Dict[str, Any]:
        """Fuehrt eine vollstaendige vergleichende Normalisierung durch."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        porter_stems = [self.porter.stem(w) for w in words]
        snowball_stems = [self.snowball.stem(w) for w in words]
        nltk_lemmas = [self.lemmatizer.lemmatize(w) for w in words]

        spacy_lemmas = []
        if self.nlp:
            doc = self.nlp(text)
            spacy_lemmas = [token.lemma_ for token in doc if not token.is_space]

        # Filterung ohne Stoppwoerter
        words_no_stop = [w for w in words if w.lower() not in self.stopwords_en and w.isalnum()]
        filtered_porter = [self.porter.stem(w) for w in words_no_stop]
        filtered_lemmas = [self.lemmatizer.lemmatize(w) for w in words_no_stop]

        return {
            "sentence_count": len(sentences),
            "word_count": len(words),
            "words": words,
            "porter_stems": porter_stems,
            "snowball_stems": snowball_stems,
            "nltk_lemmas": nltk_lemmas,
            "spacy_lemmas": spacy_lemmas,
            "freq_raw": FreqDist(words).most_common(5),
            "freq_filtered": FreqDist(filtered_lemmas).most_common(5)
        }
