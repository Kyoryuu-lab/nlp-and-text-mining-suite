"""
==============================================================================
Module:      pos_and_syntax.py
Description: Part-of-Speech (POS) classification, SpaCy dependency parsing,
             and Augmented Transition Network (ATN) state-machine modeling.
==============================================================================
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from typing import List, Tuple, Dict, Optional


def analyze_pos_tags(text: str) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]], FreqDist]:
    """Extrahiert POS-Tags und filtert nach linguistischen Kategorien."""
    words = word_tokenize(text)
    try:
        pos_tags = nltk.pos_tag(words)
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        pos_tags = nltk.pos_tag(words)

    categories = {
        "Nouns": [w for w, pos in pos_tags if pos.startswith('NN')],
        "Verbs": [w for w, pos in pos_tags if pos.startswith('VB')],
        "Adjectives": [w for w, pos in pos_tags if pos.startswith('JJ')]
    }
    tag_freq = FreqDist(tag for _, tag in pos_tags)
    return pos_tags, categories, tag_freq


def parse_dependencies_spacy(sentence: str, model_name: str = "uk_core_news_lg"):
    """Fuehrt eine Dependenzanalyse mit SpaCy durch und liefert syntaktische Wurzeln."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        return None

    doc = nlp(sentence)
    tree_data = []
    roots = []

    for sent in doc.sents:
        root_token = [token for token in sent if token.head == token]
        if root_token:
            roots.append(root_token[0].text)

    for token in doc:
        tree_data.append((token.text, token.dep_, token.head.text))

    return {"tokens": tree_data, "roots": roots}


class TransitionNetwork:
    """Implementiert ein gerichtetes Uebergangsnetzwerk (ATN) zur Grammatikmodellierung."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_transition(self, from_state: str, to_state: str, label: str):
        self.graph.add_edge(from_state, to_state, label=label)

    def render(self, save_path: Optional[str] = None):
        pos = nx.spring_layout(self.graph, seed=42)
        edge_labels = nx.get_edge_attributes(self.graph, 'label')

        plt.figure(figsize=(9, 5))
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue",
                node_size=2000, font_size=10, font_weight="bold", edge_color="gray")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=9)
        plt.title("Augmented Transition Network (ATN) Grammatikmodell")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
