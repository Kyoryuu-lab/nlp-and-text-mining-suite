<h1>nlp-and-text-mining-suite</h1>
<p>Comprehensive Natural Language Processing (NLP) and Text Mining suite in Python: morphological normalization, syntactic POS tagging, SpaCy dependency trees, Twitter mining, LDA topic modeling, extractive summarization (Sumy LSA/TextRank), ROUGE evaluation, and Flask deployment</p>

<h2>Project description</h2>
<p>This repository provides a modular, production-grade computational linguistics framework spanning lexical normalization, grammar modeling, social media text mining, and automated text summarization:</p>
<ol>
  <li>
    <dl>
      <dt><b>Morphological Processing &amp; Lexical Normalization:</b></dt>
      <dd>Implements sentence and word tokenization via NLTK.</dd>
      <dd>Benchmarks Porter and Snowball stemmers against WordNet and SpaCy (<code>en_core_web_sm</code>) lemmatizers.</dd>
      <dd>Applies frequency distribution profiling (<code>FreqDist</code>) and stopword attenuation.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Syntactic Parsing &amp; Formal Transition Networks:</b></dt>
      <dd>Extracts grammatical categories (nouns, verbs, adjectives) via Perceptron Part-of-Speech (POS) tagging.</dd>
      <dd>Parses Ukrainian dependency trees and syntactic heads using SpaCy (<code>uk_core_news_lg</code>).</dd>
      <dd>Constructs formal grammar state machines using Augmented Transition Networks (ATN) with NetworkX.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Social Analytics &amp; Unsupervised Topic Discovery:</b></dt>
      <dd>Extracts and cleans social text feeds via Tweepy cursors and regular expressions.</dd>
      <dd>Computes polarity sentiment scores using TextBlob and extracts dominant vocabulary through TF-IDF vectorization.</dd>
      <dd>Discovers latent document topics via Latent Dirichlet Allocation (LDA) using Gensim.</dd>
      <dd>Constructs topological user mention graphs with NetworkX and extracts influencer centralities.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Extractive Summarization, ROUGE Benchmarking &amp; Web Service:</b></dt>
      <dd>Scrapes online press articles using BeautifulSoup.</dd>
      <dd>Generates concise extractive summaries comparing Latent Semantic Analysis (LSA) and graph-based TextRank.</dd>
      <dd>Validates summary quality against reference baselines using ROUGE-1, ROUGE-2, and ROUGE-L metrics.</dd>
      <dd>Deploys summarization capabilities through an interactive Flask web application.</dd>
    </dl>
  </li>
</ol>

<h2>Project files</h2>
<table>
  <tr>
    <th colspan="2">nlp-and-text-mining-suite</th>
  </tr>
  <tr>
    <td>main.py</td>
    <td>Central CLI orchestrator providing execution routing across all operational sub-modules</td>
  </tr>
  <tr>
    <td>modules/preprocessing.py</td>
    <td>Tokenization, Porter/Snowball stemming, NLTK/SpaCy lemmatization, and frequency analysis</td>
  </tr>
  <tr>
    <td>modules/pos_and_syntax.py</td>
    <td>Part-of-Speech tagging, SpaCy dependency parsing, and ATN grammar state machines</td>
  </tr>
  <tr>
    <td>modules/social_analytics.py</td>
    <td>Twitter extraction, regex cleaning, TextBlob sentiment, TF-IDF, and Gensim LDA topic modeling</td>
  </tr>
  <tr>
    <td>modules/summarizer_engine.py</td>
    <td>BeautifulSoup article scraping, Sumy LSA/TextRank summarization, and ROUGE metric evaluations</td>
  </tr>
  <tr>
    <td>web_app/app.py</td>
    <td>Interactive Flask web application exposing real-time text summarization</td>
  </tr>
  <tr>
    <td>.env.example</td>
    <td>Configuration template for secure storage of Twitter/X API credentials</td>
  </tr>
  <tr>
    <td>README.md</td>
    <td>Project documentation and technical specification</td>
  </tr>
  <tr>
    <td>.gitignore</td>
    <td>Exclusion rules for bytecode, environment secrets, and plot outputs</td>
  </tr>
</table>

<h2>Methodological Formulation</h2>
<table>
  <tr>
    <th>Task</th>
    <th>Algorithm / Metric</th>
    <th>Analytical Model</th>
  </tr>
  <tr>
    <td><b>Term Importance</b></td>
    <td>TF-IDF Score</td>
    <td>$\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$</td>
  </tr>
  <tr>
    <td><b>Topic Modeling</b></td>
    <td>Latent Dirichlet Allocation</td>
    <td>$P(W, Z, \theta, \varphi | \alpha, \beta) = \prod_{i=1}^K P(\varphi_i | \beta) \prod_{j=1}^M P(\theta_j | \alpha) \prod_{t=1}^N P(Z_{j,t} | \theta_j) P(W_{j,t} | \varphi_{Z_{j,t}})$</td>
  </tr>
  <tr>
    <td><b>Summarization</b></td>
    <td>TextRank Random Walk</td>
    <td>$WS(V_i) = (1 - d) + d \sum_{V_j \in In(V_i)} \frac{w_{ji}}{\sum_{V_k \in Out(V_j)} w_{jk}} WS(V_j)$</td>
  </tr>
  <tr>
    <td><b>Evaluation</b></td>
    <td>ROUGE-N Overlap</td>
    <td>$\text{ROUGE-N} = \frac{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Ref}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$</td>
  </tr>
</table>

<h2>Installation and launch</h2>
<ol>
  <b>
    <li>
      <dl>
        <dt>Clone repository</dt>
        <dd><code>git clone https://github.com/Kyoryuu-lab/nlp-and-text-mining-suite.git</code></dd>
        <dd><code>cd nlp-and-text-mining-suite</code></dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Install dependencies &amp; linguistic models</dt>
        <dd><code>pip install nltk spacy textblob scikit-learn gensim sumy rouge networkx beautifulsoup4 requests flask</code></dd>
        <dd><code>python -m spacy download en_core_web_sm</code></dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Run complete CLI analytical suite</dt>
        <dd><code>python main.py --module all</code></dd>
        <dd>Execute individual modules: <code>python main.py --module summarize</code> (Options: <code>preprocessing</code>, <code>syntax</code>, <code>social</code>, <code>summarize</code>)</dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Launch interactive Web UI</dt>
        <dd><code>python main.py --module web</code></dd>
        <dd>Navigate to: <code>http://127.0.0.1:5000</code></dd>
      </dl>
    </li>
  </b>
</ol>

<h2>Example of console output (python main.py --module all)</h2>
<samp>
  <p>
    ==============================================================================<br />
    &nbsp;&nbsp;1. MODUL: MORPHOLOGISCHE &amp; LEXIKALISCHE NORMALISIERUNG<br />
    ==============================================================================<br />
    [*] Analyseergebnis: 2 Saetze, 22 Tokens<br />
    [*] Porter Stemming (Auszug)  : ['the', 'technolog', 'advanc', 'in', 'natur', 'languag']<br />
    [*] WordNet Lemmatisierung    : ['The', 'technological', 'advance', 'in', 'Natural', 'Language']<br />
    [*] Top-Hauefigkeiten (bereinigt): [('language', 2), ('technological', 1), ('advance', 1)]<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;2. MODUL: SYNTAKTISCHE ANALYSE &amp; GRAMMATIK-MODELLIERUNG<br />
    ==============================================================================<br />
    [*] Linguistische Kategorienfilterung:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Nouns       : Students, reports, professors<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Verbs       : write, examine<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- Adjectives  : comprehensive<br />
    [+] ATN-Zustandsgraph exportiert nach: atn_network_graph.png<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;3. MODUL: TEXT MINING, SENTIMENT &amp; TOPIC MODELING<br />
    ==============================================================================<br />
    [*] Sentiment-Polaritaetsbewertung:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- "Great breakthrough in artificial intelligence..." -&gt; POSITIV<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- "Poor performance and critical latency issues..."   -&gt; NEGATIV<br />
    [*] TF-IDF Schluesselbegriffe: intelligence, neural, artificial, development, pipeline<br />
    [*] LDA Themenextraktion:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Thema #0: 0.082*"intelligence" + 0.054*"artificial" + 0.041*"neural<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;4. MODUL: EXTRAKTIVE ZUSAMMENFASSUNG &amp; ROUGE-VALIDIERUNG<br />
    ==============================================================================<br />
    [*] Sumy LSA Zusammenfassung:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;"Artificial intelligence and automated summarization have seen rapid adoption across media agencies. Extractive approaches select central sentences directly from source texts without hallucination risk.<br />
    [*] ROUGE-Evaluation gegen Referenztext (LSA):<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- ROUGE-1: Recall=0.7857, Precision=0.7333, F1=0.7586<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- ROUGE-2: Recall=0.5833, Precision=0.5385, F1=0.5600<br />
    &nbsp;&nbsp;&nbsp;&nbsp;- ROUGE-L: Recall=0.7500, Precision=0.7000, F1=0.7241<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;VERARBEITUNG ERFOLGREICH ABGESCHLOSSEN.<br />
    ==============================================================================
  </p>
