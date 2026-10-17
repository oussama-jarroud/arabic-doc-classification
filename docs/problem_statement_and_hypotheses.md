# Problem Statement, Research Questions, and Formal Hypotheses

## 1. Problem Statement & Critical Critique of Muaad et al. (2022)

Arabic document classification is a foundational task in Natural Language Processing (NLP), challenged by rich morphological inflections, root-and-pattern derivation, orthographic ambiguities (e.g., variant Alef and Ta Marbuta forms), and prevalent diglossia. The benchmark study by **Muaad et al. (Global Transitions Proceedings 3, 2022)** investigates classical machine learning classifiers (Multinomial Naive Bayes, Linear Support Vector Machines, Random Forest, Logistic Regression) on the **Khaleej** corpus—a 7-class balanced subset of the SANAD benchmark comprising 45,500 news articles (6,500 per class across Culture, Finance, Medical, Politics, Religion, Sports, and Tech).

While Muaad et al. report high classification metrics (up to 96–98% accuracy), rigorous re-examination reveals three fundamental methodological and theoretical flaws:

1. **Mathematical Bug in Metric Definition (Equation 2)**:
   In Section 3.4 (Performance Evaluation), Muaad et al. state Equation (2) as:
   $$\text{Precision} = \frac{TN}{TN + FP}$$
   This formula mathematically defines **Specificity** (True Negative Rate), *not* Precision. True precision is strictly:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   Consequently, all published precision figures in the paper are theoretically invalidated or misattributed, necessitating standardized re-evaluation using verified implementations (`sklearn.metrics`).
  

2. **Absence of Corpus Deduplication & Cross-Split Leakage**:
   The Khaleej corpus was crawled from syndicated Gulf news outlets. Syndicated press releases and international wire stories frequently recur verbatim across archives with differing article IDs. Muaad et al. performed no deduplication check prior to partitioning, allowing identical documents to appear simultaneously in training and evaluation partitions, artificially inflating generalization performance.

3. **Weak Validation Protocol & Aggressive Root Collisions**:
   The authors evaluated models on an arbitrary single 80/20 train/test split without cross-validation, variance bounds, or statistical significance testing. Furthermore, their pipeline relies exclusively on the algorithmic **ISRI root stemmer**, which collapses rich derivational forms to tri-consonantal roots, inducing severe semantic collisions (e.g., conflating distinct vocabulary across technical and general domains into identical root tokens).

---

## 2. Research Questions (RQ1 – RQ4)

- **RQ1 (Morphological Representation)**: Can joint word-level and subword character n-gram TF-IDF representations outperform word-only unigram representations by capturing Arabic inflectional patterns while suppressing out-of-vocabulary (OOV) tokens?
- **RQ2 (Preprocessing Ablation)**: Does light orthographic normalization (unifying Alefs, Ta Marbuta, and Ya) preserve more discriminative lexical signal than aggressive ISRI root stemming?
- **RQ3 (Dimensionality vs. Classifier Family)**: How does dense dimensionality reduction via TruncatedSVD (Latent Semantic Analysis) impact linear models (LinearSVC, Logistic Regression) compared to non-linear tree ensembles (Random Forest, Gradient Boosted Trees)?
- **RQ4 (Ensemble Generalization & Robustness)**: Does out-of-fold probability stacking yield statistically significant improvements over single tuned baselines, and how robust are these models under synthetic Arabic orthographic noise?

---

## 3. Formal Testable Hypotheses (H1 – H5)

- **Hypothesis 1 (Subword N-grams + Word TF-IDF vs. Word-only Unigrams)**:
  *Statement*: Feature spaces combining subword character n-grams ($n \in [3, 5]$) with word-level TF-IDF will achieve a statistically significant higher macro-F1 score on the held-out test set compared to word-only TF-IDF baselines.
  *Rationale*: Arabic templatic morphology embeds roots within predictable affix patterns; character n-grams retain root clues across surface inflections without requiring external linguistic morphological analyzers.

- **Hypothesis 2 (Light Normalization vs. Heavy ISRI Root Stemming)**:
  *Statement*: Light orthographic normalization will outperform heavy ISRI root stemming in macro-F1 across linear and probabilistic classifiers.
  *Rationale*: ISRI root extraction over-stems non-derived and technical words (especially prevalent in Tech and Medical categories), producing spurious homographs that degrade categorical separability.

- **Hypothesis 3 (Full Sparse TF-IDF vs. TruncatedSVD across Classifier Families)**:
  *Statement*: Linear classifiers (LinearSVC, Logistic Regression) will achieve optimal macro-F1 on full high-dimensional sparse TF-IDF spaces, whereas dense TruncatedSVD embeddings ($d \in [100, 300]$) will benefit tree-based ensembles (Random Forest, LightGBM) by alleviating extreme feature sparsity.
  *Rationale*: Linear models excel at finding separating hyperplanes in very high dimensions with low sample density; decision trees struggle with orthogonal axis splits on thousands of near-zero sparse dimensions.

- **Hypothesis 4 (Out-of-Fold Stacking Ensemble vs. Best Single Baseline)**:
  *Statement*: A stacked meta-learner trained on out-of-fold class probability predictions from diverse base models (LinearSVC, MultinomialNB, and Random Forest) will yield superior macro-F1 over any single tuned model.
  *Rationale*: Linear, probabilistic, and tree models exhibit distinct inductive biases and error distributions; stacking leverages their complementary decision boundaries.

- **Hypothesis 5 (Typographic Noise Degradation)**:
  *Statement*: Models trained with subword character n-grams will experience significantly lower relative macro-F1 degradation when subjected to synthetic orthographic noise (omission of Hamza, substitution of $ة \leftrightarrow ه$, and $ى \leftrightarrow ي$) than word-only unigram models.
  *Rationale*: Character n-grams distribute lexical representation across overlapping substrings, ensuring that single-character corruption preserves remaining n-gram activations.

---

## 4. Literature Baseline Comparison Matrix (SANAD / Khaleej)

| Study | Corpus & Domain Scope | Split Protocol & Safeguards | Preprocessing Pipeline | Feature Representation & Models | Reported Metrics | Known Limitations & Flaws |
|---|---|---|---|---|---|---|
| **Muaad et al. (2022)** | Khaleej (7 classes, 45,500 docs, 6.5k/class) | Single 80/20 train/test; **No deduplication**; No CV | Diacritics removal, punctuation removal, stop words, **ISRI root stemming** | Word TF-IDF + Chi-Square selection ($k=1200$); MNB, SVM, RF, LR | Accuracy: 96.7% (SVM), 94.8% (MNB) | **Eq. 2 defines Specificity as Precision**; single split; over-stemming collisions; no leakage control |
| **Elnagar et al. (2020)** | Full SANAD (Al-Arabiya, Al-Khaleej, Akhbar-Al-Yaom; ~194k docs) | 80/10/10 train/val/test; Stratified; Fixed seed | Light normalization, diacritics removal, custom stop words | Word embeddings + Deep Learning (CNN, Bi-LSTM, GRU) | Accuracy: 95.8% (Khaleej subset), 96.9% (Full SANAD) | High computational footprint; deep architectures lack interpretability for resource-constrained deployment |
| **Einea et al. (2019)** | SANAD sample subsets (7 & 10 classes) | 70/15/15 train/val/test | Orthographic normalization, tokenization | Pre-trained Word2Vec (CBOW & Skip-gram) + Shallow CNN / RNN | Macro-F1: 94.2% – 95.5% | Static embeddings sensitive to out-of-vocabulary Arabic dialectal and technical terms |
| **Boukil et al. (2018)** | Multi-source Arabic news corpora (CNC, CNN Arabic) | 10-fold cross-validation | Khoja stemmer vs. Light10 stemmer vs. ISRI | Word TF-IDF + Information Gain; Naive Bayes, SVM, KNN | Accuracy: 91.2% (Khoja), 93.6% (Light10) | Smaller dataset (~10k docs); older stemmers with dictionary dependencies; no subword n-gram analysis |
| **This Benchmark (Ours)** | Khaleej (7 classes, 45,500 docs raw) | **Strict exact-hash deduplication**; Fixed 80% `dev` / 20% held-out `test`; **5-Fold Stratified CV on `dev`** | Composable pure pipelines: raw char, light normalized, heavy ISRI | Word + Char n-gram TF-IDF; Sparse vs. SVD; Linear, Probabilistic, Tree, & Stacking | Rigorous un-fabricated evaluation via `sklearn.metrics` | Strict verification gate; test set isolated until final phase |
