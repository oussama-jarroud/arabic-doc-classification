# Phase 7: Comprehensive Evaluation Suite & Scientific Graphics

**Benchmark Corpus**: Khaleej (SANAD) Dataset — 7 Classes, 45,485 Clean Deduplicated Documents  
**Held-Out Evaluation Split**: `data/processed/light_normalized_test.joblib` (**N = 9,097 articles**)  
**Evaluation Protocol**: Strict Zero-Leakage Inference (No retraining, fitting, or peeking on test split)

---

## 1. Clean Held-Out Test Set Performance Benchmark

The three flagship models from the preceding development phases were evaluated on the unlocked, completely held-out test set of **9,097 articles**:

| Model Architecture | Feature Representation | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Eval Latency (CPU) |
|---|---|---|---|---|---|---|
| **LinearSVC** | `sparse_fusion` (Word 1-2 + Char 3-5) | **0.983621** | **0.983643** | 0.983618 | **0.983624** | 2.41 ms/doc |
| **Stacking Ensemble** | `sparse_fusion` (SVC + LR + MNB Meta) | **0.983621** | 0.983631 | **0.983620** | 0.983623 | 2.39 ms/doc |
| **ArabicCNN_1D** | Learned Word Embeddings ($D=128$) | 0.971969 | 0.972106 | 0.971964 | 0.972010 | 2.25 ms/doc |

---

## 2. Component A: Error Analysis & Model Explainability

### A. Stacking Ensemble Confusion Matrix
The confusion matrix demonstrates near-perfect class separation across all 7 categories:
`graphics/confusion_matrix_stacking.png`

- **Sports (99.5%) & Medical (99.0%)**: Exhibit the highest recall, characterized by highly specialized domain vocabularies.
- **Culture (97.1%) vs. Politics (97.6%)**: Represent the primary source of mutual confusions (34 articles), as cultural articles in the Gulf press frequently mention government sponsorships, state festivals, and diplomatic engagements.
- **Finance (97.9%) vs. Politics (97.6%)**: Exhibit 19 confusions due to coverage of national trade sanctions, ministry budgets, and sovereign wealth policies.

### B. Morphological Feature Interpretability
Extracting the top 15 positive coefficients from `LinearSVC` validates our morphological hypothesis:
`graphics/feature_importance.png`

- **Synergy of Word and Subword Features**: The feature union integrates both multi-word phrases (e.g., `[word] دبي المالي` for Finance, `[word] لكره القدم` for Sports) and subword character n-grams (e.g., `[char] ثقاف` for Culture, `[char] لاعب` for Sports, `[char] يتكس` for Tech / GITEX).
- **Subword Affix Capture**: Subwords capture root patterns (e.g. `ث-ق-ف` in `ثقافة` and `ثقافي`) across prefixations (`لثقاف`) without aggressive root-stemming collisions.

---

## 3. Component B: Rigorous Statistical Significance Testing (McNemar's Test)

We conducted paired $2 \times 2$ McNemar tests on the 9,097 individual document predictions with Edwards' continuity correction and exact binomial tests:

- **Stacking Ensemble vs. ArabicCNN_1D**:
  - Contingency Table: $n_{00} = 8,797, n_{01} = 151, n_{10} = 45, n_{11} = 104$
  - McNemar $\chi^2 = 56.2500, p = 1.3858 \times 10^{-14}$
  - **Conclusion**: Statistically significant ($p < 0.001$).
- **Stacking Ensemble vs. LinearSVC**:
  - Contingency Table: $n_{00} = 8,922, n_{01} = 26, n_{10} = 26, n_{11} = 123$
  - McNemar $\chi^2 = 0.0192, p = 1.0000$
  - **Conclusion**: No statistically significant difference ($p = 1.0000$). `LinearSVC` matches the ensemble with a simpler single-model architecture.

---

## 4. Component C: Typographic Noise Robustness (Hypothesis 5 Validation)

| Corruption Rate | LinearSVC (`sparse_fusion`) | Stacking Ensemble (`sparse_fusion`) | ArabicCNN_1D (`word_embeddings`) |
|---|---|---|---|
| **0.0% (Clean)** | **0.9836** | **0.9836** | 0.9720 |
| **5.0% Noise** | 0.9821 (-0.0015) | **0.9830** (-0.0006) | 0.9681 (-0.0039) |
| **10.0% Noise** | 0.9822 (-0.0014) | **0.9829** (-0.0007) | 0.9642 (-0.0078) |
| **20.0% Noise** | **0.9809** (-0.0027) | 0.9805 (-0.0031) | **0.9392** (-0.0328) |

**Hypothesis 5 is Confirmed**:
Subword character n-grams retain resilience against typo corruption, degrading by $< 0.3\%$ at 20% noise, whereas word-only models drop by $> 3.28\%$.

---

## 5. Component D: Efficiency Benchmarking & Pareto Frontier
`graphics/efficiency_benchmark.png`

- **Fastest Model**: `MultinomialNB` ($0.005$ ms/doc, $0.954$ Macro-F1)
- **Champion Pareto-Optimal**: `LinearSVC (sparse_fusion)` ($0.021$ ms/doc, $0.9836$ Macro-F1)
- **High-Throughput Deep Model**: `ArabicCNN_1D` ($0.166$ ms/doc GPU, $0.9720$ Macro-F1)
