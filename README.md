# Arabic Document Classification (FastAPI + Scikit-Learn + React)

A full-stack machine learning application for high-accuracy Arabic text classification across 7 domain categories, featuring a **FastAPI** inference engine powered by real trained Scikit-Learn `.joblib` ensemble pipelines and a **React + Tailwind CSS** dashboard.

---

## Architecture Overview

- **Backend:** FastAPI (Python 3.10+) serving trained Scikit-Learn pipelines with in-memory model registry and caching.
- **Frontend:** React 18, Vite, Tailwind CSS, Lucide icons.
- **NLP Preprocessing:** Custom Arabic `light_normalized` pipeline:
  - Tashkeel (diacritics) stripping
  - Tatweel (kashida) removal
  - Hamza and Alef normalization (`أ, إ, آ, ٱ -> ا`, `ى -> ي`, `ؤ -> و`, `ئ -> ي`)
  - Non-Arabic punctuation and numeral filtering
  - Arabic stopwords removal
- **Models:**
  - `StackingEnsemble_sparse_fusion_none_light_normalized.joblib` (FeatureUnion with Word & Char TF-IDF + Stacking of LinearSVC, LogisticRegression, MultinomialNB)
  - `MultinomialNB_tfidf_light_normalized.joblib` (Word TF-IDF + Multinomial Naive Bayes)

---

## 7 Supported Categories

| Arabic | English | Example Context |
| :--- | :--- | :--- |
| **سياسة** | Politics | ديبلوماسية، حكومات، علاقات دولية |
| **اقتصاد** | Finance | أسواق المال، استثمار، تضخم، شركات |
| **رياضة** | Sports | بطولات، مباريات، أندية، منتخبات |
| **صحة** | Medical | طب، أبحاث سريرية، وقاية، تغذية |
| **تكنولوجيا** | Tech | ذكاء اصطناعي، أمن سيبراني، سحابة |
| **دين** | Religion | شريعة، أخلاق، فقه، خطب ومواعظ |
| **ثقافة** | Culture | أدب، معارض كتب، فنون تشكيلية، تراث |

---

## Quick Start Guide

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11, 3.12, or 3.14)
- Node.js 18+ & npm

---

### 1. Backend Setup (FastAPI)

1. Navigate to the project root:
   ```bash
   cd NLP_Project
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health check: [http://localhost:8000/health](http://localhost:8000/health)
   - Available models: [http://localhost:8000/models](http://localhost:8000/models)

---

### 2. Frontend Setup (React + Vite)

1. Open a new terminal tab and enter the `frontend` folder:
   ```bash
   cd NLP_Project/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## API Endpoints

### `GET /models`
Returns all `.joblib` models discovered in `models/` with display names and metadata.

### `POST /predict`
Classifies input Arabic text.

**Request:**
```json
{
  "text": "حقق المنتخب فوزاً مستحقاً في المباراة النهائية وتوج باللقب.",
  "model_name": "StackingEnsemble_sparse_fusion_none_light_normalized.joblib",
  "options": {
    "remove_tashkeel": true,
    "normalize_alef": true,
    "remove_stopwords": true
  }
}
```

**Response:**
```json
{
  "category": "رياضة (Sports)",
  "confidence": 98.45,
  "probabilities": [
    {"category": "رياضة (Sports)", "score": 98.45},
    {"category": "ثقافة (Culture)", "score": 0.82},
    {"category": "سياسة (Politics)", "score": 0.41},
    {"category": "صحة (Medical)", "score": 0.32}
  ],
  "keywords": ["المنتخب", "المباراة", "النهائية", "باللقب"],
  "latency_ms": 14.2,
  "model_used": "StackingEnsemble_sparse_fusion_none_light_normalized.joblib"
}
```

---

## Project Structure

```text
NLP_Project/
├── backend/
│   └── main.py              # FastAPI server, preprocessing & model registry
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # React dashboard with live model selector
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── models/                  # Trained .joblib models
│   ├── StackingEnsemble_sparse_fusion_none_light_normalized.joblib
│   └── MultinomialNB_tfidf_light_normalized.joblib
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Ignored files (venv, node_modules, cache)
```
