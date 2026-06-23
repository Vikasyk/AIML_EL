# Smart Delivery & Traffic Management System

An **explainable, risk-aware delivery route optimisation system** that combines
classical AI search algorithms with machine learning. Unlike Google Maps (a
black box), every routing decision here is traceable, visualisable, and
explainable — the core learning objective of an undergraduate AI/ML course.

| | |
|---|---|
| **Domain** | Logistics / Smart City |
| **AI algorithms** | A\* Search · Hill Climbing (both from scratch) |
| **ML models** | XGBoost (delay regression) · Random Forest (risk classification) |
| **Data** | OpenStreetMap (Bengaluru road graph) + US Accidents Kaggle dataset |
| **No NLP / CNN** | Pure classical AI search + tabular ML |

---

## How it works (two-layer pipeline)

```
Layer 1 — ML Prediction              Layer 2 — AI Search & Optimisation
-----------------------              ---------------------------------
XGBoost  -> delay_score (0..1)       A* Search    -> globally optimal path
Random   -> risk_level  (L/M/H)      Hill Climbing-> local refinement of A*
Forest

Final edge weight = length_km + (delay_score * 0.5) + (risk_penalty * 0.3)
                    where risk_penalty: Low=0.0, Medium=0.5, High=1.0
```

End-to-end flow: OSM graph → per-edge features → XGBoost delay + RF risk →
edge weights → A* route → Hill Climbing refinement → folium map + metrics.

---

## Project structure

```
.
├── data/                 # datasets (US Accidents CSV, Bengaluru OSM GraphML)
├── models/               # saved ML models (.pkl)
├── src/
│   ├── preprocess.py     # data cleaning + feature engineering (synthetic fallback)
│   ├── train_xgboost.py  # train + save XGBoost delay regressor (+ SHAP, baseline)
│   ├── train_rf.py       # train + save Random Forest risk classifier (+ baseline)
│   ├── astar.py          # A* search from scratch (+ Dijkstra + hand-crafted test)
│   ├── hill_climb.py     # Hill Climbing local search post-processor
│   ├── pipeline.py       # full end-to-end integration
│   ├── dynamic_reroute.py# innovation: live disruption + automatic re-route
│   └── multi_stop.py     # innovation: multi-stop delivery ordering (TSP heuristic)
├── notebooks/            # Jupyter notebooks for experiments
├── outputs/              # saved maps, charts, results
├── server.py             # FastAPI JSON API wrapping src/ (backend for the React UI)
├── frontend/             # React + Vite + Tailwind single-page app (primary UI)
│   └── src/              #   api.js · state.jsx · components/ · tabs/ (4 tabs)
├── app.py                # Streamlit dashboard (legacy UI — still works)
├── requirements.txt
└── README.md
```

Every script in `src/` is **independently runnable** (each has a `__main__`
demo block); `pipeline.py` ties them all together.

---

## Setup

> This project targets **Python 3.10+**. On this machine use the `py -3.10`
> launcher (Python 3.9 is also present but is below the supported version).

```bash
# 1. Create and activate a virtual environment
py -3.10 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # Linux / Mac

# 2. Install dependencies
python -m pip install -r requirements.txt
```

---

## Downloading the datasets

The repository ships **without** the large datasets. The scripts will fall back
to a clearly-flagged **synthetic sample** so you can run everything immediately,
but for real results download:

**1. US Accidents (2016–2023)** — for ML training only
- Source: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- Save the CSV to `data/US_Accidents.csv`
- *Note:* this US dataset is used purely to train the delay/risk ML models,
  because region-specific (Bengaluru) accident data is not publicly available.
  The actual route graph is the real OSM Bengaluru network.

**2. Bengaluru road graph** — for routing
```python
import osmnx as ox
G = ox.graph_from_place("Bengaluru, India", network_type="drive")
ox.save_graphml(G, "data/bengaluru.graphml")
```
(Or just run `python src/pipeline.py` — it downloads and caches the graph on
first run if `data/bengaluru.graphml` is missing.)

---

## Running

```bash
# 1. Preprocess data (writes data/processed.pkl)
python src/preprocess.py

# 2. Train both ML models (writes models/*.pkl and outputs/*.png)
python src/train_xgboost.py
python src/train_rf.py

# 3. Validate AI algorithms on a hand-crafted test graph
python src/astar.py
python src/hill_climb.py

# 4. Run the full end-to-end pipeline (writes outputs/route_map.html)
python src/pipeline.py

# 5. Innovation demos
python src/dynamic_reroute.py
python src/multi_stop.py

# 6. Launch the interactive dashboard
#
#    Primary UI — React frontend (two processes):
#
#    Terminal A — FastAPI backend (loads graph + models, serves /api):
python -m uvicorn server:app --reload --port 8000

#    Terminal B — React dev server (Vite proxies /api -> :8000):
cd frontend
npm install        # first run only
npm run dev        # open http://localhost:5173

#    Legacy UI — the original Streamlit dashboard still works:
streamlit run app.py
```

### Architecture (React UI)

```
React (Vite, :5173) ──HTTP /api──▶ FastAPI (server.py, :8000) ──imports──▶ src/ (unchanged)
   react-leaflet maps                 graph + models loaded once             A* / TSP / models
```

The React app reproduces all four Streamlit tabs (My orders · Dynamic re-route ·
Advanced raw route tools · ML explainability). Maps are drawn natively with
**react-leaflet** — the backend returns route **coordinates as JSON** instead of
folium HTML. None of the algorithm code in `src/` changed; `server.py` simply
wraps it. The backend runs on the same Python that has the project deps
(Python 3.9 here); Node 18+ is needed for the frontend.

---

## Expected results

| Output | Expected value |
|---|---|
| XGBoost RMSE | 0.10 – 0.18 (normalised 0–1 delay) |
| XGBoost R² | 0.75 – 0.88 |
| Random Forest accuracy | 80% – 88% |
| Random Forest weighted F1 | 0.78 – 0.86 |
| A\* vs Dijkstra | A\* expands 30–60% fewer nodes |
| Hill Climbing | refines a sub-optimal route toward the optimum (15–27% in tests) |
| Final output | `outputs/route_map.html` — interactive colour-coded map |

### Verified run (synthetic fallback data)

The project was validated end-to-end on the bundled synthetic sample
(no Kaggle download required). Measured results:

| Metric | Result | Target |
|---|---|---|
| XGBoost R² | **0.90** | > 0.75 |
| XGBoost RMSE | **0.10** | 0.10–0.18 |
| Random Forest accuracy | **0.87** | > 0.80 |
| Random Forest weighted F1 | **0.87** | 0.78–0.86 |
| A\* vs Dijkstra (node expansions) | **6 vs 30** (≈80% fewer) | 30–60% fewer |
| ML routing | **avoided 100% of high-risk edges** vs the naive distance route | — |

> Real Kaggle data will shift these numbers but the pipeline is identical.

### A note on Python version

The recommended interpreter is **Python 3.10+** (use `py -3.10` on Windows).
The code also runs on Python 3.9 — it was developed and tested there with the
core stack (numpy / pandas / scikit-learn / xgboost / networkx / folium). The
optional `osmnx`, `shap`, `seaborn` and `streamlit` packages are detected at
runtime and the scripts degrade gracefully if any are missing (synthetic city
graph instead of OSM, skipped SHAP plot, matplotlib confusion matrix, etc.).

---

## Why this is different from Google Maps

Google Maps is a closed-source commercial product with no explainability. This
project is a **transparent proof-of-concept**: A\* and Hill Climbing are coded,
traced, and explained from first principles; ML predictions are backed by SHAP
values, confusion matrices, and a step-by-step A\* node-expansion trace. Every
decision is traceable, visualisable, and presentable to an evaluator.
