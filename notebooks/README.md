# Notebooks

This folder is for Jupyter notebooks used during experimentation and for
presenting results in a viva/demo.

Suggested notebooks to create here:

- **`01_data_exploration.ipynb`** — explore the US Accidents dataset: severity
  distribution, weather correlations, missing-value analysis.
- **`02_model_tuning.ipynb`** — hyperparameter experiments for XGBoost and
  Random Forest; plot train-vs-test curves to check overfitting.
- **`03_routing_demo.ipynb`** — load the pipeline, run A* + Hill Climbing on
  example routes, and display the folium maps inline.

All the heavy lifting already lives in importable functions under `../src/`, so
a notebook can simply do, for example:

```python
import sys; sys.path.insert(0, "../src")
from preprocess import load_processed
from pipeline import load_graph, load_models, assign_ml_weights, run_route

data = load_processed()
xgb, rf = load_models()
G = load_graph()
assign_ml_weights(G, xgb, rf, data)
result = run_route(G, src_node, dst_node, trace=True)
```
