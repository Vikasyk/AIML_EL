"""
pipeline.py
===========
Full end-to-end integration of the Smart Delivery & Traffic Management System.

It ties together all four components (2 ML + 2 AI):

    1. Load the Bengaluru road graph (OSM via osmnx, cached as GraphML).
       Falls back to a synthetic grid "city" if osmnx / the graph file is
       unavailable, so the pipeline always runs.
    2. Load the trained XGBoost (delay) and Random Forest (risk) models.
    3. For every edge: build a feature vector, predict delay_score + risk_level,
       and compute the routing weight
            weight = length_km + (delay_score * 0.5) + (risk_penalty * 0.3)
       with risk_penalty  Low=0.0 / Medium=0.5 / High=1.0.
    4. Convert GPS coords -> nearest graph nodes.
    5. Run A* and Hill Climbing; compare three route variants
       (shortest-distance / ML-weighted A* / ML-weighted A* + Hill Climbing).
    6. Draw the routes on an interactive folium map (A* blue, Hill Climbing green)
       and print a comparison table over several source-destination pairs.

Run:  python src/pipeline.py
"""

from __future__ import annotations

import os
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd

from astar import astar, dijkstra, haversine_km
from hill_climb import hill_climb, route_cost
from preprocess import RISK_NAMES, load_processed

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRAPHML_PATH = os.path.join(DATA_DIR, "bengaluru.graphml")

PLACE = "Bengaluru, India"
RISK_PENALTY = {0: 0.0, 1: 0.5, 2: 1.0}      # Low / Medium / High
RISK_COLOR = {0: "green", 1: "orange", 2: "red"}

# Hazard weighting per OSM road class — busier roads are nudged toward worse
# conditions so the (weather-trained) models produce spatial delay/risk
# variation across the graph. This is the documented "bridge" standing in for
# region-specific per-edge accident data, which is not publicly available.
HIGHWAY_HAZARD = {
    "motorway": 0.85, "motorway_link": 0.80, "trunk": 0.80, "trunk_link": 0.75,
    "primary": 0.70, "primary_link": 0.65, "secondary": 0.55, "secondary_link": 0.5,
    "tertiary": 0.40, "tertiary_link": 0.35, "residential": 0.20,
    "unclassified": 0.30, "living_street": 0.10, "service": 0.15,
}


# --------------------------------------------------------------------------- #
# 1. Graph loading
# --------------------------------------------------------------------------- #
def load_graph(place: str = PLACE, graphml_path: str = GRAPHML_PATH) -> nx.MultiDiGraph:
    """Load the OSM road graph; download + cache it, or build a synthetic one.

    Order of preference:
      1. Cached GraphML at ``graphml_path`` (via osmnx if available, else nx).
      2. Download from OSM with osmnx and cache as GraphML.
      3. Synthetic grid "city" (fully offline, no extra dependencies).
    """
    # Try osmnx (the real deal).
    try:
        import osmnx as ox

        if os.path.exists(graphml_path):
            print(f"[pipeline] Loading cached OSM graph -> {graphml_path}")
            return ox.load_graphml(graphml_path)
        print(f"[pipeline] Downloading OSM graph for '{place}' (first run, slow)...")
        G = ox.graph_from_place(place, network_type="drive")
        os.makedirs(DATA_DIR, exist_ok=True)
        ox.save_graphml(G, graphml_path)
        print(f"[pipeline] Saved -> {graphml_path}")
        return G
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 (download can fail offline)
        print(f"[pipeline] osmnx graph unavailable ({exc}).")

    # Plain GraphML load without osmnx.
    if os.path.exists(graphml_path):
        print(f"[pipeline] Loading cached GraphML (networkx) -> {graphml_path}")
        return nx.read_graphml(graphml_path)

    print("[pipeline] *** Using SYNTHETIC grid city (osmnx/graph not available) ***")
    return build_synthetic_city()


def build_synthetic_city(rows: int = 14, cols: int = 14, seed: int = 7) -> nx.MultiDiGraph:
    """A grid of intersections near Bengaluru, edges weighted by real distance.

    Mimics an osmnx graph: nodes carry 'x'(lon)/'y'(lat); edges carry
    'length'(metres) and 'highway'(road class).
    """
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph()
    lat0, lon0, step = 12.95, 77.55, 0.004
    for i in range(rows):
        for j in range(cols):
            nid = i * cols + j
            # Slight jitter so the grid looks like real streets.
            G.add_node(
                nid,
                x=lon0 + j * step + rng.normal(0, 0.0004),
                y=lat0 + i * step + rng.normal(0, 0.0004),
            )

    hwys = ["residential", "tertiary", "secondary", "primary"]
    hwy_p = [0.45, 0.25, 0.20, 0.10]

    def add_edge(a, b):
        ya, xa = G.nodes[a]["y"], G.nodes[a]["x"]
        yb, xb = G.nodes[b]["y"], G.nodes[b]["x"]
        length_m = haversine_km(ya, xa, yb, xb) * 1000.0
        hwy = rng.choice(hwys, p=hwy_p)
        G.add_edge(a, b, length=length_m, highway=hwy)
        G.add_edge(b, a, length=length_m, highway=hwy)

    for i in range(rows):
        for j in range(cols):
            nid = i * cols + j
            if j + 1 < cols:
                add_edge(nid, nid + 1)
            if i + 1 < rows:
                add_edge(nid, nid + cols)
    print(f"[pipeline] Synthetic city: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.")
    return G


# --------------------------------------------------------------------------- #
# 2. Model loading
# --------------------------------------------------------------------------- #
def load_models():
    """Load the trained XGBoost + Random Forest models (train them if missing)."""
    xgb_path = os.path.join(MODELS_DIR, "xgboost_delay.pkl")
    rf_path = os.path.join(MODELS_DIR, "rf_risk.pkl")
    if not (os.path.exists(xgb_path) and os.path.exists(rf_path)):
        print("[pipeline] Models missing -> training them now...")
        from train_rf import train_rf
        from train_xgboost import train_xgboost

        train_xgboost()
        train_rf()
    with open(xgb_path, "rb") as f:
        xgb = pickle.load(f)
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    return xgb, rf


# --------------------------------------------------------------------------- #
# 3. Per-edge ML inference -> edge weights
# --------------------------------------------------------------------------- #
def _safe_encode(le, value) -> int:
    """LabelEncoder transform that maps unseen categories to 0."""
    classes = list(le.classes_)
    return classes.index(value) if value in classes else 0


def _first(value):
    """OSM attrs are sometimes lists (e.g. highway=['primary','secondary'])."""
    return value[0] if isinstance(value, list) and value else value


def assign_ml_weights(G, xgb, rf, data: dict, scenario: dict | None = None,
                      seed: int = 42) -> nx.MultiDiGraph:
    """Predict delay + risk for every edge and set the routing 'weight' attr.

    Sets these edge attributes:  length_km, delay, risk (0/1/2),
    risk_penalty, weight.  Returns the same graph (mutated in place).
    """
    scaler = data["scaler"]
    encoders = data["encoders"]
    features = data["features"]
    numeric = data["numeric_features"]

    base = {
        "Temperature(F)": 75.0,
        "Visibility(mi)": 9.0,
        "Wind_Speed(mph)": 6.0,
        "Precipitation(in)": 0.0,
        "Weather_Condition": "Clear",
        "Sunrise_Sunset": "Day",
    }
    if scenario:
        base.update(scenario)

    edges = list(G.edges(keys=True))
    rows = []
    for (u, v, k) in edges:
        attrs = G[u][v][k]
        hwy = _first(attrs.get("highway", "residential"))
        haz = HIGHWAY_HAZARD.get(hwy, 0.3)

        # Reproducible per-edge variation seeded by the node-id pair.
        er = np.random.default_rng((int(u) * 1_000_003 + int(v) + seed) & 0xFFFFFFFF)

        temp = base["Temperature(F)"] + er.normal(0, 5)
        # Busier roads -> lower effective visibility, more precipitation proxy.
        vis = float(np.clip(base["Visibility(mi)"] - haz * er.uniform(3, 7), 0.3, 10))
        wind = float(np.clip(base["Wind_Speed(mph)"] + er.normal(0, 4), 0, 45))
        precip = base["Precipitation(in)"] + (er.random() < haz * 0.4) * er.exponential(0.06)

        # Hazardous roads occasionally inherit a worse weather category.
        weather = base["Weather_Condition"]
        if er.random() < haz * 0.5:
            weather = er.choice(["Rain", "Fog", "Snow"])
        daynight = base["Sunrise_Sunset"]

        rows.append(
            {
                "Temperature(F)": temp,
                "Visibility(mi)": vis,
                "Wind_Speed(mph)": wind,
                "Precipitation(in)": precip,
                "Weather_Condition": _safe_encode(encoders["Weather_Condition"], weather),
                "Sunrise_Sunset": _safe_encode(encoders["Sunrise_Sunset"], daynight),
            }
        )

    X = pd.DataFrame(rows)[features].astype(float)
    X[numeric] = scaler.transform(X[numeric])  # same scaling as training

    delay_pred = np.clip(xgb.predict(X), 0.0, 1.0)
    risk_pred = rf.predict(X).astype(int)

    for (u, v, k), delay, risk in zip(edges, delay_pred, risk_pred):
        attrs = G[u][v][k]
        length_km = float(attrs.get("length", 100.0)) / 1000.0
        penalty = RISK_PENALTY[int(risk)]
        attrs["length_km"] = length_km
        attrs["delay"] = float(delay)
        attrs["risk"] = int(risk)
        attrs["risk_penalty"] = penalty
        attrs["weight"] = length_km + delay * 0.5 + penalty * 0.3

    print(f"[pipeline] Assigned ML weights to {len(edges)} edges "
          f"(delay mean={delay_pred.mean():.3f}, "
          f"high-risk edges={int((risk_pred == 2).sum())}).")
    return G


# --------------------------------------------------------------------------- #
# 4. Geocoding helper
# --------------------------------------------------------------------------- #
def nearest_node(G, lon: float, lat: float):
    """Nearest graph node to a (lon, lat) point via Haversine (no osmnx needed)."""
    best, best_d = None, float("inf")
    for n, d in G.nodes(data=True):
        if "x" not in d or "y" not in d:
            continue
        dist = haversine_km(lat, lon, float(d["y"]), float(d["x"]))
        if dist < best_d:
            best, best_d = n, dist
    return best


# --------------------------------------------------------------------------- #
# 5. Route analysis
# --------------------------------------------------------------------------- #
def summarise_route(G, path) -> dict:
    """Aggregate distance / delay / risk statistics for a route."""
    if not path or len(path) < 2:
        return {"distance_km": 0, "weight": 0, "delay": 0,
                "risk_counts": {0: 0, 1: 0, 2: 0}, "edges": 0}
    dist = delay = wcost = 0.0
    risk_counts = {0: 0, 1: 0, 2: 0}
    for u, v in zip(path[:-1], path[1:]):
        attrs = min(G[u][v].values(), key=lambda a: a.get("weight", float("inf")))
        dist += attrs.get("length_km", 0.0)
        delay += attrs.get("delay", 0.0)
        wcost += attrs.get("weight", 0.0)
        risk_counts[int(attrs.get("risk", 0))] += 1
    return {
        "distance_km": dist,
        "weight": wcost,
        "delay": delay,
        "risk_counts": risk_counts,
        "edges": len(path) - 1,
    }


def run_route(G, src, dst, trace: bool = False) -> dict:
    """Compute the three route variants for one source-destination pair."""
    # (1) Shortest distance only.
    dist_path, dist_cost, _ = astar(G, src, dst, weight_attr="length_km")
    # (2) ML-weighted A* (the smart, risk-aware route).
    t0 = time.perf_counter()
    if trace:
        ml_path, ml_cost, ml_exp, ml_trace = astar(G, src, dst, weight_attr="weight",
                                                    return_trace=True)
    else:
        ml_path, ml_cost, ml_exp = astar(G, src, dst, weight_attr="weight")
        ml_trace = None
    ml_time = time.perf_counter() - t0
    # Dijkstra for the efficiency comparison.
    _, _, dij_exp = dijkstra(G, src, dst, weight_attr="weight")

    # (3) Hill Climbing post-processing — faithfully seeded from the A* path.
    #     Since A* is already globally optimal under the ML cost, this usually
    #     reports 0 improvements, confirming optimality (HC never worsens it).
    hc_path, hc_cost, hc_iters = hill_climb(G, ml_path, astar_fn=astar,
                                            weight_attr="weight")

    # Hill-Climbing *capability* demo: starting from the naive distance route,
    # local search refines it under the ML-aware cost (a real, non-trivial
    # reduction toward the optimum). Kept separate so the variant table stays
    # monotonic.
    naive_ml_cost = route_cost(G, dist_path, "weight")
    ref_path, ref_cost, ref_iters = hill_climb(G, dist_path, astar_fn=astar,
                                               weight_attr="weight")
    improvement = (100.0 * (naive_ml_cost - ref_cost) / naive_ml_cost
                   if naive_ml_cost else 0.0)

    return {
        "src": src, "dst": dst,
        "dist": {"path": dist_path, "cost": dist_cost, **summarise_route(G, dist_path)},
        "ml": {"path": ml_path, "cost": ml_cost, "expanded": ml_exp,
               "dij_expanded": dij_exp, "time": ml_time, **summarise_route(G, ml_path)},
        "hc": {"path": hc_path, "cost": hc_cost, "iters": hc_iters,
               **summarise_route(G, hc_path)},
        "hc_demo": {"naive_ml_cost": naive_ml_cost, "refined_cost": ref_cost,
                    "iters": ref_iters, "improvement": improvement},
        "trace": ml_trace,
    }


# --------------------------------------------------------------------------- #
# 6. Visualisation
# --------------------------------------------------------------------------- #
def make_folium_map(G, result, out_path: str = None):
    """Draw the ML-A* (blue) and Hill-Climbing (green) routes on a folium map."""
    try:
        import folium
    except ImportError:
        print("[pipeline] folium not installed -> skipping map.")
        return None

    out_path = out_path or os.path.join(OUTPUTS_DIR, "route_map.html")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def coords(path):
        return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in path]

    ml_coords = coords(result["ml"]["path"])
    center = ml_coords[len(ml_coords) // 2]
    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # ML-weighted A* route (blue).
    folium.PolyLine(ml_coords, color="blue", weight=6, opacity=0.75,
                    tooltip="ML-weighted A* route").add_to(fmap)
    # Hill Climbing route (green, dashed so overlaps stay visible).
    folium.PolyLine(coords(result["hc"]["path"]), color="green", weight=4,
                    opacity=0.9, dash_array="8",
                    tooltip="A* + Hill Climbing route").add_to(fmap)

    # Risk-coloured markers along the ML route.
    path = result["ml"]["path"]
    for u, v in zip(path[:-1], path[1:]):
        attrs = min(G[u][v].values(), key=lambda a: a.get("weight", float("inf")))
        y, x = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
        folium.CircleMarker(
            (y, x), radius=3, color=RISK_COLOR[int(attrs.get("risk", 0))],
            fill=True, fill_opacity=0.9,
        ).add_to(fmap)

    folium.Marker(ml_coords[0], tooltip="Source",
                  icon=folium.Icon(color="darkblue", icon="play")).add_to(fmap)
    folium.Marker(ml_coords[-1], tooltip="Destination",
                  icon=folium.Icon(color="red", icon="flag")).add_to(fmap)

    fmap.save(out_path)
    print(f"[pipeline] Saved interactive map -> {out_path}")
    return out_path


def print_comparison_table(results: list[dict]) -> None:
    """Print the three-variant comparison across all tested route pairs."""
    print("\n" + "=" * 92)
    print(f"{'Pair':<6}{'Variant':<14}{'Dist(km)':>10}{'Cost':>9}"
          f"{'Delay':>8}{'High-risk edges':>17}{'HC iters':>10}")
    print("-" * 92)
    for i, r in enumerate(results, 1):
        rows = [
            ("Distance", r["dist"], ""),
            ("ML A*", r["ml"], ""),
            ("ML A*+HC", r["hc"], r["hc"]["iters"]),
        ]
        for name, v, extra in rows:
            print(f"{i:<6}{name:<14}{v['distance_km']:>10.2f}{v['cost']:>9.2f}"
                  f"{v['delay']:>8.2f}{v['risk_counts'][2]:>17}{str(extra):>10}")
        ml, dist = r["ml"], r["dist"]
        saved = 100.0 * (dist["risk_counts"][2] - ml["risk_counts"][2]) / max(dist["risk_counts"][2], 1)
        print(f"       -> A* expanded {ml['expanded']} vs Dijkstra {ml['dij_expanded']} "
              f"nodes; high-risk edges cut {saved:+.0f}%")
        print("-" * 92)
    print("=" * 92)

    # Hill-Climbing refinement demo: improving the naive distance route under
    # the ML-aware cost function.
    print("\nHill Climbing refinement (naive distance route -> ML-aware cost):")
    print(f"  {'Pair':<6}{'Naive ML-cost':>15}{'Refined cost':>15}"
          f"{'Iters':>8}{'Improvement':>14}")
    for i, r in enumerate(results, 1):
        d = r["hc_demo"]
        print(f"  {i:<6}{d['naive_ml_cost']:>15.3f}{d['refined_cost']:>15.3f}"
              f"{d['iters']:>8}{d['improvement']:>13.1f}%")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    data = load_processed()
    xgb, rf = load_models()
    G = load_graph()
    assign_ml_weights(G, xgb, rf, data)

    # Pick several source-destination pairs across the graph.
    node_ids = list(G.nodes())
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(5):
        s, t = rng.choice(node_ids, 2, replace=False)
        pairs.append((s, t))

    results = []
    for idx, (s, t) in enumerate(pairs):
        results.append(run_route(G, s, t, trace=(idx == 0)))

    print_comparison_table(results)

    # Explainability: print the A* node-expansion trace for the first route.
    first = results[0]
    if first["trace"]:
        print("\nA* node-expansion trace for route #1 (first 12 steps):")
        print(f"  {'#':<4}{'node':<14}{'g':>9}{'h':>9}{'f':>9}")
        for i, (node, g, h, f) in enumerate(first["trace"][:12], 1):
            print(f"  {i:<4}{str(node):<14}{g:>9.3f}{h:>9.3f}{f:>9.3f}")

    # Map for the first route.
    make_folium_map(G, first)

    print("\n[pipeline] Done. Open outputs/route_map.html to view the routes.")


if __name__ == "__main__":
    main()
