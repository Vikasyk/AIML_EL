"""
multi_stop.py  (innovation feature)
===================================
Multi-stop delivery routing: given one source and several delivery
destinations, find a good visiting order and the full multi-leg route.

Approach
--------
* Build a cost matrix between all stops using **A* edge costs** (the ML-aware
  routing weight), NOT raw straight-line distance.
* Find a visiting order with a TSP heuristic:
      1. Nearest-neighbour construction (source fixed as the start),
      2. 2-opt local search improvement.
* Stitch the per-leg A* paths into one continuous route.
* Visualise every leg in a different colour on a single folium map.

Run:  python src/multi_stop.py
"""

from __future__ import annotations

import os

from astar import astar
from pipeline import OUTPUTS_DIR, assign_ml_weights, load_graph, load_models
from preprocess import load_processed

LEG_COLORS = ["blue", "green", "purple", "orange", "darkred",
              "cadetblue", "darkgreen", "darkpurple", "red", "black"]


# --------------------------------------------------------------------------- #
# Cost matrix + TSP heuristics
# --------------------------------------------------------------------------- #
def build_cost_matrix(G, stops, weight_attr: str = "weight"):
    """Pairwise A* costs + paths between every pair of stops.

    Returns ``(cost[n][n], paths{(i,j): node_list})``. Unreachable pairs get
    cost = +inf.
    """
    n = len(stops)
    cost = [[0.0] * n for _ in range(n)]
    paths: dict[tuple[int, int], list] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path, c, _ = astar(G, stops[i], stops[j], weight_attr=weight_attr)
            cost[i][j] = c if path else float("inf")
            paths[(i, j)] = path
    return cost, paths


def nearest_neighbour_order(cost, start: int = 0) -> list[int]:
    """Greedy nearest-neighbour visiting order (open tour from ``start``)."""
    n = len(cost)
    unvisited = set(range(n)) - {start}
    order = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: cost[cur][j])
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return order


def tour_cost(order, cost) -> float:
    """Total cost of an open tour (sum of consecutive legs)."""
    return sum(cost[order[i]][order[i + 1]] for i in range(len(order) - 1))


def two_opt(order, cost) -> list[int]:
    """2-opt improvement on an open tour, keeping the start node fixed."""
    best = order[:]
    best_cost = tour_cost(best, cost)
    improved = True
    while improved:
        improved = False
        # i starts at 1 so the source (index 0) stays first.
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_cost = tour_cost(candidate, cost)
                if cand_cost < best_cost - 1e-9:
                    best, best_cost = candidate, cand_cost
                    improved = True
        # loop again if anything improved
    return best


def stitch_route(order, paths) -> list:
    """Concatenate per-leg A* node paths into one route (no duplicate joins)."""
    full = []
    legs = []
    for a, b in zip(order[:-1], order[1:]):
        leg = paths[(a, b)]
        legs.append(leg)
        if not full:
            full.extend(leg)
        else:
            full.extend(leg[1:])  # skip the repeated join node
    return full, legs


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def make_multistop_map(G, stops, order, legs, out_path: str | None = None):
    """Draw each leg in a different colour, with numbered stop markers."""
    try:
        import folium
    except ImportError:
        print("[multi_stop] folium not installed -> skipping map.")
        return None

    out_path = out_path or os.path.join(OUTPUTS_DIR, "multi_stop_map.html")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def coords(path):
        return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in path]

    all_pts = coords([stops[i] for i in order])
    fmap = folium.Map(location=all_pts[len(all_pts) // 2], zoom_start=13,
                      tiles="cartodbpositron")

    for idx, leg in enumerate(legs):
        folium.PolyLine(coords(leg), color=LEG_COLORS[idx % len(LEG_COLORS)],
                        weight=5, opacity=0.8,
                        tooltip=f"Leg {idx + 1}").add_to(fmap)

    for visit_idx, stop_idx in enumerate(order):
        node = stops[stop_idx]
        y, x = float(G.nodes[node]["y"]), float(G.nodes[node]["x"])
        label = "Depot" if visit_idx == 0 else f"Stop {visit_idx}"
        folium.Marker((y, x), tooltip=f"{label} (node {node})",
                      icon=folium.Icon(color="darkblue" if visit_idx == 0 else "green",
                                       icon="home" if visit_idx == 0 else "shopping-cart",
                                       prefix="fa")).add_to(fmap)
    fmap.save(out_path)
    print(f"[multi_stop] Saved map -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def plan_multistop(G, source, destinations):
    """Full multi-stop plan. Returns a result dict."""
    stops = [source] + list(destinations)
    cost, paths = build_cost_matrix(G, stops)

    naive_order = list(range(len(stops)))           # visit in given order
    nn_order = nearest_neighbour_order(cost, start=0)
    opt_order = two_opt(nn_order, cost)

    full_path, legs = stitch_route(opt_order, paths)
    return {
        "stops": stops, "cost": cost,
        "naive_order": naive_order, "naive_cost": tour_cost(naive_order, cost),
        "nn_order": nn_order, "nn_cost": tour_cost(nn_order, cost),
        "opt_order": opt_order, "opt_cost": tour_cost(opt_order, cost),
        "full_path": full_path, "legs": legs, "paths": paths,
    }


def main() -> None:
    import numpy as np

    data = load_processed()
    xgb, rf = load_models()
    G = load_graph()
    assign_ml_weights(G, xgb, rf, data)

    nodes = list(G.nodes())
    rng = np.random.default_rng(11)
    chosen = rng.choice(nodes, 6, replace=False)  # 1 depot + 5 deliveries
    source, destinations = chosen[0], list(chosen[1:])

    res = plan_multistop(G, source, destinations)

    print("========== MULTI-STOP DELIVERY ROUTING ==========")
    print(f"Depot         : {res['stops'][0]}")
    print(f"Deliveries    : {res['stops'][1:]}")
    print(f"\nVisiting order (index into stops):")
    print(f"  As-given       : {res['naive_order']}  cost={res['naive_cost']:.3f}")
    print(f"  Nearest-neighbr: {res['nn_order']}  cost={res['nn_cost']:.3f}")
    print(f"  + 2-opt (final): {res['opt_order']}  cost={res['opt_cost']:.3f}")

    improvement = (100.0 * (res["naive_cost"] - res["opt_cost"]) / res["naive_cost"]
                   if res["naive_cost"] else 0.0)
    print(f"\n=> Optimised order is {improvement:.1f}% cheaper than the as-given order.")
    print(f"=> Full multi-leg route: {len(res['full_path'])} nodes "
          f"across {len(res['legs'])} legs.")
    print("=================================================")

    make_multistop_map(G, res["stops"], res["opt_order"], res["legs"])


if __name__ == "__main__":
    main()
