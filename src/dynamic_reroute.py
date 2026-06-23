"""
dynamic_reroute.py  (innovation feature)
========================================
Simulate a **live disruption** along an in-progress delivery and automatically
re-optimise the remaining route.

Scenario
--------
1. A delivery vehicle starts following the ML-weighted A* route.
2. Part-way along, a "disruption event" fires: the next planned road segment's
   risk suddenly spikes to High (e.g. a fresh accident report), inflating its
   weight.
3. The system detects the problem at the vehicle's current node and re-runs A*
   from there to the destination using the updated weights.
4. We show a before/after comparison of the route and its cost.

Demo line: *"the system detects a problem mid-route and re-optimises automatically."*

Run:  python src/dynamic_reroute.py
"""

from __future__ import annotations

import os

from astar import astar
from hill_climb import route_cost
from pipeline import (
    OUTPUTS_DIR,
    RISK_COLOR,
    assign_ml_weights,
    load_graph,
    load_models,
)
from preprocess import load_processed

DISRUPTION_PENALTY = 6.0  # km-equivalent spike added to the blocked segment


def apply_disruption(G, u, v, penalty: float = DISRUPTION_PENALTY) -> list[tuple]:
    """Spike the risk/weight of edge u->v to simulate a new accident/blockage."""
    changed = []
    for k, attrs in G[u][v].items():
        old_weight = attrs.get("weight", 0.0)
        attrs["risk"] = 2
        attrs["risk_penalty"] = 1.0
        attrs["delay"] = 1.0
        attrs["weight"] = attrs.get("length_km", 0.0) + 0.5 + 0.3 + penalty
        changed.append((k, old_weight, attrs["weight"]))
    return changed


def simulate_and_reroute(G, src, dst, ahead_frac: float = 0.4):
    """Run A*, fire a disruption ahead of the vehicle, and re-route.

    Returns a dict with the original route, the vehicle's current node, the
    disrupted edge, and the new re-optimised route.
    """
    # --- Original plan --------------------------------------------------- #
    original_path, original_cost, _ = astar(G, src, dst, weight_attr="weight")
    if not original_path or len(original_path) < 3:
        raise ValueError("Need a route with at least 3 nodes to demo a reroute.")

    # --- Vehicle position + disruption ahead ----------------------------- #
    pos = max(1, min(int(len(original_path) * ahead_frac), len(original_path) - 2))
    current_node = original_path[pos]
    blocked = (original_path[pos], original_path[pos + 1])

    # Cost the vehicle still expected to pay on the original plan.
    remaining_before = route_cost(G, original_path[pos:], "weight")

    apply_disruption(G, *blocked)
    remaining_after_blocked = route_cost(G, original_path[pos:], "weight")

    # --- Automatic re-route from the current node ------------------------ #
    new_tail, new_tail_cost, expanded = astar(G, current_node, dst, weight_attr="weight")
    new_full = original_path[:pos] + new_tail

    rerouted = blocked[1] not in new_tail or new_tail[: 2] != list(blocked)

    return {
        "src": src, "dst": dst,
        "original_path": original_path, "original_cost": original_cost,
        "current_node": current_node, "blocked": blocked,
        "remaining_before": remaining_before,
        "remaining_after_blocked": remaining_after_blocked,
        "new_tail": new_tail, "new_tail_cost": new_tail_cost,
        "new_full": new_full, "expanded": expanded,
        "avoided_block": rerouted,
    }


def make_reroute_map(G, result, out_path: str | None = None):
    """Original route (blue), blocked segment (red), re-route (green)."""
    try:
        import folium
    except ImportError:
        print("[reroute] folium not installed -> skipping map.")
        return None

    out_path = out_path or os.path.join(OUTPUTS_DIR, "dynamic_reroute_map.html")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    def coords(path):
        return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in path]

    orig = coords(result["original_path"])
    fmap = folium.Map(location=orig[len(orig) // 2], zoom_start=14,
                      tiles="cartodbpositron")

    folium.PolyLine(orig, color="blue", weight=5, opacity=0.6,
                    tooltip="Original plan").add_to(fmap)
    folium.PolyLine(coords(result["new_full"]), color="green", weight=4,
                    opacity=0.9, dash_array="6", tooltip="Re-optimised route").add_to(fmap)
    # Blocked segment in red.
    bu, bv = result["blocked"]
    folium.PolyLine(coords([bu, bv]), color="red", weight=8, opacity=0.9,
                    tooltip="DISRUPTION (risk spike)").add_to(fmap)
    folium.Marker(coords([result["current_node"]])[0], tooltip="Vehicle position",
                  icon=folium.Icon(color="orange", icon="truck", prefix="fa")).add_to(fmap)
    folium.Marker(orig[0], tooltip="Source",
                  icon=folium.Icon(color="darkblue", icon="play")).add_to(fmap)
    folium.Marker(orig[-1], tooltip="Destination",
                  icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
    fmap.save(out_path)
    print(f"[reroute] Saved map -> {out_path}")
    return out_path


def main() -> None:
    data = load_processed()
    xgb, rf = load_models()
    G = load_graph()
    assign_ml_weights(G, xgb, rf, data)

    # Find a source/destination pair with a reasonably long route.
    import numpy as np

    nodes = list(G.nodes())
    rng = np.random.default_rng(3)
    result = None
    for _ in range(30):
        s, t = rng.choice(nodes, 2, replace=False)
        path, _, _ = astar(G, s, t, weight_attr="weight")
        if path and len(path) >= 6:
            result = simulate_and_reroute(G, s, t)
            break
    if result is None:
        print("[reroute] Could not find a long-enough route; try again.")
        return

    print("========== DYNAMIC RE-ROUTING SIMULATION ==========")
    print(f"Source -> Destination : {result['src']} -> {result['dst']}")
    print(f"Original route        : {len(result['original_path'])} nodes, "
          f"cost={result['original_cost']:.3f}")
    print(f"Vehicle reaches node  : {result['current_node']}")
    print(f"!! DISRUPTION on edge  : {result['blocked'][0]} -> {result['blocked'][1]} "
          f"(risk spiked to HIGH)")
    print(f"Remaining cost (plan) : {result['remaining_before']:.3f}")
    print(f"Remaining cost (if we ignore disruption): "
          f"{result['remaining_after_blocked']:.3f}  <- now expensive")
    print(f"Re-optimised tail     : {len(result['new_tail'])} nodes, "
          f"cost={result['new_tail_cost']:.3f}  (A* expanded {result['expanded']} nodes)")
    print(f"Avoided blocked road  : {'YES' if result['avoided_block'] else 'no alt path'}")
    saved = result["remaining_after_blocked"] - result["new_tail_cost"]
    print(f"=> Re-routing saved    : {saved:.3f} cost units vs driving into the disruption")
    print("===================================================")

    make_reroute_map(G, result)


if __name__ == "__main__":
    main()
