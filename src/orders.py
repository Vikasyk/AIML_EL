"""
orders.py
=========
Delivery-person-centric workflow.

    1. A pool of pending orders is generated (in real life this would come
       from an orders database / API — here we simulate it on the depot's
       road graph so the whole thing is demoable offline).
    2. The delivery person SELECTS one order they want to deliver.
    3. The system shows which OTHER pending orders are cheap to add to that
       same trip (small extra cost over the selected order's own route).
    4. For whichever orders the delivery person picks (selected + any extras),
       the system finds the best VISITING ORDER (TSP heuristic, reusing
       multi_stop.py) and draws the final route on a map.

This module only generates/screens orders and stitches results together —
all the actual pathfinding/TSP logic is reused from astar.py and
multi_stop.py, no new search algorithm is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astar import astar
from multi_stop import build_cost_matrix, nearest_neighbour_order, stitch_route, tour_cost, two_opt
from pipeline import summarise_route


@dataclass
class Order:
    order_id: str
    drop: int                # graph node id of the customer's location
    distance_km: float = 0.0
    cost: float = 0.0        # ML-weighted route cost from depot
    delay: float = 0.0
    risk_high: int = 0


# --------------------------------------------------------------------------- #
# 1. Generate the pending-order pool
# --------------------------------------------------------------------------- #
def generate_order_pool(G, depot, n_orders: int = 10, seed: int | None = None) -> list[Order]:
    """Simulate `n_orders` pending deliveries from the depot to random nodes.

    Each order's cost/distance is computed once with A* so the order list can
    be shown to the delivery person immediately (e.g. sorted by distance).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    candidate_nodes = [n for n in G.nodes() if n != depot]
    chosen = rng.choice(candidate_nodes, size=min(n_orders, len(candidate_nodes)), replace=False)

    orders: list[Order] = []
    for i, node in enumerate(chosen):
        path, cost, _ = astar(G, depot, node, weight_attr="weight")
        if path is None:
            continue
        summary = summarise_route(G, path)
        orders.append(Order(
            order_id=f"ORD-{i + 1:03d}",
            drop=node,
            distance_km=round(summary["distance_km"], 2),
            cost=round(cost, 2),
            delay=round(summary["delay"], 2),
            risk_high=summary["risk_counts"].get(2, 0),
        ))
    orders.sort(key=lambda o: o.distance_km)
    return orders


# --------------------------------------------------------------------------- #
# 2. Given one selected order, suggest cheap-to-add extra orders
# --------------------------------------------------------------------------- #
def suggest_combinable_orders(
    G, depot, selected: Order, pool: list[Order], max_extra_pct: float = 25.0,
    max_suggestions: int = 5,
) -> list[dict]:
    """For every other order in the pool, estimate the extra cost of also
    visiting it on the depot -> selected.drop trip, and keep the cheap ones.

    Approximation used: extra_cost = cost(depot -> candidate -> selected.drop)
    - cost(depot -> selected.drop). This is the standard "insert before
    destination" check; the *final* visiting order is decided properly later
    by the TSP step in `plan_combined_route`, so this is just a quick filter.
    """
    base_cost = selected.cost
    suggestions = []
    for cand in pool:
        if cand.order_id == selected.order_id:
            continue
        leg1, c1, _ = astar(G, depot, cand.drop, weight_attr="weight")
        leg2, c2, _ = astar(G, cand.drop, selected.drop, weight_attr="weight")
        if leg1 is None or leg2 is None:
            continue
        combined_cost = c1 + c2
        extra_cost = combined_cost - base_cost
        extra_pct = (extra_cost / base_cost * 100.0) if base_cost > 0 else float("inf")
        if extra_pct <= max_extra_pct:
            suggestions.append({
                "order": cand,
                "extra_cost": round(extra_cost, 2),
                "extra_pct": round(extra_pct, 1),
            })
    suggestions.sort(key=lambda s: s["extra_cost"])
    return suggestions[:max_suggestions]


# --------------------------------------------------------------------------- #
# 3. Combine selected + chosen extra orders into one optimal route
# --------------------------------------------------------------------------- #
def plan_combined_route(G, depot, chosen_orders: list[Order]) -> dict:
    """Find the best order to visit `chosen_orders` (TSP heuristic), starting
    and ending the trip at the depot. Reuses multi_stop.py's machinery."""
    stops = [depot] + [o.drop for o in chosen_orders]
    cost, paths = build_cost_matrix(G, stops, weight_attr="weight")

    nn_order = nearest_neighbour_order(cost, start=0)
    opt_order = two_opt(nn_order, cost)
    full_path, legs = stitch_route(opt_order, paths)
    naive_order = list(range(len(stops)))
    naive_path, naive_legs = stitch_route(naive_order, paths)

    visiting_order_ids = [chosen_orders[i - 1].order_id for i in opt_order[1:]]

    return {
        "stops": stops,
        "chosen_orders": chosen_orders,
        "opt_order": opt_order,
        "naive_order": naive_order,
        "opt_cost": tour_cost(opt_order, cost),
        "naive_cost": tour_cost(naive_order, cost),
        "visiting_order_ids": visiting_order_ids,
        "full_path": full_path,
        "legs": legs,
        "naive_path": naive_path,
        "naive_legs": naive_legs,
    }


def make_orders_map(G, depot, result: dict, out_path: str | None = None):
    """Draw the depot, every chosen stop (numbered in visiting order) and the
    stitched route on a folium map."""
    import folium

    LEG_COLORS = ["blue", "green", "purple", "orange", "darkred", "cadetblue"]

    def latlon(n):
        d = G.nodes[n]
        return float(d["y"]), float(d["x"])

    fmap = folium.Map(location=latlon(depot), zoom_start=12, tiles="cartodbpositron")
    folium.Marker(latlon(depot), tooltip="Depot (start)",
                  icon=folium.Icon(color="darkblue", icon="home", prefix="fa")).add_to(fmap)

    for idx, leg in enumerate(result["legs"]):
        folium.PolyLine([latlon(n) for n in leg],
                         color=LEG_COLORS[idx % len(LEG_COLORS)],
                         weight=5, opacity=0.8, tooltip=f"Leg {idx + 1}").add_to(fmap)

    for visit_idx, stop_idx in enumerate(result["opt_order"][1:], start=1):
        order = result["chosen_orders"][stop_idx - 1]
        folium.Marker(latlon(order.drop),
                      tooltip=f"Stop {visit_idx}: {order.order_id}",
                      icon=folium.Icon(color="green", icon="shopping-cart", prefix="fa")
                      ).add_to(fmap)

    if out_path:
        fmap.save(out_path)
        return out_path
    return None
