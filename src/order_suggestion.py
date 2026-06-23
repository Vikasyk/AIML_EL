"""
order_suggestion.py
====================
"I already picked up delivery order X (source -> destination). What OTHER
pending orders can I also fulfil along the way, without a big detour?"

This is the reverse of multi_stop.py:
    * multi_stop.py   -> you already KNOW all the stops, find the best ORDER.
    * order_suggestion.py -> you have ONE committed route, and a POOL of other
      pending orders; rank which of those are cheap to "pick up on the way".

Approach
--------
For every candidate order (pickup, drop):
    1. Compute the best way to insert (pickup -> drop) into the existing
       source -> destination route. We try the two natural insertion
       patterns and keep whichever is cheaper:
         a) source -> pickup -> drop -> destination
         b) source -> pickup -> destination, with drop handled as a
            mini-detour right after pickup (covered by pattern a in this
            simple version; kept as one pattern for clarity).
    2. extra_cost = cost(with detour) - cost(original route)
    3. If extra_cost <= threshold (absolute km-cost or % of original route),
       it's a "suggested" order. Sort all suggestions by extra_cost.

This reuses the existing `astar()` function from astar.py — no new search
algorithm is implemented, we just call A* a few extra times with different
intermediate stops.
"""

from __future__ import annotations

from dataclasses import dataclass

from astar import astar


@dataclass
class CandidateOrder:
    order_id: str
    pickup: int   # graph node id
    drop: int     # graph node id


@dataclass
class Suggestion:
    order_id: str
    pickup: int
    drop: int
    extra_cost: float
    extra_pct: float
    combined_path: list
    insertion: str  # human-readable description of where it was inserted


def _route_cost(G, a, b, weight_attr="weight"):
    """Shortest A* cost between two nodes. Returns inf if unreachable."""
    _, cost, _ = astar(G, a, b, weight_attr=weight_attr)
    return cost


def _full_path_cost(G, nodes: list, weight_attr="weight"):
    """Sum of A* leg costs across an ordered list of nodes, and the
    concatenated path (node list) across all legs."""
    total = 0.0
    full_path = [nodes[0]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        path, cost, _ = astar(G, a, b, weight_attr=weight_attr)
        if path is None:
            return float("inf"), None
        total += cost
        full_path.extend(path[1:])
    return total, full_path


def suggest_orders(
    G,
    source,
    destination,
    candidates: list[CandidateOrder],
    weight_attr: str = "weight",
    max_extra_pct: float = 20.0,
    max_suggestions: int = 5,
) -> dict:
    """Rank candidate orders by how cheaply they fit onto the committed
    source -> destination route.

    Parameters
    ----------
    G : the (ML-weighted) road graph.
    source, destination : node ids of the delivery you already committed to.
    candidates : pool of other pending orders to screen.
    weight_attr : which edge attribute to use as cost ('weight' = ML cost).
    max_extra_pct : only suggest orders that add at most this % extra cost
        over the original route.
    max_suggestions : cap on how many suggestions to return.

    Returns
    -------
    dict with:
        original_cost   : cost of the committed route alone
        original_path   : node list of the committed route
        suggestions     : list[Suggestion], sorted cheapest-detour first
        rejected        : list of (order_id, reason) for orders that didn't
                           make the cut (too expensive or unreachable)
    """
    original_cost, original_path = _full_path_cost(G, [source, destination], weight_attr)
    if original_path is None:
        raise ValueError("No path exists between source and destination.")

    suggestions: list[Suggestion] = []
    rejected: list[tuple] = []

    for cand in candidates:
        # Pattern: source -> pickup -> drop -> destination
        nodes = [source, cand.pickup, cand.drop, destination]
        combined_cost, combined_path = _full_path_cost(G, nodes, weight_attr)

        if combined_path is None:
            rejected.append((cand.order_id, "unreachable"))
            continue

        extra_cost = combined_cost - original_cost
        extra_pct = (extra_cost / original_cost * 100.0) if original_cost > 0 else float("inf")

        if extra_pct <= max_extra_pct:
            suggestions.append(
                Suggestion(
                    order_id=cand.order_id,
                    pickup=cand.pickup,
                    drop=cand.drop,
                    extra_cost=round(extra_cost, 3),
                    extra_pct=round(extra_pct, 1),
                    combined_path=combined_path,
                    insertion="source -> pickup -> drop -> destination",
                )
            )
        else:
            rejected.append((cand.order_id, f"+{extra_pct:.1f}% too costly"))

    suggestions.sort(key=lambda s: s.extra_cost)
    suggestions = suggestions[:max_suggestions]

    return {
        "original_cost": round(original_cost, 3),
        "original_path": original_path,
        "suggestions": suggestions,
        "rejected": rejected,
    }


def make_suggestion_map(G, source, destination, result: dict, out_path: str | None = None):
    """Draw the committed route (blue) plus each suggested detour (orange)
    on a folium map. Returns the saved HTML path."""
    import folium

    def node_latlon(n):
        d = G.nodes[n]
        return float(d["y"]), float(d["x"])

    center = node_latlon(source)
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Committed route.
    folium.PolyLine(
        [node_latlon(n) for n in result["original_path"]],
        color="blue", weight=5, opacity=0.8, tooltip="Committed route",
    ).add_to(m)
    folium.Marker(node_latlon(source), tooltip="Source",
                  icon=folium.Icon(color="blue", icon="home")).add_to(m)
    folium.Marker(node_latlon(destination), tooltip="Destination",
                  icon=folium.Icon(color="blue", icon="flag")).add_to(m)

    colors = ["orange", "green", "purple", "darkred", "cadetblue"]
    for i, s in enumerate(result["suggestions"]):
        color = colors[i % len(colors)]
        folium.PolyLine(
            [node_latlon(n) for n in s.combined_path],
            color=color, weight=3, opacity=0.6, dash_array="6",
            tooltip=f"{s.order_id} (+{s.extra_cost:.2f} cost, +{s.extra_pct:.1f}%)",
        ).add_to(m)
        folium.Marker(node_latlon(s.pickup), tooltip=f"{s.order_id} pickup",
                      icon=folium.Icon(color=color, icon="shopping-cart")).add_to(m)
        folium.Marker(node_latlon(s.drop), tooltip=f"{s.order_id} drop",
                      icon=folium.Icon(color=color, icon="ok")).add_to(m)

    if out_path:
        m.save(out_path)
        return out_path
    return None


def main() -> None:
    """Tiny smoke test using the synthetic/test graph from astar.py."""
    from astar import build_test_graph

    G, source, destination = build_test_graph()
    nodes = list(G.nodes.keys())

    candidates = [
        CandidateOrder(order_id="ORD-1", pickup="B", drop="D"),
        CandidateOrder(order_id="ORD-2", pickup="C", drop="F"),
    ]

    result = suggest_orders(G, source, destination, candidates, weight_attr="weight")
    print(f"Original cost: {result['original_cost']:.3f}")
    for s in result["suggestions"]:
        print(f"  {s.order_id}: +{s.extra_cost:.3f} cost (+{s.extra_pct:.1f}%) "
              f"via {s.insertion}")
    for order_id, reason in result["rejected"]:
        print(f"  rejected {order_id}: {reason}")


if __name__ == "__main__":
    main()