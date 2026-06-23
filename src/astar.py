"""
astar.py
========
A* informed search **implemented from scratch** (only ``heapq`` + ``math`` from
the standard library), plus a Dijkstra implementation for comparison.

A* evaluates each node with  f(n) = g(n) + h(n)  where
    g(n) = actual cost from the source to n,
    h(n) = Haversine straight-line distance from n to the target (admissible).

The functions work directly on an OSMnx / NetworkX ``MultiDiGraph`` (nodes carry
'x'=lon and 'y'=lat; edges carry a 'weight' attribute), and also on the tiny
``SimpleGraph`` defined below so the algorithm can be validated with **zero
external dependencies**.

Run ``python src/astar.py`` to:
  * validate A* against a brute-force shortest path on a hand-crafted graph,
  * confirm A* and Dijkstra return the same (optimal) cost,
  * show A* expands fewer nodes than Dijkstra thanks to the heuristic,
  * print a step-by-step node-expansion trace (for viva/demo).
"""

from __future__ import annotations

import heapq
import math
from itertools import count

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- #
# Heuristic and edge-weight helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def node_heuristic(G, n, target) -> float:
    """Admissible heuristic h(n): Haversine distance n -> target in km.

    If either node lacks coordinates (e.g. an abstract test graph), the
    heuristic degrades gracefully to 0, turning A* into uniform-cost search
    (still optimal, just without the geometric speed-up).
    """
    try:
        n1, n2 = G.nodes[n], G.nodes[target]
        return haversine_km(n1["y"], n1["x"], n2["y"], n2["x"])
    except (KeyError, TypeError):
        return 0.0


def edge_weight(edge_attrs: dict, weight_attr: str = "weight") -> float:
    """Read an edge's cost.

    Order of preference:
      1. the requested ``weight_attr`` (e.g. the ML-computed 'weight'),
      2. 'length' (OSM stores metres) converted to km,
      3. a large default so unknown edges are avoided.
    """
    if weight_attr in edge_attrs and edge_attrs[weight_attr] is not None:
        return float(edge_attrs[weight_attr])
    if "length" in edge_attrs and edge_attrs["length"] is not None:
        return float(edge_attrs["length"]) / 1000.0
    return 100.0


def _min_parallel_edge_weight(edge_dict: dict, weight_attr: str) -> float:
    """A MultiGraph can have parallel edges u->v; pick the cheapest."""
    return min(edge_weight(attrs, weight_attr) for attrs in edge_dict.values())


# --------------------------------------------------------------------------- #
# A* search
# --------------------------------------------------------------------------- #
def astar(G, source, target, weight_attr: str = "weight", return_trace: bool = False):
    """A* shortest path from ``source`` to ``target`` on graph ``G``.

    Returns ``(path, total_cost, nodes_expanded)``; if ``return_trace`` is True,
    returns ``(path, total_cost, nodes_expanded, trace)`` where ``trace`` is the
    ordered list of ``(node, g, h, f)`` tuples as nodes are popped/expanded.

    On failure returns ``(None, inf, nodes_expanded[, trace])``.
    """
    counter = count()  # tie-breaker so heap never compares node ids
    open_heap = [(node_heuristic(G, source, target), next(counter), source)]
    g_cost = {source: 0.0}
    came_from = {source: None}
    closed = set()
    nodes_expanded = 0
    trace: list[tuple] = []

    while open_heap:
        f, _, u = heapq.heappop(open_heap)
        if u in closed:
            continue  # stale heap entry — already finalised
        closed.add(u)
        nodes_expanded += 1

        if return_trace:
            g_u = g_cost[u]
            trace.append((u, g_u, f - g_u, f))

        if u == target:
            path = _reconstruct(came_from, target)
            result = (path, g_cost[target], nodes_expanded)
            return (*result, trace) if return_trace else result

        for v, edge_dict in G[u].items():
            if v in closed:
                continue
            w = _min_parallel_edge_weight(edge_dict, weight_attr)
            tentative_g = g_cost[u] + w
            if v not in g_cost or tentative_g < g_cost[v]:
                g_cost[v] = tentative_g
                came_from[v] = u
                f_v = tentative_g + node_heuristic(G, v, target)
                heapq.heappush(open_heap, (f_v, next(counter), v))

    result = (None, float("inf"), nodes_expanded)
    return (*result, trace) if return_trace else result


def dijkstra(G, source, target, weight_attr: str = "weight"):
    """Uniform-cost (Dijkstra) search — same as A* but with h(n) = 0.

    Used to prove A* optimality (equal cost) and efficiency (A* expands fewer
    nodes). Returns ``(path, total_cost, nodes_expanded)``.
    """
    counter = count()
    open_heap = [(0.0, next(counter), source)]
    g_cost = {source: 0.0}
    came_from = {source: None}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        g, _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        nodes_expanded += 1

        if u == target:
            return _reconstruct(came_from, target), g_cost[target], nodes_expanded

        for v, edge_dict in G[u].items():
            if v in closed:
                continue
            w = _min_parallel_edge_weight(edge_dict, weight_attr)
            tentative_g = g_cost[u] + w
            if v not in g_cost or tentative_g < g_cost[v]:
                g_cost[v] = tentative_g
                came_from[v] = u
                heapq.heappush(open_heap, (tentative_g, next(counter), v))

    return None, float("inf"), nodes_expanded


def _reconstruct(came_from: dict, target):
    """Walk the came_from chain back to the source and reverse it."""
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]


# --------------------------------------------------------------------------- #
# Minimal NetworkX-compatible graph (for dependency-free testing)
# --------------------------------------------------------------------------- #
class SimpleGraph:
    """Tiny undirected multigraph mimicking the NetworkX access pattern.

    Supports the exact operations A*/Dijkstra use:
        G.nodes[n]      -> {'x': lon, 'y': lat}
        G[u]            -> {v: {key: {attr: value}}}
    """

    def __init__(self):
        self.nodes: dict = {}
        self._adj: dict = {}

    def add_node(self, n, x=None, y=None):
        self.nodes[n] = {"x": x, "y": y}
        self._adj.setdefault(n, {})

    def add_edge(self, u, v, weight, bidirectional=True):
        self._adj.setdefault(u, {})[v] = {0: {"weight": float(weight)}}
        if bidirectional:
            self._adj.setdefault(v, {})[u] = {0: {"weight": float(weight)}}

    def __getitem__(self, u):
        return self._adj.get(u, {})


def build_test_graph():
    """Hand-crafted 8-node geographic graph with a known optimal path.

    Coordinates are small offsets near Bengaluru. Each edge weight equals the
    Haversine distance between its endpoints, which makes the straight-line
    heuristic admissible (and lets A* genuinely beat Dijkstra on node count).
    Returns ``(G, source, target)``.
    """
    # node: (lon=x, lat=y)
    coords = {
        "A": (77.560, 12.960),  # source
        "B": (77.565, 12.965),
        "C": (77.570, 12.962),
        "D": (77.568, 12.972),
        "E": (77.575, 12.970),
        "F": (77.580, 12.965),
        "G": (77.585, 12.972),
        "H": (77.590, 12.968),  # target
    }
    G = SimpleGraph()
    for n, (x, y) in coords.items():
        G.add_node(n, x=x, y=y)

    def w(u, v):  # edge weight = true geographic distance
        (x1, y1), (x2, y2) = coords[u], coords[v]
        return haversine_km(y1, x1, y2, x2)

    edges = [
        ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "E"), ("C", "F"),
        ("D", "E"), ("E", "G"), ("F", "G"), ("F", "H"),
        ("G", "H"), ("E", "F"),
    ]
    for u, v in edges:
        G.add_edge(u, v, w(u, v))
    return G, "A", "H"


def brute_force_shortest(G, source, target, weight_attr="weight"):
    """Enumerate ALL simple paths (feasible for <=10 nodes) -> true optimum.

    Returns ``(best_path, best_cost)``. Independent ground truth for validation.
    """
    best = {"path": None, "cost": float("inf")}

    def dfs(node, visited, path, cost):
        if cost >= best["cost"]:
            return
        if node == target:
            best["path"], best["cost"] = list(path), cost
            return
        for v, edge_dict in G[node].items():
            if v in visited:
                continue
            w = _min_parallel_edge_weight(edge_dict, weight_attr)
            visited.add(v)
            path.append(v)
            dfs(v, visited, path, cost + w)
            path.pop()
            visited.remove(v)

    dfs(source, {source}, [source], 0.0)
    return best["path"], best["cost"]


# --------------------------------------------------------------------------- #
# Self-validation / demo
# --------------------------------------------------------------------------- #
def main() -> None:
    G, src, dst = build_test_graph()

    print("========== A* VALIDATION (hand-crafted 8-node graph) ==========")
    a_path, a_cost, a_exp, trace = astar(G, src, dst, return_trace=True)
    d_path, d_cost, d_exp = dijkstra(G, src, dst)
    bf_path, bf_cost = brute_force_shortest(G, src, dst)

    print(f"A*       path: {a_path}  cost={a_cost:.4f}  expanded={a_exp}")
    print(f"Dijkstra path: {d_path}  cost={d_cost:.4f}  expanded={d_exp}")
    print(f"BruteForce   : {bf_path}  cost={bf_cost:.4f}  (ground truth)")

    # --- Correctness + optimality assertions ---------------------------- #
    assert a_path[0] == src and a_path[-1] == dst, "A* path endpoints wrong"
    assert math.isclose(a_cost, bf_cost, rel_tol=1e-9), "A* is NOT optimal!"
    assert math.isclose(a_cost, d_cost, rel_tol=1e-9), "A* != Dijkstra cost!"
    print("\n[OK] A* path is optimal (matches brute force and Dijkstra).")

    # --- Efficiency ----------------------------------------------------- #
    if d_exp > 0:
        saved = 100.0 * (d_exp - a_exp) / d_exp
        print(f"[OK] A* expanded {a_exp} vs Dijkstra {d_exp} nodes "
              f"({saved:+.0f}% fewer - heuristic pruning).")

    # --- Step-by-step trace (explainability) ---------------------------- #
    print("\nA* node-expansion trace  (f = g + h):")
    print(f"  {'#':<3}{'node':<6}{'g(km)':>9}{'h(km)':>9}{'f(km)':>9}")
    for i, (node, g, h, f) in enumerate(trace, 1):
        print(f"  {i:<3}{str(node):<6}{g:>9.4f}{h:>9.4f}{f:>9.4f}")
    print("===============================================================")


if __name__ == "__main__":
    main()
