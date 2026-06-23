"""
hill_climb.py
=============
Hill Climbing local search used as a **post-processing refinement** of a route.

Algorithm (as specified)
-------------------------
* Initial state : an existing route (list of node IDs).
* Neighbourhood : for each intermediate node i, replace it with a fresh
                  A*-computed sub-path from node (i-1) to node (i+1).
* Move rule     : accept the neighbour only if it STRICTLY reduces total cost.
* Termination   : when no neighbour improves the route (a local optimum).
* Output        : (improved_path, final_cost, n_improvement_iterations).

Note on usage
-------------
If you seed Hill Climbing with a route that is already globally optimal under
the same cost function (e.g. the ML-weighted A* path), it will correctly report
0 improvements — that confirms optimality. To *demonstrate* the search reducing
cost, seed it with a sub-optimal route (e.g. the distance-only route refined
under the ML-aware cost), which is exactly what ``pipeline.py`` does.

Run ``python src/hill_climb.py`` for a self-contained demo.
"""

from __future__ import annotations

from astar import SimpleGraph, _min_parallel_edge_weight, astar


def route_cost(G, path, weight_attr: str = "weight") -> float:
    """Sum of edge weights along ``path``. Returns +inf for a broken path."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_dict = G[u].get(v)
        if edge_dict is None:
            return float("inf")
        total += _min_parallel_edge_weight(edge_dict, weight_attr)
    return total


def hill_climb(G, initial_path, astar_fn=astar, weight_attr: str = "weight",
               max_iters: int = 1000):
    """Refine ``initial_path`` via node-replacement hill climbing.

    Parameters
    ----------
    G            : graph (NetworkX MultiDiGraph or SimpleGraph)
    initial_path : starting route (list of node IDs)
    astar_fn     : A* function used to compute alternative sub-paths
    weight_attr  : edge attribute used as cost
    max_iters    : safety cap on total accepted moves

    Returns ``(path, cost, iterations)``.
    """
    current = list(initial_path)
    current_cost = route_cost(G, current, weight_attr)
    iterations = 0

    improved = True
    while improved and iterations < max_iters:
        improved = False
        # Try to bypass each intermediate node in turn.
        for i in range(1, len(current) - 1):
            prev_node, nxt_node = current[i - 1], current[i + 1]

            # Cost of the two edges we would replace: prev -> cur -> nxt
            old_segment_cost = route_cost(G, current[i - 1:i + 2], weight_attr)

            # Alternative sub-path prev -> ... -> nxt via A*.
            alt = astar_fn(G, prev_node, nxt_node, weight_attr=weight_attr)
            alt_path, alt_cost = alt[0], alt[1]
            if alt_path is None or len(alt_path) < 2:
                continue

            # Build the candidate route and only keep a STRICT improvement.
            candidate = current[:i - 1] + alt_path + current[i + 2:]
            candidate = _dedupe_consecutive(candidate)
            cand_cost = route_cost(G, candidate, weight_attr)

            if cand_cost < current_cost - 1e-9:
                current, current_cost = candidate, cand_cost
                iterations += 1
                improved = True
                break  # restart the sweep from a fresh, improved route

    return current, current_cost, iterations


def _dedupe_consecutive(path):
    """Remove accidental immediate repeats (e.g. [A, B, B, C] -> [A, B, C])."""
    out = [path[0]]
    for node in path[1:]:
        if node != out[-1]:
            out.append(node)
    return out


# --------------------------------------------------------------------------- #
# Self-contained demo
# --------------------------------------------------------------------------- #
def _build_demo_graph():
    """Graph with cheap shortcuts the seed route ignores.

    The seed A-P-B-Q-T takes expensive detours through P and Q, while cheaper
    direct edges A-B and B-T exist. Single-node replacement via A* can bypass
    P and then Q, so Hill Climbing demonstrably reduces cost. (No coordinates
    are set, so the heuristic is 0 and A* behaves as uniform-cost search.)
    """
    G = SimpleGraph()
    for n in ["A", "P", "B", "Q", "T"]:
        G.add_node(n)  # no coords -> admissible h = 0
    G.add_edge("A", "P", 5)
    G.add_edge("P", "B", 5)
    G.add_edge("A", "B", 3)   # cheap shortcut bypassing P
    G.add_edge("B", "Q", 5)
    G.add_edge("Q", "T", 5)
    G.add_edge("B", "T", 4)   # cheap shortcut bypassing Q
    return G, "A", "T"


def main() -> None:
    G, src, dst = _build_demo_graph()

    # Optimal route for reference.
    opt_path, opt_cost, _ = astar(G, src, dst)

    # Seed Hill Climbing with a deliberately sub-optimal route.
    seed = ["A", "P", "B", "Q", "T"]
    seed_cost = route_cost(G, seed)

    hc_path, hc_cost, iters = hill_climb(G, seed)

    print("========== HILL CLIMBING DEMO ==========")
    print(f"Seed route     : {seed}  cost={seed_cost:.4f}")
    print(f"Hill Climbing  : {hc_path}  cost={hc_cost:.4f}  iterations={iters}")
    print(f"A* optimum     : {opt_path}  cost={opt_cost:.4f}")

    reduction = 100.0 * (seed_cost - hc_cost) / seed_cost if seed_cost else 0.0
    print(f"\n[OK] Hill Climbing reduced cost by {reduction:.1f}% "
          f"(over {iters} improving move(s)).")
    assert hc_cost <= seed_cost + 1e-9, "Hill Climbing must never worsen the route!"

    # Seeding with the already-optimal A* path -> should report 0 improvements.
    _, _, opt_iters = hill_climb(G, opt_path)
    print(f"[OK] Seeded with optimal A* path -> {opt_iters} improvements "
          f"(confirms local optimum).")
    print("========================================")


if __name__ == "__main__":
    main()
