"""
server.py  —  FastAPI backend for the React frontend
====================================================
Wraps the existing Smart Delivery & Traffic Management ``src/`` code (A*, Hill
Climbing, multi-stop TSP, dynamic re-route, XGBoost delay + Random Forest risk
models) as a JSON REST API so a React single-page app can drive it.

This replaces the role Streamlit (``app.py``) used to play: ``app.py`` imported
the same ``src/`` functions and rendered the results itself. Here we instead
return plain JSON (route coordinates, metrics, tables) and let react-leaflet do
the drawing on the client. None of the algorithm code in ``src/`` is changed.

The graph + ML models are loaded ONCE at import (mirroring Streamlit's
``@st.cache_resource``). Per-client state (depot, generated order pool) is kept
in a small in-memory session store keyed by an ``X-Session-Id`` header the React
client sends. The ML edge-weighting mutates the shared graph in place, so all
graph-touching work is serialised behind a single lock — fine for this demo
(Streamlit was single-user too).

Run:  uvicorn server:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make src/ importable (same trick as app.py).
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dynamic_reroute import simulate_and_reroute  # noqa: E402
from orders import (  # noqa: E402
    generate_order_pool,
    plan_combined_route,
    suggest_combinable_orders,
)
from astar import astar, dijkstra  # noqa: E402
from pipeline import (  # noqa: E402
    OUTPUTS_DIR,
    RISK_COLOR,
    assign_ml_weights,
    load_graph,
    load_models,
    nearest_node,
    run_route,
    summarise_route,
)
from preprocess import RISK_NAMES, load_processed  # noqa: E402

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Snow"]
ML_IMAGES = {
    "xgb_importance": "xgb_importance.png",
    "shap_xgboost": "shap_xgboost.png",
    "rf_confusion_matrix": "rf_confusion_matrix.png",
}

# --------------------------------------------------------------------------- #
# Load graph + models once (heavy — mirrors @st.cache_resource).
# --------------------------------------------------------------------------- #
print("[server] Loading processed data, ML models and road graph...")
DATA, XGB, RF = (lambda: (load_processed(), *load_models()))()
G = load_graph()
print("[server] Ready.")

# A single lock guards every graph-mutating / routing operation, because
# assign_ml_weights() rewrites edge weights on the shared graph in place.
LOCK = threading.Lock()

# Process-global weather weighting (last-writer-wins, like a single Streamlit
# session). Per-client depot + order pool live in SESSIONS.
GRAPH_WEIGHTED_FOR: Optional[tuple] = None
SESSIONS: dict[str, dict] = {}


def _session(sid: Optional[str]) -> dict:
    key = sid or "_global"
    return SESSIONS.setdefault(key, {"depot_node": None, "depot_latlon": None,
                                     "order_pool": None})


def graph_bounds(g):
    ys = [float(d["y"]) for _, d in g.nodes(data=True) if "y" in d]
    xs = [float(d["x"]) for _, d in g.nodes(data=True) if "x" in d]
    return min(ys), max(ys), min(xs), max(xs)


def path_coords(path) -> list[list[float]]:
    """Node-id path -> [[lat, lon], ...] (the conversion folium used)."""
    return [[float(G.nodes[n]["y"]), float(G.nodes[n]["x"])] for n in path]


def node_latlon(n) -> list[float]:
    return [float(G.nodes[n]["y"]), float(G.nodes[n]["x"])]


def _subsample_even(seq: list, max_items: int) -> list:
    """Evenly pick at most ``max_items`` from ``seq``, preserving order and always
    keeping the first and last element. Unlike a head-truncation this keeps the
    overall *shape* of a long A* expansion (start -> finish) when compressed."""
    n = len(seq)
    if max_items <= 0 or n <= max_items:
        return list(seq)
    # Pick indices spread across [0, n-1] inclusive of both ends.
    idxs = {round(i * (n - 1) / (max_items - 1)) for i in range(max_items)}
    return [seq[i] for i in sorted(idxs)]


def expansion_overlay(trace: list, path: list, max_nodes: int) -> dict:
    """Turn a raw A* ``trace`` (ordered (node, g, h, f) tuples) into the payload
    the animated overlay needs: each expanded node's lat/lon + g/h/f + the order
    in which A* expanded it, evenly subsampled to ``max_nodes``."""
    sampled = _subsample_even(list(enumerate(trace)), max_nodes)
    nodes = [
        {
            "order": order,                       # original expansion index
            "latlon": node_latlon(node),
            "g": round(g, 3),
            "h": round(h, 3),
            "f": round(f, 3),
        }
        for order, (node, g, h, f) in sampled
    ]
    return {
        "total_expanded": len(trace),
        "sampled": len(nodes),
        "nodes": nodes,
        "route": path_coords(path),
    }


# --------------------------------------------------------------------------- #
# Reverse geocoding (lat/lon -> human address) via Nominatim — the same service
# osmnx uses. Cached in-memory by rounded lat/lon so we never hit the geocoder
# twice for the same point. Falls back to the raw coordinate on any failure so
# a slow/blocked lookup never breaks an order row.
# --------------------------------------------------------------------------- #
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "SmartDelivery/1.0 (course demo project)"}
_geocode_cache: dict[tuple, str] = {}
_geocode_lock = threading.Lock()


def reverse_geocode(lat: float, lon: float, timeout: float = 4.0) -> str:
    """Return a short human address for (lat, lon), cached by rounded coords."""
    key = (round(float(lat), 4), round(float(lon), 4))
    with _geocode_lock:
        if key in _geocode_cache:
            return _geocode_cache[key]

    address = None
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"format": "jsonv2", "lat": key[0], "lon": key[1], "zoom": 17},
            headers=NOMINATIM_HEADERS,
            timeout=timeout,
        )
        if resp.ok:
            data = resp.json()
            addr = data.get("address", {})
            # Build a compact "place, locality" label rather than the full string.
            parts = [
                addr.get("road")
                or addr.get("neighbourhood")
                or addr.get("suburb")
                or data.get("name"),
                addr.get("suburb") or addr.get("city_district") or addr.get("city")
                or addr.get("town") or addr.get("village"),
            ]
            parts = [p for p in parts if p]
            address = ", ".join(dict.fromkeys(parts)) or data.get("display_name")
    except Exception:  # noqa: BLE001 — any geocoder failure -> fall back to coords
        address = None

    # Fallback: raw lat/lon rounded to 4 decimals (never break the row).
    result = address or f"{key[0]:.4f}, {key[1]:.4f}"
    with _geocode_lock:
        _geocode_cache[key] = result
    return result


# --------------------------------------------------------------------------- #
# Live weather (Open-Meteo, no API key) -> existing scenario categories.
# Fetched for the depot's own lat/lon so the weather + day/night reflect the
# location the user actually picked (current location or a point on the map),
# not a fixed city. Cached per rounded coordinate for ~10 minutes to avoid
# hammering the API.
# --------------------------------------------------------------------------- #
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BLR_LAT, BLR_LON = 12.9716, 77.5946  # graph-centre fallback only
_live_weather_cache: dict[tuple, dict] = {}  # rounded (lat, lon) -> {ts, data}
_live_weather_lock = threading.Lock()
LIVE_WEATHER_TTL = 600  # seconds


def _wmo_to_category(code: int) -> str:
    """Map an Open-Meteo WMO weather code to a model weather category."""
    if code in (0, 1):
        return "Clear"
    if code in (2, 3):
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snow"
    # 51-67 drizzle/rain, 80-82 rain showers, 95-99 thunderstorm
    return "Rain"


def _approx_visibility_mi(category: str, precip_in: float) -> float:
    """Open-Meteo's *current* block doesn't expose visibility, so we APPROXIMATE
    it from the mapped weather category and precipitation. This is a rough
    stand-in (clearly not a measured value) kept inside the scenario's 0.5–10 mi
    range that the ML re-weighting expects."""
    base = {"Clear": 9.5, "Cloudy": 8.0, "Rain": 4.0, "Fog": 1.0, "Snow": 2.0}.get(category, 8.0)
    base -= min(3.0, precip_in * 20.0)  # heavier rain -> lower visibility
    return round(max(0.5, min(10.0, base)) * 2) / 2  # snap to 0.5 steps


def fetch_live_weather(lat: float = BLR_LAT, lon: float = BLR_LON) -> dict:
    """Fetch the current conditions at (lat, lon) and map them onto the scenario."""
    now = time.time()
    key = (round(float(lat), 2), round(float(lon), 2))
    with _live_weather_lock:
        entry = _live_weather_cache.get(key)
        if entry and (now - entry["ts"]) < LIVE_WEATHER_TTL:
            return {**entry["data"], "cached": True}

    resp = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": key[0],
            "longitude": key[1],
            "current": "temperature_2m,precipitation,weather_code,cloud_cover",
            "daily": "sunrise,sunset",
            "timezone": "auto",
        },
        timeout=8,
    )
    resp.raise_for_status()
    j = resp.json()
    cur = j.get("current", {})
    daily = j.get("daily", {})

    code = int(cur.get("weather_code", 1))
    category = _wmo_to_category(code)
    temp_c = float(cur.get("temperature_2m", 0.0))
    precip_mm = float(cur.get("precipitation", 0.0))
    precip_in = round(min(0.5, precip_mm / 25.4), 2)  # mm -> in, clamp to slider range
    visibility_mi = _approx_visibility_mi(category, precip_in)

    # Day/night from current local time vs today's sunrise/sunset (timezone=auto).
    daynight = "Day"
    try:
        now_t = datetime.fromisoformat(cur["time"])
        sunrise = datetime.fromisoformat(daily["sunrise"][0])
        sunset = datetime.fromisoformat(daily["sunset"][0])
        daynight = "Day" if sunrise <= now_t <= sunset else "Night"
    except Exception:  # noqa: BLE001
        pass

    summary = (f"Live @ {key[0]:.3f}, {key[1]:.3f}: {category}, {round(temp_c)}°C, "
               f"{precip_in:.2f} in rain, {daynight} — as of just now")
    data = {
        "weather": category,
        "daynight": daynight,
        "visibility_mi": visibility_mi,
        "precipitation_in": precip_in,
        "temperature_c": round(temp_c, 1),
        "weather_code": code,
        "summary": summary,
        "as_of": cur.get("time"),
        "cached": False,
    }
    with _live_weather_lock:
        _live_weather_cache[key] = {"ts": now, "data": data}
    return data


def _ensure_depot(sess: dict):
    """Default the depot to the graph centre if the client never set one."""
    if sess["depot_node"] is None:
        ymin, ymax, xmin, xmax = graph_bounds(G)
        lat, lon = (ymin + ymax) / 2, (xmin + xmax) / 2
        sess["depot_latlon"] = [lat, lon]
        sess["depot_node"] = nearest_node(G, lon, lat)
    return sess["depot_node"]


def _ensure_pool(sess: dict):
    pool = sess.get("order_pool")
    if not pool:
        raise HTTPException(status_code=409,
                            detail="No order pool yet. Call /api/orders first.")
    return pool


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="Smart Delivery & Traffic Management API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class DepotReq(BaseModel):
    mode: str = "Lat/Lon"          # "Address" | "Lat/Lon"
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ScenarioReq(BaseModel):
    weather: str = "Clear"
    daynight: str = "Day"
    visibility: float = 9.0
    precipitation: float = 0.0


class OrdersReq(BaseModel):
    n_pool: int = 10


class SuggestReq(BaseModel):
    selected_id: str
    max_extra_pct: float = 25.0


class PlanReq(BaseModel):
    selected_id: str
    extra_ids: list[str] = []
    max_trace_nodes: int = 2000   # cap for the animated A* expansion overlay


class RerouteReq(BaseModel):
    selected_id: str


class RouteReq(BaseModel):
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    trace: bool = True


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/meta")
def meta():
    ymin, ymax, xmin, xmax = graph_bounds(G)
    return {
        "bounds": {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax},
        "center": [(ymin + ymax) / 2, (xmin + xmax) / 2],
        "weather_options": WEATHER_OPTIONS,
        "risk_names": RISK_NAMES,
        "is_synthetic": bool(DATA.get("is_synthetic")),
        "weighted": GRAPH_WEIGHTED_FOR is not None,
    }


@app.get("/api/live-weather")
def live_weather(lat: float = BLR_LAT, lon: float = BLR_LON):
    """Current conditions at (lat, lon) from Open-Meteo, mapped onto the scenario.
    The client passes the chosen depot location so the weather + day/night follow
    wherever the user is, then applies it via the existing /api/scenario flow."""
    try:
        return fetch_live_weather(lat, lon)
    except Exception as exc:  # noqa: BLE001 — never crash; let the UI fall back
        raise HTTPException(status_code=502,
                            detail=f"Live weather unavailable: {exc}")


@app.post("/api/depot")
def set_depot(req: DepotReq, x_session_id: Optional[str] = Header(default=None)):
    sess = _session(x_session_id)
    ymin, ymax, xmin, xmax = graph_bounds(G)
    if req.mode == "Address" and req.address:
        try:
            import osmnx as ox
            lat, lon = ox.geocode(req.address)
        except Exception:
            lat, lon = (ymin + ymax) / 2, (xmin + xmax) / 2
    else:
        lat = req.lat if req.lat is not None else (ymin + ymax) / 2
        lon = req.lon if req.lon is not None else (xmin + xmax) / 2
    with LOCK:
        node = nearest_node(G, lon, lat)
    sess["depot_latlon"] = [lat, lon]
    sess["depot_node"] = node
    return {"lat": lat, "lon": lon, "node": str(node)}


@app.post("/api/scenario")
def set_scenario(req: ScenarioReq, x_session_id: Optional[str] = Header(default=None)):
    global GRAPH_WEIGHTED_FOR
    scenario = {
        "Weather_Condition": req.weather,
        "Sunrise_Sunset": req.daynight,
        "Visibility(mi)": req.visibility,
        "Precipitation(in)": req.precipitation,
    }
    key = (req.weather, req.daynight, round(req.visibility, 2), round(req.precipitation, 3))
    high_risk = None
    with LOCK:
        if GRAPH_WEIGHTED_FOR != key:
            assign_ml_weights(G, XGB, RF, DATA, scenario=scenario)
            GRAPH_WEIGHTED_FOR = key
            # Pools were costed under the old weather — invalidate every one.
            for s in SESSIONS.values():
                s["order_pool"] = None
            high_risk = sum(1 for _, _, a in G.edges(keys=False, data=True)
                            if int(a.get("risk", 0)) == 2)
    return {"ok": True, "weighted_for": list(key), "high_risk_edges": high_risk}


def _order_dto(o, from_address: Optional[str] = None, to_address: Optional[str] = None):
    dto = {
        "order_id": o.order_id,
        "distance_km": o.distance_km,
        "cost": o.cost,
        "delay": o.delay,
        "risk_high": o.risk_high,
        "latlon": node_latlon(o.drop),
    }
    if from_address is not None:
        dto["from_address"] = from_address
    if to_address is not None:
        dto["to_address"] = to_address
    return dto


def _route_summary_row(label: str, summary: dict, cost: float):
    return {
        "variant": label,
        "distance_km": round(summary["distance_km"], 2),
        "cost": round(cost, 2),
        "delay": round(summary["delay"], 2),
        "low": summary["risk_counts"][0],
        "medium": summary["risk_counts"][1],
        "high": summary["risk_counts"][2],
    }


def _plan_technical(result: dict, max_trace_nodes: int = 2000):
    opt_summary = summarise_route(G, result["full_path"])
    naive_summary = summarise_route(G, result["naive_path"])

    expanded = 0
    dij_expanded = 0
    trace_rows = []
    expansion = None
    stops = result["stops"]
    for leg_idx, (a, b) in enumerate(zip(result["opt_order"][:-1], result["opt_order"][1:])):
        src, dst = stops[a], stops[b]
        if leg_idx == 0:
            leg_path, _, exp, trace = astar(
                G, src, dst, weight_attr="weight", return_trace=True)
            trace_rows = [
                {"node": str(n), "g": round(g, 3), "h": round(h, 3), "f": round(f, 3)}
                for (n, g, h, f) in trace[:30]
            ]
            # Full (subsampled) expansion with coordinates for the animated map.
            expansion = expansion_overlay(trace, leg_path or [], max_trace_nodes)
            expansion["src"] = node_latlon(src)
            expansion["dst"] = node_latlon(dst)
        else:
            _, _, exp = astar(G, src, dst, weight_attr="weight")
        expanded += exp
        _, _, dij_exp = dijkstra(G, src, dst, weight_attr="weight")
        dij_expanded += dij_exp

    return {
        "metrics": {
            "distance_km": round(opt_summary["distance_km"], 2),
            "cost": round(result["opt_cost"], 2),
            "expanded": expanded,
            "dij_expanded": dij_expanded,
            "expanded_delta": expanded - dij_expanded,
            "high_risk": opt_summary["risk_counts"][2],
            "high_risk_delta": (
                opt_summary["risk_counts"][2] - naive_summary["risk_counts"][2]
            ),
        },
        "variants": [
            _route_summary_row("As-picked order", naive_summary, result["naive_cost"]),
            _route_summary_row("Optimised visiting order", opt_summary, result["opt_cost"]),
        ],
        "trace": trace_rows,
        "expansion": expansion,
    }


@app.post("/api/orders")
def make_orders(req: OrdersReq, x_session_id: Optional[str] = Header(default=None)):
    if GRAPH_WEIGHTED_FOR is None:
        raise HTTPException(status_code=409,
                            detail="Set a weather scenario first (POST /api/scenario).")
    sess = _session(x_session_id)
    with LOCK:
        depot = _ensure_depot(sess)
        pool = generate_order_pool(G, depot, n_orders=req.n_pool)
    sess["order_pool"] = pool

    # Reverse-geocode the depot once (the "From" for every row) and each order's
    # drop point (its "To"). Lookups are cached; failures fall back to coords.
    depot_lat, depot_lon = sess["depot_latlon"]
    from_address = reverse_geocode(depot_lat, depot_lon)
    orders = []
    for o in pool:
        d_lat, d_lon = node_latlon(o.drop)
        orders.append(_order_dto(o, from_address=from_address,
                                 to_address=reverse_geocode(d_lat, d_lon)))

    return {
        "depot": {"node": str(depot), "latlon": sess["depot_latlon"],
                  "address": from_address},
        "from_address": from_address,
        "orders": orders,
    }


@app.post("/api/orders/suggest")
def suggest(req: SuggestReq, x_session_id: Optional[str] = Header(default=None)):
    sess = _session(x_session_id)
    pool = _ensure_pool(sess)
    selected = next((o for o in pool if o.order_id == req.selected_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown order id.")
    with LOCK:
        depot = _ensure_depot(sess)
        suggestions = suggest_combinable_orders(
            G, depot, selected, pool, max_extra_pct=req.max_extra_pct)
    return {
        "selected": _order_dto(selected),
        "suggestions": [{
            "order": _order_dto(s["order"]),
            "extra_cost": s["extra_cost"],
            "extra_pct": s["extra_pct"],
        } for s in suggestions],
    }


@app.post("/api/orders/plan")
def plan(req: PlanReq, x_session_id: Optional[str] = Header(default=None)):
    sess = _session(x_session_id)
    pool = _ensure_pool(sess)
    selected = next((o for o in pool if o.order_id == req.selected_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown order id.")
    chosen = [selected] + [o for o in pool if o.order_id in req.extra_ids]
    with LOCK:
        depot = _ensure_depot(sess)
        result = plan_combined_route(G, depot, chosen)
        technical = _plan_technical(result, req.max_trace_nodes)

    # Draw exactly two route lines on the map:
    #   1) the normal/as-picked route through the selected orders
    #   2) the optimised visiting-order route
    # This keeps the map focused on the comparison the delivery person needs.
    polylines = [
        {
            "positions": path_coords(result["naive_path"]),
            "color": "#64748b",
            "weight": 5,
            "opacity": 0.65,
            "dashArray": "8",
            "label": "Normal route",
        },
        {
            "positions": path_coords(result["full_path"]),
            "color": "#16a34a",
            "weight": 6,
            "opacity": 0.9,
            "label": "Optimised route",
        },
    ]

    markers = [{"position": sess["depot_latlon"], "label": "Depot (start)", "kind": "depot"}]
    for visit_idx, stop_idx in enumerate(result["opt_order"][1:], start=1):
        order = result["chosen_orders"][stop_idx - 1]
        markers.append({
            "position": node_latlon(order.drop),
            "label": f"Stop {visit_idx}: {order.order_id}",
            "kind": "stop",
        })

    return {
        "opt_cost": round(result["opt_cost"], 2),
        "naive_cost": round(result["naive_cost"], 2),
        "saved": round(result["naive_cost"] - result["opt_cost"], 2),
        "visiting_order_ids": result["visiting_order_ids"],
        "polylines": polylines,
        "markers": markers,
        "technical": technical,
    }


@app.post("/api/reroute")
def reroute(req: RerouteReq, x_session_id: Optional[str] = Header(default=None)):
    sess = _session(x_session_id)
    pool = _ensure_pool(sess)
    selected = next((o for o in pool if o.order_id == req.selected_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Unknown order id.")
    with LOCK:
        depot = _ensure_depot(sess)
        try:
            res = simulate_and_reroute(G, depot, selected.drop)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    bu, bv = res["blocked"]
    polylines = [
        {"positions": path_coords(res["original_path"]), "color": "blue",
         "weight": 5, "opacity": 0.6, "label": "Original plan"},
        {"positions": path_coords(res["new_full"]), "color": "green",
         "weight": 4, "opacity": 0.9, "dashArray": "6", "label": "Re-optimised route"},
        {"positions": path_coords([bu, bv]), "color": "red",
         "weight": 8, "opacity": 0.9, "label": "DISRUPTION (risk spike)"},
    ]
    markers = [
        {"position": node_latlon(res["current_node"]), "label": "Vehicle position", "kind": "vehicle"},
        {"position": node_latlon(res["src"]), "label": "Source", "kind": "src"},
        {"position": node_latlon(res["dst"]), "label": "Destination", "kind": "dst"},
    ]
    return {
        "original_cost": round(res["original_cost"], 2),
        "remaining_after_blocked": round(res["remaining_after_blocked"], 2),
        "new_tail_cost": round(res["new_tail_cost"], 2),
        "saved": round(res["remaining_after_blocked"] - res["new_tail_cost"], 2),
        "blocked": [str(bu), str(bv)],
        "avoided_block": bool(res["avoided_block"]),
        "polylines": polylines,
        "markers": markers,
    }


@app.post("/api/route")
def route(req: RouteReq):
    with LOCK:
        if GRAPH_WEIGHTED_FOR is None:
            assign_ml_weights(G, XGB, RF, DATA)
        src_node = nearest_node(G, req.src_lon, req.src_lat)
        dst_node = nearest_node(G, req.dst_lon, req.dst_lat)
        if src_node == dst_node:
            raise HTTPException(status_code=400,
                                detail="Source and destination are the same node.")
        res = run_route(G, src_node, dst_node, trace=req.trace)

    ml, dist, hc = res["ml"], res["dist"], res["hc"]

    # Risk-coloured dots along the ML route (mirrors make_folium_map loop).
    ml_path = ml["path"]
    risk_dots = []
    for u, v in zip(ml_path[:-1], ml_path[1:]):
        attrs = min(G[u][v].values(), key=lambda a: a.get("weight", float("inf")))
        risk_dots.append({
            "position": node_latlon(u),
            "color": RISK_COLOR[int(attrs.get("risk", 0))],
        })

    variants = []
    for name, v in [("Shortest distance", dist), ("ML-weighted A*", ml),
                    ("ML A* + Hill Climbing", hc)]:
        variants.append({
            "variant": name,
            "distance_km": round(v["distance_km"], 2),
            "cost": round(v["cost"], 2),
            "delay": round(v["delay"], 2),
            "low": v["risk_counts"][0],
            "medium": v["risk_counts"][1],
            "high": v["risk_counts"][2],
        })

    trace_rows = [
        {"node": str(n), "g": round(g, 3), "h": round(h, 3), "f": round(f, 3)}
        for (n, g, h, f) in (res["trace"][:30] if res["trace"] else [])
    ]

    return {
        "metrics": {
            "distance_km": round(ml["distance_km"], 2),
            "cost": round(ml["cost"], 2),
            "expanded": ml["expanded"],
            "dij_expanded": ml["dij_expanded"],
            "expanded_delta": ml["expanded"] - ml["dij_expanded"],
            "high_risk": ml["risk_counts"][2],
            "high_risk_delta": ml["risk_counts"][2] - dist["risk_counts"][2],
        },
        "variants": variants,
        "polylines": [
            {"positions": path_coords(ml["path"]), "color": "blue",
             "weight": 6, "opacity": 0.75, "label": "ML-weighted A* route"},
            {"positions": path_coords(hc["path"]), "color": "green",
             "weight": 4, "opacity": 0.9, "dashArray": "8", "label": "A* + Hill Climbing"},
        ],
        "risk_dots": risk_dots,
        "markers": [
            {"position": node_latlon(src_node), "label": "Source", "kind": "src"},
            {"position": node_latlon(dst_node), "label": "Destination", "kind": "dst"},
        ],
        "trace": trace_rows,
    }


@app.get("/api/ml-status")
def ml_status():
    return {name: os.path.exists(os.path.join(OUTPUTS_DIR, fname))
            for name, fname in ML_IMAGES.items()}


@app.get("/api/ml-image/{name}")
def ml_image(name: str):
    fname = ML_IMAGES.get(name)
    if not fname:
        raise HTTPException(status_code=404, detail="Unknown image.")
    path = os.path.join(OUTPUTS_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not generated yet.")
    return FileResponse(path, media_type="image/png")
