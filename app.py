"""
app.py  —  Streamlit interactive dashboard
==========================================
Delivery-person-first workflow for the Smart Delivery & Traffic Management
System.

What the delivery person sees
------------------------------
1. A list of pending orders near the depot (id, distance, predicted delay,
   risk) — generated from the ML-weighted road graph.
2. They SELECT one order to deliver.
3. The app shows that order's own route, then suggests OTHER pending orders
   that are cheap to combine into the same trip.
4. The delivery person ticks any extra orders they also want to take.
5. The app finds the best VISITING ORDER for everything picked (TSP
   heuristic) and draws the final multi-stop route on a map.

Advanced tools (dynamic re-route, raw A* vs Hill Climbing, ML explainability)
are kept further down for anyone who wants to dig into the algorithms.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# Make src/ importable.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import streamlit.components.v1 as components  # noqa: E402

from dynamic_reroute import make_reroute_map, simulate_and_reroute  # noqa: E402
from orders import (  # noqa: E402
    generate_order_pool,
    make_orders_map,
    plan_combined_route,
    suggest_combinable_orders,
)
from pipeline import (  # noqa: E402
    OUTPUTS_DIR,
    assign_ml_weights,
    load_graph,
    load_models,
    make_folium_map,
    nearest_node,
    run_route,
)
from preprocess import RISK_NAMES, load_processed  # noqa: E402

st.set_page_config(page_title="Smart Delivery & Traffic Management",
                   page_icon="🚚", layout="wide")

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Snow"]


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading graph + ML models...")
def load_everything():
    data = load_processed()
    xgb, rf = load_models()
    G = load_graph()
    return data, xgb, rf, G


def graph_bounds(G):
    ys = [float(d["y"]) for _, d in G.nodes(data=True) if "y" in d]
    xs = [float(d["x"]) for _, d in G.nodes(data=True) if "x" in d]
    return min(ys), max(ys), min(xs), max(xs)


def embed_map(path):
    """Embed a saved folium HTML map into the Streamlit page."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=520)
    else:
        st.info("No map available.")


# --------------------------------------------------------------------------- #
# Sidebar — depot location + weather scenario
# --------------------------------------------------------------------------- #
data, xgb, rf, G = load_everything()
ymin, ymax, xmin, xmax = graph_bounds(G)

st.sidebar.title("⚙️ Controls")

st.sidebar.subheader("Depot (your starting point)")
depot_mode = st.sidebar.radio("Pick depot by", ["Address", "Lat/Lon"], horizontal=True)
if depot_mode == "Address":
    import osmnx as ox

    depot_address = st.sidebar.text_input("Depot address", "Majestic Bus Stand, Bengaluru, India")
    if st.sidebar.button("📍 Locate depot") or "depot_latlon" not in st.session_state:
        try:
            d_lat, d_lon = ox.geocode(depot_address)
            st.session_state.depot_latlon = (d_lat, d_lon)
        except Exception:
            st.session_state.depot_latlon = ((ymin + ymax) / 2, (xmin + xmax) / 2)
    depot_lat, depot_lon = st.session_state.depot_latlon
else:
    depot_lat = st.sidebar.number_input("Depot lat", value=round((ymin + ymax) / 2, 5), format="%.5f")
    depot_lon = st.sidebar.number_input("Depot lon", value=round((xmin + xmax) / 2, 5), format="%.5f")

depot_node = nearest_node(G, depot_lon, depot_lat)

st.sidebar.markdown("---")
st.sidebar.subheader("Weather scenario")
st.sidebar.caption("Re-weights every road via the ML models.")
weather = st.sidebar.selectbox("Weather condition", WEATHER_OPTIONS, index=0)
daynight = st.sidebar.radio("Time of day", ["Day", "Night"], horizontal=True)
visibility = st.sidebar.slider("Visibility (mi)", 0.5, 10.0, 9.0, 0.5)
precipitation = st.sidebar.slider("Precipitation (in)", 0.0, 0.5, 0.0, 0.01)

scenario = {
    "Weather_Condition": weather,
    "Sunrise_Sunset": daynight,
    "Visibility(mi)": visibility,
    "Precipitation(in)": precipitation,
}

scenario_key = (weather, daynight, round(visibility, 2), round(precipitation, 3))
if st.session_state.get("weighted_for") != scenario_key:
    with st.spinner("Re-weighting road graph with the ML models..."):
        assign_ml_weights(G, xgb, rf, data, scenario=scenario)
    st.session_state.weighted_for = scenario_key
    # Any previously generated order pool was costed under the old weather —
    # invalidate it so distances/costs stay correct.
    st.session_state.pop("order_pool", None)

st.sidebar.markdown("---")
n_pool = st.sidebar.slider("Pending orders to simulate", 5, 25, 10)
if st.sidebar.button("🔄 Refresh pending orders") or "order_pool" not in st.session_state:
    st.session_state.order_pool = generate_order_pool(G, depot_node, n_orders=n_pool)
    st.session_state.pop("selected_order_id", None)
    st.session_state.pop("chosen_extra_ids", None)

pool = st.session_state.order_pool

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚚 Smart Delivery & Traffic Management System")
st.caption("Pick an order. We'll show what else you can fit on the same trip, "
           "and the best order to visit everything — explainably, not as a black box.")
if data.get("is_synthetic"):
    st.warning("Models were trained on a **synthetic** sample (real US Accidents "
               "CSV not found). Download it for production-quality predictions.")

tab_orders, tab_reroute, tab_advanced, tab_ml = st.tabs(
    ["📋 My orders", "⚡ Dynamic re-route", "🛠️ Advanced: raw route tools", "📊 ML explainability"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — My orders (the main delivery-person workflow)
# --------------------------------------------------------------------------- #
with tab_orders:
    st.subheader("Step 1 — Pending orders near your depot")
    order_rows = [{
        "Order": o.order_id,
        "Distance (km)": o.distance_km,
        "Route cost": o.cost,
        "Predicted delay": o.delay,
        "High-risk edges": o.risk_high,
    } for o in pool]
    st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

    st.subheader("Step 2 — Select the order you want to deliver")
    order_ids = [o.order_id for o in pool]
    selected_id = st.selectbox("Your order", order_ids, key="selected_order_id")
    selected = next(o for o in pool if o.order_id == selected_id)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance (km)", f"{selected.distance_km:.2f}")
    m2.metric("Route cost", f"{selected.cost:.2f}")
    m3.metric("Predicted delay", f"{selected.delay:.2f}")
    m4.metric("High-risk edges", selected.risk_high)

    st.subheader("Step 3 — Other orders you can fit on the same trip")
    max_extra_pct = st.slider("Max acceptable extra cost to combine (%)", 5, 60, 25)
    suggestions = suggest_combinable_orders(
        G, depot_node, selected, pool, max_extra_pct=max_extra_pct
    )

    chosen_extra_ids: list[str] = []
    if suggestions:
        st.write("Tick any you also want to deliver on this trip:")
        for s in suggestions:
            o = s["order"]
            label = (f"{o.order_id} — +{s['extra_cost']:.2f} cost "
                     f"(+{s['extra_pct']:.1f}%), {o.distance_km:.2f} km, "
                     f"{o.risk_high} high-risk edges")
            if st.checkbox(label, key=f"extra_{o.order_id}"):
                chosen_extra_ids.append(o.order_id)
    else:
        st.info("No other pending order is cheap enough to combine at this threshold. "
                 "Try raising the slider above.")

    st.subheader("Step 4 — Best visiting order & final route")
    if st.button("🧭 Plan my trip"):
        chosen_orders = [selected] + [o for o in pool if o.order_id in chosen_extra_ids]
        result = plan_combined_route(G, depot_node, chosen_orders)

        c1, c2 = st.columns(2)
        c1.metric("Optimised visiting-order cost", f"{result['opt_cost']:.2f}")
        c2.metric("As-picked order cost", f"{result['naive_cost']:.2f}",
                  delta=f"-{result['naive_cost'] - result['opt_cost']:.2f}",
                  delta_color="inverse")

        st.write("**Visit in this order:** Depot → " +
                 " → ".join(result["visiting_order_ids"]))

        path = make_orders_map(G, depot_node, result,
                                out_path=os.path.join(OUTPUTS_DIR, "app_orders_map.html"))
        embed_map(path)

# --------------------------------------------------------------------------- #
# Tab 2 — Dynamic re-route
# --------------------------------------------------------------------------- #
with tab_reroute:
    st.subheader("Live disruption → automatic re-route")
    st.write("Simulates a risk spike on the next planned road segment for your "
             "currently selected order, and re-optimises from the vehicle's "
             "current position.")
    if st.button("⚡ Simulate disruption", key="run_reroute"):
        sel = next(o for o in pool if o.order_id == st.session_state.get("selected_order_id"))
        try:
            res = simulate_and_reroute(G, depot_node, sel.drop)
            c1, c2, c3 = st.columns(3)
            c1.metric("Original route cost", f"{res['original_cost']:.2f}")
            c2.metric("If we ignore it", f"{res['remaining_after_blocked']:.2f}",
                      delta="blocked road")
            c3.metric("Re-optimised tail", f"{res['new_tail_cost']:.2f}",
                      delta=f"saved {res['remaining_after_blocked'] - res['new_tail_cost']:.2f}",
                      delta_color="inverse")
            st.info(f"Disruption on edge {res['blocked'][0]} → {res['blocked'][1]}; "
                    f"avoided: {'YES' if res['avoided_block'] else 'no alternative'}.")
            path = make_reroute_map(G, res,
                                    out_path=os.path.join(OUTPUTS_DIR, "app_reroute_map.html"))
            embed_map(path)
        except ValueError as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- #
# Tab 3 — Advanced: raw route tools (the original single-pair A*/HC demo)
# --------------------------------------------------------------------------- #
with tab_advanced:
    st.subheader("A* + Hill Climbing on any two points (for algorithm demos)")
    c1, c2 = st.columns(2)
    src_lat = c1.number_input("Src lat", value=depot_lat, format="%.5f")
    src_lon = c2.number_input("Src lon", value=depot_lon, format="%.5f")
    dst_lat = c1.number_input("Dst lat", value=round(depot_lat + 0.03, 5), format="%.5f")
    dst_lon = c2.number_input("Dst lon", value=round(depot_lon + 0.03, 5), format="%.5f")

    if st.button("▶️ Compute route", key="run_route"):
        src_node = nearest_node(G, src_lon, src_lat)
        dst_node = nearest_node(G, dst_lon, dst_lat)
        if src_node == dst_node:
            st.error("Source and destination are the same node.")
        else:
            res = run_route(G, src_node, dst_node, trace=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Distance (km)", f"{res['ml']['distance_km']:.2f}")
            m2.metric("ML route cost", f"{res['ml']['cost']:.2f}")
            m3.metric("A* nodes expanded", res["ml"]["expanded"],
                      delta=f"{res['ml']['expanded'] - res['ml']['dij_expanded']} vs Dijkstra")
            m4.metric("High-risk edges",
                      res["ml"]["risk_counts"][2],
                      delta=f"{res['ml']['risk_counts'][2] - res['dist']['risk_counts'][2]} vs distance route",
                      delta_color="inverse")

            rows = []
            for name, v in [("Shortest distance", res["dist"]),
                            ("ML-weighted A*", res["ml"]),
                            ("ML A* + Hill Climbing", res["hc"])]:
                rows.append({
                    "Variant": name,
                    "Distance (km)": round(v["distance_km"], 2),
                    "Route cost": round(v["cost"], 2),
                    "Total delay": round(v["delay"], 2),
                    "Low": v["risk_counts"][0],
                    "Medium": v["risk_counts"][1],
                    "High": v["risk_counts"][2],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            path = make_folium_map(G, res, out_path=os.path.join(OUTPUTS_DIR, "app_route_map.html"))
            embed_map(path)

            if res["trace"]:
                with st.expander("🔍 A* node-expansion trace (f = g + h)"):
                    tr = pd.DataFrame(res["trace"][:30],
                                      columns=["node", "g", "h", "f"]).round(3)
                    st.dataframe(tr, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 4 — ML explainability
# --------------------------------------------------------------------------- #
with tab_ml:
    st.subheader("Model explainability")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost — feature importance (delay)**")
        p = os.path.join(OUTPUTS_DIR, "xgb_importance.png")
        st.image(p) if os.path.exists(p) else st.info("Run train_xgboost.py first.")
        p = os.path.join(OUTPUTS_DIR, "shap_xgboost.png")
        if os.path.exists(p):
            st.markdown("**SHAP summary**")
            st.image(p)
    with col2:
        st.markdown("**Random Forest — confusion matrix (risk)**")
        p = os.path.join(OUTPUTS_DIR, "rf_confusion_matrix.png")
        st.image(p) if os.path.exists(p) else st.info("Run train_rf.py first.")

    st.markdown("---")
    st.markdown(f"**Risk classes:** {', '.join(RISK_NAMES)} &nbsp;|&nbsp; "
                "**Edge weight** = length_km + delay×0.5 + risk_penalty×0.3")