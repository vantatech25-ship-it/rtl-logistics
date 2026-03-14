"""
main.py — RTL Logistics KPI Dashboard
FastAPI backend serving real-time metrics from TimescaleDB + Routing Engine.
Pairs with the frontend dashboard.html for a full KPI view.
"""

import asyncio
import httpx
import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

TIMESCALE_DSN      = os.getenv("TIMESCALE_DSN", "postgresql://rtt_user:rtt_pass@localhost:5432/rtt_logistics")
ROUTING_ENGINE_URL = os.getenv("ROUTING_ENGINE_URL", "http://localhost:3000")


# ─── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await asyncpg.create_pool(TIMESCALE_DSN, min_size=2, max_size=10)
    yield
    await app.state.db.close()


app = FastAPI(title="RTL Logistics KPI Dashboard", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Serve frontend ───────────────────────────────────────────────────────────
@app.get("/")
async def serve_dashboard():
    return FileResponse("dashboard.html")


# ─── /api/kpis — top-level summary metrics ───────────────────────────────────
@app.get("/api/kpis")
async def get_kpis(request: None = None):
    """One-stop endpoint: hub count, avg congestion, route stats."""
    pool = app.state.db

    # Hub stats
    hub_stats = await pool.fetchrow("""
        SELECT
            COUNT(DISTINCT hub_id)                        AS total_hubs,
            AVG(congestion_score)                         AS avg_congestion,
            AVG(load_factor)                              AS avg_load,
            COUNT(*) FILTER (WHERE time > NOW() - INTERVAL '5 min') AS recent_readings
        FROM hub_telemetry
    """)

    # Route stats
    route_stats = await pool.fetchrow("""
        SELECT
            COUNT(*)                            AS total_routes,
            COUNT(*) FILTER (WHERE completed)   AS completed_routes,
            AVG(actual_cost - predicted_cost)
              FILTER (WHERE completed)          AS avg_cost_delta,
            AVG(gnn_confidence)                 AS avg_confidence
        FROM route_executions
        WHERE time > NOW() - INTERVAL '24 hours'
    """)

    # Live fleet size
    fleet = await pool.fetchval("""
        SELECT COUNT(DISTINCT vehicle_id) FROM vehicle_pings
        WHERE time > NOW() - INTERVAL '10 min'
    """)

    return {
        "hub_count":       hub_stats["total_hubs"] or 0,
        "avg_congestion":  round(float(hub_stats["avg_congestion"] or 0), 3),
        "avg_load":        round(float(hub_stats["avg_load"] or 0), 3),
        "recent_readings": hub_stats["recent_readings"] or 0,
        "routes_24h":      route_stats["total_routes"] or 0,
        "routes_completed":route_stats["completed_routes"] or 0,
        "avg_cost_delta":  round(float(route_stats["avg_cost_delta"] or 0), 2),
        "avg_confidence":  round(float(route_stats["avg_confidence"] or 0), 1),
        "live_vehicles":   fleet or 0,
    }


# ─── /api/hubs — per-hub congestion trend ────────────────────────────────────
@app.get("/api/hubs")
async def get_hub_scores():
    """Latest GNN scores per hub, proxied from the Rust engine."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{ROUTING_ENGINE_URL}/hubs", timeout=4.0)
            return r.json()
        except Exception:
            return {"success": False, "data": [], "error": "Routing engine unreachable"}


# ─── /api/congestion-trend — last 12h time-series ────────────────────────────
@app.get("/api/congestion-trend")
async def congestion_trend(hub_id: str = "JHB"):
    pool = app.state.db
    rows = await pool.fetch("""
        SELECT
            time_bucket('30 minutes', time) AS bucket,
            AVG(congestion_score)           AS avg_cong,
            AVG(load_factor)                AS avg_load
        FROM hub_telemetry
        WHERE hub_id = $1
          AND time > NOW() - INTERVAL '12 hours'
        GROUP BY bucket
        ORDER BY bucket ASC
    """, hub_id)
    return [{"time": str(r["bucket"]), "congestion": round(float(r["avg_cong"]),3), "load": round(float(r["avg_load"]),3)} for r in rows]


# ─── /api/routes/recent — last 20 dispatched routes ─────────────────────────
@app.get("/api/routes/recent")
async def recent_routes():
    pool = app.state.db
    rows = await pool.fetch("""
        SELECT route_id, from_hub_id, to_hub_id, predicted_cost,
               actual_cost, gnn_confidence, completed, time
        FROM route_executions
        ORDER BY time DESC LIMIT 20
    """)
    return [dict(r) for r in rows]


# ─── /api/fleet — live vehicle positions ─────────────────────────────────────
@app.get("/api/fleet")
async def fleet_positions():
    pool = app.state.db
    rows = await pool.fetch("""
        SELECT DISTINCT ON (vehicle_id)
            vehicle_id, hub_id, latitude, longitude, speed_kmh, heading_deg, time
        FROM vehicle_pings
        ORDER BY vehicle_id, time DESC
    """)
    return [dict(r) for r in rows]
