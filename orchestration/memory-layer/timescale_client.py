"""
timescale_client.py
TimescaleDB time-series client for RTL Logistics.
Records and queries live telemetry: GPS pings, hub load, traffic conditions.
"""

import asyncpg
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from config import TIMESCALE_DSN


# ---------------------------------------------------------------------------
# Schema Bootstrap
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Hub telemetry: load, congestion readings over time
CREATE TABLE IF NOT EXISTS hub_telemetry (
    time            TIMESTAMPTZ     NOT NULL,
    hub_id          TEXT            NOT NULL,
    load_factor     DOUBLE PRECISION NOT NULL,
    congestion_score DOUBLE PRECISION NOT NULL,
    active          BOOLEAN         NOT NULL DEFAULT TRUE
);
SELECT create_hypertable('hub_telemetry', 'time', if_not_exists => TRUE);

-- Vehicle GPS pings
CREATE TABLE IF NOT EXISTS vehicle_pings (
    time            TIMESTAMPTZ     NOT NULL,
    vehicle_id      TEXT            NOT NULL,
    hub_id          TEXT,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    speed_kmh       DOUBLE PRECISION,
    heading_deg     DOUBLE PRECISION
);
SELECT create_hypertable('vehicle_pings', 'time', if_not_exists => TRUE);

-- Route execution log (actual vs predicted cost)
CREATE TABLE IF NOT EXISTS route_executions (
    time            TIMESTAMPTZ     NOT NULL,
    route_id        TEXT            NOT NULL,
    from_hub_id     TEXT            NOT NULL,
    to_hub_id       TEXT            NOT NULL,
    predicted_cost  DOUBLE PRECISION,
    actual_cost     DOUBLE PRECISION,
    gnn_confidence  SMALLINT,
    completed       BOOLEAN         NOT NULL DEFAULT FALSE
);
SELECT create_hypertable('route_executions', 'time', if_not_exists => TRUE);
"""


async def init_schema(conn: asyncpg.Connection) -> None:
    """Create all hypertables if they don't exist."""
    await conn.execute(SCHEMA_SQL)
    print("[TimescaleDB] Schema initialised.")


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

async def get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(TIMESCALE_DSN)


# ---------------------------------------------------------------------------
# Hub Telemetry
# ---------------------------------------------------------------------------

async def record_hub_telemetry(telemetry: Dict[str, Any]) -> None:
    """
    Insert a hub telemetry reading.
    Called by LangGraph after each GNN scoring cycle.
    """
    conn = await get_conn()
    try:
        await conn.execute("""
            INSERT INTO hub_telemetry (time, hub_id, load_factor, congestion_score, active)
            VALUES ($1, $2, $3, $4, $5)
        """,
            datetime.now(timezone.utc),
            telemetry["hub_id"],
            telemetry["load_factor"],
            telemetry["congestion_score"],
            telemetry.get("active", True),
        )
    finally:
        await conn.close()


async def get_hub_trend(hub_id: str, hours: int = 6) -> List[Dict]:
    """
    Fetch the last N hours of telemetry for a hub.
    Used by LangGraph to detect worsening congestion trends.
    """
    conn = await get_conn()
    try:
        rows = await conn.fetch("""
            SELECT time, load_factor, congestion_score
            FROM hub_telemetry
            WHERE hub_id = $1
              AND time > NOW() - INTERVAL '1 hour' * $2
            ORDER BY time DESC
        """, hub_id, hours)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Vehicle GPS Pings
# ---------------------------------------------------------------------------

async def record_vehicle_ping(ping: Dict[str, Any]) -> None:
    """Record a live GPS ping from a vehicle for Digital Twin visualization."""
    conn = await get_conn()
    try:
        await conn.execute("""
            INSERT INTO vehicle_pings
              (time, vehicle_id, hub_id, latitude, longitude, speed_kmh, heading_deg)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            datetime.now(timezone.utc),
            ping["vehicle_id"],
            ping.get("hub_id"),
            ping["latitude"],
            ping["longitude"],
            ping.get("speed_kmh"),
            ping.get("heading_deg"),
        )
    finally:
        await conn.close()


async def get_live_fleet_positions() -> List[Dict]:
    """
    Returns latest GPS ping per vehicle — used by Digital Twin to render live vehicle positions.
    """
    conn = await get_conn()
    try:
        rows = await conn.fetch("""
            SELECT DISTINCT ON (vehicle_id)
                vehicle_id, hub_id, latitude, longitude, speed_kmh, heading_deg, time
            FROM vehicle_pings
            ORDER BY vehicle_id, time DESC
        """)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Route Execution Logging
# ---------------------------------------------------------------------------

async def log_route_execution(route: Dict[str, Any]) -> None:
    """Log a route dispatched by LangGraph for performance tracking."""
    conn = await get_conn()
    try:
        await conn.execute("""
            INSERT INTO route_executions
              (time, route_id, from_hub_id, to_hub_id, predicted_cost, gnn_confidence)
            VALUES ($1, $2, $3, $4, $5, $6)
        """,
            datetime.now(timezone.utc),
            route["route_id"],
            route["from_hub_id"],
            route["to_hub_id"],
            route.get("predicted_cost"),
            route.get("gnn_confidence"),
        )
    finally:
        await conn.close()


async def update_route_completion(route_id: str, actual_cost: float) -> None:
    """Mark a route as completed and record actual vs predicted cost."""
    conn = await get_conn()
    try:
        await conn.execute("""
            UPDATE route_executions
            SET completed = TRUE, actual_cost = $1
            WHERE route_id = $2
        """, actual_cost, route_id)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test():
        conn = await get_conn()
        await init_schema(conn)
        await conn.close()
        print("[TimescaleDB] Connection and schema OK.")
    asyncio.run(_test())
