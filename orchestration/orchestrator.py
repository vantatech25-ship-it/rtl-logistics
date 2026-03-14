"""
orchestrator.py
LangGraph agent orchestrating the Living Neural Network logistics pipeline.

Graph flow:
  fetch_hubs → score_hubs → store_embeddings → request_route → log_route → END

Each node is a pure function operating on a shared AgentState TypedDict.
LangGraph handles state threading and conditional branching automatically.
"""

import asyncio
import uuid
import httpx
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "memory-layer"))

from pinecone_client import upsert_hub_embedding, find_similar_hubs
from timescale_client import record_hub_telemetry, log_route_execution

from config import ROUTING_ENGINE_URL


# ---------------------------------------------------------------------------
# Agent State — shared across all graph nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The single mutable state object that flows through the LangGraph pipeline.
    Every node reads from and writes to this dict.
    """
    from_hub_id: str
    to_hub_id: str

    # Populated by fetch_hubs
    hub_features: List[Dict[str, Any]]

    # Populated by request_route
    route_recommendation: Optional[Dict[str, Any]]

    # Tracking
    route_id: str
    errors: List[str]


# ---------------------------------------------------------------------------
# Node 1: Fetch live hub data and GNN scores from Rust engine
# ---------------------------------------------------------------------------

async def fetch_hubs(state: AgentState) -> AgentState:
    print("[LangGraph] Node: fetch_hubs")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ROUTING_ENGINE_URL}/hubs", timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

    if data.get("success"):
        state["hub_features"] = data["data"]
        print(f"[LangGraph] Fetched {len(state['hub_features'])} hub features")
    else:
        state["errors"].append("Failed to fetch hubs from routing engine")
    return state


# ---------------------------------------------------------------------------
# Node 2: Store hub embeddings in Pinecone + telemetry in TimescaleDB
# ---------------------------------------------------------------------------

async def store_memory(state: AgentState) -> AgentState:
    print("[LangGraph] Node: store_memory")
    tasks = []
    for hub in state.get("hub_features", []):
        tasks.append(asyncio.create_task(_store_single_hub(hub)))
    await asyncio.gather(*tasks)
    return state


async def _store_single_hub(hub: Dict[str, Any]) -> None:
    # Vector memory (Pinecone)
    upsert_hub_embedding(hub)
    # Time-series telemetry (TimescaleDB)
    await record_hub_telemetry(hub)


# ---------------------------------------------------------------------------
# Node 3: Request GNN-optimised route from Rust engine
# ---------------------------------------------------------------------------

async def request_route(state: AgentState) -> AgentState:
    print(f"[LangGraph] Node: request_route ({state['from_hub_id']} → {state['to_hub_id']})")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ROUTING_ENGINE_URL}/route",
            json={"from_hub_id": state["from_hub_id"], "to_hub_id": state["to_hub_id"]},
            timeout=10.0,
        )

    if resp.status_code == 200 and resp.json().get("success"):
        state["route_recommendation"] = resp.json()["data"]
        print(f"[LangGraph] Route confidence: {state['route_recommendation']['confidence']}%")
    else:
        # Fallback: try to find similar hub via Pinecone and reroute
        print("[LangGraph] Primary route failed — querying Pinecone for alternative hub")
        ref_hub = state["hub_features"][0] if state["hub_features"] else {}
        alternatives = find_similar_hubs(ref_hub, top_k=3)
        state["errors"].append(f"Route failed. Pinecone alternatives: {[a['id'] for a in alternatives]}")
    return state


# ---------------------------------------------------------------------------
# Node 4: Log dispatched route to TimescaleDB
# ---------------------------------------------------------------------------

async def log_route(state: AgentState) -> AgentState:
    print("[LangGraph] Node: log_route")
    rec = state.get("route_recommendation")
    if rec:
        await log_route_execution({
            "route_id": state["route_id"],
            "from_hub_id": state["from_hub_id"],
            "to_hub_id": state["to_hub_id"],
            "predicted_cost": rec.get("adjusted_cost"),
            "gnn_confidence": rec.get("confidence"),
        })
    return state


# ---------------------------------------------------------------------------
# Conditional edge: should we abort if route failed?
# ---------------------------------------------------------------------------

def check_route(state: AgentState) -> str:
    if state.get("route_recommendation"):
        return "log_route"
    return END


# ---------------------------------------------------------------------------
# Build the LangGraph state machine
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_hubs",   fetch_hubs)
    graph.add_node("store_memory", store_memory)
    graph.add_node("request_route", request_route)
    graph.add_node("log_route",    log_route)

    graph.set_entry_point("fetch_hubs")
    graph.add_edge("fetch_hubs",    "store_memory")
    graph.add_edge("store_memory",  "request_route")
    graph.add_conditional_edges("request_route", check_route, {
        "log_route": "log_route",
        END: END,
    })
    graph.add_edge("log_route", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run a dispatch
# ---------------------------------------------------------------------------

async def dispatch_route(from_hub_id: str, to_hub_id: str) -> Dict[str, Any]:
    """
    Main entry point called by any external system (webhook, scheduler, UI).
    Returns the final agent state including the route recommendation.
    """
    app = build_graph()
    initial_state: AgentState = {
        "from_hub_id": from_hub_id,
        "to_hub_id": to_hub_id,
        "hub_features": [],
        "route_recommendation": None,
        "route_id": str(uuid.uuid4()),
        "errors": [],
    }
    result = await app.ainvoke(initial_state)
    return result


if __name__ == "__main__":
    result = asyncio.run(dispatch_route("JHB", "CPT"))
    import json
    print("\n=== Final Agent State ===")
    print(json.dumps(result, indent=2, default=str))
