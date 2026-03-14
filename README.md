# RTL Logistics — Living Neural Network

> A full-stack intelligent logistics platform powered by GNN routing, vector memory, time-series telemetry, LangGraph orchestration, and a real-time 3D Digital Twin.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    core-digital-twin (:8080)                    │
│              Three.js 3D Globe  •  Route Dispatcher             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /route
┌──────────────────────────▼──────────────────────────────────────┐
│              routing-engine (:3000) — Rust + Axum               │
│          Petgraph Dijkstra  •  GNN Congestion Scorer            │
└────────────┬──────────────────────────────────────┬─────────────┘
             │ HTTP                                  │ Hub features
┌────────────▼────────────┐           ┌─────────────▼─────────────┐
│  orchestration (Python) │           │   memory-layer (Python)   │
│  LangGraph StateGraph   │◄──────────►  Pinecone  +  TimescaleDB │
│  4-node pipeline        │           │  vectors  +  hypertables  │
└────────────┬────────────┘           └───────────────────────────┘
             │
┌────────────▼────────────┐
│  dashboard (:8000/8080) │
│  FastAPI + Chart.js     │
│  KPI  •  Trend  •  Fleet│
└─────────────────────────┘
```

## Quick Start (Docker)

```powershell
# 1. Copy secrets file
Copy-Item .env.example .env
# Edit .env and add your PINECONE_API_KEY

# 2. Launch all services
docker compose up --build

# 3. Open in browser
#    Digital Twin:  http://localhost:8080
#    KPI Dashboard: http://localhost:8000
#    Routing API:   http://localhost:3000/health
```

## Services

| Service         | Port | Tech                         |
|-----------------|------|------------------------------|
| routing-engine  | 3000 | Rust, Axum, petgraph (GNN)   |
| digital-twin    | 8080 | Three.js, Nginx              |
| orchestration   | —    | Python, LangGraph            |
| dashboard       | 8000 | FastAPI, Chart.js            |
| timescaledb     | 5432 | PostgreSQL + TimescaleDB     |

## GNN Training

```powershell
cd routing-engine
pip install -r gnn_requirements.txt
python gnn_model.py   # trains GraphSAGE, saves gnn_weights.pt
```

## Cloud Deploy

**Railway:**
```bash
railway up
```

**GCP Cloud Run:**
```bash
gcloud builds submit --config cloudbuild.yaml --substitutions=_PROJECT_ID=$(gcloud config get-value project)
```

## Routing API

```bash
# Get hub GNN scores
curl http://localhost:3000/hubs

# Dispatch a GNN-optimised route
curl -X POST http://localhost:3000/route \
  -H "Content-Type: application/json" \
  -d '{"from_hub_id":"JHB","to_hub_id":"CPT"}'
```
