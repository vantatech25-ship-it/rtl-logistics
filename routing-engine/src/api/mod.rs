use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::gnn::GnnRouter;
use crate::graph::{Hub, LogisticsNetwork, RouteEdge};

/// Shared application state (thread-safe live network graph).
pub type SharedNetwork = Arc<RwLock<LogisticsNetwork>>;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RouteRequest {
    pub from_hub_id: String,
    pub to_hub_id: String,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn ok(data: T) -> Json<Self> {
        Json(Self { success: true, data: Some(data), error: None })
    }
    fn err(msg: impl Into<String>) -> Json<Self> {
        Json(Self { success: false, data: None, error: Some(msg.into()) })
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /health — liveness check for Docker/K8s probes
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "service": "routing-engine" }))
}

/// GET /hubs — list all hubs and their GNN congestion scores
async fn list_hubs(State(network): State<SharedNetwork>) -> impl IntoResponse {
    let net = network.read().unwrap();
    let scores = GnnRouter::score_hubs(&net);
    ApiResponse::ok(scores)
}

/// POST /route — request GNN-optimised route between two hubs
/// Body: { "from_hub_id": "JHB", "to_hub_id": "CPT" }
async fn get_route(
    State(network): State<SharedNetwork>,
    Json(req): Json<RouteRequest>,
) -> impl IntoResponse {
    info!("Route request: {} → {}", req.from_hub_id, req.to_hub_id);
    let net = network.read().unwrap();
    match GnnRouter::recommend_route(&net, &req.from_hub_id, &req.to_hub_id) {
        Some(rec) => (StatusCode::OK, ApiResponse::ok(rec)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            ApiResponse::<()>::err(format!(
                "No route found between '{}' and '{}'",
                req.from_hub_id, req.to_hub_id
            )),
        )
            .into_response(),
    }
}

/// POST /hubs — add a new hub to the live network
async fn add_hub(
    State(network): State<SharedNetwork>,
    Json(hub): Json<Hub>,
) -> impl IntoResponse {
    let mut net = network.write().unwrap();
    let id = hub.id.clone();
    net.add_hub(hub);
    ApiResponse::ok(serde_json::json!({ "added_hub_id": id }))
}

#[derive(Debug, Deserialize)]
pub struct AddRouteRequest {
    pub from_hub_id: String,
    pub to_hub_id: String,
    pub edge: RouteEdge,
}

/// POST /routes — connect two hubs with a route edge
async fn add_route(
    State(network): State<SharedNetwork>,
    Json(req): Json<AddRouteRequest>,
) -> impl IntoResponse {
    let mut net = network.write().unwrap();
    match net.add_route(&req.from_hub_id, &req.to_hub_id, req.edge) {
        Some(_) => ApiResponse::ok(serde_json::json!({ "status": "edge added" })),
        None => ApiResponse::err("One or both hub IDs not found"),
    }
}

// ---------------------------------------------------------------------------
// Server bootstrap
// ---------------------------------------------------------------------------

pub async fn start_server(network: SharedNetwork) {
    let cors = CorsLayer::permissive(); // allow LangGraph / Digital Twin clients

    let app = Router::new()
        .route("/health",  get(health))
        .route("/hubs",    get(list_hubs).post(add_hub))
        .route("/route",   post(get_route))
        .route("/routes",  post(add_route))
        .with_state(network)
        .layer(cors);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    info!("🚀 Routing Engine API listening on http://{}", addr);
    axum::serve(listener, app).await.unwrap();
}
