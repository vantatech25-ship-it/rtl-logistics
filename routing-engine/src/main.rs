mod api;
mod graph;
mod gnn;

use std::sync::{Arc, RwLock};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::api::{start_server, SharedNetwork};
use crate::graph::LogisticsNetwork;

#[tokio::main]
async fn main() {
    // Initialize tracing / structured logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "routing_engine=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🧠 RTL Logistics — Living Neural Network Routing Engine v0.1.0");

    // Build live logistics graph and seed demo hubs
    let mut network = LogisticsNetwork::new();
    network.seed_demo_network();

    tracing::info!("📍 Demo network seeded: {} hubs loaded", network.hub_index.len());

    // Wrap in Arc<RwLock<>> for safe concurrent API access
    let shared_network: SharedNetwork = Arc::new(RwLock::new(network));

    // Start REST API server (blocks until shutdown)
    start_server(shared_network).await;
}
