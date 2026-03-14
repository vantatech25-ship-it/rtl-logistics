use crate::graph::{Hub, LogisticsNetwork, PathResult};
use serde::{Deserialize, Serialize};

/// GNN-inspired route scoring using hub features as a message-passing simulation.
/// When a full PyTorch/tch-rs GNN model is trained, this module will load it
/// and run inference. For now this uses a feature-weighted heuristic that
/// mirrors what a GNN aggregation step would compute.
pub struct GnnRouter;

/// Aggregated node features for GNN input (mirror of a graph embedding vector).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubFeatures {
    pub hub_id: String,
    /// Normalised load 0–1
    pub load_factor: f64,
    /// Number of direct connections (degree centrality proxy)
    pub connectivity: usize,
    /// Computed GNN-style congestion score (lower = better to route through)
    pub congestion_score: f64,
}

impl GnnRouter {
    /// Scores every hub in the network and returns a sorted list.
    /// Lower congestion_score = better candidate for routing through.
    pub fn score_hubs(network: &LogisticsNetwork) -> Vec<HubFeatures> {
        let mut features: Vec<HubFeatures> = Vec::new();

        for (hub_id, &node_idx) in &network.hub_index {
            let hub: &Hub = &network.graph[node_idx];
            if !hub.active { continue; }

            // Degree = number of neighbours (connectivity proxy)
            let connectivity = network.graph.neighbors(node_idx).count();

            // GNN aggregation heuristic:
            //   congestion_score = load_factor * 0.6 + (1 / (connectivity + 1)) * 0.4
            // A heavily loaded, poorly connected hub gets a high congestion penalty.
            let congestion_score =
                hub.load_factor * 0.6
                + (1.0 / (connectivity as f64 + 1.0)) * 0.4;

            features.push(HubFeatures {
                hub_id: hub_id.clone(),
                load_factor: hub.load_factor,
                connectivity,
                congestion_score,
            });
        }

        // Sort ascending by congestion score (best hub first)
        features.sort_by(|a, b| a.congestion_score.partial_cmp(&b.congestion_score).unwrap());
        features
    }

    /// Recommends the optimal route, weighted by GNN hub scores.
    /// Applies a congestion surcharge on paths that traverse congested hubs.
    pub fn recommend_route(
        network: &LogisticsNetwork,
        from_id: &str,
        to_id: &str,
    ) -> Option<GnnRouteRecommendation> {
        let base_path = network.shortest_path(from_id, to_id)?;
        let hub_scores = Self::score_hubs(network);

        // Compute a congestion surcharge: average score of from/to hubs
        let from_score = hub_scores.iter().find(|h| h.hub_id == from_id)
            .map(|h| h.congestion_score).unwrap_or(0.5);
        let to_score = hub_scores.iter().find(|h| h.hub_id == to_id)
            .map(|h| h.congestion_score).unwrap_or(0.5);

        let gnn_adjustment = ((from_score + to_score) / 2.0) as f32;
        let adjusted_cost = base_path.total_cost * (1.0 + gnn_adjustment);

        // Confidence 0–100: inverse of gnn_adjustment
        let confidence = ((1.0 - gnn_adjustment as f64) * 100.0).clamp(0.0, 100.0) as u8;

        Some(GnnRouteRecommendation {
            path: base_path,
            gnn_adjustment,
            adjusted_cost,
            confidence,
            hub_features: hub_scores,
        })
    }
}

/// The final recommendation returned by the GNN router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnRouteRecommendation {
    pub path: PathResult,
    /// Fractional cost adjustment from GNN congestion model (0.0 = no adjustment)
    pub gnn_adjustment: f32,
    /// Final recommended cost after GNN scoring
    pub adjusted_cost: f32,
    /// Route confidence score 0–100
    pub confidence: u8,
    /// Full hub feature vectors for Digital Twin visualization
    pub hub_features: Vec<HubFeatures>,
}
