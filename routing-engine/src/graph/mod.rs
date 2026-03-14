use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::dijkstra;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A logistics hub or vehicle stop in the network graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hub {
    pub id: String,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    /// Current load factor 0.0 (empty) to 1.0 (full capacity)
    pub load_factor: f64,
    /// Whether the hub is currently active/reachable
    pub active: bool,
}

/// Properties of a route edge connecting two hubs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEdge {
    /// Base route weight (combination of distance, time, cost)
    pub base_weight: f32,
    /// Live traffic multiplier (1.0 = no congestion, 2.0 = double travel time)
    pub traffic_factor: f32,
    /// Weather/road hazard penalty
    pub hazard_penalty: f32,
}

impl RouteEdge {
    /// Returns the effective composite weight for pathfinding.
    pub fn effective_weight(&self) -> f32 {
        self.base_weight * self.traffic_factor + self.hazard_penalty
    }
}

/// The live logistics network graph.
pub struct LogisticsNetwork {
    pub graph: UnGraph<Hub, RouteEdge>,
    /// Maps hub ID → NodeIndex for fast lookup
    pub hub_index: HashMap<String, NodeIndex>,
}

impl LogisticsNetwork {
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            hub_index: HashMap::new(),
        }
    }

    /// Adds a hub node to the graph.
    pub fn add_hub(&mut self, hub: Hub) -> NodeIndex {
        let id = hub.id.clone();
        let idx = self.graph.add_node(hub);
        self.hub_index.insert(id, idx);
        idx
    }

    /// Connects two hubs with a route edge.
    pub fn add_route(&mut self, from_id: &str, to_id: &str, edge: RouteEdge) -> Option<()> {
        let from = *self.hub_index.get(from_id)?;
        let to = *self.hub_index.get(to_id)?;
        self.graph.add_edge(from, to, edge);
        Some(())
    }

    /// Returns the optimal path between two hubs using Dijkstra's algorithm.
    pub fn shortest_path(&self, from_id: &str, to_id: &str) -> Option<PathResult> {
        let from = *self.hub_index.get(from_id)?;
        let to = *self.hub_index.get(to_id)?;

        // Edge cost function uses effective weight
        let costs = dijkstra(&self.graph, from, Some(to), |e| {
            e.weight().effective_weight()
        });

        let total_cost = *costs.get(&to)?;

        // Collect hub names along path (simplified: just endpoints for now)
        let from_hub = &self.graph[from];
        let to_hub = &self.graph[to];

        Some(PathResult {
            from: from_hub.name.clone(),
            to: to_hub.name.clone(),
            total_cost,
            hubs_traversed: vec![from_hub.name.clone(), to_hub.name.clone()],
        })
    }

    /// Seeds the network with a sample RTL logistics graph for demo/testing.
    pub fn seed_demo_network(&mut self) {
        let hubs = vec![
            Hub { id: "JHB".into(), name: "Johannesburg Hub".into(), latitude: -26.2041, longitude: 28.0473, load_factor: 0.75, active: true },
            Hub { id: "CPT".into(), name: "Cape Town Hub".into(), latitude: -33.9249, longitude: 18.4241, load_factor: 0.55, active: true },
            Hub { id: "DBN".into(), name: "Durban Hub".into(), latitude: -29.8587, longitude: 31.0218, load_factor: 0.80, active: true },
            Hub { id: "PE".into(),  name: "Gqeberha Hub".into(),  latitude: -33.9608, longitude: 25.6022, load_factor: 0.40, active: true },
        ];

        for h in hubs { self.add_hub(h); }

        let routes = vec![
            ("JHB", "CPT", RouteEdge { base_weight: 1400.0, traffic_factor: 1.2, hazard_penalty: 0.0 }),
            ("JHB", "DBN", RouteEdge { base_weight: 560.0,  traffic_factor: 1.1, hazard_penalty: 10.0 }),
            ("JHB", "PE",  RouteEdge { base_weight: 1060.0, traffic_factor: 1.0, hazard_penalty: 5.0 }),
            ("CPT", "PE",  RouteEdge { base_weight: 750.0,  traffic_factor: 1.0, hazard_penalty: 0.0 }),
            ("DBN", "PE",  RouteEdge { base_weight: 760.0,  traffic_factor: 1.3, hazard_penalty: 15.0 }),
        ];

        for (f, t, e) in routes { self.add_route(f, t, e); }
    }
}

/// Result of a pathfinding query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    pub from: String,
    pub to: String,
    pub total_cost: f32,
    pub hubs_traversed: Vec<String>,
}
