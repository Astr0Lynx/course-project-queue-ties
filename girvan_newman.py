"""Girvan-Newman community detection tailored for stock-network exploration.

This version integrates with the project's shared ``src`` helpers for synthetic
data generation and graph construction while keeping the core Girvan-Newman
implementation free of external dependencies beyond the Python standard
library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from graph_utils import Graph, load_graph_from_json, save_graph_to_json

SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

try:  # pragma: no cover - optional dependency for richer sample generation
    from data_generation import StockDataGenerator
    from graph import build_graph_from_correlation
except ModuleNotFoundError:  # Fallback for environments without numpy/pandas
    StockDataGenerator = None  # type: ignore[assignment]
    build_graph_from_correlation = None  # type: ignore[assignment]


Community = List[Any]
Partition = List[Community]


def betweenness_centrality(graph: Graph) -> Dict[Tuple[Any, Any], float]:
    """Compute edge betweenness centrality for an undirected graph."""

    betweenness: Dict[Tuple[Any, Any], float] = defaultdict(float)
    nodes = graph.nodes()

    for source in nodes:
        stack: List[Any] = []
        predecessors: Dict[Any, List[Any]] = {node: [] for node in nodes}
        shortest_paths: Dict[Any, float] = {node: 0.0 for node in nodes}
        shortest_paths[source] = 1.0
        distances: Dict[Any, int] = {node: -1 for node in nodes}
        distances[source] = 0
        queue: deque[Any] = deque([source])

        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in graph.neighbors(vertex):
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
                if distances[neighbor] == distances[vertex] + 1:
                    shortest_paths[neighbor] += shortest_paths[vertex]
                    predecessors[neighbor].append(vertex)

        dependencies: Dict[Any, float] = {node: 0.0 for node in nodes}

        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                if shortest_paths[node] == 0:
                    continue
                contribution = (shortest_paths[pred] / shortest_paths[node]) * (
                    1 + dependencies[node]
                )
                edge = tuple(sorted((pred, node)))
                betweenness[edge] += contribution
                dependencies[pred] += contribution

    for edge in list(betweenness.keys()):
        betweenness[edge] /= 2.0

    return dict(betweenness)


def modularity(graph: Graph, communities: Sequence[Iterable[Any]]) -> float:
    """Compute the modularity score for a given node partition."""

    total_weight = graph.total_edge_weight()
    if total_weight == 0:
        return 0.0

    degrees = {node: graph.degree(node) for node in graph.nodes()}
    norm = 2.0 * total_weight
    score = 0.0

    for community in communities:
        community_nodes = list(community)
        for node_u in community_nodes:
            for node_v in community_nodes:
                actual = graph.get_edge_weight(node_u, node_v)
                expected = (degrees[node_u] * degrees[node_v]) / norm
                score += actual - expected

    return score / norm


def _connected_components(graph: Graph) -> List[List[Any]]:
    """Return connected components as lists of nodes."""

    visited = set()
    components: List[List[Any]] = []

    for node in graph.nodes():
        if node in visited:
            continue
        queue: deque[Any] = deque([node])
        component: List[Any] = []
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    return components


def girvan_newman(graph: Graph, max_iterations: int | None = None) -> Tuple[Partition, float]:
    """Detect communities by iteratively removing high-betweenness edges."""

    if graph.number_of_nodes() == 0:
        return [], 0.0

    initial_components = _connected_components(graph)
    if len(initial_components) > 1:
        raise ValueError("Girvan-Newman expects a connected graph as input.")

    working_graph = graph.copy()
    best_partition: Partition = [sorted(graph.nodes())]
    best_modularity = modularity(graph, best_partition)
    iteration = 0

    while working_graph.number_of_edges() > 0:
        if max_iterations is not None and iteration >= max_iterations:
            break

        betweenness = betweenness_centrality(working_graph)
        if not betweenness:
            break

        max_betweenness = max(betweenness.values())
        edges_to_remove = [
            edge for edge, value in betweenness.items() if abs(value - max_betweenness) < 1e-9
        ]

        for node_u, node_v in edges_to_remove:
            working_graph.remove_edge(node_u, node_v)

        components = _connected_components(working_graph)
        if len(components) > 1:
            current_partition = [sorted(component) for component in components]
            current_modularity = modularity(graph, current_partition)
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_partition = current_partition

        iteration += 1

    return best_partition, best_modularity


def _convert_stock_graph(stock_graph: Any) -> Graph:
    """Translate the shared stock graph structure into the algorithm graph."""

    algorithm_graph = Graph()
    for node in stock_graph.get_nodes():
        algorithm_graph.add_node(node)
    for node_u, node_v, weight in stock_graph.get_edges():
        algorithm_graph.add_edge(node_u, node_v, weight)
    return algorithm_graph


def _build_manual_stock_graph() -> Graph:
    """Fallback manual graph used when the data generator is unavailable."""

    graph = Graph()
    edges = [
        ("AAPL", "MSFT", 0.9),
        ("AAPL", "GOOG", 0.85),
        ("MSFT", "GOOG", 0.88),
        ("GOOG", "NVDA", 0.8),
        ("NVDA", "TSLA", 0.75),
        ("AMZN", "TSLA", 0.7),
        ("AMZN", "AAPL", 0.78),
        ("JPM", "BAC", 0.83),
        ("JPM", "GS", 0.8),
        ("BAC", "GS", 0.82),
        ("MSFT", "JPM", 0.4),
        ("AMZN", "BAC", 0.45),
    ]

    for node_u, node_v, weight in edges:
        graph.add_edge(node_u, node_v, weight)

    return graph


def _connect_components_with_correlations(
    graph: Graph, correlation_lookup: Dict[str, Dict[str, float]]
) -> Graph:
    """Augment ``graph`` with strongest cross-component edges until connected."""

    components = _connected_components(graph)
    while len(components) > 1:
        best_weight = -1.0
        best_edge: Tuple[str, str] | None = None
        for idx, component in enumerate(components):
            for other_idx in range(idx + 1, len(components)):
                other_component = components[other_idx]
                for node_u in component:
                    for node_v in other_component:
                        weight = correlation_lookup.get(node_u, {}).get(node_v, 0.0)
                        if weight > best_weight:
                            best_weight = weight
                            best_edge = (node_u, node_v)
        if best_edge is None:
            node_u = components[0][0]
            node_v = components[1][0]
            best_weight = 1e-3
        else:
            node_u, node_v = best_edge
            best_weight = max(best_weight, 1e-3)
        graph.add_edge(node_u, node_v, best_weight)
        components = _connected_components(graph)
    return graph


def _build_sample_stock_graph(
    num_stocks: int = 18,
    scenario: str = "normal",
    threshold: float = 0.75,
    min_threshold: float = 0.2,
) -> Graph:
    """Create a sample graph using the shared data generator when available."""

    if StockDataGenerator is None or build_graph_from_correlation is None:
        return _build_manual_stock_graph()

    generator = StockDataGenerator(seed=42)
    returns, corr_matrix, attributes = generator.generate_dataset(
        num_stocks=num_stocks,
        scenario=scenario,
    )

    current_threshold = threshold
    densest_graph: Graph | None = None
    while current_threshold >= min_threshold:
        stock_graph = build_graph_from_correlation(
            correlation_matrix=corr_matrix,
            stock_attributes=attributes,
            threshold=current_threshold,
        )
        algorithm_graph = _convert_stock_graph(stock_graph)
        if len(_connected_components(algorithm_graph)) == 1:
            return algorithm_graph
        if densest_graph is None or algorithm_graph.number_of_edges() > densest_graph.number_of_edges():
            densest_graph = algorithm_graph
        current_threshold -= 0.05

    if densest_graph is not None:
        correlation_lookup = {
            stock: {neighbor: abs(value) for neighbor, value in values.items()}
            for stock, values in corr_matrix.to_dict().items()
        }
        connected_graph = _connect_components_with_correlations(densest_graph, correlation_lookup)
        if len(_connected_components(connected_graph)) == 1:
            return connected_graph
    return _build_manual_stock_graph()


def _print_partition(communities: Partition, modularity_score: float) -> None:
    """Pretty-print detected communities and their modularity score."""

    print("Detected communities:")
    for idx, community in enumerate(communities, start=1):
        print(f"  Community {idx}: {community}")
    print(f"Modularity score: {modularity_score:.4f}")


def main() -> None:
    """Demonstrate Girvan-Newman on a sample stock graph serialized to JSON."""

    sample_graph = _build_sample_stock_graph()
    json_path = "stock_graph.json"

    save_graph_to_json(sample_graph, json_path)
    loaded_graph = load_graph_from_json(json_path)

    communities, score = girvan_newman(loaded_graph)
    _print_partition(communities, score)


if __name__ == "__main__":
    main()
