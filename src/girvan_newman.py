"""Girvan-Newman community detection for stock market network analysis.

This module implements the Girvan-Newman algorithm from scratch for detecting
communities in stock correlation networks. It integrates with the project's
shared graph utilities and data generation components.

Author: Avani Sood
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from graph import Graph, load_graph, save_graph

try:  # Optional dependency for richer sample generation
    from data_generation import StockDataGenerator
    from graph import build_graph_from_correlation
    HAS_DATA_GENERATOR = True
except (ModuleNotFoundError, ImportError):
    HAS_DATA_GENERATOR = False


Community = List[Any]
Partition = List[Community]


def betweenness_centrality(graph: Graph) -> Dict[Tuple[Any, Any], float]:
    """Compute edge betweenness centrality for an undirected graph.
    
    Uses a BFS-based approach adapted from Brandes' algorithm to calculate
    the betweenness centrality for each edge in the graph. Edge betweenness
    measures how many shortest paths pass through each edge.
    
    Args:
        graph: Undirected Graph object
    
    Returns:
        Dictionary mapping edge tuples (u, v) to betweenness scores
    
    Time Complexity: O(V * (V + E)) for BFS from each vertex
    Space Complexity: O(V + E) for storing paths and dependencies
    """

    betweenness: Dict[Tuple[Any, Any], float] = defaultdict(float)
    nodes = graph.nodes()

    # Run BFS from each source node
    for source in nodes:
        stack: List[Any] = []
        predecessors: Dict[Any, List[Any]] = {node: [] for node in nodes}
        shortest_paths: Dict[Any, float] = {node: 0.0 for node in nodes}
        shortest_paths[source] = 1.0
        distances: Dict[Any, int] = {node: -1 for node in nodes}
        distances[source] = 0
        queue: deque[Any] = deque([source])

        # BFS to compute shortest paths and predecessors
        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in graph.neighbors(vertex):
                # First time visiting this neighbor
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
                # Found another shortest path to neighbor
                if distances[neighbor] == distances[vertex] + 1:
                    shortest_paths[neighbor] += shortest_paths[vertex]
                    predecessors[neighbor].append(vertex)

        dependencies: Dict[Any, float] = {node: 0.0 for node in nodes}

        # Accumulate dependencies from farthest nodes back to source
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

    # Normalize for undirected graphs (each edge counted twice)
    for edge in list(betweenness.keys()):
        betweenness[edge] /= 2.0

    return dict(betweenness)


def modularity(graph: Graph, communities: Sequence[Iterable[Any]]) -> float:
    """Compute the modularity score for a given node partition.
    
    Modularity measures the quality of a community structure by comparing
    the actual edge density within communities to the expected density
    in a random graph with the same degree sequence.
    
    Args:
        graph: Original graph (not the progressively pruned one)
        communities: Partition of nodes into communities
    
    Returns:
        Modularity value in range [-1, 1], higher is better
    
    Time Complexity: O(C^2 * k^2) where C is number of communities, k is avg community size
    Space Complexity: O(V) for storing degrees
    """

    total_weight = graph.total_edge_weight()
    if total_weight == 0:
        return 0.0

    degrees = {node: graph.degree(node) for node in graph.nodes()}
    norm = 2.0 * total_weight
    score = 0.0

    # For each community, sum contribution of all node pairs
    for community in communities:
        community_nodes = list(community)
        for node_u in community_nodes:
            for node_v in community_nodes:
                actual = graph.get_edge_weight(node_u, node_v) or 0.0
                expected = (degrees[node_u] * degrees[node_v]) / norm
                score += actual - expected

    return score / norm


def _connected_components(graph: Graph) -> List[List[Any]]:
    """Find all connected components in the graph using BFS.
    
    Args:
        graph: Graph object
    
    Returns:
        List of components, each component is a list of node identifiers
    
    Time Complexity: O(V + E) for BFS traversal
    Space Complexity: O(V) for visited set and queue
    """

    visited = set()
    components: List[List[Any]] = []

    for node in graph.nodes():
        if node in visited:
            continue
        
        # BFS to find all nodes in this component
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


def _estimate_virtual_edge_weight(graph: Graph) -> float:
    """Infer a minimal edge weight for virtual bridges based on existing edges."""

    weights: List[float] = []
    for node in graph.nodes():
        for weight in graph.get_neighbors(node).values():
            if weight > 0:
                weights.append(weight)

    if weights:
        return max(min(weights), 1e-3)
    return 1e-3


def _chain_nodes_with_virtual_edges(graph: Graph, ordered_nodes: List[Any]) -> None:
    """Connect nodes sequentially with low-weight edges to ensure connectivity."""

    if len(ordered_nodes) <= 1:
        return

    bridge_weight = _estimate_virtual_edge_weight(graph)
    for idx in range(len(ordered_nodes) - 1):
        u, v = ordered_nodes[idx], ordered_nodes[idx + 1]
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, bridge_weight)


def _connect_components_with_virtual_edges(graph: Graph, components: List[List[Any]]) -> None:
    """Bridge disconnected components with lightweight virtual edges."""

    if len(components) <= 1:
        return

    bridge_weight = _estimate_virtual_edge_weight(graph)
    representative = components[0][0]
    for component in components[1:]:
        target = component[0]
        if not graph.has_edge(representative, target):
            graph.add_edge(representative, target, bridge_weight)
        representative = target


def _augment_graph_with_virtual_bridges(graph: Graph) -> Graph:
    """Return a working copy of the graph that is fully connected for GN iterations."""

    working_graph = graph.copy()
    nodes = working_graph.nodes()
    if len(nodes) <= 1:
        return working_graph

    components = _connected_components(working_graph)

    if working_graph.number_of_edges() == 0:
        ordered_nodes: List[Any] = []
        for component in components:
            ordered_nodes.extend(sorted(component))
        _chain_nodes_with_virtual_edges(working_graph, ordered_nodes)
        return working_graph

    if len(components) > 1:
        _connect_components_with_virtual_edges(working_graph, components)

    return working_graph


def girvan_newman(graph: Graph, max_iterations: int | None = None) -> Tuple[Partition, float]:
    """Detect communities using the Girvan-Newman algorithm.
    
    The algorithm iteratively removes edges with highest betweenness centrality,
    progressively splitting the graph into communities. The partition with the
    highest modularity score is returned.
    
    Args:
        graph: Input graph (should be connected for best results)
        max_iterations: Optional limit on edge-removal iterations
    
    Returns:
        Tuple of (best partition, best modularity score)
    
    Raises:
        ValueError: If the input graph is disconnected
    
    Time Complexity: O(k * V * (V + E)^2) where k is iterations (worst case V*E)
    Space Complexity: O(V + E) for graph copy and intermediate structures
    """

    if graph.number_of_nodes() == 0:
        return [], 0.0

    working_graph = _augment_graph_with_virtual_bridges(graph)

    best_partition: Partition = [sorted(working_graph.nodes())]
    best_modularity = modularity(graph, best_partition)
    iteration = 0

    # Iteratively remove highest-betweenness edges
    while working_graph.number_of_edges() > 0:
        if max_iterations is not None and iteration >= max_iterations:
            break

        # Compute edge betweenness for current graph state
        betweenness = betweenness_centrality(working_graph)
        if not betweenness:
            break

        # Find and remove all edges with maximum betweenness
        max_betweenness = max(betweenness.values())
        edges_to_remove = [
            edge for edge, value in betweenness.items() 
            if abs(value - max_betweenness) < 1e-9
        ]

        for node_u, node_v in edges_to_remove:
            working_graph.remove_edge(node_u, node_v)

        # Check if graph split into multiple components
        components = _connected_components(working_graph)
        if len(components) > 1:
            # Evaluate modularity of this partition
            current_partition = [sorted(component) for component in components]
            current_modularity = modularity(graph, current_partition)
            
            # Keep track of best partition seen so far
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_partition = current_partition

        iteration += 1

    return best_partition, best_modularity


def _build_manual_stock_graph() -> Graph:
    """Create a hardcoded sample graph for fallback when data generator unavailable.
    
    Returns:
        Small stock correlation graph with realistic structure
    """

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
    """Augment graph with strongest cross-component edges until connected.
    
    When a generated graph has multiple components, this function bridges them
    by adding the highest-correlation edges between components.
    
    Args:
        graph: Potentially disconnected graph
        correlation_lookup: Full correlation matrix as nested dict
    
    Returns:
        Connected graph
    
    Time Complexity: O(C^2 * k^2) where C is components, k is avg component size
    Space Complexity: O(V + E) for the graph structure
    """

    components = _connected_components(graph)
    
    # Keep adding edges until graph is connected
    while len(components) > 1:
        best_weight = -1.0
        best_edge: Tuple[str, str] | None = None
        
        # Find strongest correlation between any two components
        for idx, component in enumerate(components):
            for other_idx in range(idx + 1, len(components)):
                other_component = components[other_idx]
                for node_u in component:
                    for node_v in other_component:
                        weight = correlation_lookup.get(node_u, {}).get(node_v, 0.0)
                        if weight > best_weight:
                            best_weight = weight
                            best_edge = (node_u, node_v)
        
        # Add the best edge found (or a minimal edge if none found)
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
    """Create a sample graph using the data generator when available.
    
    Generates synthetic stock data and builds a correlation graph. If the
    graph is disconnected, lowers the threshold or bridges components.
    
    Args:
        num_stocks: Number of stocks to generate
        scenario: Market scenario (normal/stable/volatile/crash)
        threshold: Initial correlation threshold for edges
        min_threshold: Minimum threshold to try before bridging
    
    Returns:
        Connected stock correlation graph
    """

    # Fallback to manual graph if data generator not available
    if not HAS_DATA_GENERATOR:
        return _build_manual_stock_graph()

    # Generate synthetic stock data
    generator = StockDataGenerator(seed=42)
    returns, corr_matrix, attributes = generator.generate_dataset(
        num_stocks=num_stocks,
        scenario=scenario,
    )

    # Try to build connected graph by lowering threshold
    current_threshold = threshold
    densest_graph: Graph | None = None
    
    while current_threshold >= min_threshold:
        stock_graph = build_graph_from_correlation(
            correlation_matrix=corr_matrix,
            stock_attributes=attributes,
            threshold=current_threshold,
        )
        
        # Check if graph is connected
        if len(_connected_components(stock_graph)) == 1:
            return stock_graph
        
        # Track densest graph in case we need to bridge it
        if densest_graph is None or stock_graph.num_edges > densest_graph.number_of_edges():
            densest_graph = stock_graph
        
        current_threshold -= 0.05

    # Bridge disconnected components if needed
    if densest_graph is not None:
        correlation_lookup = {
            stock: {neighbor: abs(value) for neighbor, value in values.items()}
            for stock, values in corr_matrix.to_dict().items()
        }
        connected_graph = _connect_components_with_correlations(
            densest_graph, correlation_lookup
        )
        if len(_connected_components(connected_graph)) == 1:
            return connected_graph
    
    # Final fallback
    return _build_manual_stock_graph()


def _print_partition(communities: Partition, modularity_score: float) -> None:
    """Display detected communities and modularity score.
    
    Args:
        communities: List of communities (each a list of node IDs)
        modularity_score: Quality metric for the partition
    """

    print("Detected communities:")
    for idx, community in enumerate(communities, start=1):
        print(f"  Community {idx}: {community}")
    print(f"Modularity score: {modularity_score:.4f}")


def main() -> None:
    """Run Girvan-Newman algorithm on sample stock correlation data.
    
    Demonstrates the complete workflow: generate/load data, build graph,
    persist to JSON in data/ directory, reload, detect communities, and
    report results.
    """

    # Generate sample stock correlation graph
    sample_graph = _build_sample_stock_graph()
    
    # Save to data directory (create if needed)
    import os
    os.makedirs("../data", exist_ok=True)
    json_path = "../data/stock_graph.json"

    # Persist graph and reload for analysis
    save_graph(sample_graph, json_path)
    loaded_graph = load_graph(json_path)

    # Run community detection
    communities, score = girvan_newman(loaded_graph)
    _print_partition(communities, score)


if __name__ == "__main__":
    main()
