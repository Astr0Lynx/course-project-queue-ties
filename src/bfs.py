"""
Breadth-First Search (BFS) Algorithm
Author: Guntesh Singh
Description: Implementation of BFS for finding shortest paths and correlation chains
             in stock graphs. Includes variant for prioritizing low-volatility routes.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import deque


def bfs_shortest_path(graph, start: str, end: str) -> Optional[List[str]]:
    """
    Find shortest path between two stocks using BFS.
    
    BFS guarantees finding the shortest path in an unweighted graph.
    In our context, this finds the shortest correlation chain between two stocks.
    
    Args:
        graph: Graph object
        start: Starting stock
        end: Target stock
    
    Returns:
        List of stocks forming shortest path, or None if no path exists
    
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for queue and visited set
    """
    if start not in graph.get_nodes() or end not in graph.get_nodes():
        return None
    
    if start == end:
        return [start]
    
    # Queue stores tuples of (current_node, path_to_node)
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # Explore neighbors
        neighbors = graph.get_neighbors(current)
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            
            # Build path to neighbor
            new_path = path + [neighbor]
            
            # Check if we reached the target
            if neighbor == end:
                return new_path
            
            # Add to queue for further exploration
            queue.append((neighbor, new_path))
            visited.add(neighbor)
    
    # No path found
    return None


def bfs_shortest_path_volatility_weighted(graph, 
                                          start: str, 
                                          end: str,
                                          stock_attributes: Dict[str, Dict]) -> Optional[List[str]]:
    """
    Find path prioritizing low-volatility stocks (safer correlation chain).
    
    This variant of BFS prefers paths through stable stocks by exploring
    lower-volatility neighbors first (using a priority-like approach).
    
    Args:
        graph: Graph object
        start: Starting stock
        end: Target stock
        stock_attributes: Dictionary of stock attributes (must have 'volatility')
    
    Returns:
        List of stocks forming low-volatility path, or None if no path exists
    
    Time Complexity: O(V + E log V) due to sorting neighbors
    Space Complexity: O(V)
    """
    if start not in graph.get_nodes() or end not in graph.get_nodes():
        return None
    
    if start == end:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # Get neighbors and sort by volatility (stable stocks first)
        neighbors = graph.get_neighbors(current)
        neighbor_list = list(neighbors.keys())
        
        # Sort by volatility (ascending - lower volatility first)
        neighbor_list.sort(
            key=lambda n: stock_attributes.get(n, {}).get('volatility', float('inf'))
        )
        
        for neighbor in neighbor_list:
            if neighbor in visited:
                continue
            
            new_path = path + [neighbor]
            
            if neighbor == end:
                return new_path
            
            queue.append((neighbor, new_path))
            visited.add(neighbor)
    
    return None


def bfs_all_paths_bounded(graph, 
                         start: str, 
                         max_length: int) -> Dict[str, List[str]]:
    """
    Find shortest paths from start to all reachable stocks within max_length.
    
    Args:
        graph: Graph object
        start: Starting stock
        max_length: Maximum path length to explore
    
    Returns:
        Dictionary mapping stocks to their shortest paths from start
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if start not in graph.get_nodes():
        return {}
    
    paths = {start: [start]}
    queue = deque([(start, [start], 0)])  # (node, path, depth)
    visited = {start}
    
    while queue:
        current, path, depth = queue.popleft()
        
        # Don't explore beyond max_length
        if depth >= max_length:
            continue
        
        neighbors = graph.get_neighbors(current)
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            
            new_path = path + [neighbor]
            paths[neighbor] = new_path
            
            queue.append((neighbor, new_path, depth + 1))
            visited.add(neighbor)
    
    return paths


def bfs_connected_component(graph, start: str) -> Set[str]:
    """
    Find all stocks in the same connected component as start using BFS.
    
    Args:
        graph: Graph object
        start: Starting stock
    
    Returns:
        Set of all stocks in the same component
    
    Time Complexity: O(V + E) for the component
    Space Complexity: O(V)
    """
    if start not in graph.get_nodes():
        return set()
    
    component = {start}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        
        neighbors = graph.get_neighbors(current)
        for neighbor in neighbors:
            if neighbor not in component:
                component.add(neighbor)
                queue.append(neighbor)
    
    return component


def calculate_path_metrics(path: List[str], 
                          graph,
                          stock_attributes: Dict[str, Dict]) -> Dict:
    """
    Calculate metrics for a given path through the graph.
    
    Args:
        path: List of stocks forming a path
        graph: Graph object
        stock_attributes: Dictionary of stock attributes
    
    Returns:
        Dictionary with path metrics
    """
    if not path or len(path) < 2:
        return {
            'length': len(path),
            'avg_correlation': 0.0,
            'avg_volatility': 0.0,
            'min_correlation': 0.0,
            'total_volatility': 0.0
        }
    
    # Calculate correlation along path
    correlations = []
    for i in range(len(path) - 1):
        weight = graph.get_edge_weight(path[i], path[i + 1])
        if weight is not None:
            correlations.append(weight)
    
    # Calculate volatility along path
    volatilities = [
        stock_attributes.get(stock, {}).get('volatility', 0.0)
        for stock in path
    ]
    
    return {
        'length': len(path),
        'avg_correlation': sum(correlations) / len(correlations) if correlations else 0.0,
        'min_correlation': min(correlations) if correlations else 0.0,
        'avg_volatility': sum(volatilities) / len(volatilities) if volatilities else 0.0,
        'total_volatility': sum(volatilities),
        'path': path
    }


def analyze_graph_connectivity(graph) -> Dict:
    """
    Analyze overall connectivity of the graph using BFS.
    
    Args:
        graph: Graph object
    
    Returns:
        Dictionary with connectivity analysis
    """
    nodes = graph.get_nodes()
    if not nodes:
        return {'num_components': 0, 'components': [], 'avg_component_size': 0}
    
    visited = set()
    components = []
    
    for node in nodes:
        if node not in visited:
            # Find component using BFS
            component = bfs_connected_component(graph, node)
            components.append(component)
            visited.update(component)
    
    component_sizes = [len(c) for c in components]
    
    return {
        'num_components': len(components),
        'component_sizes': component_sizes,
        'avg_component_size': sum(component_sizes) / len(component_sizes),
        'largest_component_size': max(component_sizes) if component_sizes else 0,
        'smallest_component_size': min(component_sizes) if component_sizes else 0
    }


def calculate_average_path_length(graph, sample_size: int = 100) -> float:
    """
    Calculate average shortest path length in the graph (sampling approach).
    
    Args:
        graph: Graph object
        sample_size: Number of random node pairs to sample
    
    Returns:
        Average shortest path length
    
    Time Complexity: O(sample_size * (V + E))
    """
    import random
    
    nodes = graph.get_nodes()
    if len(nodes) < 2:
        return 0.0
    
    total_length = 0
    count = 0
    
    # Sample random pairs
    for _ in range(min(sample_size, len(nodes) * (len(nodes) - 1) // 2)):
        start, end = random.sample(nodes, 2)
        path = bfs_shortest_path(graph, start, end)
        
        if path:
            total_length += len(path) - 1  # Number of edges in path
            count += 1
    
    return total_length / count if count > 0 else 0.0


def main():
    """
    Example usage and testing of BFS algorithms.
    """
    print("=== BFS Algorithm Demo ===\n")
    
    # Import graph module for example
    from graph import Graph
    
    # Create example graph
    graph = Graph()
    
    # Add nodes
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    for stock in stocks:
        graph.add_node(stock, {"volatility": 0.02 + hash(stock) % 10 / 100})
    
    # Add edges (correlations)
    edges = [
        ("AAPL", "MSFT", 0.8),
        ("MSFT", "GOOGL", 0.7),
        ("GOOGL", "AMZN", 0.6),
        ("AAPL", "GOOGL", 0.65),
        ("TSLA", "META", 0.5),
        ("META", "NVDA", 0.7),
    ]
    
    for s1, s2, weight in edges:
        graph.add_edge(s1, s2, weight)
    
    print(f"Graph: {graph}")
    
    # Test shortest path
    print("\n--- Shortest Path ---")
    path = bfs_shortest_path(graph, "AAPL", "AMZN")
    if path:
        print(f"Path from AAPL to AMZN: {' -> '.join(path)}")
        print(f"Path length: {len(path) - 1} edges")
    else:
        print("No path found")
    
    # Test connectivity analysis
    print("\n--- Connectivity Analysis ---")
    analysis = analyze_graph_connectivity(graph)
    print(f"Number of components: {analysis['num_components']}")
    print(f"Component sizes: {analysis['component_sizes']}")
    
    # Test component finding
    print("\n--- Connected Components ---")
    comp = bfs_connected_component(graph, "AAPL")
    print(f"Component containing AAPL: {sorted(comp)}")


if __name__ == "__main__":
    main()
