"""
Unit Tests for BFS Algorithm
Author: Guntesh Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bfs import (
    bfs_shortest_path, 
    bfs_shortest_path_volatility_weighted,
    bfs_all_paths_bounded,
    bfs_connected_component,
    calculate_path_metrics,
    analyze_graph_connectivity
)
from graph import Graph


def create_test_graph():
    """Create a test graph for BFS tests."""
    graph = Graph()
    
    # Create a graph with known structure
    #     A --- B --- C
    #     |           |
    #     D --- E --- F
    #           |
    #           G
    
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    for node in nodes:
        volatility = 0.01 + ord(node) % 5 / 100  # Different volatilities
        graph.add_node(node, {"volatility": volatility})
    
    edges = [
        ("A", "B", 0.8),
        ("B", "C", 0.7),
        ("A", "D", 0.6),
        ("D", "E", 0.75),
        ("E", "F", 0.7),
        ("C", "F", 0.65),
        ("E", "G", 0.5),
    ]
    
    for n1, n2, weight in edges:
        graph.add_edge(n1, n2, weight)
    
    return graph


def test_bfs_shortest_path_exists():
    """Test BFS finds shortest path when it exists."""
    graph = create_test_graph()
    
    path = bfs_shortest_path(graph, "A", "F")
    
    assert path is not None
    assert path[0] == "A"
    assert path[-1] == "F"
    # Shortest path should be A-D-E-F (length 4)
    assert len(path) == 4


def test_bfs_shortest_path_same_node():
    """Test BFS with start == end."""
    graph = create_test_graph()
    
    path = bfs_shortest_path(graph, "A", "A")
    
    assert path == ["A"]


def test_bfs_shortest_path_not_exists():
    """Test BFS when no path exists."""
    graph = Graph()
    
    # Create disconnected graph
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_edge("A", "B", 0.5)
    
    path = bfs_shortest_path(graph, "A", "C")
    
    assert path is None


def test_bfs_connected_component():
    """Test finding connected component."""
    graph = create_test_graph()
    
    component = bfs_connected_component(graph, "A")
    
    # All nodes should be in same component
    assert len(component) == 7
    assert "A" in component and "G" in component


def test_bfs_connected_component_disconnected():
    """Test component finding in disconnected graph."""
    graph = Graph()
    
    # Two separate components
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_edge("A", "B", 0.5)
    graph.add_edge("C", "D", 0.5)
    
    component = bfs_connected_component(graph, "A")
    
    assert len(component) == 2
    assert "A" in component and "B" in component
    assert "C" not in component and "D" not in component


def test_bfs_all_paths_bounded():
    """Test finding all paths within distance."""
    graph = create_test_graph()
    
    paths = bfs_all_paths_bounded(graph, "A", max_length=2)
    
    # Should reach A, B, D (depth 1) and C, E (depth 2)
    assert "A" in paths
    assert "B" in paths
    assert "D" in paths
    assert "E" in paths
    
    # Should NOT reach F or G (depth 3)
    assert "F" not in paths or len(paths["F"]) - 1 <= 2


def test_calculate_path_metrics():
    """Test path metrics calculation."""
    graph = create_test_graph()
    stock_attrs = graph.node_attributes
    
    path = ["A", "B", "C"]
    metrics = calculate_path_metrics(path, graph, stock_attrs)
    
    assert metrics['length'] == 3
    assert metrics['avg_correlation'] > 0
    assert 'avg_volatility' in metrics


def test_analyze_graph_connectivity():
    """Test graph connectivity analysis."""
    graph = Graph()
    
    # Create graph with 2 components
    for node in ["A", "B", "C", "D", "E"]:
        graph.add_node(node)
    
    graph.add_edge("A", "B", 0.5)
    graph.add_edge("B", "C", 0.5)
    graph.add_edge("D", "E", 0.5)
    
    analysis = analyze_graph_connectivity(graph)
    
    assert analysis['num_components'] == 2
    assert len(analysis['component_sizes']) == 2
    assert sorted(analysis['component_sizes']) == [2, 3]


def test_bfs_volatility_weighted():
    """Test volatility-weighted BFS."""
    graph = Graph()
    
    # Create path with different volatilities
    graph.add_node("A", {"volatility": 0.02})
    graph.add_node("B", {"volatility": 0.01})  # Low volatility
    graph.add_node("C", {"volatility": 0.05})  # High volatility
    graph.add_node("D", {"volatility": 0.02})
    
    graph.add_edge("A", "B", 0.7)
    graph.add_edge("A", "C", 0.7)
    graph.add_edge("B", "D", 0.7)
    graph.add_edge("C", "D", 0.7)
    
    stock_attrs = graph.node_attributes
    
    path = bfs_shortest_path_volatility_weighted(graph, "A", "D", stock_attrs)
    
    # Should prefer path through B (lower volatility)
    assert path is not None
    assert "B" in path


def run_all_tests():
    """Run all tests."""
    print("Running BFS Tests...\n")
    
    tests = [
        ("Shortest Path - Exists", test_bfs_shortest_path_exists),
        ("Shortest Path - Same Node", test_bfs_shortest_path_same_node),
        ("Shortest Path - Not Exists", test_bfs_shortest_path_not_exists),
        ("Connected Component", test_bfs_connected_component),
        ("Connected Component - Disconnected", test_bfs_connected_component_disconnected),
        ("All Paths Bounded", test_bfs_all_paths_bounded),
        ("Path Metrics", test_calculate_path_metrics),
        ("Connectivity Analysis", test_analyze_graph_connectivity),
        ("Volatility Weighted Path", test_bfs_volatility_weighted),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: Unexpected error - {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
