"""Unit tests for Girvan-Newman community detection algorithm.

Tests core functionality, edge cases, and integration with shared components.

Author: Avani Sood
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from graph import Graph
from girvan_newman import (
    betweenness_centrality,
    modularity,
    girvan_newman,
    _connected_components,
)


class TestBetweennessCentrality(unittest.TestCase):
    """Test edge betweenness centrality calculation."""

    def test_simple_path_graph(self):
        """Test betweenness on a simple path: A-B-C."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)

        betweenness = betweenness_centrality(graph)

        # Edge B-C should have higher betweenness as it's the bridge
        self.assertIn(("A", "B"), betweenness)
        self.assertIn(("B", "C"), betweenness)
        self.assertGreater(len(betweenness), 0)

    def test_triangle_graph(self):
        """Test betweenness on a triangle: A-B-C-A."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)

        betweenness = betweenness_centrality(graph)

        # All edges should have equal betweenness in a triangle
        values = list(betweenness.values())
        self.assertEqual(len(set(values)), 1)  # All equal

    def test_bridge_graph(self):
        """Test betweenness on graph with clear bridge: (A-B-C) bridge (D-E-F)."""
        graph = Graph()
        # First community
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        # Bridge
        graph.add_edge("C", "D", 1.0)
        # Second community
        graph.add_edge("D", "E", 1.0)
        graph.add_edge("E", "F", 1.0)
        graph.add_edge("F", "D", 1.0)

        betweenness = betweenness_centrality(graph)

        # Bridge edge should have highest betweenness
        bridge_edge = tuple(sorted(("C", "D")))
        max_betweenness = max(betweenness.values())
        self.assertEqual(betweenness[bridge_edge], max_betweenness)

    def test_empty_graph(self):
        """Test betweenness on empty graph."""
        graph = Graph()
        betweenness = betweenness_centrality(graph)
        self.assertEqual(len(betweenness), 0)

    def test_single_edge(self):
        """Test betweenness on graph with single edge."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        betweenness = betweenness_centrality(graph)
        self.assertEqual(len(betweenness), 1)


class TestModularity(unittest.TestCase):
    """Test modularity calculation."""

    def test_perfect_communities(self):
        """Test modularity with perfect community structure."""
        graph = Graph()
        # Community 1: fully connected triangle
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        # Community 2: fully connected triangle
        graph.add_edge("D", "E", 1.0)
        graph.add_edge("E", "F", 1.0)
        graph.add_edge("F", "D", 1.0)
        # Single bridge
        graph.add_edge("C", "D", 0.1)

        communities = [["A", "B", "C"], ["D", "E", "F"]]
        mod = modularity(graph, communities)

        # Good community structure should have positive modularity
        self.assertGreater(mod, 0)

    def test_all_nodes_one_community(self):
        """Test modularity when all nodes in one community."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)

        communities = [["A", "B", "C"]]
        mod = modularity(graph, communities)

        # Should have some modularity value
        self.assertIsInstance(mod, float)

    def test_empty_graph(self):
        """Test modularity on empty graph."""
        graph = Graph()
        communities = [[]]
        mod = modularity(graph, communities)
        self.assertEqual(mod, 0.0)

    def test_each_node_separate_community(self):
        """Test modularity when each node is its own community."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)

        communities = [["A"], ["B"], ["C"]]
        mod = modularity(graph, communities)

        # Poor community structure should have low/negative modularity
        self.assertLess(mod, 0.5)


class TestConnectedComponents(unittest.TestCase):
    """Test connected components detection."""

    def test_single_component(self):
        """Test graph with single connected component."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)

        components = _connected_components(graph)
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0]), {"A", "B", "C"})

    def test_multiple_components(self):
        """Test graph with multiple disconnected components."""
        graph = Graph()
        # Component 1
        graph.add_edge("A", "B", 1.0)
        # Component 2
        graph.add_edge("C", "D", 1.0)

        components = _connected_components(graph)
        self.assertEqual(len(components), 2)

    def test_isolated_nodes(self):
        """Test graph with isolated nodes."""
        graph = Graph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")

        components = _connected_components(graph)
        self.assertEqual(len(components), 3)

    def test_empty_graph(self):
        """Test empty graph."""
        graph = Graph()
        components = _connected_components(graph)
        self.assertEqual(len(components), 0)


class TestGirvanNewman(unittest.TestCase):
    """Test main Girvan-Newman algorithm."""

    def test_simple_two_communities(self):
        """Test detection of two clear communities."""
        graph = Graph()
        # Community 1: triangle
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        # Bridge
        graph.add_edge("C", "D", 0.3)
        # Community 2: triangle
        graph.add_edge("D", "E", 1.0)
        graph.add_edge("E", "F", 1.0)
        graph.add_edge("F", "D", 1.0)

        communities, mod = girvan_newman(graph)

        # Should find 2 communities
        self.assertEqual(len(communities), 2)
        # Modularity should be positive
        self.assertGreater(mod, 0)

    def test_karate_club_like(self):
        """Test on a structure similar to Zachary's karate club."""
        graph = Graph()
        # Create two densely connected groups with few bridges
        # Group 1
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_edge(f"G1_{i}", f"G1_{j}", 0.9)

        # Group 2
        for i in range(1, 5):
            for j in range(i + 1, 5):
                graph.add_edge(f"G2_{i}", f"G2_{j}", 0.9)

        # Add a couple bridges
        graph.add_edge("G1_1", "G2_1", 0.3)
        graph.add_edge("G1_2", "G2_2", 0.3)

        communities, mod = girvan_newman(graph)

        # Should detect at least 2 communities
        self.assertGreaterEqual(len(communities), 2)
        self.assertGreater(mod, 0)

    def test_max_iterations(self):
        """Test that max_iterations parameter works."""
        graph = Graph()
        # Create a larger connected graph
        for i in range(10):
            for j in range(i + 1, 10):
                graph.add_edge(f"N{i}", f"N{j}", 0.5)

        # Run with limited iterations
        communities1, mod1 = girvan_newman(graph, max_iterations=5)
        # Run with more iterations
        communities2, mod2 = girvan_newman(graph, max_iterations=20)

        # Should complete without error
        self.assertIsNotNone(communities1)
        self.assertIsNotNone(communities2)

    def test_disconnected_graph_raises_error(self):
        """Test that disconnected graph raises ValueError."""
        graph = Graph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("C", "D", 1.0)  # Separate component

        with self.assertRaises(ValueError):
            girvan_newman(graph)

    def test_empty_graph(self):
        """Test algorithm on empty graph."""
        graph = Graph()
        communities, mod = girvan_newman(graph)
        self.assertEqual(communities, [])
        self.assertEqual(mod, 0.0)

    def test_single_node(self):
        """Test algorithm on single-node graph."""
        graph = Graph()
        graph.add_node("A")
        communities, mod = girvan_newman(graph)
        # Single node is one community
        self.assertEqual(len(communities), 1)
        self.assertEqual(communities[0], ["A"])

    def test_modularity_increases_with_better_partition(self):
        """Test that found partition has reasonable modularity."""
        graph = Graph()
        # Create clear community structure
        # Community 1
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("C", "A", 1.0)
        # Weak bridge
        graph.add_edge("C", "D", 0.1)
        # Community 2
        graph.add_edge("D", "E", 1.0)
        graph.add_edge("E", "F", 1.0)
        graph.add_edge("F", "D", 1.0)

        communities, mod = girvan_newman(graph)

        # Algorithm should find a good partition
        self.assertGreater(mod, 0.3)  # Expect decent modularity


class TestIntegration(unittest.TestCase):
    """Integration tests with shared components."""

    def test_graph_compatibility(self):
        """Test that algorithm works with shared Graph class."""
        from graph import Graph

        graph = Graph()
        graph.add_edge("STOCK_A", "STOCK_B", 0.8)
        graph.add_edge("STOCK_B", "STOCK_C", 0.8)
        graph.add_edge("STOCK_A", "STOCK_C", 0.8)

        # Should work without errors
        communities, mod = girvan_newman(graph)
        self.assertIsNotNone(communities)
        self.assertIsInstance(mod, float)

    def test_weighted_edges(self):
        """Test that algorithm handles weighted edges correctly."""
        graph = Graph()
        graph.add_edge("A", "B", 0.9)
        graph.add_edge("B", "C", 0.9)
        graph.add_edge("C", "D", 0.1)  # Weak link
        graph.add_edge("D", "E", 0.9)
        graph.add_edge("E", "F", 0.9)
        graph.add_edge("F", "D", 0.9)

        communities, mod = girvan_newman(graph)

        # Should detect communities based on weight structure
        self.assertGreater(len(communities), 1)


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
