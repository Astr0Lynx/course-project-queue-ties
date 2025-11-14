"""
Graph Representation Module for Stock Market Tangle Project
Author: Guntesh Singh
Description: Provides helper functions for graph representation and manipulation
             using adjacency lists. Shared with the team for consistent graph structure.
"""

from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import json


class Graph:
    """
    Graph representation using adjacency list structure.
    
    This class provides a shared graph structure for the team to use.
    Uses dict of dicts for efficient edge weight storage and neighbor lookup.
    
    Structure:
        adjacency_list = {
            node1: {neighbor1: weight1, neighbor2: weight2, ...},
            node2: {neighbor1: weight1, ...},
            ...
        }
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.adjacency_list: Dict[str, Dict[str, float]] = {}
        self.node_attributes: Dict[str, Dict] = {}
        self.num_nodes = 0
        self.num_edges = 0
    
    def add_node(self, node: str, attributes: Optional[Dict] = None) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: Node identifier (stock name)
            attributes: Optional dictionary of node attributes
        """
        if node not in self.adjacency_list:
            self.adjacency_list[node] = {}
            self.num_nodes += 1
        
        if attributes:
            self.node_attributes[node] = attributes
    
    def add_edge(self, node1: str, node2: str, weight: float = 1.0) -> None:
        """
        Add an undirected edge between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            weight: Edge weight (correlation value)
        """
        # Ensure nodes exist
        if node1 not in self.adjacency_list:
            self.add_node(node1)
        if node2 not in self.adjacency_list:
            self.add_node(node2)
        
        # Add edge (undirected, so add both directions)
        if node2 not in self.adjacency_list[node1]:
            self.adjacency_list[node1][node2] = weight
            self.adjacency_list[node2][node1] = weight
            self.num_edges += 1
    
    def remove_edge(self, node1: str, node2: str) -> None:
        """
        Remove an edge between two nodes.
        
        Args:
            node1: First node
            node2: Second node
        """
        if node1 in self.adjacency_list and node2 in self.adjacency_list[node1]:
            del self.adjacency_list[node1][node2]
            del self.adjacency_list[node2][node1]
            self.num_edges -= 1
    
    def get_neighbors(self, node: str) -> Dict[str, float]:
        """
        Get all neighbors of a node with edge weights.
        
        Args:
            node: Node identifier
        
        Returns:
            Dictionary mapping neighbor nodes to edge weights
        """
        return self.adjacency_list.get(node, {})
    
    def get_nodes(self) -> List[str]:
        """
        Get list of all nodes in the graph.
        
        Returns:
            List of node identifiers
        """
        return list(self.adjacency_list.keys())
    
    def get_edges(self) -> List[Tuple[str, str, float]]:
        """
        Get list of all edges in the graph.
        
        Returns:
            List of tuples (node1, node2, weight)
        """
        edges = []
        seen = set()
        
        for node1, neighbors in self.adjacency_list.items():
            for node2, weight in neighbors.items():
                # Avoid duplicates in undirected graph
                edge_key = tuple(sorted([node1, node2]))
                if edge_key not in seen:
                    edges.append((node1, node2, weight))
                    seen.add(edge_key)
        
        return edges
    
    def get_degree(self, node: str) -> int:
        """
        Get degree (number of neighbors) of a node.
        
        Args:
            node: Node identifier
        
        Returns:
            Number of neighbors
        """
        return len(self.adjacency_list.get(node, {}))
    
    def has_edge(self, node1: str, node2: str) -> bool:
        """
        Check if an edge exists between two nodes.
        
        Args:
            node1: First node
            node2: Second node
        
        Returns:
            True if edge exists, False otherwise
        """
        return node2 in self.adjacency_list.get(node1, {})
    
    def get_edge_weight(self, node1: str, node2: str) -> Optional[float]:
        """
        Get weight of edge between two nodes.
        
        Args:
            node1: First node
            node2: Second node
        
        Returns:
            Edge weight or None if edge doesn't exist
        """
        return self.adjacency_list.get(node1, {}).get(node2)
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return (f"Graph(nodes={self.num_nodes}, edges={self.num_edges})")
    
    def __repr__(self) -> str:
        """Representation of the graph."""
        return self.__str__()


def build_graph_from_correlation(correlation_matrix: pd.DataFrame,
                                 stock_attributes: Dict[str, Dict],
                                 threshold: float = 0.5) -> Graph:
    """
    Build a stock correlation graph from correlation matrix.
    
    Edges are created between stocks with correlation above threshold.
    Edge weights represent correlation strength.
    
    Args:
        correlation_matrix: Pandas DataFrame with stock correlations
        stock_attributes: Dictionary of stock attributes
        threshold: Minimum correlation to create an edge (default 0.5)
    
    Returns:
        Graph object representing stock correlations
    
    Time Complexity: O(V^2) where V is number of stocks
    """
    graph = Graph()
    
    # Add all stocks as nodes with their attributes
    for stock in correlation_matrix.columns:
        attributes = stock_attributes.get(stock, {})
        graph.add_node(stock, attributes)
    
    # Add edges based on correlation threshold
    stocks = list(correlation_matrix.columns)
    for i in range(len(stocks)):
        for j in range(i + 1, len(stocks)):
            stock1, stock2 = stocks[i], stocks[j]
            correlation = correlation_matrix.loc[stock1, stock2]
            
            # Add edge if correlation is above threshold
            # Use absolute value to capture both positive and negative correlations
            if abs(correlation) >= threshold:
                graph.add_edge(stock1, stock2, abs(correlation))
    
    return graph


def build_graph_from_correlation_with_volatility_filter(
        correlation_matrix: pd.DataFrame,
        stock_attributes: Dict[str, Dict],
        correlation_threshold: float = 0.5,
        max_volatility: float = 0.1) -> Graph:
    """
    Build graph with additional volatility filtering for stable paths.
    
    This variant only includes stocks below a volatility threshold,
    useful for finding low-risk correlation paths (for BFS with volatility).
    
    Args:
        correlation_matrix: Pandas DataFrame with stock correlations
        stock_attributes: Dictionary of stock attributes
        correlation_threshold: Minimum correlation to create edge
        max_volatility: Maximum volatility to include stock
    
    Returns:
        Graph with only low-volatility stocks
    """
    graph = Graph()
    
    # Filter stocks by volatility
    valid_stocks = [
        stock for stock, attrs in stock_attributes.items()
        if attrs.get('volatility', float('inf')) <= max_volatility
    ]
    
    # Add filtered stocks as nodes
    for stock in valid_stocks:
        attributes = stock_attributes.get(stock, {})
        graph.add_node(stock, attributes)
    
    # Add edges between valid stocks
    for i in range(len(valid_stocks)):
        for j in range(i + 1, len(valid_stocks)):
            stock1, stock2 = valid_stocks[i], valid_stocks[j]
            correlation = correlation_matrix.loc[stock1, stock2]
            
            if abs(correlation) >= correlation_threshold:
                graph.add_edge(stock1, stock2, abs(correlation))
    
    return graph


def get_graph_statistics(graph: Graph) -> Dict:
    """
    Calculate basic statistics about the graph.
    
    Args:
        graph: Graph object
    
    Returns:
        Dictionary with graph statistics
    """
    degrees = [graph.get_degree(node) for node in graph.get_nodes()]
    
    stats = {
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0,
        'density': (2 * graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))) 
                   if graph.num_nodes > 1 else 0
    }
    
    return stats


def save_graph(graph: Graph, filename: str) -> None:
    """
    Save graph to a JSON file.
    
    Args:
        graph: Graph object
        filename: Output filename
    """
    graph_data = {
        'adjacency_list': graph.adjacency_list,
        'node_attributes': graph.node_attributes,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges
    }
    
    with open(filename, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph saved to {filename}")


def load_graph(filename: str) -> Graph:
    """
    Load graph from a JSON file.
    
    Args:
        filename: Input filename
    
    Returns:
        Graph object
    """
    with open(filename, 'r') as f:
        graph_data = json.load(f)
    
    graph = Graph()
    graph.adjacency_list = graph_data['adjacency_list']
    graph.node_attributes = graph_data['node_attributes']
    graph.num_nodes = graph_data['num_nodes']
    graph.num_edges = graph_data['num_edges']
    
    return graph


def main():
    """
    Example usage of graph utilities.
    """
    print("=== Graph Representation Example ===\n")
    
    # Create a simple example graph
    graph = Graph()
    
    # Add nodes with attributes
    graph.add_node("STOCK_A", {"volatility": 0.02, "category": "stable"})
    graph.add_node("STOCK_B", {"volatility": 0.03, "category": "moderate"})
    graph.add_node("STOCK_C", {"volatility": 0.05, "category": "volatile"})
    graph.add_node("STOCK_D", {"volatility": 0.02, "category": "stable"})
    
    # Add edges
    graph.add_edge("STOCK_A", "STOCK_B", 0.7)
    graph.add_edge("STOCK_B", "STOCK_C", 0.6)
    graph.add_edge("STOCK_A", "STOCK_D", 0.8)
    graph.add_edge("STOCK_C", "STOCK_D", 0.5)
    
    print(graph)
    print(f"\nGraph Statistics:")
    stats = get_graph_statistics(graph)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nNodes: {graph.get_nodes()}")
    print(f"\nNeighbors of STOCK_A: {graph.get_neighbors('STOCK_A')}")


if __name__ == "__main__":
    main()
