"""
DFS (Depth-First Search) Algorithm Implementation for Stock Market Tangle Project
Author: Khushi Dhingra
Description: Provides DFS traversal capabilities for stock correlation graphs.
             Includes path finding, connectivity analysis, and cycle detection.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import time


class DFS:
    """
    Depth-First Search implementation for graph traversal and analysis.
    
    Provides both recursive and iterative implementations with various
    applications for stock market network analysis.
    """
    
    def __init__(self, graph):
        """
        Initialize DFS with a graph.
        
        Args:
            graph: Graph object from graph.py
        """
        self.graph = graph
        self.visited = set()
        self.traversal_order = []
        self.parent = {}
        self.discovery_time = {}
        self.finish_time = {}
        self.time_counter = 0
    
    def reset(self):
        """Reset internal state for new traversal."""
        self.visited.clear()
        self.traversal_order.clear()
        self.parent.clear()
        self.discovery_time.clear()
        self.finish_time.clear()
        self.time_counter = 0
    
    def dfs_recursive(self, start_node: str, target: Optional[str] = None) -> List[str]:
        """
        Recursive DFS traversal from a starting node.
        
        Args:
            start_node: Starting node for traversal
            target: Optional target node to search for
        
        Returns:
            List of nodes in traversal order
            
        Time Complexity: O(V + E)
        Space Complexity: O(V) for recursion stack
        """
        if start_node not in self.graph.get_nodes():
            return []
        
        self.reset()
        self._dfs_recursive_helper(start_node, target)
        return self.traversal_order.copy()
    
    def _dfs_recursive_helper(self, node: str, target: Optional[str] = None):
        """Helper function for recursive DFS."""
        self.visited.add(node)
        self.traversal_order.append(node)
        self.discovery_time[node] = self.time_counter
        self.time_counter += 1
        
        # Early termination if target found
        if target and node == target:
            return True
        
        # Visit all unvisited neighbors
        neighbors = self.graph.get_neighbors(node)
        for neighbor in sorted(neighbors.keys()):  # Sort for consistent ordering
            if neighbor not in self.visited:
                self.parent[neighbor] = node
                if self._dfs_recursive_helper(neighbor, target):
                    return True
        
        self.finish_time[node] = self.time_counter
        self.time_counter += 1
        return False
    
    def dfs_iterative(self, start_node: str, target: Optional[str] = None) -> List[str]:
        """
        Iterative DFS traversal using stack.
        
        Args:
            start_node: Starting node for traversal
            target: Optional target node to search for
        
        Returns:
            List of nodes in traversal order
            
        Time Complexity: O(V + E)
        Space Complexity: O(V) for stack
        """
        if start_node not in self.graph.get_nodes():
            return []
        
        self.reset()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            
            if node not in self.visited:
                self.visited.add(node)
                self.traversal_order.append(node)
                
                # Early termination if target found
                if target and node == target:
                    break
                
                # Add neighbors to stack (reverse sorted for consistent ordering)
                neighbors = self.graph.get_neighbors(node)
                for neighbor in sorted(neighbors.keys(), reverse=True):
                    if neighbor not in self.visited:
                        stack.append(neighbor)
                        if neighbor not in self.parent:
                            self.parent[neighbor] = node
        
        return self.traversal_order.copy()
    
    def find_path(self, start: str, end: str, method: str = "recursive") -> Optional[List[str]]:
        """
        Find path between two nodes using DFS.
        
        Args:
            start: Starting node
            end: Target node
            method: "recursive" or "iterative"
        
        Returns:
            Path as list of nodes, or None if no path exists
        """
        if method == "recursive":
            self.dfs_recursive(start, end)
        else:
            self.dfs_iterative(start, end)
        
        if end not in self.visited:
            return None
        
        # Reconstruct path from parent relationships
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        path.reverse()
        return path if path[0] == start else None
    
    def find_connected_components(self) -> List[List[str]]:
        """
        Find all connected components in the graph.
        
        Returns:
            List of components, each component is a list of nodes
            
        Time Complexity: O(V + E)
        """
        self.reset()
        components = []
        nodes = self.graph.get_nodes()
        
        for node in nodes:
            if node not in self.visited:
                component_nodes = []
                self._dfs_component_helper(node, component_nodes)
                components.append(component_nodes)
        
        return components
    
    def _dfs_component_helper(self, node: str, component_nodes: List[str]):
        """Helper function for finding connected components."""
        self.visited.add(node)
        component_nodes.append(node)
        
        neighbors = self.graph.get_neighbors(node)
        for neighbor in neighbors:
            if neighbor not in self.visited:
                self._dfs_component_helper(neighbor, component_nodes)
    
    def is_connected(self) -> bool:
        """
        Check if the graph is connected.
        
        Returns:
            True if graph is connected, False otherwise
        """
        nodes = self.graph.get_nodes()
        if not nodes:
            return True
        
        # Start DFS from first node
        self.dfs_recursive(nodes[0])
        
        # Graph is connected if all nodes were visited
        return len(self.visited) == len(nodes)
    
    def has_cycle(self) -> bool:
        """
        Detect if the graph has cycles using DFS.
        
        Returns:
            True if cycle exists, False otherwise
            
        Note: For undirected graphs, we ignore back edges to parent
        """
        self.reset()
        nodes = self.graph.get_nodes()
        
        for node in nodes:
            if node not in self.visited:
                if self._has_cycle_helper(node, None):
                    return True
        
        return False
    
    def _has_cycle_helper(self, node: str, parent: Optional[str]) -> bool:
        """Helper function for cycle detection."""
        self.visited.add(node)
        
        neighbors = self.graph.get_neighbors(node)
        for neighbor in neighbors:
            if neighbor not in self.visited:
                if self._has_cycle_helper(neighbor, node):
                    return True
            elif neighbor != parent:
                # Found back edge (cycle)
                return True
        
        return False
    
    def get_connectivity_info(self) -> Dict:
        """
        Get comprehensive connectivity information about the graph.
        
        Returns:
            Dictionary with connectivity statistics
        """
        components = self.find_connected_components()
        
        info = {
            'is_connected': len(components) == 1,
            'num_components': len(components),
            'component_sizes': [len(comp) for comp in components],
            'largest_component_size': max([len(comp) for comp in components]) if components else 0,
            'has_cycle': self.has_cycle(),
            'total_nodes': len(self.graph.get_nodes()),
            'total_edges': self.graph.num_edges
        }
        
        # Add component details for stock market analysis
        info['components'] = components
        
        # Calculate connectivity ratio (largest component size / total nodes)
        if info['total_nodes'] > 0:
            info['connectivity_ratio'] = info['largest_component_size'] / info['total_nodes']
        else:
            info['connectivity_ratio'] = 0.0
        
        return info
    
    def benchmark_performance(self, start_node: str, num_runs: int = 10) -> Dict:
        """
        Benchmark performance of recursive vs iterative DFS.
        
        Args:
            start_node: Node to start traversal from
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with performance metrics
        """
        if start_node not in self.graph.get_nodes():
            return {'error': 'Start node not in graph'}
        
        # Benchmark recursive DFS
        recursive_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.dfs_recursive(start_node)
            end_time = time.time()
            recursive_times.append(end_time - start_time)
        
        # Benchmark iterative DFS
        iterative_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.dfs_iterative(start_node)
            end_time = time.time()
            iterative_times.append(end_time - start_time)
        
        return {
            'recursive_avg_time': sum(recursive_times) / len(recursive_times),
            'iterative_avg_time': sum(iterative_times) / len(iterative_times),
            'recursive_min_time': min(recursive_times),
            'iterative_min_time': min(iterative_times),
            'nodes_traversed': len(self.traversal_order),
            'graph_size': len(self.graph.get_nodes()),
            'num_runs': num_runs
        }


def analyze_market_connectivity(graph, stock_attributes: Dict) -> Dict:
    """
    Analyze market connectivity using DFS for stock correlation networks.
    
    Args:
        graph: Stock correlation graph
        stock_attributes: Dictionary of stock attributes
    
    Returns:
        Dictionary with market connectivity analysis
    """
    dfs = DFS(graph)
    connectivity_info = dfs.get_connectivity_info()
    
    # Add sector analysis for connected components
    sector_analysis = defaultdict(lambda: defaultdict(int))
    
    for component in connectivity_info['components']:
        component_sectors = defaultdict(int)
        for stock in component:
            if stock in stock_attributes:
                sector = stock_attributes[stock].get('sector', 'Unknown')
                component_sectors[sector] += 1
        
        # Find dominant sector in this component
        if component_sectors:
            dominant_sector = max(component_sectors.items(), key=lambda x: x[1])
            sector_analysis[f'Component_{len(component)}_stocks'] = {
                'dominant_sector': dominant_sector[0],
                'sector_distribution': dict(component_sectors),
                'stocks': component
            }
    
    connectivity_info['sector_analysis'] = dict(sector_analysis)
    
    return connectivity_info