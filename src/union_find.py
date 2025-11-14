"""
Union-Find (Disjoint Set Union) Algorithm
Author: Guntesh Singh
Description: Implementation of Union-Find data structure for finding connected
             components (market segments) in the stock correlation graph.
             Uses path compression and union by rank optimizations.
"""

from typing import Dict, List, Set, Tuple


class UnionFind:
    """
    Union-Find data structure for efficient connected component detection.
    
    This implementation uses:
    - Path compression: During find(), compress paths for faster future lookups
    - Union by rank: Attach smaller trees under larger ones to keep tree flat
    
    Time Complexity:
        - find(): O(α(n)) amortized, where α is inverse Ackermann (nearly O(1))
        - union(): O(α(n)) amortized
        - Overall for m operations on n elements: O(m * α(n)) ≈ O(m)
    
    Space Complexity: O(n) for parent and rank arrays
    """
    
    def __init__(self, elements: List[str]):
        """
        Initialize Union-Find structure with given elements.
        
        Args:
            elements: List of element identifiers (stock names)
        """
        # Each element starts as its own parent (separate set)
        self.parent: Dict[str, str] = {elem: elem for elem in elements}
        
        # Rank represents approximate depth of tree (for union by rank)
        self.rank: Dict[str, int] = {elem: 0 for elem in elements}
        
        # Number of disjoint sets
        self.num_components = len(elements)
        
        # Track component sizes
        self.component_size: Dict[str, int] = {elem: 1 for elem in elements}
    
    def find(self, elem: str) -> str:
        """
        Find the root/representative of the set containing elem.
        
        Uses path compression: all nodes on path point directly to root.
        
        Args:
            elem: Element to find root for
        
        Returns:
            Root element of the set containing elem
        
        Time Complexity: O(α(n)) amortized
        """
        if elem not in self.parent:
            raise ValueError(f"Element {elem} not in Union-Find structure")
        
        # Path compression: make elem point directly to root
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])
        
        return self.parent[elem]
    
    def union(self, elem1: str, elem2: str) -> bool:
        """
        Unite the sets containing elem1 and elem2.
        
        Uses union by rank: attach tree with smaller rank under tree with larger rank.
        
        Args:
            elem1: First element
            elem2: Second element
        
        Returns:
            True if union was performed (elements were in different sets),
            False if elements were already in same set
        
        Time Complexity: O(α(n)) amortized
        """
        root1 = self.find(elem1)
        root2 = self.find(elem2)
        
        # Already in same set
        if root1 == root2:
            return False
        
        # Union by rank: attach smaller tree under larger tree
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            self.component_size[root2] += self.component_size[root1]
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            self.component_size[root1] += self.component_size[root2]
        else:
            # Equal rank: make one root and increase its rank
            self.parent[root2] = root1
            self.rank[root1] += 1
            self.component_size[root1] += self.component_size[root2]
        
        # Decrease number of components
        self.num_components -= 1
        return True
    
    def connected(self, elem1: str, elem2: str) -> bool:
        """
        Check if two elements are in the same set.
        
        Args:
            elem1: First element
            elem2: Second element
        
        Returns:
            True if elements are in same component, False otherwise
        
        Time Complexity: O(α(n)) amortized
        """
        return self.find(elem1) == self.find(elem2)
    
    def get_component_size(self, elem: str) -> int:
        """
        Get size of the component containing elem.
        
        Args:
            elem: Element to query
        
        Returns:
            Number of elements in the component
        """
        root = self.find(elem)
        return self.component_size[root]
    
    def get_components(self) -> Dict[str, Set[str]]:
        """
        Get all connected components.
        
        Returns:
            Dictionary mapping root elements to sets of all elements in component
        
        Time Complexity: O(n * α(n))
        """
        components: Dict[str, Set[str]] = {}
        
        for elem in self.parent.keys():
            root = self.find(elem)
            if root not in components:
                components[root] = set()
            components[root].add(elem)
        
        return components
    
    def get_component_list(self) -> List[Set[str]]:
        """
        Get all connected components as a list of sets.
        
        Returns:
            List of sets, each containing elements in a component
        """
        return list(self.get_components().values())
    
    def get_num_components(self) -> int:
        """
        Get the number of disjoint components.
        
        Returns:
            Number of connected components
        """
        return self.num_components


def find_market_segments(graph) -> Tuple[UnionFind, List[Set[str]]]:
    """
    Find market segments (connected components) in stock correlation graph.
    
    This function uses Union-Find to identify groups of stocks that are
    connected through correlation relationships.
    
    Args:
        graph: Graph object representing stock correlations
    
    Returns:
        Tuple of (UnionFind object, list of component sets)
    
    Time Complexity: O(E * α(V)) where E is edges, V is vertices
    Space Complexity: O(V)
    """
    # Initialize Union-Find with all stocks
    nodes = graph.get_nodes()
    uf = UnionFind(nodes)
    
    # Process all edges to build components
    edges = graph.get_edges()
    for node1, node2, weight in edges:
        uf.union(node1, node2)
    
    # Get final components
    components = uf.get_component_list()
    
    return uf, components


def analyze_market_segments(uf: UnionFind, 
                           graph,
                           stock_attributes: Dict[str, Dict]) -> Dict:
    """
    Analyze characteristics of market segments.
    
    Args:
        uf: UnionFind object with identified components
        graph: Graph object
        stock_attributes: Dictionary of stock attributes
    
    Returns:
        Dictionary with segment analysis results
    """
    components = uf.get_components()
    
    analysis = {
        'num_segments': uf.get_num_components(),
        'segments': []
    }
    
    for root, stocks in components.items():
        stocks_list = list(stocks)
        
        # Calculate average volatility and other metrics for segment
        volatilities = [stock_attributes.get(s, {}).get('volatility', 0) 
                       for s in stocks_list]
        avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
        
        # Count categories
        categories = [stock_attributes.get(s, {}).get('category', 'unknown') 
                     for s in stocks_list]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Dominant category
        dominant_category = max(category_counts.items(), 
                               key=lambda x: x[1])[0] if category_counts else "unknown"
        
        segment_info = {
            'size': len(stocks_list),
            'stocks': stocks_list[:10],  # Show first 10 for readability
            'avg_volatility': avg_volatility,
            'dominant_category': dominant_category,
            'category_distribution': category_counts
        }
        
        analysis['segments'].append(segment_info)
    
    # Sort segments by size (largest first)
    analysis['segments'].sort(key=lambda x: x['size'], reverse=True)
    
    return analysis


def main():
    """
    Example usage and testing of Union-Find algorithm.
    """
    print("=== Union-Find Algorithm Demo ===\n")
    
    # Create example with stocks
    stocks = [f"STOCK_{i}" for i in range(10)]
    uf = UnionFind(stocks)
    
    print(f"Initial components: {uf.get_num_components()}")
    
    # Simulate connections (edges in graph)
    connections = [
        ("STOCK_0", "STOCK_1"),
        ("STOCK_1", "STOCK_2"),
        ("STOCK_3", "STOCK_4"),
        ("STOCK_5", "STOCK_6"),
        ("STOCK_6", "STOCK_7"),
        ("STOCK_8", "STOCK_9"),
    ]
    
    print("\nCreating connections:")
    for s1, s2 in connections:
        uf.union(s1, s2)
        print(f"  Connected {s1} - {s2}")
    
    print(f"\nFinal components: {uf.get_num_components()}")
    
    # Show components
    components = uf.get_component_list()
    print("\nMarket Segments:")
    for i, component in enumerate(sorted(components, key=len, reverse=True), 1):
        print(f"  Segment {i} (size {len(component)}): {sorted(component)}")
    
    # Test connectivity
    print("\nConnectivity tests:")
    test_pairs = [("STOCK_0", "STOCK_2"), ("STOCK_0", "STOCK_5"), ("STOCK_3", "STOCK_4")]
    for s1, s2 in test_pairs:
        connected = uf.connected(s1, s2)
        print(f"  {s1} and {s2}: {'Connected' if connected else 'Not connected'}")


if __name__ == "__main__":
    main()
