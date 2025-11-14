"""
Unit Tests for Union-Find Algorithm
Author: Guntesh Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from union_find import UnionFind, find_market_segments, analyze_market_segments


def test_union_find_initialization():
    """Test Union-Find initialization."""
    elements = ["A", "B", "C", "D"]
    uf = UnionFind(elements)
    
    assert uf.get_num_components() == 4
    assert all(uf.find(e) == e for e in elements)


def test_union_find_union():
    """Test union operation."""
    elements = ["A", "B", "C", "D"]
    uf = UnionFind(elements)
    
    # Union A and B
    result = uf.union("A", "B")
    assert result == True  # Union performed
    assert uf.get_num_components() == 3
    assert uf.connected("A", "B")
    
    # Try to union again (should return False)
    result = uf.union("A", "B")
    assert result == False


def test_union_find_connectivity():
    """Test connectivity checks."""
    elements = ["A", "B", "C", "D", "E"]
    uf = UnionFind(elements)
    
    uf.union("A", "B")
    uf.union("B", "C")
    
    assert uf.connected("A", "C")
    assert not uf.connected("A", "D")


def test_union_find_components():
    """Test component retrieval."""
    elements = [f"S{i}" for i in range(6)]
    uf = UnionFind(elements)
    
    # Create two components
    uf.union("S0", "S1")
    uf.union("S1", "S2")
    uf.union("S3", "S4")
    
    components = uf.get_component_list()
    assert len(components) == 3  # 3 components
    
    # Check sizes
    sizes = sorted([len(c) for c in components], reverse=True)
    assert sizes == [3, 2, 1]


def test_union_find_component_size():
    """Test component size tracking."""
    elements = ["A", "B", "C", "D"]
    uf = UnionFind(elements)
    
    uf.union("A", "B")
    uf.union("B", "C")
    
    assert uf.get_component_size("A") == 3
    assert uf.get_component_size("B") == 3
    assert uf.get_component_size("C") == 3
    assert uf.get_component_size("D") == 1


def test_find_market_segments():
    """Test market segment detection with a graph."""
    from graph import Graph
    
    graph = Graph()
    stocks = [f"STOCK_{i}" for i in range(8)]
    for stock in stocks:
        graph.add_node(stock, {"volatility": 0.02})
    
    # Create two segments
    graph.add_edge("STOCK_0", "STOCK_1", 0.7)
    graph.add_edge("STOCK_1", "STOCK_2", 0.6)
    graph.add_edge("STOCK_3", "STOCK_4", 0.8)
    graph.add_edge("STOCK_4", "STOCK_5", 0.7)
    
    uf, components = find_market_segments(graph)
    
    # Should have 4 components (2 pairs, 2 singles)
    assert uf.get_num_components() == 4
    assert len(components) == 4


def test_path_compression():
    """Test that path compression works correctly."""
    elements = [f"N{i}" for i in range(10)]
    uf = UnionFind(elements)
    
    # Create a chain: N0-N1-N2-N3-N4
    for i in range(4):
        uf.union(f"N{i}", f"N{i+1}")
    
    # Find should compress paths
    root = uf.find("N4")
    
    # All should have same root
    for i in range(5):
        assert uf.find(f"N{i}") == root


def run_all_tests():
    """Run all tests."""
    print("Running Union-Find Tests...\n")
    
    tests = [
        ("Initialization", test_union_find_initialization),
        ("Union Operation", test_union_find_union),
        ("Connectivity", test_union_find_connectivity),
        ("Components", test_union_find_components),
        ("Component Size", test_union_find_component_size),
        ("Market Segments", test_find_market_segments),
        ("Path Compression", test_path_compression),
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
