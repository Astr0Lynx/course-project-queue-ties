"""
Demonstration Script for Khushi's Algorithm Implementations
Author: Khushi Dhingra
Description: Demonstrates DFS, PageRank, and Market Disruption algorithms
             working together on sample stock correlation data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import StockDataGenerator
from src.graph import build_graph_from_correlation
from src.dfs import DFS, analyze_market_connectivity
from src.pagerank import PageRank, identify_market_influencers
from src.market_disruption import MarketDisruptionSimulator
from src.testing import run_complete_test_suite

def run_demo():
    """Run a comprehensive demonstration of all implemented algorithms."""
    
    print("=" * 80)
    print("STOCK MARKET TANGLE - ALGORITHM DEMONSTRATION")
    print("Author: Khushi Dhingra")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. GENERATING SAMPLE MARKET DATA")
    print("-" * 40)
    
    generator = StockDataGenerator(seed=42)
    returns, corr_matrix, stock_attributes = generator.generate_dataset(50, scenario="normal")
    graph = build_graph_from_correlation(corr_matrix, stock_attributes, threshold=0.5)
    
    print(f"✓ Generated {graph.num_nodes} stocks with {graph.num_edges} correlations")
    print(f"✓ Network density: {(2 * graph.num_edges) / (graph.num_nodes * (graph.num_nodes - 1)):.3f}")
    
    # Demonstrate DFS algorithms
    print("\n2. DFS ALGORITHM DEMONSTRATION")
    print("-" * 40)
    
    dfs = DFS(graph)
    nodes = graph.get_nodes()
    
    if nodes:
        start_node = nodes[0]
        target_node = nodes[-1] if len(nodes) > 1 else nodes[0]
        
        # Basic traversal
        recursive_traversal = dfs.dfs_recursive(start_node)
        iterative_traversal = dfs.dfs_iterative(start_node)
        
        print(f"✓ Recursive DFS from {start_node}: {len(recursive_traversal)} nodes visited")
        print(f"✓ Iterative DFS from {start_node}: {len(iterative_traversal)} nodes visited")
        print(f"✓ Traversal consistency: {set(recursive_traversal) == set(iterative_traversal)}")
        
        # Path finding
        path = dfs.find_path(start_node, target_node)
        if path:
            print(f"✓ Path from {start_node} to {target_node}: {len(path)} hops")
        else:
            print(f"✓ No path found from {start_node} to {target_node}")
        
        # Connected components
        components = dfs.find_connected_components()
        print(f"✓ Connected components: {len(components)}")
        print(f"✓ Largest component: {max(len(comp) for comp in components)} stocks")
        
        # Connectivity analysis
        connectivity_info = dfs.get_connectivity_info()
        print(f"✓ Network connectivity ratio: {connectivity_info['connectivity_ratio']:.3f}")
        print(f"✓ Network has cycles: {connectivity_info['has_cycle']}")
    
    # Demonstrate PageRank algorithm
    print("\n3. PAGERANK ALGORITHM DEMONSTRATION")
    print("-" * 40)
    
    pagerank = PageRank(graph, damping_factor=0.85)
    scores = pagerank.calculate_pagerank(max_iterations=100, tolerance=1e-6)
    
    print(f"✓ PageRank converged in {pagerank.iterations} iterations")
    print(f"✓ Score sum: {sum(scores.values()):.6f} (should be ~1.0)")
    
    # Top influencers
    top_stocks = pagerank.get_top_stocks(5)
    print("✓ Top 5 Market Influencers:")
    for i, (stock, score) in enumerate(top_stocks, 1):
        sector = stock_attributes.get(stock, {}).get('sector', 'Unknown')
        print(f"   {i}. {stock} (Score: {score:.4f}, Sector: {sector})")
    
    # Sector influence analysis
    sector_analysis = pagerank.sector_influence_analysis(stock_attributes)
    if 'sector_ranking' in sector_analysis:
        print("\n✓ Top 3 Most Influential Sectors:")
        for i, (sector, data) in enumerate(sector_analysis['sector_ranking'][:3], 1):
            print(f"   {i}. {sector}: {data['total_influence']:.4f} total influence")
    
    # Hidden influencers
    centrality_comparison = pagerank.compare_with_degree_centrality()
    hidden = centrality_comparison.get('hidden_influencers', [])
    if hidden:
        print(f"✓ Found {len(hidden)} hidden influencers (high PageRank, low degree)")
    
    # Demonstrate Market Disruption Simulations
    print("\n4. MARKET DISRUPTION SIMULATION DEMONSTRATION")
    print("-" * 40)
    
    simulator = MarketDisruptionSimulator(graph, stock_attributes)
    
    # Market crash simulation
    print("Running market crash simulation...")
    crash_result = simulator.simulate_market_crash(severity=0.6)
    print(f"✓ Market crash simulation completed")
    print(f"  - Edges added: {crash_result['edges_added']}")
    print(f"  - Network fragility: {crash_result['network_fragility']:.3f}")
    
    # Sector collapse simulation
    sectors = set(attrs.get('sector', 'Unknown') for attrs in stock_attributes.values() 
                 if attrs.get('sector', 'Unknown') != 'Unknown')
    
    if sectors:
        test_sector = list(sectors)[0]
        print(f"\nRunning {test_sector} sector collapse simulation...")
        sector_result = simulator.simulate_sector_collapse(test_sector)
        print(f"✓ Sector collapse simulation completed")
        print(f"  - Stocks removed: {len(sector_result['stocks_removed'])}")
        print(f"  - Systemic risk score: {sector_result['systemic_risk_score']:.3f}")
    
    # Key player removal simulation
    print("\nRunning key player removal simulation...")
    removal_result = simulator.simulate_key_player_removal('top_pagerank', 3)
    print(f"✓ Key player removal simulation completed")
    print(f"  - Players removed: {len(removal_result['players_removed'])}")
    print(f"  - Network resilience: {removal_result['network_resilience_score']:.3f}")
    
    # Performance demonstration
    print("\n5. PERFORMANCE BENCHMARKING")
    print("-" * 40)
    
    if nodes:
        # DFS performance
        dfs_perf = dfs.benchmark_performance(nodes[0], num_runs=10)
        print(f"✓ DFS Performance (10 runs average):")
        print(f"  - Recursive: {dfs_perf['recursive_avg_time']:.6f}s")
        print(f"  - Iterative: {dfs_perf['iterative_avg_time']:.6f}s")
        
        # PageRank performance
        pr_perf = pagerank.benchmark_performance(max_iterations=100, num_runs=5)
        print(f"✓ PageRank Performance (5 runs average):")
        print(f"  - Weighted: {pr_perf['weighted_avg_time']:.6f}s")
        print(f"  - Unweighted: {pr_perf['unweighted_avg_time']:.6f}s")
    
    print("\n6. MARKET INSIGHTS SUMMARY")
    print("-" * 40)
    
    # Overall market analysis
    market_connectivity = analyze_market_connectivity(graph, stock_attributes)
    market_influence = identify_market_influencers(graph, stock_attributes, top_n=10)
    
    print(f"✓ Market Network Analysis:")
    print(f"  - Connectivity ratio: {market_connectivity['connectivity_ratio']:.3f}")
    print(f"  - Number of components: {market_connectivity['num_components']}")
    print(f"  - Has market cycles: {market_connectivity['has_cycle']}")
    
    print(f"\n✓ Market Influence Analysis:")
    print(f"  - Top influencer: {market_influence['top_influencers'][0][0]} (Score: {market_influence['top_influencers'][0][1]:.4f})")
    print(f"  - Influence concentration: {market_influence['sector_influence']['sector_concentration']:.3f}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("All algorithms working correctly with sample market data.")
    print("=" * 80)


def run_quick_test():
    """Run a quick test to verify all algorithms work correctly."""
    
    print("\nRunning quick correctness test...")
    
    try:
        # Generate small test data
        generator = StockDataGenerator(seed=42)
        returns, corr_matrix, stock_attributes = generator.generate_dataset(20, scenario="normal")
        graph = build_graph_from_correlation(corr_matrix, stock_attributes, threshold=0.5)
        
        # Test DFS
        dfs = DFS(graph)
        nodes = graph.get_nodes()
        if nodes:
            traversal = dfs.dfs_recursive(nodes[0])
            components = dfs.find_connected_components()
            assert len(traversal) > 0, "DFS traversal failed"
            assert len(components) > 0, "Connected components failed"
        
        # Test PageRank
        pagerank = PageRank(graph)
        if graph.num_nodes > 0:
            scores = pagerank.calculate_pagerank()
            assert len(scores) == graph.num_nodes, "PageRank scoring failed"
            assert abs(sum(scores.values()) - 1.0) < 1e-3, "PageRank normalization failed"
        
        # Test Market Disruption
        simulator = MarketDisruptionSimulator(graph, stock_attributes)
        crash_result = simulator.simulate_market_crash(severity=0.3)
        assert 'scenario' in crash_result, "Market disruption failed"
        
        print("✓ All algorithms passed quick test!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Run quick test first
    if run_quick_test():
        # Run full demonstration
        run_demo()
        
        # Optionally run comprehensive test suite
        print("\nTo run comprehensive test suite, uncomment the following line:")
        print("# run_complete_test_suite(save_results=True)")
        
    else:
        print("Quick test failed. Please check algorithm implementations.")