"""
Main Integration Script - Stock Market Tangle Project
Author: Guntesh Singh
Description: Demonstrates the complete workflow from data generation to analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import StockDataGenerator
from graph import build_graph_from_correlation, get_graph_statistics, save_graph
from union_find import find_market_segments, analyze_market_segments
from bfs import (
    bfs_shortest_path,
    bfs_shortest_path_volatility_weighted,
    analyze_graph_connectivity,
    calculate_path_metrics
)
import json


def main():
    """Main workflow demonstrating all implemented features."""
    
    print("="*70)
    print("STOCK MARKET TANGLE - COMPLETE WORKFLOW")
    print("Implementation by Guntesh Singh")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n" + "="*70)
    print("STEP 1: DATA GENERATION")
    print("="*70)
    
    generator = StockDataGenerator(seed=42)
    num_stocks = 100
    scenario = "normal"
    
    print(f"\nGenerating {num_stocks} stocks in '{scenario}' market scenario...")
    returns, corr_matrix, stock_attributes = generator.generate_dataset(
        num_stocks, 
        scenario=scenario
    )
    
    print(f"✓ Generated {len(returns.columns)} stocks with {len(returns)} days of data")
    print(f"✓ Correlation matrix: {corr_matrix.shape}")
    print(f"✓ Stock attributes: {len(stock_attributes)} stocks")
    
    # Save dataset
    generator.save_dataset(returns, corr_matrix, stock_attributes, 
                          output_dir="data", prefix="demo")
    
    # Step 2: Build graph
    print("\n" + "="*70)
    print("STEP 2: GRAPH CONSTRUCTION")
    print("="*70)
    
    correlation_threshold = 0.5
    print(f"\nBuilding graph with correlation threshold: {correlation_threshold}")
    
    graph = build_graph_from_correlation(
        corr_matrix, 
        stock_attributes, 
        correlation_threshold
    )
    
    stats = get_graph_statistics(graph)
    print(f"\n✓ Graph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Avg Degree: {stats['avg_degree']:.2f}")
    print(f"  Max Degree: {stats['max_degree']}")
    
    # Save graph
    save_graph(graph, "data/demo_graph.json")
    
    # Step 3: Find market segments using Union-Find
    print("\n" + "="*70)
    print("STEP 3: MARKET SEGMENTATION (Union-Find)")
    print("="*70)
    
    print("\nFinding connected components (market segments)...")
    uf, components = find_market_segments(graph)
    
    print(f"\n✓ Found {uf.get_num_components()} market segments")
    
    # Analyze segments
    segment_analysis = analyze_market_segments(uf, graph, stock_attributes)
    
    print("\nTop 5 Largest Segments:")
    for i, segment in enumerate(segment_analysis['segments'][:5], 1):
        print(f"\n  Segment {i}:")
        print(f"    Size: {segment['size']} stocks")
        print(f"    Avg Volatility: {segment['avg_volatility']:.4f}")
        print(f"    Dominant Category: {segment['dominant_category']}")
        print(f"    Sample stocks: {', '.join(segment['stocks'][:5])}")
    
    # Step 4: Analyze paths using BFS
    print("\n" + "="*70)
    print("STEP 4: PATH ANALYSIS (BFS)")
    print("="*70)
    
    # Connectivity analysis
    print("\nAnalyzing graph connectivity...")
    connectivity = analyze_graph_connectivity(graph)
    
    print(f"\n✓ Connectivity Analysis:")
    print(f"  Connected Components: {connectivity['num_components']}")
    print(f"  Largest Component: {connectivity['largest_component_size']} stocks")
    print(f"  Average Component Size: {connectivity['avg_component_size']:.2f}")
    
    # Find some example paths
    print("\n" + "-"*70)
    print("Example Shortest Paths:")
    print("-"*70)
    
    nodes = graph.get_nodes()
    if len(nodes) >= 2:
        # Test a few random paths
        import random
        random.seed(42)
        
        for i in range(3):
            start, end = random.sample(nodes, 2)
            
            # Standard BFS
            path = bfs_shortest_path(graph, start, end)
            
            if path:
                print(f"\n  Path {i+1}: {start} → {end}")
                print(f"    Route: {' → '.join(path[:5])}{'...' if len(path) > 5 else ''}")
                print(f"    Length: {len(path)-1} hops")
                
                # Calculate metrics
                metrics = calculate_path_metrics(path, graph, stock_attributes)
                print(f"    Avg Correlation: {metrics['avg_correlation']:.3f}")
                print(f"    Avg Volatility: {metrics['avg_volatility']:.4f}")
                
                # Compare with volatility-weighted path
                vol_path = bfs_shortest_path_volatility_weighted(
                    graph, start, end, stock_attributes
                )
                
                if vol_path and vol_path != path:
                    vol_metrics = calculate_path_metrics(vol_path, graph, stock_attributes)
                    print(f"    Low-volatility route volatility: {vol_metrics['avg_volatility']:.4f}")
    
    # Step 5: Summary and export
    print("\n" + "="*70)
    print("STEP 5: SUMMARY AND EXPORT")
    print("="*70)
    
    summary = {
        'dataset': {
            'num_stocks': num_stocks,
            'scenario': scenario,
            'correlation_threshold': correlation_threshold
        },
        'graph': stats,
        'segmentation': {
            'num_segments': segment_analysis['num_segments'],
            'largest_segment_size': segment_analysis['segments'][0]['size'] if segment_analysis['segments'] else 0,
        },
        'connectivity': connectivity
    }
    
    # Save summary
    with open('results/demo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Complete workflow summary:")
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/demo_* : Dataset files")
    print("  - results/demo_summary.json : Analysis summary")
    print("\nAll algorithms implemented from scratch:")
    print("  ✓ Data Generation (synthetic stock data)")
    print("  ✓ Graph Representation (adjacency list)")
    print("  ✓ Union-Find (with path compression & union by rank)")
    print("  ✓ BFS (shortest paths & connectivity)")
    print("="*70)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()
