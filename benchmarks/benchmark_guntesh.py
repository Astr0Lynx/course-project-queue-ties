"""
Benchmarking Script for Guntesh's Algorithms
Author: Guntesh Singh
Description: Benchmarks Union-Find and BFS performance across different scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import psutil
import json
from typing import Dict, List
import numpy as np

from data_generation import StockDataGenerator
from graph import build_graph_from_correlation, get_graph_statistics
from union_find import find_market_segments, analyze_market_segments
from bfs import (
    bfs_shortest_path, 
    bfs_shortest_path_volatility_weighted,
    calculate_average_path_length,
    analyze_graph_connectivity
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class BenchmarkRunner:
    """Run benchmarks for Union-Find and BFS algorithms."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize benchmark runner."""
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)
    
    def benchmark_union_find(self, graph, stock_attributes: Dict, 
                            scenario: str, num_stocks: int) -> Dict:
        """
        Benchmark Union-Find algorithm.
        
        Args:
            graph: Graph object
            stock_attributes: Stock attributes dictionary
            scenario: Market scenario name
            num_stocks: Number of stocks
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"  Benchmarking Union-Find...")
        
        # Measure time and memory
        mem_before = get_memory_usage()
        start_time = time.time()
        
        uf, components = find_market_segments(graph)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        # Analyze results
        analysis = analyze_market_segments(uf, graph, stock_attributes)
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        
        result = {
            'algorithm': 'Union-Find',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'num_components': uf.get_num_components(),
            'component_sizes': [len(c) for c in components],
            'largest_component': max([len(c) for c in components]) if components else 0,
        }
        
        return result
    
    def benchmark_bfs(self, graph, stock_attributes: Dict,
                     scenario: str, num_stocks: int, num_samples: int = 50) -> Dict:
        """
        Benchmark BFS algorithm.
        
        Args:
            graph: Graph object
            stock_attributes: Stock attributes dictionary
            scenario: Market scenario name
            num_stocks: Number of stocks
            num_samples: Number of random paths to test
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"  Benchmarking BFS...")
        
        nodes = graph.get_nodes()
        if len(nodes) < 2:
            return None
        
        # Test standard BFS
        mem_before = get_memory_usage()
        start_time = time.time()
        
        path_lengths = []
        for _ in range(min(num_samples, len(nodes) * (len(nodes) - 1) // 2)):
            start, end = np.random.choice(nodes, 2, replace=False)
            path = bfs_shortest_path(graph, start, end)
            if path:
                path_lengths.append(len(path) - 1)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        # Connectivity analysis
        connectivity = analyze_graph_connectivity(graph)
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        
        result = {
            'algorithm': 'BFS',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'num_samples': len(path_lengths),
            'avg_path_length': avg_path_length,
            'max_path_length': max(path_lengths) if path_lengths else 0,
            'connectivity': connectivity,
        }
        
        return result
    
    def run_benchmark_suite(self, sizes: List[int] = [50, 100, 200],
                          scenarios: List[str] = ["stable", "normal", "volatile", "crash"]):
        """
        Run complete benchmark suite across different graph sizes and scenarios.
        
        Args:
            sizes: List of graph sizes to test
            scenarios: List of market scenarios to test
        """
        print("="*60)
        print("BENCHMARK SUITE: Union-Find and BFS")
        print("="*60)
        
        generator = StockDataGenerator(seed=42)
        
        for size in sizes:
            for scenario in scenarios:
                print(f"\n{'='*60}")
                print(f"Testing: {size} stocks, {scenario} scenario")
                print('='*60)
                
                # Generate data
                print("  Generating data...")
                returns, corr_matrix, stock_attrs = generator.generate_dataset(
                    size, scenario=scenario
                )
                
                # Build graph
                print("  Building graph...")
                threshold = 0.5 if scenario != "crash" else 0.3  # Lower threshold for crash
                graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
                
                stats = get_graph_statistics(graph)
                print(f"  Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges, "
                      f"density: {stats['density']:.4f}")
                
                # Benchmark Union-Find
                uf_result = self.benchmark_union_find(graph, stock_attrs, scenario, size)
                self.results.append(uf_result)
                print(f"    Union-Find: {uf_result['runtime_seconds']:.4f}s, "
                      f"{uf_result['num_components']} components")
                
                # Benchmark BFS
                bfs_result = self.benchmark_bfs(graph, stock_attrs, scenario, size)
                if bfs_result:
                    self.results.append(bfs_result)
                    print(f"    BFS: {bfs_result['runtime_seconds']:.4f}s, "
                          f"avg path: {bfs_result['avg_path_length']:.2f}")
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filepath}")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group by algorithm
        uf_results = [r for r in self.results if r['algorithm'] == 'Union-Find']
        bfs_results = [r for r in self.results if r['algorithm'] == 'BFS']
        
        if uf_results:
            print("\nUnion-Find Performance:")
            print(f"  Average runtime: {np.mean([r['runtime_seconds'] for r in uf_results]):.4f}s")
            print(f"  Average memory: {np.mean([r['memory_mb'] for r in uf_results]):.2f}MB")
            print(f"  Fastest: {min([r['runtime_seconds'] for r in uf_results]):.4f}s")
            print(f"  Slowest: {max([r['runtime_seconds'] for r in uf_results]):.4f}s")
        
        if bfs_results:
            print("\nBFS Performance:")
            print(f"  Average runtime: {np.mean([r['runtime_seconds'] for r in bfs_results]):.4f}s")
            print(f"  Average memory: {np.mean([r['memory_mb'] for r in bfs_results]):.2f}MB")
            print(f"  Average path length: {np.mean([r['avg_path_length'] for r in bfs_results]):.2f}")
            print(f"  Fastest: {min([r['runtime_seconds'] for r in bfs_results]):.4f}s")
            print(f"  Slowest: {max([r['runtime_seconds'] for r in bfs_results]):.4f}s")


def main():
    """Run benchmarks."""
    print("\n" + "="*60)
    print("GUNTESH'S ALGORITHM BENCHMARKS")
    print("Union-Find and BFS Performance Analysis")
    print("="*60 + "\n")
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir="results")
    
    # Run benchmark suite
    sizes = [50, 100, 200]
    scenarios = ["stable", "normal", "volatile", "crash"]
    
    runner.run_benchmark_suite(sizes, scenarios)
    
    # Print summary
    runner.print_summary()
    
    # Save results
    runner.save_results("guntesh_benchmarks.json")


if __name__ == "__main__":
    main()
