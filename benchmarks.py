"""
Universal Benchmarking Script for All Algorithms
Author: Guntesh Singh (Data Foundation Team)
Description: Unified benchmark script that can run any algorithm with CLI support

Usage:
    # Run all available algorithms
    python benchmarks.py
    
    # Run specific algorithm
    python benchmarks.py union_find
    python benchmarks.py bfs
    python benchmarks.py union-find  # Also works (normalized)
    python benchmarks.py BFS  # Case-insensitive
    
    # Show help
    python benchmarks.py --help

Output:
    Results are saved to results/<algorithm>_benchmarks.json
    If no algorithm is specified, saves to results/all_benchmarks.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import psutil
import json
from typing import Dict, List, Optional
import numpy as np

# Import shared components
from data_generation import StockDataGenerator
from graph import build_graph_from_correlation, get_graph_statistics


def normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm name for matching."""
    if name.endswith('.py'):
        name = name[:-3]
    return name.lower().replace('-', '_').replace(' ', '_')


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class UniversalBenchmarkRunner:
    """Universal benchmark runner for all algorithms."""
    
    # Registry of available algorithms
    ALGORITHMS = {
        'union_find': {
            'display_name': 'Union-Find',
            'import_path': 'union_find',
            'functions': ['find_market_segments', 'analyze_market_segments']
        },
        'bfs': {
            'display_name': 'BFS',
            'import_path': 'bfs',
            'functions': ['bfs_shortest_path', 'analyze_graph_connectivity']
        }
    }
    
    def __init__(self, output_dir: str = "results"):
        """Initialize benchmark runner."""
        self.output_dir = output_dir
        self.results = []
        os.makedirs(output_dir, exist_ok=True)
    
    def benchmark_union_find(self, graph, stock_attributes: Dict, 
                            scenario: str, num_stocks: int) -> Dict:
        """Benchmark Union-Find algorithm."""
        from union_find import find_market_segments, analyze_market_segments
        
        print(f"  Benchmarking Union-Find...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        uf, components = find_market_segments(graph)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
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
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Components: {result['num_components']} | Largest: {result['largest_component']} stocks")
        
        return result
    
    def benchmark_bfs(self, graph, stock_attributes: Dict,
                     scenario: str, num_stocks: int, num_samples: int = 50) -> Dict:
        """Benchmark BFS algorithm."""
        from bfs import bfs_shortest_path, analyze_graph_connectivity
        
        print(f"  Benchmarking BFS...")
        
        nodes = graph.get_nodes()
        if len(nodes) < 2:
            print(f"    ⚠ Skipped (insufficient nodes)")
            return None
        
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
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Paths found: {len(path_lengths)}/{num_samples} | Avg length: {avg_path_length:.2f}")
        
        return result
    
    def run_benchmark(self, algorithm: str, sizes: List[int], scenarios: List[str]) -> List[Dict]:
        """
        Run benchmark for a specific algorithm.
        
        Args:
            algorithm: Normalized algorithm name
            sizes: List of graph sizes to test
            scenarios: List of market scenarios to test
        
        Returns:
            List of benchmark results
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algo_info = self.ALGORITHMS[algorithm]
        display_name = algo_info['display_name']
        
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {display_name}")
        print('='*60)
        
        generator = StockDataGenerator(seed=42)
        results = []
        
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
                threshold = 0.5 if scenario != "crash" else 0.3
                graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
                
                stats = get_graph_statistics(graph)
                print(f"  Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges, "
                      f"density: {stats['density']:.4f}")
                
                # Run benchmark based on algorithm
                if algorithm == 'union_find':
                    result = self.benchmark_union_find(graph, stock_attrs, scenario, size)
                elif algorithm == 'bfs':
                    result = self.benchmark_bfs(graph, stock_attrs, scenario, size)
                else:
                    print(f"  ⚠ No benchmark implementation for {algorithm}")
                    continue
                
                if result:
                    results.append(result)
        
        return results
    
    def run_all_benchmarks(self, sizes: List[int], scenarios: List[str]) -> List[Dict]:
        """Run benchmarks for all available algorithms."""
        print("\n" + "="*60)
        print("UNIVERSAL BENCHMARK SUITE - ALL ALGORITHMS")
        print("="*60)
        
        all_results = []
        
        for algorithm in self.ALGORITHMS.keys():
            try:
                results = self.run_benchmark(algorithm, sizes, scenarios)
                all_results.extend(results)
            except Exception as e:
                print(f"\n⚠ Error benchmarking {algorithm}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save benchmark results to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n> Results saved to {filepath}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary of benchmark results."""
        if not results:
            print("\nNo results to summarize.")
            return
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Group by algorithm
        by_algorithm = {}
        for r in results:
            algo = r['algorithm']
            if algo not in by_algorithm:
                by_algorithm[algo] = []
            by_algorithm[algo].append(r)
        
        for algo, algo_results in by_algorithm.items():
            print(f"\n{algo} Performance:")
            runtimes = [r['runtime_seconds'] for r in algo_results]
            memories = [r['memory_mb'] for r in algo_results]
            
            print(f"  Average runtime: {np.mean(runtimes):.4f}s")
            print(f"  Average memory: {np.mean(memories):.2f}MB")
            print(f"  Fastest: {min(runtimes):.4f}s")
            print(f"  Slowest: {max(runtimes):.4f}s")
            print(f"  Total test cases: {len(algo_results)}")


def main():
    """Main entry point with CLI support."""
    # Parse command line arguments
    algorithm_filter = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Handle help flag
        if arg in ['--help', '-h', 'help']:
            print(__doc__)
            print("\nAvailable algorithms:")
            runner = UniversalBenchmarkRunner()
            for algo, info in runner.ALGORITHMS.items():
                print(f"  - {algo} ({info['display_name']})")
            print("\nExamples:")
            print("  python benchmarks.py")
            print("  python benchmarks.py union_find")
            print("  python benchmarks.py bfs")
            sys.exit(0)
        
        # Normalize algorithm name
        algorithm_filter = normalize_algorithm_name(arg)
        
        # Validate algorithm name
        runner = UniversalBenchmarkRunner()
        if algorithm_filter not in runner.ALGORITHMS:
            print(f"Error: Unknown algorithm '{arg}'")
            print(f"Available algorithms: {', '.join(runner.ALGORITHMS.keys())}")
            print("Run with --help for more information")
            sys.exit(1)
    
    # Setup
    print("\n" + "="*60)
    if algorithm_filter:
        runner = UniversalBenchmarkRunner()
        display_name = runner.ALGORITHMS[algorithm_filter]['display_name']
        print(f"ALGORITHM BENCHMARK - {display_name}")
    else:
        print("UNIVERSAL ALGORITHM BENCHMARKS")
    print("="*60 + "\n")
    
    # Configuration
    sizes = [50, 100, 200]
    scenarios = ["stable", "normal", "volatile", "crash"]
    
    # Create runner and execute
    runner = UniversalBenchmarkRunner()
    
    if algorithm_filter:
        # Run specific algorithm
        results = runner.run_benchmark(algorithm_filter, sizes, scenarios)
    else:
        # Run all algorithms
        results = runner.run_all_benchmarks(sizes, scenarios)
    
    # Print summary
    runner.print_summary(results)
    
    # Save results
    if algorithm_filter:
        filename = f"{algorithm_filter}_benchmarks.json"
    else:
        filename = "all_benchmarks.json"
    
    runner.save_results(results, filename)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    if algorithm_filter:
        print(f"\nTo visualize: python visualize_results.py {algorithm_filter}")
    else:
        print("\nTo visualize: python visualize_results.py")


if __name__ == "__main__":
    main()
