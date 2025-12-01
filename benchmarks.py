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
    python benchmarks.py louvain
    python benchmarks.py girvan_newman
    python benchmarks.py dfs
    python benchmarks.py pagerank
    python benchmarks.py market_disruption
    
    # Names are normalized (case-insensitive, hyphens/underscores)
    python benchmarks.py Girvan-Newman  # Also works
    python benchmarks.py market-disruption  # Also works
    
    # Show help
    python benchmarks.py --help

Output:
    Results are saved to results/<algorithm>_benchmarks.json
    If no algorithm is specified, saves to results/all_benchmarks.json

Note:
    Algorithms from different branches:
    - union_find, bfs: guntesh-data-foundation
    - louvain: main / saanvi-louvian  
    - girvan_newman: avani-girvan-newman
    - dfs, pagerank, market_disruption: main
    
    Make sure the required algorithm files are in src/ before running!
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
            'functions': ['find_market_segments', 'analyze_market_segments'],
            'branch': 'guntesh-data-foundation'
        },
        'bfs': {
            'display_name': 'BFS',
            'import_path': 'bfs',
            'functions': ['bfs_shortest_path', 'analyze_graph_connectivity'],
            'branch': 'guntesh-data-foundation'
        },
        'louvain': {
            'display_name': 'Louvain',
            'import_path': 'louvain',
            'functions': ['louvain', 'compute_modularity'],
            'branch': 'main / saanvi-louvian'
        },
        'girvan_newman': {
            'display_name': 'Girvan-Newman',
            'import_path': 'girvan_newman',
            'functions': ['girvan_newman', 'modularity', 'betweenness_centrality'],
            'branch': 'avani-girvan-newman'
        },
        'dfs': {
            'display_name': 'DFS',
            'import_path': 'dfs',
            'functions': ['DFS', 'analyze_market_connectivity'],
            'branch': 'main'
        },
        'pagerank': {
            'display_name': 'PageRank',
            'import_path': 'pagerank',
            'functions': ['PageRank', 'identify_market_influencers'],
            'branch': 'main'
        },
        'market_disruption': {
            'display_name': 'Market Disruption',
            'import_path': 'market_disruption',
            'functions': ['MarketDisruptionSimulator'],
            'branch': 'main'
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
    
    def benchmark_louvain(self, graph, stock_attributes: Dict,
                         scenario: str, num_stocks: int) -> Dict:
        """Benchmark Louvain algorithm."""
        from louvain import louvain
        
        print(f"  Benchmarking Louvain...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        communities, modularity_score = louvain(graph)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        num_communities = len(set(communities.values()))
        
        result = {
            'algorithm': 'Louvain',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'num_communities': num_communities,
            'modularity': modularity_score,
        }
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Communities: {num_communities} | Modularity: {modularity_score:.4f}")
        
        return result
    
    def benchmark_girvan_newman(self, graph, stock_attributes: Dict,
                               scenario: str, num_stocks: int) -> Dict:
        """Benchmark Girvan-Newman algorithm."""
        from girvan_newman import girvan_newman, modularity
        
        print(f"  Benchmarking Girvan-Newman...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        communities, modularity_score = girvan_newman(graph, max_iterations=10)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        num_communities = len(communities)
        
        result = {
            'algorithm': 'Girvan-Newman',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'num_communities': num_communities,
            'modularity': modularity_score,
        }
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Communities: {num_communities} | Modularity: {modularity_score:.4f}")
        
        return result
    
    def benchmark_dfs(self, graph, stock_attributes: Dict,
                     scenario: str, num_stocks: int) -> Dict:
        """Benchmark DFS algorithm."""
        from dfs import DFS, analyze_market_connectivity
        
        print(f"  Benchmarking DFS...")
        
        nodes = graph.get_nodes()
        if len(nodes) < 1:
            print(f"    Warning: Skipped (insufficient nodes)")
            return None
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        dfs = DFS(graph)
        components = dfs.find_connected_components()
        connectivity_info = dfs.get_connectivity_info()
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        
        result = {
            'algorithm': 'DFS',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'num_components': len(components),
            'connectivity_ratio': connectivity_info.get('connectivity_ratio', 0),
            'has_cycle': connectivity_info.get('has_cycle', False),
        }
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Components: {len(components)} | Connectivity: {result['connectivity_ratio']:.3f}")
        
        return result
    
    def benchmark_pagerank(self, graph, stock_attributes: Dict,
                          scenario: str, num_stocks: int) -> Dict:
        """Benchmark PageRank algorithm."""
        from pagerank import PageRank
        
        print(f"  Benchmarking PageRank...")
        
        if graph.num_nodes < 1:
            print(f"    Warning: Skipped (insufficient nodes)")
            return None
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        pr = PageRank(graph, damping_factor=0.85)
        scores = pr.calculate_pagerank(max_iterations=100, tolerance=1e-6)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        
        top_stocks = pr.get_top_stocks(5)
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        
        result = {
            'algorithm': 'PageRank',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'iterations': pr.iterations,
            'avg_score': avg_score,
            'top_score': top_stocks[0][1] if top_stocks else 0,
        }
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Iterations: {pr.iterations} | Top score: {result['top_score']:.4f}")
        
        return result
    
    def benchmark_market_disruption(self, graph, stock_attributes: Dict,
                                   scenario: str, num_stocks: int) -> Dict:
        """Benchmark Market Disruption Simulator."""
        from market_disruption import MarketDisruptionSimulator
        
        print(f"  Benchmarking Market Disruption...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        simulator = MarketDisruptionSimulator(graph, stock_attributes)
        crash_result = simulator.simulate_market_crash(severity=0.5)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = mem_after - mem_before
        
        result = {
            'algorithm': 'Market Disruption',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'num_edges': graph.num_edges,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'network_fragility': crash_result.get('network_fragility', 0),
            'edges_added': crash_result.get('edges_added', 0),
        }
        
        print(f"    > Runtime: {runtime*1000:.2f}ms | Memory: {memory_used:.2f}MB")
        print(f"    > Fragility: {result['network_fragility']:.3f} | Edges added: {result['edges_added']}")
        
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
                elif algorithm == 'louvain':
                    result = self.benchmark_louvain(graph, stock_attrs, scenario, size)
                elif algorithm == 'girvan_newman':
                    result = self.benchmark_girvan_newman(graph, stock_attrs, scenario, size)
                elif algorithm == 'dfs':
                    result = self.benchmark_dfs(graph, stock_attrs, scenario, size)
                elif algorithm == 'pagerank':
                    result = self.benchmark_pagerank(graph, stock_attrs, scenario, size)
                elif algorithm == 'market_disruption':
                    result = self.benchmark_market_disruption(graph, stock_attrs, scenario, size)
                else:
                    print(f"  Warning: No benchmark implementation for {algorithm}")
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
                branch_info = info.get('branch', 'unknown')
                print(f"  - {algo} ({info['display_name']}) [from: {branch_info}]")
            print("\nNote: Some algorithms may only be available on specific branches.")
            print("Make sure the required algorithm files are present in src/")
            print("\nExamples:")
            print("  python benchmarks.py")
            print("  python benchmarks.py union_find")
            print("  python benchmarks.py louvain")
            print("  python benchmarks.py girvan_newman")
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
