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
    
    # Names are normalized (case-insensitive, hyphens/underscores)
    python benchmarks.py Girvan-Newman  # Also works
    python benchmarks.py DFS  # Also works
    
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
    - dfs, pagerank: main
    
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
        'node2vec': {
            'display_name': 'Node2Vec',
            'import_path': 'node2vec',
            'functions': ['Node2Vec'],
            'branch': 'main / vaibhavi-node2vec'
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
        from src.union_find import find_market_segments, analyze_market_segments
        
        print(f"  Benchmarking Union-Find...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        uf, components = find_market_segments(graph)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        analysis = analyze_market_segments(uf, graph, stock_attributes)
        
        runtime = end_time - start_time
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
        
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
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Components: {result['num_components']} | Largest: {result['largest_component']} stocks")
        
        return result
    
    def benchmark_bfs(self, graph, stock_attributes: Dict,
                     scenario: str, num_stocks: int, num_samples: int = 50) -> Dict:
        """Benchmark BFS algorithm."""
        from src.bfs import bfs_shortest_path, analyze_graph_connectivity
        
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
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
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
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Paths found: {len(path_lengths)}/{num_samples} | Avg length: {avg_path_length:.4f}")
        
        return result
    
    def benchmark_louvain(self, graph, stock_attributes: Dict,
                          scenario: str, num_stocks: int) -> Dict:
        """Benchmark Louvain algorithm."""
        from src.louvain import louvain

        print(f"  Benchmarking Louvain...")

        mem_before = get_memory_usage()
        start_time = time.time()

        communities, modularity_score = louvain(graph)

        end_time = time.time()
        mem_after = get_memory_usage()

        runtime = end_time - start_time
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
        num_communities = len(communities)

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

        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Communities: {num_communities} | Modularity: {modularity_score:.4f}")

        return result
    
    def benchmark_girvan_newman(self, graph, stock_attributes: Dict,
                                scenario: str, num_stocks: int) -> Dict:
        """Benchmark Girvan-Newman algorithm."""
        from src.girvan_newman import girvan_newman, modularity
        
        print(f"  Benchmarking Girvan-Newman...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        communities, modularity_score = girvan_newman(graph, max_iterations=10)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
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
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Communities: {num_communities} | Modularity: {modularity_score:.4f}")
        
        return result
    
    def benchmark_dfs(self, graph, stock_attributes: Dict,
                     scenario: str, num_stocks: int) -> Dict:
        """Benchmark DFS algorithm."""
        from src.dfs import DFS, analyze_market_connectivity
        
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
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
        
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
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Components: {len(components)} | Connectivity: {result['connectivity_ratio']:.4f}")
        
        return result
    
    def benchmark_pagerank(self, graph, stock_attributes: Dict,
                          scenario: str, num_stocks: int) -> Dict:
        """Benchmark PageRank algorithm."""
        from src.pagerank import PageRank
        
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
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
        
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
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Iterations: {pr.iterations} | Top score: {result['top_score']:.4f}")
        
        return result
    
    def benchmark_node2vec(self, graph, stock_attributes: Dict,
                          scenario: str, num_stocks: int) -> Dict:
        """Benchmark Node2Vec algorithm."""
        from src.node2vec import Node2Vec
        
        print(f"  Benchmarking Node2Vec...")
        
        nodes = graph.get_nodes()
        if len(nodes) < 2:
            print(f"    Warning: Skipped (insufficient nodes)")
            return None
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        # Use smaller parameters for benchmarking speed
        n2v = Node2Vec(graph, walk_length=30, num_walks=10, embedding_dim=32, 
                      window_size=5, epochs=1, learning_rate=0.01)
        embeddings = n2v.learn_embeddings()
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        runtime = end_time - start_time
        memory_used = max(0, mem_after - mem_before)  # Prevent negative memory
        
        result = {
            'algorithm': 'Node2Vec',
            'scenario': scenario,
            'num_stocks': num_stocks,
            'runtime_seconds': runtime,
            'memory_mb': memory_used,
            'embedding_dim': len(next(iter(embeddings.values()))) if embeddings else 0,
            'num_embeddings': len(embeddings)
        }
        
        print(f"    > Runtime: {runtime*1000:.4f}ms | Memory: {memory_used:.4f}MB")
        print(f"    > Embeddings: {len(embeddings)} nodes | Dim: {result['embedding_dim']}")
        
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
                # Realistic correlation thresholds for stock market analysis
                if scenario == "stable":
                    threshold = 0.35  # Stable: strong sector correlations
                elif scenario == "normal":
                    threshold = 0.45  # Normal: moderate correlations
                elif scenario == "volatile":
                    threshold = 0.40  # Volatile: reduced but present correlations
                else:  # crash
                    threshold = 0.30  # Crash: only strongest correlations survive
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
                elif algorithm == 'node2vec':
                    result = self.benchmark_node2vec(graph, stock_attrs, scenario, size)
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
                print(f"\n{'='*60}")
                print(f"Starting {self.ALGORITHMS[algorithm]['display_name']} benchmarks...")
                print(f"{'='*60}")
                results = self.run_benchmark(algorithm, sizes, scenarios)
                if results:
                    all_results.extend(results)
                    print(f"✓ {self.ALGORITHMS[algorithm]['display_name']}: {len(results)} test cases completed")
                else:
                    print(f"⚠ {self.ALGORITHMS[algorithm]['display_name']}: No results generated")
            except Exception as e:
                print(f"\n✗ Error benchmarking {self.ALGORITHMS[algorithm]['display_name']}: {e}")
                import traceback
                traceback.print_exc()
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
            
            print(f"  Average runtime: {np.mean(runtimes)*1000:.4f}ms ({np.mean(runtimes):.4f}s)")
            print(f"  Average memory: {np.mean(memories):.4f}MB")
            print(f"  Fastest: {min(runtimes)*1000:.4f}ms")
            print(f"  Slowest: {max(runtimes)*1000:.4f}ms")
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
    
    # Configuration - Realistic stock market sizes
    sizes = [100, 250, 500]
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
