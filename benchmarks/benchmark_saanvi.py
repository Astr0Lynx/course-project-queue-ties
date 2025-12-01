

"""
Benchmark Report for Louvain Community Detection
Author: Saanvi Jain

Purpose:
        - Evaluate Louvain algorithm on synthetic stock correlation graphs
        - Test all standard sizes (50, 100, 200, 500) and market scenarios
        - Report runtime, memory, number of communities, modularity
        - Save results to results/saanvi_benchmarks.json
        - Print summary table for easy comparison
"""

import sys
import os
import time
import psutil
import json

# Add src/ directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import StockDataGenerator
from graph import build_graph_from_correlation
from louvain import louvain

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_summary_table(results):
    print("\nSummary Table (Louvain)")
    print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Stocks", "Scenario", "Edges", "Time(s)", "Mem(MB)", "Comms", "Modularity"))
    print("-" * 68)
    for r in results:
        print("{:<8} {:<10} {:<10} {:<10.3f} {:<10.2f} {:<10} {:<10.4f}".format(
            r['num_stocks'], r['scenario'], r['num_edges'],
            r['runtime_seconds'], r['memory_mb'], r['num_communities'], r['modularity']))

def main():
    generator = StockDataGenerator(seed=42)
    sizes = [50, 100, 200, 500]
    scenarios = ["stable", "normal", "volatile", "crash"]
    results = []

    print("\n=== Louvain Community Detection Benchmark ===")
    print("Author: Saanvi Jain\n")
    print("Testing sizes:", sizes)
    print("Market scenarios:", scenarios)
    print("-------------------------------------------\n")

    for size in sizes:
        for scenario in scenarios:
            print(f"[RUN] {size} stocks | {scenario:8s}")
            returns, corr_matrix, stock_attrs = generator.generate_dataset(
                num_stocks=size,
                scenario=scenario
            )
            threshold = 0.5 if scenario != "crash" else 0.3
            graph = build_graph_from_correlation(
                corr_matrix,
                stock_attrs,
                threshold
            )

            mem_before = get_memory_usage()
            start_time = time.time()
            partition, mod_score = louvain(graph, random_state=42)
            end_time = time.time()
            mem_after = get_memory_usage()

            num_communities = len(partition)

            results.append({
                "algorithm": "Louvain",
                "scenario": scenario,
                "num_stocks": size,
                "num_edges": graph.num_edges,
                "runtime_seconds": end_time - start_time,
                "memory_mb": mem_after - mem_before,
                "num_communities": num_communities,
                "modularity": mod_score
            })

            print(f"    Communities: {num_communities:3d} | Modularity: {mod_score:.4f} | Time: {(end_time-start_time):.3f}s | Mem: {(mem_after-mem_before):.2f}MB")

    os.makedirs("results", exist_ok=True)
    output_file = "results/saanvi_benchmarks.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print_summary_table(results)
    print("\nBenchmarks complete. Results saved to results/saanvi_benchmarks.json")
    print("Run 'python visualize_results.py' to generate charts.")

if __name__ == "__main__":
    main()
