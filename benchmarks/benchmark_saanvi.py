"""
Benchmark Script for Louvain Community Detection
Author: Saanvi Jain

Runs Louvain on all 12 standard testcases:
(50, 100, 200 stocks) × (stable, normal, volatile, crash)

Outputs results to:
results/saanvi_louvain_benchmarks.json
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
    """Return memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    print("\n📌 Running Louvain Benchmarks...\n")

    generator = StockDataGenerator(seed=42)

    sizes = [50, 100, 200, 500]
    scenarios = ["stable", "normal", "volatile", "crash"]

    results = []

    for size in sizes:
        for scenario in scenarios:
            print(f"➡️  Testcase: {size} stocks — {scenario}")

            # Generate synthetic stock data
            returns, corr_matrix, stock_attrs = generator.generate_dataset(
                num_stocks=size,
                scenario=scenario
            )

            # Threshold rule from TESTCASES.md
            threshold = 0.5 if scenario != "crash" else 0.3

            # Build correlation graph
            graph = build_graph_from_correlation(
                corr_matrix,
                stock_attrs,
                threshold
            )

            # Benchmark Louvain
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

            print(f"   ✔ communities={num_communities}, modularity={mod_score:.4f}, time={(end_time-start_time):.4f}s")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    output_file = "results/saanvi_benchmarks.json"

    # Save JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n🎉 Benchmark complete!")
    print(f"📁 Results saved to: {output_file}\n")
    print("Now run:  python visualize_results.py")
    print("to generate all charts for your report + presentation.")


if __name__ == "__main__":
    main()
