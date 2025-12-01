# Testing & Benchmarking Infrastructure - Usage Guide

**Quick Start Guide for Team Members**  
This document explains how to use the shared testing infrastructure to benchmark your algorithms and generate visualizations.

---

## Table of Contents
1. [Setup](#setup)
2. [Running Benchmarks](#running-benchmarks)
3. [Benchmark Output Format](#benchmark-output-format)
4. [Generating Visualizations](#generating-visualizations)
5. [Complete Example Workflow](#complete-example-workflow)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd course-project-queue-ties

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Git Bash):
. venv/Scripts/activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull Latest Shared Components

The `main` branch contains:
- `src/data_generation.py` - Stock data generator (seed=42 for reproducibility)
- `src/graph.py` - Graph representation class
- `visualize_results.py` - Dynamic visualization script
- `TESTCASES.md` - Standardized test cases

```bash
git checkout main
git pull origin main
```

### 3. Create Your Own Branch

```bash
git checkout -b yourname-algorithm
```

---

## Running Benchmarks

### Universal Benchmark Script (Recommended)

The easiest way to run benchmarks is using the unified `benchmarks.py` script:

```bash
# Activate virtual environment
. venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # macOS/Linux

# Run all algorithms
python benchmarks.py

# Run specific algorithm
python benchmarks.py union_find
python benchmarks.py bfs
python benchmarks.py union-find  # Also works (normalized)

# Get help
python benchmarks.py --help
```

**Output:** Results saved to `results/<algorithm>_benchmarks.json`

### Manual Benchmark Implementation

If you want to create a custom benchmark script:

### Step 1: Import Shared Components

In your algorithm file or benchmark script:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import psutil
from data_generation import StockDataGenerator
from graph import build_graph_from_correlation

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024
```

### Step 2: Generate Test Data

Use the **exact same parameters** as defined in `TESTCASES.md`:

```python
# Initialize generator with seed=42 for reproducibility
generator = StockDataGenerator(seed=42)

# Test configurations (from TESTCASES.md)
sizes = [50, 100, 200]
scenarios = ["stable", "normal", "volatile", "crash"]

results = []

for size in sizes:
    for scenario in scenarios:
        # Generate data
        returns, corr_matrix, stock_attrs = generator.generate_dataset(
            num_stocks=size, 
            scenario=scenario
        )
        
        # Build graph
        threshold = 0.5 if scenario != "crash" else 0.3
        graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
```

### Step 3: Benchmark Your Algorithm

```python
        # Benchmark your algorithm
        mem_before = get_memory_usage()
        start_time = time.time()
        
        # >>> YOUR ALGORITHM HERE <<<
        # Example: result = your_algorithm(graph, stock_attrs)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        # Record results
        result_entry = {
            'algorithm': 'YourAlgorithmName',  # e.g., 'DFS', 'PageRank', 'Louvain'
            'scenario': scenario,
            'num_stocks': size,
            'num_edges': graph.num_edges,
            'runtime_seconds': end_time - start_time,
            'memory_mb': mem_after - mem_before,
            # Add your algorithm-specific metrics below:
            # 'num_components': ...,  # for Union-Find/community detection
            # 'avg_path_length': ...,  # for BFS/DFS
            # 'modularity': ...,       # for Girvan-Newman/Louvain
            # 'top_nodes': ...,        # for PageRank
            # etc.
        }
        
        results.append(result_entry)
```

### Step 4: Save Results to JSON

```python
import json

# Save to results/ directory
os.makedirs('results', exist_ok=True)
output_file = 'results/yourname_benchmarks.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
```

---

## Benchmark Output Format

Your JSON file should follow this structure:

```json
[
  {
    "algorithm": "DFS",
    "scenario": "stable",
    "num_stocks": 50,
    "num_edges": 167,
    "runtime_seconds": 0.0015,
    "memory_mb": 0.08,
    "num_components": 4,
    "avg_depth": 3.2
  },
  {
    "algorithm": "DFS",
    "scenario": "normal",
    "num_stocks": 50,
    "num_edges": 9,
    "runtime_seconds": 0.0003,
    "memory_mb": 0.02,
    "num_components": 43,
    "avg_depth": 1.1
  }
]
```

### Required Fields
- `algorithm` (string) - Your algorithm name
- `scenario` (string) - One of: "stable", "normal", "volatile", "crash"
- `num_stocks` (int) - Graph size: 50, 100, or 200
- `num_edges` (int) - Number of edges in the graph
- `runtime_seconds` (float) - Execution time in seconds
- `memory_mb` (float) - Memory usage in megabytes

### Optional Algorithm-Specific Fields

**For Union-Find / Community Detection:**
- `num_components` - Number of connected components/communities
- `component_sizes` - List of component sizes
- `largest_component` - Size of largest component
- `modularity` - Modularity score

**For BFS / DFS / Traversal:**
- `avg_path_length` - Average shortest path length
- `max_path_length` - Maximum path length
- `connectivity` - Connectivity percentage

**For PageRank / Centrality:**
- `top_nodes` - List of top-k influential nodes
- `avg_rank` - Average rank score
- `rank_distribution` - Rank statistics

**For Node2Vec / Embeddings:**
- `embedding_dim` - Dimensionality of embeddings
- `similarity_scores` - Similarity metrics
- `recommendation_accuracy` - Accuracy score

---

## Generating Visualizations

### Using the Dynamic Visualizer

The `visualize_results.py` script **automatically detects** your algorithm and metrics:

```bash
# Activate virtual environment
. venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # macOS/Linux

# Option 1: Auto-detect all algorithms (default)
python visualize_results.py

# Option 2: Filter by specific algorithm (saves to results/<algorithm>/ folder)
python visualize_results.py <algorithm_name>

# Examples:
python visualize_results.py union_find
python visualize_results.py girvan_newman
python visualize_results.py bfs
python visualize_results.py dfs

# Note: Algorithm name matches your Python filename without .py extension
# e.g., if your file is src/girvan_newman.py, use: python visualize_results.py girvan_newman

# Get help
python visualize_results.py --help
```

**How it works:**
- Without an algorithm name: Generates visualizations for ALL algorithms found in benchmark results, saves to `results/`
- With an algorithm name: Filters results for that specific algorithm, saves to `results/<algorithm_name>/`
- Automatically looks for `results/<algorithm_name>_benchmarks.json` first
- Falls back to any `*_benchmarks.json` file and filters by algorithm field

### What Gets Generated

The visualizer creates:

1. **Per-Algorithm Runtime Plots**
   - `results/youralgorithm_runtime.png` - Runtime vs graph size by scenario

2. **Combined Runtime Comparison**
   - `results/runtime_comparison_all_algorithms.png` - Compare all algorithms

3. **Components Analysis** (if `num_components` exists)
   - `results/components_analysis.png` - Bar chart of components by scenario

4. **Path Length Analysis** (if `avg_path_length` exists)
   - `results/youralgorithm_path_length.png` - Path lengths by scenario

5. **Graph Density**
   - `results/graph_density.png` - Density across scenarios

6. **Scalability Analysis**
   - `results/scalability_analysis.png` - Runtime vs edges scatter plot

7. **Component Distribution**
   - `results/component_distribution.png` - Histograms of component sizes

8. **Summary Table**
   - `results/summary_table.png` - Tabular summary

### Customizing Visualizations

If you need custom plots, modify `visualize_results.py` or create your own script:

```python
import json
import matplotlib.pyplot as plt

# Load your results
with open('results/yourname_benchmarks.json', 'r') as f:
    results = json.load(f)

# Create custom plot
sizes = [r['num_stocks'] for r in results]
runtimes = [r['runtime_seconds'] for r in results]

plt.plot(sizes, runtimes, marker='o')
plt.xlabel('Number of Stocks')
plt.ylabel('Runtime (s)')
plt.title('My Algorithm Performance')
plt.savefig('results/my_custom_plot.png')
```

---

## Complete Example Workflow

Here's a full example for implementing and benchmarking DFS:

### File: `benchmarks/benchmark_dfs.py`

```python
"""
Benchmark script for DFS algorithm
Author: [Your Name]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import psutil
import json
from data_generation import StockDataGenerator
from graph import build_graph_from_correlation

# Your DFS implementation
def dfs(graph, start, visited=None):
    """DFS traversal implementation."""
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph.get_neighbors(start):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    # Setup
    generator = StockDataGenerator(seed=42)
    sizes = [50, 100, 200]
    scenarios = ["stable", "normal", "volatile", "crash"]
    results = []
    
    print("Running DFS Benchmarks...")
    
    for size in sizes:
        for scenario in scenarios:
            print(f"  Testing: {size} stocks, {scenario} scenario")
            
            # Generate data
            returns, corr_matrix, stock_attrs = generator.generate_dataset(size, scenario=scenario)
            threshold = 0.5 if scenario != "crash" else 0.3
            graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
            
            # Benchmark DFS
            mem_before = get_memory_usage()
            start_time = time.time()
            
            # Run DFS from each node to find components
            visited_global = set()
            num_components = 0
            for node in graph.get_nodes():
                if node not in visited_global:
                    component = dfs(graph, node)
                    visited_global.update(component)
                    num_components += 1
            
            end_time = time.time()
            mem_after = get_memory_usage()
            
            # Record results
            results.append({
                'algorithm': 'DFS',
                'scenario': scenario,
                'num_stocks': size,
                'num_edges': graph.num_edges,
                'runtime_seconds': end_time - start_time,
                'memory_mb': mem_after - mem_before,
                'num_components': num_components
            })
            
            print(f"    ✓ Runtime: {(end_time - start_time)*1000:.2f}ms | Components: {num_components}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/dfs_benchmarks.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Benchmarks complete! Results saved to results/dfs_benchmarks.json")
    print("Run 'python visualize_results.py' to generate charts.")

if __name__ == "__main__":
    main()
```

### Running the Benchmark

```bash
# Activate environment
. venv/Scripts/activate

# Run your benchmark
python benchmarks/benchmark_dfs.py

# Generate visualizations
python visualize_results.py
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'data_generation'`

**Solution:** 
- Make sure you're in the correct directory
- Add the src path: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))`
- Verify `src/data_generation.py` exists

### Different Results Than Expected

**Problem:** Graph has different number of edges than in TESTCASES.md

**Solution:**
- Verify you're using `seed=42` when creating `StockDataGenerator`
- Check threshold: `0.5` for most scenarios, `0.3` for crash
- Make sure you're using the shared `build_graph_from_correlation` function

### Visualizer Not Finding Results

**Problem:** `FileNotFoundError: No benchmark json file found`

**Solution:**
- Ensure your JSON file is in `results/` directory
- File must end with `_benchmarks.json`
- Or pass the file explicitly: `python visualize_results.py results/yourfile.json`

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'matplotlib'`

**Solution:**
```bash
. venv/Scripts/activate
pip install -r requirements.txt
```

### Git Conflicts

**Problem:** Merge conflicts when pulling from main

**Solution:**
```bash
# Stash your changes
git stash

# Pull latest from main
git checkout main
git pull origin main

# Return to your branch and apply changes
git checkout yourname-algorithm
git merge main
git stash pop
```

---

## Best Practices

### ✅ DO:
- Use `seed=42` for reproducibility
- Test all 12 test cases (3 sizes × 4 scenarios)
- Save results as `results/yourname_benchmarks.json`
- Include both required and algorithm-specific metrics
- Commit your benchmark script to your branch
- Document any deviations in your README

### ❌ DON'T:
- Modify `src/data_generation.py` or `src/graph.py` without team discussion
- Use different seeds or thresholds
- Hard-code algorithm names in the visualizer
- Commit large result files to git (add to `.gitignore`)

---

## Questions?

- **Data generation issues:** Contact Guntesh Singh
- **Graph representation:** Check `src/graph.py` documentation
- **Test cases:** See `TESTCASES.md`
- **Visualization:** See `visualize_results.py` source code

---

## Quick Reference Commands

```bash
# Setup
git clone <repo-url>
cd course-project-queue-ties
python -m venv venv
. venv/Scripts/activate
pip install -r requirements.txt

# Create your branch
git checkout -b yourname-algorithm

# Run benchmarks (unified script)
python benchmarks.py                  # All algorithms
python benchmarks.py union_find       # Specific algorithm
python benchmarks.py --help           # Show help

# Generate visualizations
python visualize_results.py           # All algorithms
python visualize_results.py bfs       # Specific algorithm
python visualize_results.py --help    # Show help

# Commit your work
git add src/your_algorithm.py
git commit -m "Add [algorithm] implementation"
git push origin yourname-algorithm
```

---

**Last Updated:** December 1, 2025  
**Maintained By:** Guntesh Singh (Data Foundation Lead)
