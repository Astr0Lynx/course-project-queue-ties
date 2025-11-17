# Standard Test Cases for Stock Market Tangle Project

This document defines the **standardized test cases** that all team members should use to benchmark their algorithms. This ensures direct comparability of results across all implementations.

---

## Test Configuration Overview

All algorithms should be tested on:
- **3 graph sizes**: 50, 100, 200 stocks
- **4 market scenarios**: stable, normal, volatile, crash
- **Total**: 12 test cases (3 sizes × 4 scenarios)

---

## Data Generation Parameters

### Common Settings
- **Random seed**: `42` (for reproducibility)
- **Library**: Use `src/data_generation.py` → `StockDataGenerator`
- **Correlation threshold for edges**: 
  - Stable/Normal/Volatile: `0.5`
  - Crash: `0.3` (lower threshold to capture stressed correlations)

### Graph Construction
```python
from src.data_generation import StockDataGenerator
from src.graph import build_graph_from_correlation

generator = StockDataGenerator(seed=42)
returns, corr_matrix, stock_attrs = generator.generate_dataset(num_stocks, scenario=scenario)
threshold = 0.5 if scenario != "crash" else 0.3
graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
```

---

## Test Case Specifications

### Test Case 1: Small Stable Market
- **Size**: 50 stocks
- **Scenario**: `"stable"`
- **Expected Properties**:
  - High connectivity (~136 edges)
  - Few components (2-5)
  - Large dominant component

### Test Case 2: Small Normal Market
- **Size**: 50 stocks
- **Scenario**: `"normal"`
- **Expected Properties**:
  - Moderate connectivity (~9 edges)
  - Many components (40-45)
  - Fragmented structure

### Test Case 3: Small Volatile Market
- **Size**: 50 stocks
- **Scenario**: `"volatile"`
- **Expected Properties**:
  - Very low connectivity (~0 edges)
  - Maximum fragmentation (50 components)
  - All isolated nodes

### Test Case 4: Small Crash Market
- **Size**: 50 stocks
- **Scenario**: `"crash"`
- **Expected Properties**:
  - Minimal connectivity (~0-2 edges)
  - Near-complete fragmentation
  - Stress-induced correlations only

### Test Case 5: Medium Stable Market
- **Size**: 100 stocks
- **Scenario**: `"stable"`
- **Expected Properties**:
  - High connectivity (~270 edges)
  - Very few components (1-3)
  - Dominant giant component (95%+ nodes)

### Test Case 6: Medium Normal Market
- **Size**: 100 stocks
- **Scenario**: `"normal"`
- **Expected Properties**:
  - Low-moderate connectivity (~30 edges)
  - Many components (70-80)
  - Small clusters

### Test Case 7: Medium Volatile Market
- **Size**: 100 stocks
- **Scenario**: `"volatile"`
- **Expected Properties**:
  - Negligible connectivity (~0 edges)
  - Complete fragmentation (100 components)
  - No correlation patterns

### Test Case 8: Medium Crash Market
- **Size**: 100 stocks
- **Scenario**: `"crash"`
- **Expected Properties**:
  - Minimal connectivity (~2-5 edges)
  - Near-complete fragmentation (95-98 components)
  - Few extreme correlations

### Test Case 9: Large Stable Market
- **Size**: 200 stocks
- **Scenario**: `"stable"`
- **Expected Properties**:
  - Moderate-high connectivity (~300 edges)
  - Moderate components (15-25)
  - Several large clusters

### Test Case 10: Large Normal Market
- **Size**: 200 stocks
- **Scenario**: `"normal"`
- **Expected Properties**:
  - Low connectivity (~30-35 edges)
  - High fragmentation (160-170 components)
  - Sparse network

### Test Case 11: Large Volatile Market
- **Size**: 200 stocks
- **Scenario**: `"volatile"`
- **Expected Properties**:
  - Minimal connectivity (~0-1 edges)
  - Complete fragmentation (199-200 components)
  - Extreme market stress

### Test Case 12: Large Crash Market
- **Size**: 200 stocks
- **Scenario**: `"crash"`
- **Expected Properties**:
  - Very low connectivity (~5-10 edges)
  - Near-complete fragmentation (190-195 components)
  - Crisis-level correlations

---

## Benchmark Metrics to Report

### Required Metrics (All Algorithms)
1. **Runtime** (seconds or milliseconds)
2. **Memory usage** (MB)
3. **Graph statistics**:
   - Number of nodes
   - Number of edges
   - Graph density

### Algorithm-Specific Metrics

#### Union-Find / Connected Components
- Number of components
- Largest component size
- Average component size
- Component size distribution

#### BFS / DFS / Traversal Algorithms
- Average shortest path length
- Maximum path length
- Connectivity percentage (reachable node pairs)
- Number of samples tested

#### PageRank / Centrality Algorithms
- Top-k influential nodes
- Average rank score
- Rank distribution
- Rank stability across scenarios

#### Girvan-Newman / Louvain / Community Detection
- Modularity score
- Number of communities
- Community size distribution
- Bridge edges identified

#### Node2Vec / Embeddings
- Embedding dimensionality
- Similarity scores
- Recommendation accuracy
- Diversification metrics

---

## Benchmark Execution Template

```python
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Setup
from src.data_generation import StockDataGenerator
from src.graph import build_graph_from_correlation

sizes = [50, 100, 200]
scenarios = ["stable", "normal", "volatile", "crash"]
results = []

generator = StockDataGenerator(seed=42)

for size in sizes:
    for scenario in scenarios:
        print(f"Testing: {size} stocks, {scenario} scenario")
        
        # Generate data
        returns, corr_matrix, stock_attrs = generator.generate_dataset(size, scenario=scenario)
        
        # Build graph
        threshold = 0.5 if scenario != "crash" else 0.3
        graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
        
        # Benchmark your algorithm
        mem_before = get_memory_usage()
        start_time = time.time()
        
        # >>> YOUR ALGORITHM HERE <<<
        # result = your_algorithm(graph, stock_attrs)
        
        end_time = time.time()
        mem_after = get_memory_usage()
        
        # Record results
        results.append({
            'algorithm': 'YourAlgorithmName',
            'scenario': scenario,
            'num_stocks': size,
            'num_edges': graph.num_edges,
            'runtime_seconds': end_time - start_time,
            'memory_mb': mem_after - mem_before,
            # ... add algorithm-specific metrics
        })
```

---

## Results Storage Format

Save results as JSON in `results/yourname_benchmarks.json`:

```json
[
  {
    "algorithm": "AlgorithmName",
    "scenario": "stable",
    "num_stocks": 50,
    "num_edges": 167,
    "runtime_seconds": 0.0012,
    "memory_mb": 0.05,
    "metric1": 4,
    "metric2": 47
  },
  ...
]
```

---

## Expected Output Example (Guntesh's Results)

Reference benchmark results from Union-Find and BFS:

| Size | Scenario | Edges | Components | Avg Runtime (ms) |
|------|----------|-------|------------|------------------|
| 50   | stable   | 167   | 4          | 0.00             |
| 50   | normal   | 9     | 43         | 0.00             |
| 50   | volatile | 0     | 50         | 0.00             |
| 50   | crash    | 0     | 50         | 0.00             |
| 100  | stable   | 274   | 2          | 0.00             |
| 100  | normal   | 32    | 74         | 0.00             |
| 100  | volatile | 0     | 100        | 0.00             |
| 100  | crash    | 2     | 98         | 0.00             |
| 200  | stable   | 296   | 21         | 0.00             |
| 200  | normal   | 33    | 167        | 0.00             |
| 200  | volatile | 1     | 199        | 0.00             |
| 200  | crash    | 6     | 194        | 0.00             |

---

## Notes for Team Members

1. **Use the same seed (42)** to ensure everyone generates identical graphs
2. **Use the same threshold logic** (0.5 for most, 0.3 for crash)
3. **Report all 12 test cases** for complete comparison
4. **Save JSON results** in `results/` folder with your name
5. **Create visualizations** using the data (see `visualize_results.py` for examples)
6. **Document any deviations** from these test cases in your README

---

## Questions?

Contact: Guntesh Singh (data generation and graph setup lead)

Last Updated: November 17, 2025
