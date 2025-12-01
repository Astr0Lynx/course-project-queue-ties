
# Standard Test Cases for Stock Market Tangle Project

This document defines the standardized test cases for benchmarking algorithms in this codebase. 

**Graph Sizes:** 100, 250, 500 stocks
**Market Scenarios:** stable, normal, volatile, crash
**Total Test Cases:** 12 (3 sizes × 4 scenarios)


**Random seed:** `42` (for reproducibility)
**Library:** Use `src/data_generation.py` → `StockDataGenerator`
**Correlation threshold for edges:**
  - Stable: `0.40`
  - Normal: `0.44`
  - Volatile: `0.68`
  - Crash: `0.10` (lowest threshold for stress)

### Graph Construction

```python
from src.data_generation import StockDataGenerator
from src.graph import build_graph_from_correlation

generator = StockDataGenerator(seed=42)
returns, corr_matrix, stock_attrs = generator.generate_dataset(num_stocks, scenario=scenario)
if scenario == "crash":
  threshold = 0.10
elif scenario == "stable":
  threshold = 0.40
elif scenario == "normal":
  threshold = 0.44
else:
  threshold = 0.68
graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
```


## Test Case Specifications

### Test Case 1: Stable Market (100 stocks)
- **Size**: 100 stocks
- **Scenario**: `"stable"` (threshold: 0.40)
- **Expected Properties**:
  - Sector separation, few components
  - Moderate connectivity

### Test Case 2: Normal Market (100 stocks)
- **Size**: 100 stocks
- **Scenario**: `"normal"` (threshold: 0.44)
- **Expected Properties**:
  - Moderate fragmentation
  - Moderate connectivity

### Test Case 3: Volatile Market (100 stocks)
- **Size**: 100 stocks
- **Scenario**: `"volatile"` (threshold: 0.68)
- **Expected Properties**:
  - Near complete isolation
  - Many components

### Test Case 4: Crash Market (100 stocks)
- **Size**: 100 stocks
- **Scenario**: `"crash"` (threshold: 0.10)
- **Expected Properties**:
  - Panic selling, high connectivity, usually 1 component

### Test Case 5: Stable Market (250 stocks)
- **Size**: 250 stocks
- **Scenario**: `"stable"` (threshold: 0.40)
- **Expected Properties**:
  - Sector separation, few components
  - Moderate connectivity

### Test Case 6: Normal Market (250 stocks)
- **Size**: 250 stocks
- **Scenario**: `"normal"` (threshold: 0.44)
- **Expected Properties**:
  - Moderate fragmentation
  - Moderate connectivity

### Test Case 7: Volatile Market (250 stocks)
- **Size**: 250 stocks
- **Scenario**: `"volatile"` (threshold: 0.68)
- **Expected Properties**:
  - Near complete isolation
  - Many components

### Test Case 8: Crash Market (250 stocks)
- **Size**: 250 stocks
- **Scenario**: `"crash"` (threshold: 0.10)
- **Expected Properties**:
  - Panic selling, high connectivity, usually 1 component

### Test Case 9: Stable Market (500 stocks)
- **Size**: 500 stocks
- **Scenario**: `"stable"` (threshold: 0.40)
- **Expected Properties**:
  - Sector separation, few components
  - Moderate connectivity

### Test Case 10: Normal Market (500 stocks)
- **Size**: 500 stocks
- **Scenario**: `"normal"` (threshold: 0.44)
- **Expected Properties**:
  - Moderate fragmentation
  - Moderate connectivity

### Test Case 11: Volatile Market (500 stocks)
- **Size**: 500 stocks
- **Scenario**: `"volatile"` (threshold: 0.68)
- **Expected Properties**:
  - Near complete isolation
  - Many components

### Test Case 12: Crash Market (500 stocks)
- **Size**: 500 stocks
- **Scenario**: `"crash"` (threshold: 0.10)
- **Expected Properties**:
  - Panic selling, high connectivity, usually 1 component
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

sizes = [100, 250, 500]
scenarios = ["stable", "normal", "volatile", "crash"]
results = []

generator = StockDataGenerator(seed=42)

for size in sizes:
  for scenario in scenarios:
    print(f"Testing: {size} stocks, {scenario} scenario")
    # Generate data
    returns, corr_matrix, stock_attrs = generator.generate_dataset(size, scenario=scenario)
    # Build graph
    if scenario == "crash":
      threshold = 0.10
    elif scenario == "stable":
      threshold = 0.40
    elif scenario == "normal":
      threshold = 0.44
    else:
      threshold = 0.68
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

Save results as JSON in `results/<algorithm>_benchmarks.json`:

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
| 100  | stable   | 274   | 2          | 0.00             |
| 100  | normal   | 32    | 74         | 0.00             |
| 100  | volatile | 0     | 100        | 0.00             |
| 100  | crash    | 2     | 98         | 0.00             |
| 250  | stable   | ...   | ...        | ...              |
| 250  | normal   | ...   | ...        | ...              |
| 250  | volatile | ...   | ...        | ...              |
| 250  | crash    | ...   | ...        | ...              |
| 500  | stable   | ...   | ...        | ...              |
| 500  | normal   | ...   | ...        | ...              |
| 500  | volatile | ...   | ...        | ...              |
| 500  | crash    | ...   | ...        | ...              |

---

