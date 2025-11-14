# Stock Market Tangle - Guntesh's Implementation

**Author:** Guntesh Singh  
**Team:** Queue-ties  
**Course:** Algorithm Analysis and Design

## Overview

This module implements the foundational components of the Stock Market Tangle project:
- **Data Generation**: Synthetic stock market data with realistic correlations
- **Graph Representation**: Efficient adjacency list structure for stock correlation graphs
- **Union-Find Algorithm**: Finding market segments (connected components)
- **BFS Algorithm**: Shortest paths and correlation chain analysis

All algorithms are implemented **from scratch** without using external graph libraries (no NetworkX for core algorithms).

## Project Structure

```
course-project-queue-ties/
├── src/
│   ├── data_generation.py    # Synthetic stock data generator
│   ├── graph.py               # Graph representation (shared with team)
│   ├── union_find.py          # Union-Find algorithm
│   └── bfs.py                 # Breadth-First Search algorithm
├── tests/
│   ├── test_union_find.py     # Union-Find tests
│   └── test_bfs.py            # BFS tests
├── benchmarks/
│   └── benchmark_guntesh.py   # Performance benchmarking script
├── data/                      # Generated datasets (created at runtime)
├── results/                   # Benchmark and analysis results
├── main.py                    # Main integration script
├── requirements.txt           # Python dependencies
├── PLAN.md                    # Team plan and task distribution
└── README.md                  # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd c:/Users/Guntesh/Desktop/foo/aad/course-project-queue-ties
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Run Complete Workflow

```bash
python main.py
```

This demonstrates the entire pipeline:
1. Generate synthetic stock data
2. Build correlation graph
3. Find market segments (Union-Find)
4. Analyze correlation paths (BFS)
5. Export results

### Generate Data Only

```bash
cd src
python data_generation.py
```

This creates multiple datasets in the `data/` directory for different market scenarios.

### Run Tests

```bash
# Test Union-Find
python tests/test_union_find.py

# Test BFS
python tests/test_bfs.py
```

### Run Benchmarks

```bash
python benchmarks/benchmark_guntesh.py
```

This benchmarks both algorithms across different:
- Graph sizes (50, 100, 200 stocks)
- Market scenarios (stable, normal, volatile, crash)

Results are saved to `results/guntesh_benchmarks.json`.

## Algorithm Details

### 1. Union-Find (Disjoint Set Union)

**Purpose:** Identify connected components (market segments) in the stock correlation graph.

**Implementation:**
- Path compression for faster lookups
- Union by rank to keep trees balanced
- Time complexity: O(E × α(V)) ≈ O(E) where α is inverse Ackermann function
- Space complexity: O(V)

**Use cases:**
- Finding stocks that are correlated (directly or indirectly)
- Identifying market sectors/clusters
- Detecting isolated stock groups

**Key Functions:**
```python
from union_find import UnionFind, find_market_segments

uf = UnionFind(stock_list)
uf.union("STOCK_A", "STOCK_B")
connected = uf.connected("STOCK_A", "STOCK_C")
components = uf.get_component_list()
```

### 2. Breadth-First Search (BFS)

**Purpose:** Find shortest correlation chains between stocks and analyze connectivity.

**Implementation:**
- Standard BFS using queue (deque)
- Volatility-weighted variant prioritizes low-risk paths
- Time complexity: O(V + E)
- Space complexity: O(V)

**Use cases:**
- Finding shortest path between two stocks
- Analyzing correlation chains
- Computing average path lengths
- Connectivity analysis

**Key Functions:**
```python
from bfs import bfs_shortest_path, analyze_graph_connectivity

path = bfs_shortest_path(graph, "STOCK_A", "STOCK_Z")
connectivity = analyze_graph_connectivity(graph)
low_vol_path = bfs_shortest_path_volatility_weighted(graph, start, end, attributes)
```

### 3. Graph Representation

**Structure:** Adjacency list using `dict of dicts`
```python
{
    "STOCK_A": {"STOCK_B": 0.75, "STOCK_C": 0.62},
    "STOCK_B": {"STOCK_A": 0.75, "STOCK_D": 0.80},
    ...
}
```

**Features:**
- Efficient neighbor lookup: O(1)
- Edge weight storage (correlation values)
- Node attributes (volatility, stability, etc.)
- Undirected graph support

### 4. Data Generation

**Features:**
- Realistic stock return correlations using factor model
- Multiple market scenarios (stable, normal, volatile, crash)
- Configurable volatility ranges
- Stock attributes: volatility, stability, mean return, category

**Scenarios:**
- **Stable**: Low volatility (0.005-0.02), positive returns
- **Normal**: Moderate volatility (0.01-0.05), small positive returns
- **Volatile**: High volatility (0.03-0.08), variable returns
- **Crash**: Very high volatility (0.05-0.15), negative returns

## Experimental Setup

### Datasets

Benchmark suite tests on:
- **Small**: 50 stocks
- **Medium**: 100 stocks
- **Large**: 200 stocks

Each with 4 market scenarios (12 total configurations).

### Metrics

**Union-Find:**
- Number of connected components
- Component size distribution
- Runtime and memory usage

**BFS:**
- Average shortest path length
- Path finding success rate
- Runtime and memory usage
- Connectivity statistics

### Hardware/Software

- **OS:** Windows 11
- **Python:** 3.x
- **CPU:** [Will be recorded during benchmarks]
- **RAM:** [Will be recorded during benchmarks]

## Team Integration

### Shared Components

This implementation provides **shared components** for the team:

1. **`graph.py`**: Common graph structure everyone will use
2. **`data_generation.py`**: Generate datasets for all algorithms
3. **Helper functions**: Graph building, statistics, save/load

### For Other Team Members

To use the graph structure in your algorithms:

```python
from graph import Graph, load_graph

# Option 1: Build from correlation data
from graph import build_graph_from_correlation
graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold=0.5)

# Option 2: Load pre-built graph
graph = load_graph("data/demo_graph.json")

# Use in your algorithm
nodes = graph.get_nodes()
neighbors = graph.get_neighbors(node)
edges = graph.get_edges()
```

## Results and Analysis

Results are exported to `results/` directory:

- **Benchmark results**: `guntesh_benchmarks.json`
- **Demo summary**: `demo_summary.json`
- **Generated graphs**: `data/*.json`

### Sample Output Format

```json
{
  "algorithm": "Union-Find",
  "scenario": "normal",
  "num_stocks": 100,
  "runtime_seconds": 0.0234,
  "num_components": 5,
  "largest_component": 67
}
```

## Complexity Analysis

### Union-Find
- **Time:** O(m × α(n)) where m = operations, n = elements, α ≈ constant
- **Space:** O(n) for parent and rank arrays
- **Optimizations:** Path compression, union by rank

### BFS
- **Time:** O(V + E) for standard BFS
- **Space:** O(V) for queue and visited set
- **Variants:** 
  - Volatility-weighted: O(V + E log V) due to neighbor sorting
  - All paths bounded: O(V + E) with depth limit

## Known Limitations

- Synthetic data may not capture all real market behaviors
- Correlation threshold is a simplification (could use dynamic thresholds)
- Memory usage grows with graph size (expected for adjacency list)

## Future Enhancements

Potential bonus features:
- Dynamic threshold selection based on scenario
- Multi-level market segments (hierarchical Union-Find)
- Weighted BFS using correlation strength as edge weights
- Temporal analysis (how segments change over time)

## References

- Course materials on graph algorithms
- "Introduction to Algorithms" (CLRS) - Union-Find and BFS chapters
- Project guidelines and problem statement

## Contact

**Guntesh Singh**  
Team: Queue-ties  
Branch: `guntesh-data-foundation`

---

**Note:** This implementation is part of a team project. Other team members are implementing DFS, PageRank, Girvan-Newman, Louvain, and Node2Vec algorithms.
