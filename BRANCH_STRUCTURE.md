# Repository Structure Summary

## Branch Organization

### `main` branch (Shared Components)
**Purpose:** Foundation code that ALL team members use

**Contents:**
- ✅ `src/data_generation.py` - Synthetic stock data generator
- ✅ `src/graph.py` - Graph representation (Graph class + helper functions)
- ✅ `requirements.txt` - Required dependencies
- ✅ `.gitignore` - Ignore patterns for Python/data files

**Why these are shared:**
- Everyone needs to generate the same synthetic stock market data
- Everyone needs to work with the same graph structure
- Ensures consistency across all algorithms

---

### `guntesh-data-foundation` branch (Your Work)
**Purpose:** Guntesh's specific algorithm implementations

**Contents:**
- ✅ `src/data_generation.py` - (same as main)
- ✅ `src/graph.py` - (same as main)
- ✅ `src/union_find.py` - **YOUR Union-Find algorithm**
- ✅ `src/bfs.py` - **YOUR BFS algorithm**
- ✅ `tests/test_union_find.py` - Tests for Union-Find
- ✅ `tests/test_bfs.py` - Tests for BFS
- ✅ `benchmarks/benchmark_guntesh.py` - Performance benchmarking
- ✅ `main.py` - Integration demo
- ✅ `README_GUNTESH.md` - Your documentation
- ✅ `visualize_results.py` - Visualization scripts

---

## Git Status

✅ **Main branch pushed to remote** - Team can now use shared components
✅ **Your branch pushed to remote** - Your work is backed up and separate

---

## For Your Teammates

They should:
1. Clone the repo
2. Work on `main` branch to access shared `data_generation.py` and `graph.py`
3. Create their own branches (e.g., `teammate-algorithm-name`)
4. Import and use the shared modules:
   ```python
   from src.data_generation import StockDataGenerator
   from src.graph import Graph, build_graph_from_correlation
   ```

---

## Next Steps

When your team is ready to integrate:
1. Each person finishes their algorithm on their own branch
2. Create pull requests to merge individual algorithms into main
3. Review and test integrations
4. Final project combines all algorithms working together

---

## Current Commit Status

**Main branch:**
- Latest commit: "Add shared data generation and graph modules for team use"
- Contains: data_generation.py, graph.py, requirements.txt, .gitignore

**Your branch (guntesh-data-foundation):**
- Latest commit: "Add detailed per-test-case performance output to benchmarks and visualization script"
- Contains: All shared files + your Union-Find + BFS + tests + benchmarks + visualizations
