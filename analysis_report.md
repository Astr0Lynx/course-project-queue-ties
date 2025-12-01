## Results & Analysis Report — Course Project: Queue-Ties

Summary
- Purpose: Present benchmark results for algorithms in this codebase (Union-Find, BFS, DFS, Louvain, Girvan-Newman, PageRank, Node2Vec).
- Data source: JSON files written by `benchmarks.py` in `results/` (e.g. `girvan_newman_benchmarks.json`, `all_benchmarks.json`).

Metrics used
- Wall-clock time (runtime_seconds): end-to-end elapsed time measured by perf_counter/time.
- Memory usage (memory_mb): resident memory difference measured before/after each benchmark step.
- Solution quality:
  - num_components / largest_component / connectivity_ratio (connectivity metrics)
  - modularity (community quality when available)
  - top influence score / influence concentration (PageRank-based quality)
  - embedding_dim / num_embeddings (Node2Vec output size)
- Secondary: iterations performed (PageRank), avg_path_length (BFS), num_communities (community algorithms).

How to reproduce
1. Run the benchmark for the algorithm you care about:
   - Single algorithm: python3 benchmarks.py girvan_newman
   - All algorithms: python3 benchmarks.py
2. Result files will be saved under `results/` (see `runner.save_results`).

Quick interpretation (high-level)
- DFS and Union-Find: expected near-linear performance O(V + E) — very fast for moderate graphs; observed runtimes should be low and scale roughly linearly as size increases.
- BFS: O(V + E) per source; sampling multiple pairs increases runtime linearly with number of samples; expect higher runtimes than a single DFS traversal.
- PageRank: iterative algorithm O(k*(V + E)), runtime scales with iterations and graph density.
- Louvain: heuristic; typically scales near linear or n log n in practice for sparse graphs; modularity quality usually good.
- Girvan-Newman: expensive — repeatedly recomputes edge betweenness; expected high runtime (often super-linear or worse) and may be skipped if implementation not installed. If benchmarks skipped, you will see "Skipped" messages.
- Node2Vec: expensive due to random walks + embedding training; runtime dominated by walk_count*walk_length and training epochs.

Reproducible plotting snippet
- Save the following snippet to `results/plot_benchmarks.py` and run it (requires matplotlib, pandas, numpy).

```python
# filepath: /home/saanvi-jain/sem3/course-project-queue-ties/results/plot_benchmarks.py
import json, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = os.path.dirname(__file__) or '.'
files = glob.glob(os.path.join(RESULT_DIR, '*_benchmarks.json'))
if not files:
    print("No benchmark JSON files found in results/. Run benchmarks first.")
    exit(1)

rows = []
for f in files:
    try:
        data = json.load(open(f))
    except Exception:
        continue
    # data may be list of results
    if isinstance(data, list):
        for r in data:
            r['_source_file'] = os.path.basename(f)
            rows.append(r)
    elif isinstance(data, dict):
        # possibly single result
        r = data
        r['_source_file'] = os.path.basename(f)
        rows.append(r)

if not rows:
    print("No rows parsed from JSON files.")
    exit(1)

df = pd.DataFrame(rows)

# Normalize algorithm naming
df['algorithm'] = df['algorithm'].fillna(df['_source_file'].str.replace('_benchmarks.json','',regex=False).str.title())

# Plot 1: Average runtime per algorithm
group = df.groupby('algorithm')['runtime_seconds'].agg(['mean','median','min','max','count']).sort_values('mean')
plt.figure(figsize=(10,6))
group['mean'].plot(kind='bar', color='C0', yerr=(group['max']-group['min']).values, capsize=4)
plt.ylabel('Mean runtime (s)')
plt.title('Algorithm mean runtime (errorbars: min-max)')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,'figure_runtime_comparison.png'))
print("Saved figure_runtime_comparison.png")

# Plot 2: Memory usage comparison
if 'memory_mb' in df.columns:
    mem = df.groupby('algorithm')['memory_mb'].mean().sort_values()
    plt.figure(figsize=(10,6))
    mem.plot(kind='bar', color='C1')
    plt.ylabel('Mean memory delta (MB)')
    plt.title('Algorithm mean memory usage')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,'figure_memory_comparison.png'))
    print("Saved figure_memory_comparison.png")

# Table: Quality metrics for community algorithms (modularity / num_communities)
quality_cols = []
if 'modularity' in df.columns:
    qdf = df[['algorithm','num_communities','modularity']].dropna(subset=['modularity'])
    if not qdf.empty:
        qdf = qdf.sort_values('modularity', ascending=False)
        qdf.to_csv(os.path.join(RESULT_DIR,'community_quality.csv'), index=False)
        print("Saved community_quality.csv")

# Save combined CSV for ad-hoc analysis
df.to_csv(os.path.join(RESULT_DIR,'benchmarks_combined.csv'), index=False)
print("Saved benchmarks_combined.csv")
print(df.groupby('algorithm').agg({'runtime_seconds':['mean','std','count'],'memory_mb':['mean','std']}))
```

Example visualizations and tables you should include in the report
- figure_runtime_comparison.png — bar chart of mean runtime with min/max error bars.
- figure_memory_comparison.png — mean memory delta by algorithm.
- community_quality.csv — table of modularity and number of communities per run for Louvain / Girvan-Newman (if present).
- benchmarks_combined.csv — raw flattened data for downstream analysis.

Comparing empirical performance to theoretical complexity
- Collect runtime scaling across sizes (100, 250, 500). Plot runtime vs size on linear and log-log axes.
- Expected slopes:
  - DFS / BFS / Union-Find: slope ≈ 1 (linear) in log-log (runtime ∝ n^1) for sparse graphs (E ∝ n).
  - PageRank: slope ≈ 1 per iteration; if iterations constant, runtime ∝ n. If convergence iterations grow with size/density, slope may be higher.
  - Girvan-Newman: runtime grows rapidly; often impossible to run for >500 nodes. If skipped, record as not scalable.
  - Node2Vec: runtime depends on num_walks * walk_length and embedding training; scales roughly linearly with those parameters and with graph size.

Why you may see these results (discussion)
- Graph density affects all algorithms: denser graphs increase E, hurting BFS/PageRank/Node2Vec more than DFS/Union-Find.
- Implementation details:
  - Python-level overhead and data structures (dict-based adjacency lists) raise constants; algorithms with heavy Python loops (Girvan-Newman, Node2Vec training loops) show much larger runtimes.
  - External dependencies: some algorithm implementations may not be present or use networkx; missing implementations cause benchmarks to skip or use fallback, affecting observed results.
- System noise: measuring short-running algorithms may produce near-zero memory deltas or noisy runtimes; aggregate across multiple runs for stability.

Recommendations
- For large-scale benchmarking, increase problem sizes gradually and run each configuration multiple times to average out noise.
- For expensive algorithms (Girvan-Newman, Node2Vec), reduce parameters (fewer iterations, smaller walk counts) when benchmarking, and treat results qualitatively.
- Cache generated graphs / correlation matrices when comparing multiple algorithms to ensure consistent inputs.

Appendix: Example interpretation template (fill with your numbers)
- Algorithm X (100 nodes): mean runtime Y s, memory Z MB, quality Q.
- Observed scaling between 100→500: runtime increased by factor F, expected by theory ~G — explain difference.

## Algorithm Performance Summary

| Algorithm      | Mean Runtime (s) | Min Runtime (s) | Max Runtime (s) | Std Runtime (s) | Mean Memory (MB) | Min Memory (MB) | Max Memory (MB) | Std Memory (MB) |
|----------------|------------------|-----------------|-----------------|-----------------|------------------|-----------------|-----------------|-----------------|
| Union-Find     | 0.0012           | 0.0010          | 0.0015          | 0.0002          | 0.05             | 0.04            | 0.06            | 0.01            |
| BFS            | 0.0050           | 0.0045          | 0.0055          | 0.0003          | 0.07             | 0.06            | 0.08            | 0.01            |
| Louvain        | 0.0200           | 0.0180          | 0.0220          | 0.0015          | 0.10             | 0.09            | 0.11            | 0.01            |
| Girvan-Newman  | 0.1500           | 0.1400          | 0.1600          | 0.0080          | 0.30             | 0.28            | 0.32            | 0.02            |
| DFS            | 0.0010           | 0.0009          | 0.0012          | 0.0001          | 0.05             | 0.04            | 0.06            | 0.01            |
| PageRank       | 0.0100           | 0.0090          | 0.0110          | 0.0005          | 0.08             | 0.07            | 0.09            | 0.01            |
| Node2Vec       | 0.0500           | 0.0480          | 0.0520          | 0.0010          | 0.15             | 0.14            | 0.16            | 0.01            |

---

## Solution Quality Table (Community Algorithms)

| Algorithm      | Scenario | Graph Size | Num Communities | Modularity |
|----------------|----------|------------|-----------------|-----------|
| Louvain        | stable   | 100        | 5               | 0.42      |
| Louvain        | crash    | 100        | 1               | 0.01      |
| Girvan-Newman  | stable   | 100        | 4               | 0.39      |
| Girvan-Newman  | crash    | 100        | 1               | 0.00      |

---

## Scaling Table (Runtime vs Graph Size)

| Algorithm      | Scenario | Graph Size | Runtime (s) | Memory (MB) |
|----------------|----------|------------|-------------|-------------|
| DFS            | stable   | 100        | 0.0010      | 0.05        |
| DFS            | stable   | 250        | 0.0025      | 0.06        |
| DFS            | stable   | 500        | 0.0050      | 0.07        |
| BFS            | stable   | 100        | 0.0050      | 0.07        |
| BFS            | stable   | 250        | 0.0125      | 0.08        |
| BFS            | stable   | 500        | 0.0250      | 0.09        |
| Louvain        | stable   | 100        | 0.0200      | 0.10        |
| Louvain        | stable   | 250        | 0.0500      | 0.12        |
| Louvain        | stable   | 500        | 0.1000      | 0.15        |

---

## Connectivity Table (DFS/Union-Find)

| Algorithm   | Scenario | Graph Size | Num Components | Largest Component | Connectivity Ratio |
|-------------|----------|------------|----------------|-------------------|-------------------|
| DFS         | stable   | 100        | 1              | 100               | 1.00              |
| DFS         | normal   | 100        | 41             | 60                | 0.60              |
| Union-Find  | crash    | 100        | 1              | 100               | 1.00              |

---

## PageRank Quality Table

| Scenario | Graph Size | Top Score | Avg Score | Iterations |
|----------|------------|-----------|-----------|------------|
| stable   | 100        | 0.032     | 0.010     | 100        |
| crash    | 100        | 0.020     | 0.010     | 100        |

---

## Node2Vec Embedding Table

| Scenario | Graph Size | Embedding Dim | Num Embeddings | Runtime (s) | Memory (MB) |
|----------|------------|---------------|---------------|-------------|-------------|
| stable   | 100        | 32            | 100           | 0.050       | 0.15        |
| crash    | 100        | 32            | 100           | 0.048       | 0.14        |

---

*Replace the numbers above with your actual results from the CSVs produced by your benchmarks.*
(End of report)
