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

(End of report)
