# Girvan-Newman Community Detection – Theory, Implementation, and Results

## Algorithm Theory Refresher
### High-level pipeline
1. **Edge betweenness centrality.** Compute $BC(e)$ for every edge, where
   $$BC(e) = \sum_{s \ne t \in V} \frac{\sigma_{st}(e)}{\sigma_{st}},$$
   $\sigma_{st}$ is the number of shortest $s\!\to\!t$ paths, and $\sigma_{st}(e)$ counts those that traverse edge $e$.
2. **Remove structural bridges.** Delete all edges attaining the current maximum betweenness to peel apart bridges between densely knit regions.
3. **Re-evaluate components.** Recompute connected components; each becomes a tentative community.
4. **Score via modularity.** Measure structure quality with modularity
   $$Q = \frac{1}{2m} \sum_{c \in \mathcal{C}} \sum_{i,j \in c} \left(A_{ij} - \frac{k_i k_j}{2m}\right),$$
   where $m$ is the total edge weight, $A_{ij}$ is the adjacency weight, and $k_i$ is the weighted degree of node $i$.
5. **Track the best partition.** Keep the partition that yields the highest $Q$ until no edges remain or an iteration cap is reached.

### Complexity and practical challenges
- Computing betweenness exactly with Brandes' BFS-based method is $O(V \cdot (V+E))$ per full pass because we run a multi-source BFS from every vertex and accumulate dependencies.
- GN recomputes betweenness **after each edge removal wave**, so in the dense case we pay the above cost roughly $E$ times, giving $O(E \cdot V \cdot (V+E))$ $\approx O((V+E)^2)$.
- Memory pressure arises from storing predecessor lists, shortest-path counts, and dependency stacks for every node. The implementation keeps these as plain Python dicts/lists to minimize overhead while staying readable.
- To keep work bounded, `girvan_newman()` exposes `max_iterations`, and the helper `_build_sample_stock_graph()` gradually relaxes correlation thresholds or bridges components to avoid pathological disconnected inputs.

## Implementation Walk-through (`src/girvan_newman.py`)
### Graph substrate
- Uses the shared `Graph` class from `src/graph.py` (adjacency list stored as nested dicts with weights, plus node attributes for sector metadata).
- Utility functions `load_graph_from_json()`/`save_graph_to_json()` ensure experiments are reproducible and file-format compatible with the rest of the project.

### Edge betweenness: `betweenness_centrality(graph)`
- Implements Brandes' algorithm explicitly: for every source node it runs BFS, tracks predecessor lists, number of shortest paths, and a dependency stack.
- Because the graph is undirected, the contribution for each edge is halved at the end (each shortest path is discovered twice, once from each endpoint order).
- Time: $O(V \cdot (V+E))$; Space: $O(V+E)$.

### Quality scoring: `modularity(graph, communities)`
- Reuses the **original** graph to evaluate modularity even as edges are removed from the working copy.
- Weighted degrees come from `Graph.degree()`, and `Graph.get_edge_weight()` provides $A_{ij}$.
- Nested loops over nodes inside each community dominate runtime with $O(C^2 k^2)$ in the worst case (where $C$ is community count, $k$ average community size). For the data sizes we tested ($\le 32$ nodes) this remained negligible.

### Main loop: `girvan_newman(graph, max_iterations=None)`
- Defensive guard: `_connected_components()` ensures the input graph is connected; otherwise a `ValueError` is raised so users can fix data generation.
- Operates on a copy via `graph.copy()` to preserve caller data.
- Iterative process:
  1. Compute edge betweenness on the current working graph.
  2. Remove **all** edges that achieve the max betweenness (ties handled via a tolerance of $1\mathrm{e}{-9}$).
  3. Run `_connected_components()`; if multiple components appear, evaluate modularity and update the best partition tracker.
- Stops when the graph becomes edgeless or the optional iteration budget is exhausted. Returns `(best_partition, best_modularity)`.

### Data-generation helpers
- `_build_sample_stock_graph()` leans on `StockDataGenerator` (if available) to create synthetic returns, compute correlations, and attach attributes such as `volatility`, `stability`, `category`, and `mean_return` to each node.
- If strict thresholds disconnect the graph, `_connect_components_with_correlations()` stitches components together using the strongest surviving cross-component correlations to honor GN's connected-graph requirement.
- A hard-coded `_build_manual_stock_graph()` exists as a fallback when numpy/pandas are missing.

## Experimental Setup for Sector Analysis
- **Data generation.** Used `_build_sample_stock_graph()` to synthesize four market scenarios (`stable`, `normal`, `volatile`, `crash`) with 32 stocks each. Initial correlation threshold set to 0.65 and relaxed down to 0.25 before bridging components when needed. Attributes attach a categorical label (`stable`, `moderate`, `volatile`) derived from simulated volatility.
- **Execution.** Ran `girvan_newman()` on each scenario and captured: node/edge counts, number of discovered communities, modularity, per-community size, volatility-category histogram, and average stability (inverse volatility) inside each community. Script available upon request (uses `girvan_newman.analyze_scenario`, stored in the execution log for reproducibility).

## Results & Analysis – Sector Cohesion vs. Theory
| Scenario | Nodes | Edges | Communities | Modularity | Dominant composition & notes |
| --- | --- | --- | --- | --- | --- |
| Stable | 32 | 260 | 19 | 0.0825 | Overly dense correlations keep betweenness values nearly uniform, so GN repeatedly removes bridges, yielding many singleton/size-2 clusters with a mix of `stable` and `moderate` nodes. Low $Q$ reflects weak separation. |
| Normal | 32 | 92 | 3 | 0.2714 | Two medium clusters (sizes 17 & 13) containing blends of `moderate`/`volatile` stocks plus one tiny bridge component. GN identifies intuitive splits, but categories remain mixed because absolute correlations between moderate & volatile names stay high enough to keep them co-resident. |
| Volatile | 32 | 31 | 5 | 0.6304 | Sparse network with clear bridges; each community is >80% `volatile` with average stability $\approx 0.94$. Strong sector cohesion emerges and modularity is high, matching theory (few cross-sector ties). |
| Crash | 32 | 31 | 6 | 0.6524 | Similar sparsity but even lower stability ($\approx 0.91$). GN isolates homogeneous `volatile` pockets, underscoring its strength when market stress amplifies sector separation. |

### Sector cohesion takeaways
- **Dense/stable regimes.** When correlations hover near one another (stable scenario) GN struggles: betweenness values tie frequently, causing near-uniform pruning that fragments what should be a single sector. Theoretical expectations warn about this—GN assumes sparse bridges, so a high-density graph violates the premise and depresses $Q$.
- **Mixed regimes.** In the normal scenario the algorithm still finds a respectable $Q$, but categories remain mixed because volatility-derived sectors do not align perfectly with correlation structure. This highlights that GN detects **structural** communities, not necessarily pre-defined sectors.
- **Stress regimes.** Under volatile/crash conditions, correlations polarize: intra-sector edges stay relatively strong while inter-sector edges vanish. GN excels here, quickly surfacing cohesive, single-category clusters with modularity >0.63, agreeing with theoretical predictions that GN shines when a few edges act as inter-sector bridges.

### Accuracy vs. theoretical expectations
- **Agreement:** In sparse, high-contrast graphs (volatile/crash), GN's edge-removal strategy aligns with the theoretical "bridge removal" intuition; communities resemble ground-truth volatility groupings and modularity peaks.
- **Deviations:** In dense graphs the theoretical $O((V+E)^2)$ recomputation becomes expensive while producing marginal gains—edge betweenness changes slowly, so we incur heavy cost for little structural benefit. Moreover, sector labels derived from volatility may not match the correlation-driven topology, so "accuracy" should be interpreted as **modularity maximization** rather than semantic sector recovery.
- **Mitigations:**
  - Limit iterations via `max_iterations` when $Q$ plateaus to avoid quadratic blow-ups.
  - Pre-filter edges with low informational value (e.g., keep top-$k$ correlations per node) before GN.
  - For dense markets, pair GN with Louvain or spectral clustering to validate whether low modularity truly reflects a lack of sector separation rather than algorithmic limitations.

## Recommendations & Next Steps
1. **Hybrid evaluation:** Combine GN results with sector metadata to produce confusion matrices (category vs. community) for future presentations.
2. **Performance guardrails:** Instrument runtime/edge counts per iteration to empirically confirm the $O((V+E)^2)$ growth and set practical limits for larger synthetic universes (>200 nodes).
3. **Visualization:** Use matplotlib to plot modularity vs. iteration and sector purity per community—useful for the report's Results & Analysis section.
4. **Benchmark parity:** Run Louvain on the same scenarios to showcase the trade-off: Louvain should be faster but may produce slightly lower purity in stress regimes, underscoring GN's accuracy advantage when edges truly act as bridges.

With the above documentation, the Girvan-Newman section now covers theory, implementation, complexity, and empirical behavior, ready to be integrated into the overall Results & Analysis narrative for the course project.
