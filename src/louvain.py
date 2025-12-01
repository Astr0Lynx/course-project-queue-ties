"""Louvain community detection for stock correlation graphs.

This module implements a lightweight Louvain method suitable for the
project's `Graph` representation (adjacency-list based). It provides:

- `louvain` : main function that returns a partition and modularity
- utility helpers: modularity calculation, graph aggregation, strengths
- example `main()` showing usage with a small sample graph

Author: Generated for Saanvi Jain (adapted to project Graph API)
"""

from typing import Dict, List, Tuple, Iterable
import random
import copy

from graph import Graph


def _total_edge_weight(graph: Graph) -> float:
    """Return total weight (m) of the graph (sum of unique undirected edges)."""
    return sum(weight for _, _, weight in graph.get_edges())


def _node_strengths(graph: Graph) -> Dict[str, float]:
    """Compute node strengths (sum of weights of incident edges).

    Returns:
        dict node -> strength (float)
    """
    strengths: Dict[str, float] = {}
    for node in graph.get_nodes():
        neigh = graph.get_neighbors(node)
        strengths[node] = sum(float(w) for w in neigh.values())
    return strengths


def _partition_to_communities(partition_map: Dict[str, int]) -> List[List[str]]:
    communities: Dict[int, List[str]] = {}
    for node, comm in partition_map.items():
        communities.setdefault(comm, []).append(node)
    return [sorted(nodes) for nodes in communities.values()]


def modularity(graph: Graph, partition: Iterable[Iterable[str]]) -> float:
    """Compute modularity of `partition` on `graph`.

    Uses weighted modularity definition:
    Q = (1/(2m)) * sum_{ij} [ A_ij - (k_i k_j) / (2m) ] delta(c_i, c_j)
    """
    communities = [list(c) for c in partition]
    m = _total_edge_weight(graph)
    if m == 0:
        return 0.0

    strengths = _node_strengths(graph)
    norm = 2.0 * m
    node_index = {node: idx for idx, node in enumerate(graph.get_nodes())}

    # Build quick lookup for adjacency weights
    adj = {}
    for u, v, w in graph.get_edges():
        adj.setdefault(u, {})[v] = float(w)
        adj.setdefault(v, {})[u] = float(w)

    q = 0.0
    for community in communities:
        for i in community:
            for j in community:
                Aij = adj.get(i, {}).get(j, 0.0)
                ki = strengths.get(i, 0.0)
                kj = strengths.get(j, 0.0)
                q += Aij - (ki * kj) / norm

    return q / norm


def _aggregate_graph(graph: Graph, partition_map: Dict[str, int]) -> Graph:
    """Aggregate `graph` according to `partition_map` (node->community id).

    New nodes are community ids (as strings). Edge weight between two
    communities is the sum of weights between their member nodes.
    """
    agg = Graph()
    # create nodes for each community
    comm_nodes = set(partition_map.values())
    for cid in comm_nodes:
        agg.add_node(str(cid), {})

    # accumulate weights
    weights: Dict[Tuple[str, str], float] = {}
    for u, v, w in graph.get_edges():
        cu = str(partition_map[u])
        cv = str(partition_map[v])
        key = tuple(sorted((cu, cv)))
        weights[key] = weights.get(key, 0.0) + float(w)

    for (cu, cv), w in weights.items():
        if cu == cv:
            # self-loop: add as undirected edge (keep weight as connection strength)
            # Graph expects undirected edges; adding edge will create symmetric entries
            # For a self-loop we still add an edge but ensure nodes exist
            if cu not in agg.get_nodes():
                agg.add_node(cu)
            # Represent self-loop as a tiny edge to itself (no-op for community id)
            # Skip adding self-loop to avoid Graph's undirected duplicate handling
            continue
        else:
            # add or increment edge between communities
            if not agg.has_edge(cu, cv):
                agg.add_edge(cu, cv, w)
            else:
                # increment existing weight
                existing = agg.get_edge_weight(cu, cv) or 0.0
                # remove and re-add to set new weight (Graph doesn't have set_edge)
                agg.remove_edge(cu, cv)
                agg.add_edge(cu, cv, existing + w)

    return agg


def louvain(graph: Graph, resolution: float = 1.0, random_state: int | None = None,
            max_passes: int = 10) -> Tuple[List[List[str]], float]:
    """Run the Louvain method on `graph`.

    Args:
        graph: input Graph (uses adjacency weights)
        resolution: resolution parameter (keeps 1.0 for standard modularity)
        random_state: seed for node ordering
        max_passes: maximum number of phase-1 passes per level

    Returns:
        (partition, modularity)
    """
    if random_state is not None:
        random.seed(random_state)

    working_graph = graph
    # Initial partition: each node in its own community
    partition_map = {node: idx for idx, node in enumerate(working_graph.get_nodes())}
    strengths = _node_strengths(working_graph)
    m = _total_edge_weight(working_graph)
    if m == 0:
        return _partition_to_communities(partition_map), 0.0

    current_mod = modularity(working_graph, _partition_to_communities(partition_map))
    improvement = True
    while improvement:
        improvement = False
        passes = 0
        while passes < max_passes:
            passes += 1
            moved = 0
            nodes = working_graph.get_nodes()
            random.shuffle(nodes)

            # Community weights
            comm_weight: Dict[int, float] = {}
            for node, comm in partition_map.items():
                comm_weight[comm] = comm_weight.get(comm, 0.0) + strengths.get(node, 0.0)

            for node in nodes:
                node_comm = partition_map.get(node, None)
                if node_comm is None:
                    # Assign to new community if missing
                    node_comm = max(comm_weight.keys(), default=0) + 1
                    partition_map[node] = node_comm
                    comm_weight[node_comm] = strengths.get(node, 0.0)
                node_ki = strengths.get(node, 0.0)

                # Neighboring community weights
                neighbor_comms: Dict[int, float] = {}
                for nbr, w in working_graph.get_neighbors(node).items():
                    nbr_comm = partition_map.get(nbr, None)
                    if nbr_comm is None:
                        nbr_comm = max(comm_weight.keys(), default=0) + 1
                        partition_map[nbr] = nbr_comm
                        comm_weight[nbr_comm] = strengths.get(nbr, 0.0)
                    neighbor_comms[nbr_comm] = neighbor_comms.get(nbr_comm, 0.0) + float(w)

                # Remove node from current community temporarily
                comm_weight[node_comm] -= node_ki

                # Find best community to move to
                best_delta = 0.0
                best_comm = node_comm
                for comm, k_i_in in neighbor_comms.items():
                    sum_tot = comm_weight.get(comm, 0.0)
                    delta_q = (k_i_in - (node_ki * sum_tot) / (2.0 * m)) / (2.0 * m)
                    if delta_q * resolution > best_delta:
                        best_delta = delta_q * resolution
                        best_comm = comm

                # Move if modularity improves
                if best_comm != node_comm and best_delta > 1e-12:
                    partition_map[node] = best_comm
                    comm_weight[best_comm] = comm_weight.get(best_comm, 0.0) + node_ki
                    moved += 1
                else:
                    comm_weight[node_comm] = comm_weight.get(node_comm, 0.0) + node_ki

            if moved > 0:
                improvement = True
            else:
                break

        # Phase 2: aggregate graph and repeat if improvement
        # Compact community ids
        mapping = {}
        next_id = 0
        new_partition = {}
        for node, comm in partition_map.items():
            if comm not in mapping:
                mapping[comm] = next_id
                next_id += 1
            new_partition[node] = mapping[comm]

        new_graph = _aggregate_graph(working_graph, new_partition)
        # After aggregation, partition_map = {node: int(node) for node in new_graph.get_nodes()}
        partition_map = {node: int(node) for node in new_graph.get_nodes()}
        strengths = _node_strengths(new_graph)
        working_graph = new_graph
        m = _total_edge_weight(working_graph)

        # Always return partition as a list of lists (communities)
        communities = _partition_to_communities(new_partition)
        new_mod = modularity(graph, communities)

        if new_mod - current_mod > 1e-7 and new_graph.get_nodes():
            improvement = True
            current_mod = new_mod
        else:
            # Final output: partition is always a list of lists
            return communities, new_mod

    # Final output: partition is always a list of lists
    return _partition_to_communities(partition_map), current_mod


def _build_sample_graph() -> Graph:
    g = Graph()
    # small example with two dense clusters lightly connected
    cluster1 = ["A", "B", "C", "D"]
    cluster2 = ["E", "F", "G"]
    for u in cluster1:
        for v in cluster1:
            if u < v:
                g.add_edge(u, v, 1.0)
    for u in cluster2:
        for v in cluster2:
            if u < v:
                g.add_edge(u, v, 1.0)
    # weak cross links
    g.add_edge("C", "E", 0.2)
    g.add_edge("D", "F", 0.15)
    return g


def main() -> None:
    """Example runner for Louvain."""
    g = _build_sample_graph()
    partition, score = louvain(g, random_state=42)
    print("Detected communities:")
    for i, comm in enumerate(partition, 1):
        print(f"  Community {i}: {comm}")
    print(f"Modularity: {score:.4f}")


if __name__ == "__main__":
    main()
