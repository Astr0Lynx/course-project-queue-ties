"""
Louvain Community Detection Algorithm (From Scratch)
Author: Saanvi Jain

Implements the Louvain modularity-optimization algorithm using the shared
Graph structure (src/graph.py). Weighted edges supported.
"""

from collections import defaultdict
import math

def compute_modularity(graph, communities):
    """
    Compute modularity for a given community assignment.
    communities: dict[node] = community_id
    """
    m = 0
    for u in graph.get_nodes():
        for v, w in graph.get_neighbors(u).items():
            m += w
    m /= 2  # undirected graph

    if m == 0:
        return 0

    # Sum of degrees per community
    community_degree = defaultdict(float)
    # Sum of internal weights per community
    community_internal = defaultdict(float)

    for u in graph.get_nodes():
        cu = communities[u]
        for v, w in graph.get_neighbors(u).items():
            if communities[v] == cu:
                community_internal[cu] += w
            community_degree[cu] += w

    modularity = 0
    for c in community_degree:
        modularity += (community_internal[c] / (2*m)) - (community_degree[c] / (2*m))**2

    return modularity


def louvain(graph):
    """
    Complete Louvain algorithm.

    Returns:
        communities (dict): node -> community id
        final_modularity (float)
    """
    # INITIAL PHASE: each node is its own community
    nodes = graph.get_nodes()
    communities = {node: i for i, node in enumerate(nodes)}

    improved = True
    while improved:
        improved = False

        # ---- PHASE 1: Local modularity optimization ----
        local_improved = True
        while local_improved:
            local_improved = False
            for node in nodes:
                current_comm = communities[node]

                # Compute degree sum from node to each community
                comm_weights = defaultdict(float)
                for nbr, w in graph.get_neighbors(node).items():
                    comm_weights[communities[nbr]] += w

                # Remove node from its community temporarily
                communities[node] = -1

                # Best gain
                best_comm = current_comm
                best_gain = -1e18

                # Try placing node in all neighbor communities
                for comm, w_to_comm in comm_weights.items():
                    communities[node] = comm
                    mod = compute_modularity(graph, communities)
                    if mod > best_gain:
                        best_gain = mod
                        best_comm = comm

                # Move to best community found
                if best_comm != current_comm:
                    communities[node] = best_comm
                    local_improved = True
                else:
                    communities[node] = current_comm

            if local_improved:
                improved = True

        # ---- PHASE 2: Graph aggregation ----
        # Build super-graph
        new_graph = defaultdict(lambda: defaultdict(float))
        comm_map = defaultdict(list)

        # Invert mapping (community → list of nodes)
        for node, comm in communities.items():
            comm_map[comm].append(node)

        # Rebuild weighted edges between "supernodes"
        for ci, nodes_i in comm_map.items():
            for cj, nodes_j in comm_map.items():
                weight_sum = 0
                for u in nodes_i:
                    for v in nodes_j:
                        w = graph.get_edge_weight(u, v)
                        if w:
                            weight_sum += w
                if weight_sum > 0:
                    new_graph[ci][cj] = weight_sum

        # Convert aggregated adjacency list to Graph object if changed
        if len(new_graph) == len(communities):
            # No further simplification possible → end
            break

        # Build new Graph object
        from graph import Graph
        agg = Graph()
        for u in new_graph:
            agg.add_node(str(u))
        for u in new_graph:
            for v, w in new_graph[u].items():
                if u != v:
                    agg.add_edge(str(u), str(v), w)

        # Update communities mapping: each supernode is its own community
        communities = {node: i for i, node in enumerate(agg.get_nodes())}
        graph = agg

    final_modularity = compute_modularity(graph, communities)
    return communities, final_modularity
