"""Utility module providing a simple undirected weighted Graph class and I/O helpers.

Only Python's standard library is used to keep the implementation lightweight and
portable for stock market network analysis experiments.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple


class Graph:
    """Undirected weighted graph based on adjacency dictionaries.

    Nodes can be any hashable type. Edge weights default to 1.0, making the
    graph suitable for unweighted or weighted network analyses.
    """

    def __init__(self) -> None:
        self._adjacency: Dict[Any, Dict[Any, float]] = {}

    def add_node(self, node: Any) -> None:
        """Add a node to the graph if it does not already exist."""

        if node not in self._adjacency:
            self._adjacency[node] = {}

    def add_edge(self, node_u: Any, node_v: Any, weight: float = 1.0) -> None:
        """Insert an undirected edge between ``node_u`` and ``node_v``.

        Args:
            node_u (Any): First endpoint.
            node_v (Any): Second endpoint.
            weight (float): Optional weight representing relationship strength.
        """

        if node_u == node_v:
            raise ValueError("Self-loops are not supported in this graph implementation.")

        self.add_node(node_u)
        self.add_node(node_v)
        self._adjacency[node_u][node_v] = weight
        self._adjacency[node_v][node_u] = weight

    def remove_edge(self, node_u: Any, node_v: Any) -> None:
        """Remove the undirected edge between ``node_u`` and ``node_v`` if present."""

        if node_v in self._adjacency.get(node_u, {}):
            del self._adjacency[node_u][node_v]
        if node_u in self._adjacency.get(node_v, {}):
            del self._adjacency[node_v][node_u]

    def neighbors(self, node: Any) -> Dict[Any, float]:
        """Return a mapping of neighbors to edge weights for ``node``."""

        return self._adjacency.get(node, {})

    def get_edge_weight(self, node_u: Any, node_v: Any) -> float:
        """Return the weight of an edge, or 0.0 if the edge does not exist."""

        return self._adjacency.get(node_u, {}).get(node_v, 0.0)

    def nodes(self) -> List[Any]:
        """Return a list of nodes currently in the graph."""

        return list(self._adjacency.keys())

    def number_of_nodes(self) -> int:
        """Return the number of nodes in the graph."""

        return len(self._adjacency)

    def number_of_edges(self) -> int:
        """Return the count of undirected edges in the graph."""

        total = sum(len(neighbors) for neighbors in self._adjacency.values())
        return total // 2  # each undirected edge is stored twice

    def degree(self, node: Any) -> float:
        """Return the weighted degree of ``node``."""

        return sum(self._adjacency.get(node, {}).values())

    def total_edge_weight(self) -> float:
        """Return the total weight across all undirected edges."""

        return sum(self.degree(node) for node in self._adjacency) / 2.0

    def copy(self) -> "Graph":
        """Return a deep copy of the graph."""

        new_graph = Graph()
        new_graph._adjacency = deepcopy(self._adjacency)
        return new_graph


def save_graph_to_json(graph: Graph, filename: str) -> None:
    """Persist the graph structure to disk in a simple JSON format.

    The JSON schema is ``{"nodes": [...], "edges": [[u, v, weight], ...]}``.
    Edges are stored with ``u < v`` (based on lexical ordering) to avoid
    duplicates when reconstructing the undirected graph.
    """

    nodes = graph.nodes()
    edges: List[Tuple[Any, Any, float]] = []
    for node_u in nodes:
        for node_v, weight in graph.neighbors(node_u).items():
            if node_u < node_v:
                edges.append((node_u, node_v, weight))

    payload = {
        "nodes": nodes,
        "edges": [[u, v, weight] for u, v, weight in edges],
    }

    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_graph_from_json(filename: str) -> Graph:
    """Load a graph from the JSON format produced by ``save_graph_to_json``."""

    with open(filename, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    graph = Graph()
    for node in payload.get("nodes", []):
        graph.add_node(node)

    for edge in payload.get("edges", []):
        node_u, node_v, weight = edge
        graph.add_edge(node_u, node_v, float(weight))

    return graph
