"""Semantic Dependency Graph G_S with three edge types."""

from typing import List, Set, Dict, Tuple, Optional
import networkx as nx
from .types import Rule, EdgeType


class SemanticDepGraph:
    """Directed graph over rules with sequential, trigger_overlap, and tool_conflict edges."""

    def __init__(self, rules: List[Rule] = None):
        self.graph = nx.DiGraph()
        self._rule_ids: Set[str] = set()
        if rules:
            for r in rules:
                self.add_node(r.rule_id)

    def add_node(self, rule_id: str) -> None:
        self.graph.add_node(rule_id)
        self._rule_ids.add(rule_id)

    def add_edge(self, from_id: str, to_id: str, edge_type: EdgeType, weight: float = 0.0) -> None:
        if from_id not in self.graph:
            self.add_node(from_id)
        if to_id not in self.graph:
            self.add_node(to_id)
        self.graph.add_edge(from_id, to_id, type=edge_type, weight=weight)

    def add_sequential_edge(self, from_id: str, to_id: str, weight: float) -> None:
        self.add_edge(from_id, to_id, EdgeType.SEQUENTIAL, weight)

    def add_trigger_overlap_edge(self, id1: str, id2: str, weight: float) -> None:
        self.add_edge(id1, id2, EdgeType.TRIGGER_OVERLAP, weight)
        self.add_edge(id2, id1, EdgeType.TRIGGER_OVERLAP, weight)

    def add_tool_conflict_edge(self, id1: str, id2: str) -> None:
        self.add_edge(id1, id2, EdgeType.TOOL_CONFLICT, 1.0)
        self.add_edge(id2, id1, EdgeType.TOOL_CONFLICT, 1.0)

    @property
    def nodes(self) -> Set[str]:
        return set(self.graph.nodes())

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def has_edge(self, from_id: str, to_id: str) -> bool:
        return self.graph.has_edge(from_id, to_id)

    def get_edge_weight(self, from_id: str, to_id: str) -> float:
        if self.graph.has_edge(from_id, to_id):
            return self.graph[from_id][to_id].get("weight", 0.0)
        return 0.0

    def get_edge_type(self, from_id: str, to_id: str) -> Optional[EdgeType]:
        if self.graph.has_edge(from_id, to_id):
            return self.graph[from_id][to_id].get("type")
        return None

    def predecessors(self, rule_id: str) -> Set[str]:
        if rule_id not in self.graph:
            return set()
        return set(self.graph.predecessors(rule_id))

    def successors(self, rule_id: str) -> Set[str]:
        if rule_id not in self.graph:
            return set()
        return set(self.graph.successors(rule_id))

    def neighbors(self, rule_id: str) -> Set[str]:
        if rule_id not in self.graph:
            return set()
        return set(self.graph.neighbors(rule_id))

    def neighbors_within_distance(self, rule_id: str, distance: int) -> Set[str]:
        """All nodes within graph distance d from rule_id (undirected)."""
        if rule_id not in self.graph:
            return set()
        ug = self.graph.to_undirected()
        result = set()
        for n in ug.nodes():
            try:
                d = nx.shortest_path_length(ug, source=rule_id, target=n)
                if d <= distance:
                    result.add(n)
            except nx.NetworkXNoPath:
                continue
        return result

    def reachable_from(self, rule_id: str) -> Set[str]:
        if rule_id not in self.graph:
            return set()
        return set(nx.descendants(self.graph, rule_id))

    def path_exists(self, from_id: str, to_id: str) -> bool:
        if from_id not in self.graph or to_id not in self.graph:
            return False
        return nx.has_path(self.graph, from_id, to_id)

    def get_sequential_edges_from(self, rule_id: str) -> List[Tuple[str, float]]:
        result = []
        for succ in self.graph.successors(rule_id):
            if self.graph[rule_id][succ].get("type") == EdgeType.SEQUENTIAL:
                result.append((succ, self.graph[rule_id][succ].get("weight", 0.0)))
        return result

    def get_trigger_overlap_pairs(self) -> List[Tuple[str, str]]:
        """Return all trigger-overlapping rule pairs (undirected, unique)."""
        pairs = set()
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == EdgeType.TRIGGER_OVERLAP:
                pair = tuple(sorted([u, v]))
                pairs.add(pair)
        return list(pairs)

    @property
    def max_degree(self) -> int:
        if self.graph.number_of_nodes() == 0:
            return 0
        ug = self.graph.to_undirected()
        return max(dict(ug.degree()).values(), default=0)

    def copy(self) -> "SemanticDepGraph":
        g = SemanticDepGraph()
        g.graph = self.graph.copy()
        g._rule_ids = set(self._rule_ids)
        return g

    def subgraph(self, node_ids: Set[str]) -> "SemanticDepGraph":
        g = SemanticDepGraph()
        g.graph = self.graph.subgraph(node_ids).copy()
        g._rule_ids = node_ids & set(self.graph.nodes())
        return g

    def remove_node(self, rule_id: str) -> None:
        if rule_id in self.graph:
            self.graph.remove_node(rule_id)
            self._rule_ids.discard(rule_id)

    def __repr__(self) -> str:
        return f"SemanticDepGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"
