"""Module 5: Incremental Update Engine."""

from typing import List, Dict, Set
from ..core.types import (
    Rule, Conflict, Module2Output, Module3Output, Module5Output
)
from ..core.graph import SemanticDepGraph
from ..utils.config import Config


class IncrementalUpdateEngine:
    """Updates graph, conflict matrix, and influence sets after conflict resolution.

    Restricted to rules within graph distance 2 of each modified rule.
    """

    def __init__(self, sbert_encoder, nli_classifier, config: Config):
        self.sbert = sbert_encoder
        self.nli = nli_classifier
        self.config = config

    def run(self, merged_rules: List[Rule],
            graph: SemanticDepGraph,
            conflicts: List[Conflict],
            influence_sets: Dict[str, Set[str]],
            resolved_conflicts: List[Conflict]) -> Module5Output:
        """Update all structures restricted to distance-2 neighborhood."""

        # Identify affected rules from resolved conflicts
        resolved_rule_ids = set()
        for c in resolved_conflicts:
            resolved_rule_ids.update(c.involved_rules)

        # Get all rules within distance 2
        affected = self.affected_rules(resolved_rule_ids, graph)

        # Recompute edges for affected neighborhood
        updated_graph = self.recompute_edges(graph, affected, merged_rules)

        # Recompute conflicts in affected neighborhood
        updated_conflicts = self.recompute_conflicts(conflicts, affected, resolved_rule_ids)

        # Recompute influence sets
        updated_influence = self.recompute_influence(influence_sets, updated_graph, affected)

        return Module5Output(
            updated_graph=updated_graph,
            updated_conflicts=updated_conflicts,
            updated_influence_sets=updated_influence,
        )

    def affected_rules(self, resolved_rule_ids: Set[str],
                        graph: SemanticDepGraph) -> Set[str]:
        """All rules within graph distance 2 of any resolved rule."""
        affected = set()
        for rid in resolved_rule_ids:
            affected.update(graph.neighbors_within_distance(rid, 2))
        affected.update(resolved_rule_ids)
        return affected

    def recompute_edges(self, graph: SemanticDepGraph,
                         affected: Set[str],
                         merged_rules: List[Rule]) -> SemanticDepGraph:
        """Recompute edges within the affected subgraph."""
        updated = graph.copy()
        theta_seq = self.config.get("thresholds.theta_seq", 0.7)
        theta_trig = self.config.get("thresholds.theta_trig", 0.6)

        rule_map = {r.rule_id: r for r in merged_rules}

        affected_list = sorted(affected & set(rule_map.keys()))
        for i, rid_i in enumerate(affected_list):
            ri = rule_map.get(rid_i)
            if ri is None:
                continue
            for rid_j in affected_list[i + 1:]:
                rj = rule_map.get(rid_j)
                if rj is None:
                    continue

                # Remove old edges
                if updated.has_edge(rid_i, rid_j):
                    updated.graph.remove_edge(rid_i, rid_j)
                if updated.has_edge(rid_j, rid_i):
                    updated.graph.remove_edge(rid_j, rid_i)

                # Sequential dependency check
                if ri.postcondition and rj.precondition:
                    ent = self.nli.entailment_score(ri.postcondition, rj.precondition)
                    if ent > theta_seq:
                        updated.add_sequential_edge(rid_i, rid_j, ent)

                if rj.postcondition and ri.precondition:
                    ent = self.nli.entailment_score(rj.postcondition, ri.precondition)
                    if ent > theta_seq:
                        updated.add_sequential_edge(rid_j, rid_i, ent)

                # Trigger overlap
                if ri.embedding is not None and rj.embedding is not None:
                    trig_sim = self.sbert.cosine_sim(
                        ri.embedding[:ri.embedding.shape[0] // 5] if ri.embedding is not None else 0,
                        rj.embedding[:rj.embedding.shape[0] // 5] if rj.embedding is not None else 0,
                    )
                    if trig_sim > theta_trig:
                        updated.add_trigger_overlap_edge(rid_i, rid_j, trig_sim)

                # Tool conflict
                if ri.tools & rj.tools:
                    updated.add_tool_conflict_edge(rid_i, rid_j)

        return updated

    def recompute_conflicts(self, conflicts: List[Conflict],
                             affected: Set[str],
                             resolved_ids: Set[str]) -> List[Conflict]:
        """Remove resolved conflicts, keep unaffected ones."""
        remaining = []
        for c in conflicts:
            is_resolved = False
            for rid in c.involved_rules:
                if rid in resolved_ids:
                    is_resolved = True
                    break
            if not is_resolved:
                remaining.append(c)
        return remaining

    def recompute_influence(self, influence_sets: Dict[str, Set[str]],
                             graph: SemanticDepGraph,
                             affected: Set[str]) -> Dict[str, Set[str]]:
        """Recompute influence sets for affected nodes."""
        updated = dict(influence_sets)
        for node in affected:
            updated[node] = graph.reachable_from(node)
        return updated
