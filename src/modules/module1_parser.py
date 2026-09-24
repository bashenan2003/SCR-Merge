"""Module 1: Semantic Parser and Rule Alignment."""

from typing import List, Dict, Set
from ..core.types import (
    Rule, Field, AlignmentResult, Module1Output, EdgeType
)
from ..core.graph import SemanticDepGraph
from ..utils.hungarian import build_alignment_map
from ..utils.config import Config


class SemanticParserAndAlignment:
    """Extracts rules, encodes them, aligns across trajectories, builds dependency graph."""

    def __init__(self, t5_parser, sbert_encoder, nli_classifier, config: Config):
        self.t5 = t5_parser
        self.sbert = sbert_encoder
        self.nli = nli_classifier
        self.config = config

    def run(self, S0_raw: str, S_k_raw: Dict[int, str],
            edit_sets: Dict[int, "EditSet"] = None) -> Module1Output:
        """Full Module 1 pipeline.

        Args:
            S0_raw: Base document text.
            S_k_raw: {traj_k: document_text} for each trajectory.
            edit_sets: Optional pre-computed edit sets.

        Returns:
            Module1Output with rules, alignment, graph, embeddings.
        """
        # Step 1: Parse all documents
        rules_by_traj = {0: self.t5.parse_document(S0_raw, trajectory_k=0, doc_id="S0")}
        for k, raw_text in S_k_raw.items():
            rules_by_traj[k] = self.t5.parse_document(raw_text, trajectory_k=k, doc_id=f"S{k}")

        # Step 2: Encode all rules
        for traj_rules in rules_by_traj.values():
            self.sbert.encode_rules(traj_rules)

        # Step 3: Align rules across trajectories
        alignment = build_alignment_map(
            rules_by_traj,
            base_traj=0,
            threshold=self.config.get("thresholds.alignment_threshold", 0.5),
        )

        # Step 4: Build semantic dependency graph
        base_rules = rules_by_traj[0]
        graph = self.build_graph(base_rules)

        # Step 5: Collect embeddings
        embeddings = {r.rule_id: r.embedding for r in base_rules if r.embedding is not None}

        return Module1Output(
            rules_by_traj=rules_by_traj,
            alignment=alignment,
            graph=graph,
            embeddings=embeddings,
            base_rules=base_rules,
        )

    def build_graph(self, rules: List[Rule]) -> SemanticDepGraph:
        """Construct G_S with three edge types."""
        graph = SemanticDepGraph(rules)
        theta_seq = self.config.get("thresholds.theta_seq", 0.7)
        theta_trig = self.config.get("thresholds.theta_trig", 0.6)

        for i, ri in enumerate(rules):
            for j, rj in enumerate(rules):
                if i == j:
                    continue

                # Sequential dependency: entail(ω_i, π_j) > θ_seq
                if ri.postcondition and rj.precondition:
                    ent_score = self.nli.entailment_score(ri.postcondition, rj.precondition)
                    if ent_score > theta_seq:
                        graph.add_sequential_edge(ri.rule_id, rj.rule_id, ent_score)

                # Trigger overlap: cos(τ_i, τ_j) > θ_trig
                if ri.embedding is not None and rj.embedding is not None:
                    trig_sim = self.sbert.cosine_sim(
                        self.sbert.encode_field(ri.trigger),
                        self.sbert.encode_field(rj.trigger),
                    )
                    if trig_sim > theta_trig:
                        graph.add_trigger_overlap_edge(ri.rule_id, rj.rule_id, trig_sim)

                # Tool conflict: T_i ∩ T_j ≠ ∅
                if ri.tools & rj.tools:
                    graph.add_tool_conflict_edge(ri.rule_id, rj.rule_id)

        return graph

    def compute_influence_sets(self, graph: SemanticDepGraph) -> Dict[str, Set[str]]:
        """Compute influence set for each rule: all rules reachable from it."""
        influence = {}
        for node in graph.nodes:
            influence[node] = graph.reachable_from(node)
        return influence
