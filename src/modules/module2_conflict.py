"""Module 2: Efficient Conflict Detection (Types I–IV)."""

import math
import numpy as np
from typing import List, Dict, Set, Tuple
from ..core.types import (
    Conflict, AtomicEdit, EditSet, Field, Module1Output, Module2Output
)
from ..utils.config import Config
from ..utils.math_utils import normalize_array


class ConflictDetector:
    """Detects Type I–IV conflicts across multi-trajectory edit sets."""

    def __init__(self, sbert_encoder, nli_classifier, config: Config):
        self.sbert = sbert_encoder
        self.nli = nli_classifier
        self.config = config
        self._conflict_counter = 0

    def run(self, m1_output: Module1Output,
            edit_sets: Dict[int, EditSet]) -> Module2Output:
        """Detect all conflicts and compute priorities."""
        conflicts: List[Conflict] = []

        # Group edits by rule_id for cross-trajectory comparison
        edits_by_rule: Dict[str, Dict[int, List[AtomicEdit]]] = {}
        for k, es in edit_sets.items():
            for edit in es.edits:
                if edit.rule_id not in edits_by_rule:
                    edits_by_rule[edit.rule_id] = {}
                if k not in edits_by_rule[edit.rule_id]:
                    edits_by_rule[edit.rule_id][k] = []
                edits_by_rule[edit.rule_id][k].append(edit)

        base_rules = {r.rule_id: r for r in m1_output.base_rules}
        graph = m1_output.graph

        # Type I & III: Same-rule edits from different trajectories
        for rule_id, traj_edits in edits_by_rule.items():
            traj_keys = list(traj_edits.keys())
            for i in range(len(traj_keys)):
                for j in range(i + 1, len(traj_keys)):
                    ka, kb = traj_keys[i], traj_keys[j]
                    conflicts.extend(self.detect_type_I(rule_id, traj_edits[ka], ka, traj_edits[kb], kb, base_rules))
                    conflicts.extend(self.detect_type_III(rule_id, traj_edits[ka], ka, traj_edits[kb], kb, base_rules))

        # Type II: Dependency chain breakage
        for k, es in edit_sets.items():
            for edit in es.edits:
                if edit.field == Field.POSTCONDITION:
                    conflicts.extend(
                        self.detect_type_II(edit, graph, base_rules, m1_output.rules_by_traj)
                    )

        # Type IV: Transitive causal conflicts (cross-rule)
        conflicts.extend(self.detect_type_IV(edits_by_rule, graph, m1_output))

        # Compute influence sets
        influence_sets: Dict[str, Set[str]] = {}
        for node in graph.nodes:
            influence_sets[node] = graph.reachable_from(node)

        # Compute priorities
        self._compute_priorities(conflicts, influence_sets)

        # Sort by priority (descending)
        conflicts.sort(key=lambda c: c.priority, reverse=True)

        # Build conflict matrix
        n_rules = len(m1_output.base_rules)
        conflict_matrix = self._build_conflict_matrix(conflicts, n_rules)

        return Module2Output(
            conflicts=conflicts,
            influence_sets=influence_sets,
            conflict_matrix=conflict_matrix,
        )

    def detect_type_I(self, rule_id: str,
                      edits_a: List[AtomicEdit], traj_a: int,
                      edits_b: List[AtomicEdit], traj_b: int,
                      base_rules: Dict[str, any]) -> List[Conflict]:
        """Type I: Direct Semantic Contradiction.

        C_I = cos(τ_i^a, τ_i^b) * max(0, -cos(α_i^a, α_i^b))

        When cosine-based opposition fails (common with generic BERT embeddings),
        NLI contradiction detection is used as fallback per the paper's validation
        approach (89% agreement between cosine and NLI on MergeBench).
        """
        conflicts = []
        edits_a_action = [e for e in edits_a if e.field == Field.ACTION]
        edits_b_action = [e for e in edits_b if e.field == Field.ACTION]

        if not edits_a_action or not edits_b_action:
            return conflicts

        theta_opp = self.config.get("thresholds.theta_opp", 0.3)
        theta_conflict = self.config.get("thresholds.theta_conflict", 0.7)

        base_rule = base_rules.get(rule_id)
        if base_rule is None:
            return conflicts

        trig_emb_a = self.sbert.encode_field(base_rule.trigger)
        trig_emb_b = self.sbert.encode_field(base_rule.trigger)
        trigger_cos = self.sbert.cosine_sim(trig_emb_a, trig_emb_b)

        for ea in edits_a_action:
            for eb in edits_b_action:
                if ea.v_new is None or eb.v_new is None:
                    continue
                action_cos = self.sbert.cosine_sim(
                    self.sbert.encode_field(ea.v_new),
                    self.sbert.encode_field(eb.v_new),
                )
                opposition = max(0.0, -action_cos)

                # Primary: cosine-based opposition
                if opposition >= theta_opp:
                    score = trigger_cos * opposition
                    if score > theta_conflict:
                        cid = self._next_id()
                        conflicts.append(Conflict(
                            conflict_id=cid, type="I",
                            involved_rules=[rule_id],
                            involved_fields=[Field.ACTION],
                            score=score,
                            details={
                                "trigger_cosine": trigger_cos,
                                "action_opposition": opposition,
                                "traj_a": traj_a, "traj_b": traj_b,
                                "action_a": ea.v_new, "action_b": eb.v_new,
                                "method": "cosine",
                            },
                        ))
                        continue

                # Fallback: NLI-based three-way classification
                # If either direction is classified as CONTRADICTION, flag as Type I
                class_ab = self.nli.three_way_classify(ea.v_new, eb.v_new)
                class_ba = self.nli.three_way_classify(eb.v_new, ea.v_new)
                is_contradiction = (class_ab == "CONTRADICTION" or class_ba == "CONTRADICTION")
                if is_contradiction:
                    nli_contra = max(
                        self.nli.contradiction_score(ea.v_new, eb.v_new),
                        self.nli.contradiction_score(eb.v_new, ea.v_new),
                    )
                    score = trigger_cos * nli_contra
                    if score > 0.3:
                        cid = self._next_id()
                        conflicts.append(Conflict(
                            conflict_id=cid, type="I",
                            involved_rules=[rule_id],
                            involved_fields=[Field.ACTION],
                            score=score,
                            details={
                                "trigger_cosine": trigger_cos,
                                "nli_class_ab": class_ab,
                                "nli_class_ba": class_ba,
                                "nli_contradiction": nli_contra,
                            },
                        ))
        return conflicts

    def detect_type_II(self, edit: AtomicEdit, graph, base_rules,
                       rules_by_traj: Dict[int, List]) -> List[Conflict]:
        """Type II: Dependency Chain Breakage.

        C_II = max(0, entail(ω_i^old, π_j) - entail(ω_i^new, π_j),
                      entail(ω_i^new, π_j) - entail(ω_i^old, π_j) * 𝟙[entail(ω_i^new, π_j) < θ_seq])
        """
        conflicts = []
        theta_seq = self.config.get("thresholds.theta_seq", 0.7)

        if edit.v_old is None or edit.v_new is None:
            return conflicts

        # Find downstream rules with sequential dependency
        downstream = graph.get_sequential_edges_from(edit.rule_id)
        base_rule = base_rules.get(edit.rule_id)
        if base_rule is None:
            return conflicts

        for dep_rule_id, _ in downstream:
            dep_rule = base_rules.get(dep_rule_id)
            if dep_rule is None:
                continue

            ent_old = self.nli.entailment_score(edit.v_old, dep_rule.precondition)
            ent_new = self.nli.entailment_score(edit.v_new, dep_rule.precondition)

            term1 = max(0.0, ent_old - ent_new)
            indicator = 1.0 if ent_new < theta_seq else 0.0
            term2 = max(0.0, ent_new - ent_old) * indicator
            score = max(term1, term2)

            if score > 0.1:  # Non-trivial breakage
                cid = self._next_id()
                conflicts.append(Conflict(
                    conflict_id=cid,
                    type="II",
                    involved_rules=[edit.rule_id, dep_rule_id],
                    involved_fields=[Field.POSTCONDITION, Field.PRECONDITION],
                    score=score,
                    details={
                        "upstream_rule": edit.rule_id,
                        "downstream_rule": dep_rule_id,
                        "entailment_old": ent_old,
                        "entailment_new": ent_new,
                        "trajectory_k": edit.trajectory_k,
                        "old_postcondition": edit.v_old,
                        "new_postcondition": edit.v_new,
                    },
                ))
        return conflicts

    def detect_type_III(self, rule_id: str,
                        edits_a: List[AtomicEdit], traj_a: int,
                        edits_b: List[AtomicEdit], traj_b: int,
                        base_rules: Dict[str, any]) -> List[Conflict]:
        """Type III: Scope Mismatch.

        C_III = |entail(α_i^a, α_i^b) - entail(α_i^b, α_i^a)|
        """
        conflicts = []
        edits_a_action = [e for e in edits_a if e.field == Field.ACTION]
        edits_b_action = [e for e in edits_b if e.field == Field.ACTION]

        if not edits_a_action or not edits_b_action:
            return conflicts

        for ea in edits_a_action:
            for eb in edits_b_action:
                if ea.v_new is None or eb.v_new is None:
                    continue
                ent_ab = self.nli.entailment_score(ea.v_new, eb.v_new)
                ent_ba = self.nli.entailment_score(eb.v_new, ea.v_new)
                score = abs(ent_ab - ent_ba)

                if score > 0.3:  # Significant scope mismatch
                    cid = self._next_id()
                    conflicts.append(Conflict(
                        conflict_id=cid,
                        type="III",
                        involved_rules=[rule_id],
                        involved_fields=[Field.ACTION],
                        score=score,
                        details={
                            "traj_a": traj_a, "traj_b": traj_b,
                            "action_a": ea.v_new, "action_b": eb.v_new,
                            "entail_a_to_b": ent_ab,
                            "entail_b_to_a": ent_ba,
                        },
                    ))
        return conflicts

    def detect_type_IV(self, edits_by_rule: Dict[str, Dict[int, List[AtomicEdit]]],
                       graph, m1_output: Module1Output) -> List[Conflict]:
        """Type IV: Transitive Causal Conflict.

        C_IV = |Infl(δ_a) ∩ Infl(δ_b)| / |Infl(δ_a) ∪ Infl(δ_b)|
               * max_{r_k ∈ Infl(δ_a) ∩ Infl(δ_b)} |ATE(δ_a, r_k) - ATE(δ_b, r_k)|
        """
        conflicts = []
        epsilon_ate = self.config.get("thresholds.epsilon_ate", 0.05)

        # Compute influence sets for each edit
        influence: Dict[Tuple[str, int], Set[str]] = {}
        for rule_id, traj_edits in edits_by_rule.items():
            for k in traj_edits:
                infl = graph.reachable_from(rule_id) | graph.neighbors_within_distance(rule_id, 1)
                influence[(rule_id, k)] = infl

        # Compare edits from different trajectories on different rules
        rule_ids = sorted(edits_by_rule.keys())
        for i in range(len(rule_ids)):
            for j in range(i + 1, len(rule_ids)):
                ra, rb = rule_ids[i], rule_ids[j]
                traj_keys_a = list(edits_by_rule[ra].keys())
                traj_keys_b = list(edits_by_rule[rb].keys())

                for ka in traj_keys_a:
                    for kb in traj_keys_b:
                        if ka == kb:
                            continue
                        infl_a = influence.get((ra, ka), set())
                        infl_b = influence.get((rb, kb), set())

                        intersection = infl_a & infl_b
                        if not intersection:
                            continue

                        union = infl_a | infl_b
                        overlap_ratio = len(intersection) / len(union) if union else 0.0

                        # Estimate ATE divergence (simplified: use conflict scores as proxy)
                        max_ate_div = 0.0
                        for rk in intersection:
                            ate_div = self._estimate_ate_divergence(ra, rb, rk, ka, kb, edits_by_rule)
                            max_ate_div = max(max_ate_div, ate_div)

                        score = overlap_ratio * max_ate_div
                        if score > epsilon_ate:
                            cid = self._next_id()
                            conflicts.append(Conflict(
                                conflict_id=cid,
                                type="IV",
                                involved_rules=[ra, rb],
                                involved_fields=[],
                                score=score,
                                details={
                                    "traj_a": ka, "traj_b": kb,
                                    "overlap_ratio": overlap_ratio,
                                    "max_ate_divergence": max_ate_div,
                                    "intersection_size": len(intersection),
                                },
                            ))
        return conflicts

    def _estimate_ate_divergence(self, rule_a: str, rule_b: str, target: str,
                                  traj_a: int, traj_b: int,
                                  edits_by_rule: Dict) -> float:
        """Simplified ATE divergence estimate using embedding distances."""
        # In absence of actual agent execution, use embedding-based proxy
        return 0.1  # Placeholder; real implementation requires agent execution

    def _compute_priorities(self, conflicts: List[Conflict],
                            influence_sets: Dict[str, Set[str]]) -> None:
        """Compute normalized priority for each conflict."""
        if not conflicts:
            return

        w1 = self.config.get("priority.w1", 0.6)
        w2 = self.config.get("priority.w2", 0.4)

        scores = np.array([c.score for c in conflicts])
        if len(scores) > 1:
            scores_norm = normalize_array(scores)
        else:
            scores_norm = np.array([0.0])

        max_influence = max(
            (len(influence_sets.get(rid, set())) for c in conflicts for rid in c.involved_rules),
            default=1,
        )

        for i, c in enumerate(conflicts):
            total_influence = sum(
                len(influence_sets.get(rid, set())) for rid in c.involved_rules
            )
            infl_norm = total_influence / max(max_influence, 1)
            c.priority = w1 * float(scores_norm[i]) + w2 * infl_norm

    def _build_conflict_matrix(self, conflicts: List[Conflict], n_rules: int):
        """Build rule-level conflict adjacency matrix."""
        matrix = np.zeros((n_rules, n_rules))
        rule_ids = {}
        # Map rule_ids to indices
        idx = 0
        for c in conflicts:
            for rid in c.involved_rules:
                if rid not in rule_ids:
                    rule_ids[rid] = idx
                    idx += 1
        for c in conflicts:
            for i, rid1 in enumerate(c.involved_rules):
                for rid2 in c.involved_rules[i + 1:]:
                    if rid1 in rule_ids and rid2 in rule_ids:
                        ii, jj = rule_ids[rid1], rule_ids[rid2]
                        matrix[ii, jj] = max(matrix[ii, jj], c.score)
                        matrix[jj, ii] = matrix[ii, jj]
        return matrix if len(rule_ids) == n_rules else matrix

    def _next_id(self) -> str:
        self._conflict_counter += 1
        return f"C{self._conflict_counter:04d}"
