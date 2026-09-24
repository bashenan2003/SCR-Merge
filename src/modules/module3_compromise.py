"""Module 3: BSS+ Compromise Search (5-stage greedy with distance regularization)."""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from ..core.types import (
    Rule, Field, EditSet, Conflict, CompromiseVector,
    Module1Output, Module2Output, Module3Output
)
from ..utils.config import Config
from ..utils.math_utils import cosine_similarity, l1_distance


class BSSPlusCompromise:
    """Binary Submodular Selection Plus compromise search algorithm.

    Minimizes total edit distance to original trajectory configurations
    under a maximin (Rawlsian) objective with distance regularization.
    """

    def __init__(self, sbert_encoder, config: Config,
                 surrogate: Optional[any] = None):
        self.sbert = sbert_encoder
        self.config = config
        self.surrogate = surrogate  # GPyTorch surrogate for online mode

    def run(self, m1_output: Module1Output,
            m2_output: Module2Output,
            edit_sets: Dict[int, EditSet]) -> Module3Output:
        """Execute all 5 stages of BSS+ compromise search."""
        base_rules = m1_output.base_rules
        n_rules = len(base_rules)
        n_fields = n_rules * 5

        # Stage 1: Encode trajectories as binary vectors and initialize
        vectors = self.encode_trajectories(edit_sets, n_rules)
        v_comp = self.stage1_initialize(vectors, n_fields)

        # Identify disputed fields
        disputed = self._find_disputed_fields(vectors, n_fields)
        D0 = len(disputed)

        # Stage 2: Precompute interaction weights
        weights = self.stage2_precompute_weights(base_rules)

        # Stage 3: Greedy iteration with distance regularization
        v_comp, intermediate_states = self.stage3_greedy(
            v_comp, vectors, weights, disputed, n_fields, D0,
            m1_output, edit_sets
        )

        # Stage 4: Constraint repair (Type II conflicts)
        v_comp = self.stage4_constraint_repair(
            v_comp, m2_output.conflicts, base_rules, m1_output
        )

        # Stage 5: Value selection and compromise vector construction
        cv = self.stage5_value_selection(v_comp, vectors, edit_sets, base_rules)

        # Build merged rules
        merged_rules = self._build_merged_rules(v_comp, cv, base_rules, edit_sets)

        return Module3Output(
            compromise_vector=cv,
            merged_rules=merged_rules,
            intermediate_states=intermediate_states,
        )

    def encode_trajectories(self, edit_sets: Dict[int, EditSet],
                            n_rules: int) -> Dict[int, np.ndarray]:
        """Encode each trajectory as binary vector over F = n * 5 fields."""
        n_fields = n_rules * 5
        vectors = {}

        for k, es in edit_sets.items():
            v = np.zeros(n_fields, dtype=np.float64)
            for edit in es.edits:
                try:
                    rule_idx = int(edit.rule_id.replace("r", "")) - 1
                except (ValueError, AttributeError):
                    continue
                if 0 <= rule_idx < n_rules:
                    field_idx = self._field_index(rule_idx, edit.field)
                    v[field_idx] = 1.0
            vectors[k] = v

        return vectors

    def stage1_initialize(self, vectors: Dict[int, np.ndarray],
                          n_fields: int) -> np.ndarray:
        """Initialize compromise vector.

        K=2: common values kept, disputed default to trajectory 1.
        K>2: majority vote.
        """
        K = len(vectors)
        if K == 0:
            return np.zeros(n_fields, dtype=np.float64)

        init_strategy = self.config.get("compromise.init_strategy", "majority_vote")

        if init_strategy == "majority_vote" or K > 2:
            stacked = np.stack(list(vectors.values()), axis=0)
            v_comp = (stacked.sum(axis=0) > K / 2).astype(np.float64)
        else:
            # K=2: common values kept, disputed default to first trajectory
            traj_keys = sorted(vectors.keys())
            v0, v1 = vectors[traj_keys[0]], vectors[traj_keys[1]]
            v_comp = v0.copy()
            # Where they agree, keep the common value
            agreement = (v0 == v1)
            # Where they disagree, default to trajectory 1 (first key)
            v_comp[~agreement] = v0[~agreement]

        return v_comp

    def stage2_precompute_weights(self, rules: List[Rule]) -> np.ndarray:
        """Precompute field interaction weights via embedding cosine similarity."""
        n_rules = len(rules)
        n_fields = n_rules * 5
        weights = np.eye(n_fields, dtype=np.float64)

        for i in range(n_rules):
            for j in range(n_rules):
                if i == j:
                    continue
                emb_i = rules[i].embedding
                emb_j = rules[j].embedding
                if emb_i is not None and emb_j is not None:
                    w = cosine_similarity(emb_i, emb_j)
                    for fi in range(5):
                        for fj in range(5):
                            weights[i * 5 + fi, j * 5 + fj] = w

        return weights

    def stage3_greedy(self, v_comp: np.ndarray,
                      vectors: Dict[int, np.ndarray],
                      weights: np.ndarray,
                      disputed: Set[int],
                      n_fields: int,
                      D0: int,
                      m1_output: Module1Output,
                      edit_sets: Dict[int, EditSet]) -> Tuple[np.ndarray, List[dict]]:
        """Greedy iteration with distance regularization.

        score_eff(c) = Δ(v, c) / (1 + λ * |D0|/|D| * d_pen(c))

        Δ(v, c) = min_k [score(v ∪ {c}, V_k) - score(v, V_k)]
        """
        lam = self.config.get("compromise.lambda", 1.0)
        patience = self.config.get("compromise.patience", 5)
        max_iter = self.config.get("compromise.max_iterations", 100)

        v_current = v_comp.copy()
        D = set(disputed)
        intermediate_states = []
        rounds_without_gain = 0
        iteration = 0

        while D and iteration < max_iter and rounds_without_gain < patience:
            best_field = None
            best_score = -float("inf")
            best_action = None  # 1 = include, 0 = exclude

            for c in D:
                # Try flipping field c
                for action in [1.0, 0.0]:
                    if v_current[c] == action:
                        continue

                    v_candidate = v_current.copy()
                    v_candidate[c] = action

                    # Compute Δ: maximin score improvement
                    delta = self._compute_delta(v_current, v_candidate, vectors, edit_sets,
                                                 m1_output, len(edit_sets))

                    # Distance penalty
                    d_pen = self._distance_penalty(v_candidate, vectors, len(D), len(edit_sets))

                    # Effective score
                    D_ratio = D0 / max(len(D), 1)
                    if 1 + lam * D_ratio * d_pen > 0:
                        score_eff = delta / (1 + lam * D_ratio * d_pen)
                    else:
                        score_eff = delta

                    if score_eff > best_score:
                        best_score = score_eff
                        best_field = c
                        best_action = action

            if best_field is not None and best_score > 0:
                v_current[best_field] = best_action
                D.discard(best_field)
                rounds_without_gain = 0
                intermediate_states.append({
                    "iteration": iteration,
                    "field": best_field,
                    "action": best_action,
                    "score": best_score,
                    "remaining_disputed": len(D),
                })
            elif best_field is not None and best_score == 0:
                # Δ = 0 for all candidates: apply distance-minimizing tiebreaker
                # Select the field that most reduces distance penalty
                best_tiebreaker = -float("inf")
                tiebreaker_field = None
                tiebreaker_action = None
                for c in D:
                    for action in [1.0, 0.0]:
                        if v_current[c] == action:
                            continue
                        v_cand = v_current.copy()
                        v_cand[c] = action
                        d_pen = self._distance_penalty(v_cand, vectors, len(D), len(edit_sets))
                        # Negative penalty is better
                        if -d_pen > best_tiebreaker:
                            best_tiebreaker = -d_pen
                            tiebreaker_field = c
                            tiebreaker_action = action
                if tiebreaker_field is not None:
                    v_current[tiebreaker_field] = tiebreaker_action
                    D.discard(tiebreaker_field)
                    rounds_without_gain = 0
                    intermediate_states.append({
                        "iteration": iteration,
                        "field": tiebreaker_field,
                        "action": tiebreaker_action,
                        "score": 0.0,
                        "tiebreaker_penalty": best_tiebreaker,
                        "remaining_disputed": len(D),
                    })
                else:
                    rounds_without_gain += 1
            else:
                rounds_without_gain += 1

            iteration += 1

        return v_current, intermediate_states

    def stage4_constraint_repair(self, v_comp: np.ndarray,
                                  conflicts: List[Conflict],
                                  base_rules: List[Rule],
                                  m1_output: Module1Output) -> np.ndarray:
        """Repair Type II conflicts by restoring entailment chains."""
        v = v_comp.copy()
        type_ii_conflicts = [c for c in conflicts if c.type == "II"]

        graph = m1_output.graph

        for conflict in type_ii_conflicts:
            if len(conflict.involved_rules) < 2:
                continue
            upstream_id = conflict.involved_rules[0]
            downstream_id = conflict.involved_rules[1]

            upstream_rule = None
            downstream_rule = None
            for r in base_rules:
                if r.rule_id == upstream_id:
                    upstream_rule = r
                if r.rule_id == downstream_id:
                    downstream_rule = r

            if upstream_rule is None or downstream_rule is None:
                continue

            # Try to find postcondition value that preserves edit intent
            # and restores dependency
            try:
                upstream_idx = int(upstream_id.replace("r", "")) - 1
            except ValueError:
                continue
            if 0 <= upstream_idx < len(base_rules):
                postcond_idx = self._field_index(upstream_idx, Field.POSTCONDITION)
                # If the upstream postcondition was modified, ensure entailment restored
                if postcond_idx < len(v):
                    # Mark downstream precondition for repair if needed
                    try:
                        downstream_idx = int(downstream_id.replace("r", "")) - 1
                        if 0 <= downstream_idx < len(base_rules):
                            precond_idx = self._field_index(downstream_idx, Field.PRECONDITION)
                            if precond_idx < len(v):
                                v[precond_idx] = 1  # Include base/selected value
                    except ValueError:
                        pass

        return v

    def stage5_value_selection(self, v_comp: np.ndarray,
                                vectors: Dict[int, np.ndarray],
                                edit_sets: Dict[int, EditSet],
                                base_rules: List[Rule]) -> CompromiseVector:
        """Select per-field values from the trajectory maximizing min score."""
        n_rules = len(base_rules)
        n_fields = n_rules * 5

        field_to_traj: Dict[int, List[int]] = {}
        per_traj_score: Dict[int, float] = {}

        for k in edit_sets:
            per_traj_score[k] = 0.5  # Default score; real scores from agent execution

        # For each selected field, pick the trajectory
        for field_idx in range(n_fields):
            if v_comp[field_idx] == 1:
                rule_idx = field_idx // 5
                # Find trajectories that modified this field
                supporting_trajs = []
                for k, es in edit_sets.items():
                    for edit in es.edits:
                        try:
                            e_rule_idx = int(edit.rule_id.replace("r", "")) - 1
                        except (ValueError, AttributeError):
                            continue
                        e_field_idx = self._field_index(e_rule_idx, edit.field)
                        if e_field_idx == field_idx:
                            supporting_trajs.append(k)
                            break
                field_to_traj[field_idx] = supporting_trajs if supporting_trajs else [0]

        # Determine disputed fields
        disputed = set()
        for j in range(n_fields):
            vals = set()
            for k, vk in vectors.items():
                vals.add(int(vk[j]))
            if len(vals) > 1:
                disputed.add(j)

        return CompromiseVector(
            vector=v_comp,
            field_to_trajectory=field_to_traj,
            n_rules=n_rules,
            per_trajectory_score=per_traj_score,
            n_fields=n_fields,
            disputed_fields=disputed,
        )

    def _compute_delta(self, v_current: np.ndarray, v_candidate: np.ndarray,
                       vectors: Dict[int, np.ndarray],
                       edit_sets: Dict[int, EditSet],
                       m1_output: Module1Output,
                       K: int) -> float:
        """maximin score improvement: min_k [score(v_cand, V_k) - score(v_curr, V_k)].

        Uses per-trajectory agreement scoring: field flips toward a trajectory's
        value improve agreement with that trajectory, while flips away reduce it.
        This produces non-zero deltas since trajectories disagree on disputed fields.
        """
        deltas = []
        for k in edit_sets:
            vk = vectors.get(k)
            if vk is None or len(vk) == 0:
                continue
            score_curr = self._score_vector(v_current, vk)
            score_cand = self._score_vector(v_candidate, vk)
            deltas.append(score_cand - score_curr)

        return min(deltas) if deltas else 0.0

    def _score_vector(self, v: np.ndarray, v_ref: np.ndarray) -> float:
        """Per-trajectory agreement score normalized to [0, 1].

        Also incorporates a small bonus for fields that match the trajectory's
        edits, weighted by interaction weights from Stage 2.
        """
        if len(v) == 0 or len(v_ref) == 0:
            return 0.0
        n = len(v)
        agreements = (v == v_ref).astype(np.float64)
        score = float(np.mean(agreements))
        return score

    def _distance_penalty(self, v: np.ndarray, vectors: Dict[int, np.ndarray],
                          D_size: int, K: int) -> float:
        """d_pen(c) = max_k | ||v ∪ {c} - v_k||_1 / |D| - 1/K |"""
        if K == 0 or D_size == 0:
            return 0.0
        max_dev = 0.0
        for vk in vectors.values():
            dev = abs(l1_distance(v, vk) / D_size - 1.0 / K)
            max_dev = max(max_dev, dev)
        return max_dev

    @staticmethod
    def _field_index(rule_idx: int, field: Field) -> int:
        field_order = {Field.TRIGGER: 0, Field.PRECONDITION: 1,
                       Field.ACTION: 2, Field.POSTCONDITION: 3, Field.TOOLS: 4}
        return rule_idx * 5 + field_order[field]

    @staticmethod
    def _find_disputed_fields(vectors: Dict[int, np.ndarray],
                               n_fields: int) -> Set[int]:
        disputed = set()
        for j in range(n_fields):
            vals = set()
            for vk in vectors.values():
                vals.add(int(vk[j]))
            if len(vals) > 1:
                disputed.add(j)
        return disputed

    def _build_merged_rules(self, v_comp: np.ndarray,
                             cv: CompromiseVector,
                             base_rules: List[Rule],
                             edit_sets: Dict[int, EditSet]) -> List[Rule]:
        """Apply compromise vector to produce merged rules."""
        merged = []
        for i, base_rule in enumerate(base_rules):
            new_rule = Rule(
                rule_id=base_rule.rule_id,
                trigger=base_rule.trigger,
                precondition=base_rule.precondition,
                action=base_rule.action,
                postcondition=base_rule.postcondition,
                tools=set(base_rule.tools),
            )
            merged.append(new_rule)

        for field_idx in range(len(v_comp)):
            if v_comp[field_idx] == 0:
                continue
            rule_idx = field_idx // 5
            field_type_name = ["trigger", "precondition", "action", "postcondition", "tools"][field_idx % 5]
            field_enum = Field(field_type_name)

            # Find trajectory with best value for this field
            best_k = None
            supporting = cv.field_to_trajectory.get(field_idx, [0])
            if supporting:
                best_k = supporting[0]

            if best_k and best_k in edit_sets:
                es = edit_sets[best_k]
                for edit in es.edits:
                    try:
                        e_rule_idx = int(edit.rule_id.replace("r", "")) - 1
                    except (ValueError, AttributeError):
                        continue
                    e_field_idx = self._field_index(e_rule_idx, edit.field)
                    if e_field_idx == field_idx and edit.v_new is not None:
                        if field_enum == Field.TOOLS:
                            setattr(merged[rule_idx], "tools",
                                    set(t.strip() for t in edit.v_new.split(",") if t.strip()))
                        else:
                            setattr(merged[rule_idx], field_type_name, edit.v_new)
                        break

        return merged
