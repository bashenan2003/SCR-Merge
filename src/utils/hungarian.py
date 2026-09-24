"""Hungarian algorithm for cross-version rule alignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from ..core.types import Rule, AlignmentResult
from .math_utils import cosine_similarity


def align_rules_hungarian(
    rules_a: List[Rule],
    rules_b: List[Rule],
    traj_id_a: int,
    traj_id_b: int,
    threshold: float = 0.5,
) -> List[AlignmentResult]:
    """Align rules between two trajectory versions using Hungarian matching.

    Cost matrix: C[i,j] = 1 - cos(emb_i, emb_j)
    Only pairs with cost below threshold are considered matches.
    """
    n_a, n_b = len(rules_a), len(rules_b)
    if n_a == 0 or n_b == 0:
        return []

    cost = np.ones((max(n_a, n_b), max(n_a, n_b))) * 1000.0
    for i, ra in enumerate(rules_a):
        emb_a = ra.embedding
        if emb_a is None:
            continue
        for j, rb in enumerate(rules_b):
            emb_b = rb.embedding
            if emb_b is None:
                continue
            cost[i, j] = 1.0 - cosine_similarity(emb_a, emb_b)

    row_ind, col_ind = linear_sum_assignment(cost)

    results = []
    for i, j in zip(row_ind, col_ind):
        if i < n_a and j < n_b and cost[i, j] < (1.0 - threshold):
            results.append(AlignmentResult(
                base_rule_id=rules_a[i].rule_id,
                aligned_ids={traj_id_a: rules_a[i].rule_id, traj_id_b: rules_b[j].rule_id},
                alignment_cost=float(cost[i, j]),
            ))

    return results


def build_alignment_map(
    rules_by_traj: Dict[int, List[Rule]],
    base_traj: int = 0,
    threshold: float = 0.5,
) -> Dict[str, AlignmentResult]:
    """Build alignment map across all trajectories relative to base.

    Returns dict: base_rule_id -> AlignmentResult mapping to all other trajectories.
    """
    if base_traj not in rules_by_traj:
        return {}

    base_rules = rules_by_traj[base_traj]
    alignment_map: Dict[str, AlignmentResult] = {
        r.rule_id: AlignmentResult(base_rule_id=r.rule_id, aligned_ids={base_traj: r.rule_id})
        for r in base_rules
    }

    for traj_k, traj_rules in rules_by_traj.items():
        if traj_k == base_traj:
            continue
        pairwise = align_rules_hungarian(base_rules, traj_rules, base_traj, traj_k, threshold)
        for al in pairwise:
            if al.base_rule_id in alignment_map:
                alignment_map[al.base_rule_id].aligned_ids.update(al.aligned_ids)
            else:
                alignment_map[al.base_rule_id] = al

    return alignment_map
