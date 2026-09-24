"""Score computer for per-trajectory validation scoring."""

import numpy as np
from typing import List, Dict
from ..core.types import Rule, Field


class ScoreComputer:
    """Computes validation scores for merged documents against trajectory validation sets."""

    def __init__(self, validation_sets: Dict[int, List[dict]] = None):
        self.validation_sets = validation_sets or {}

    def add_validation_set(self, trajectory_k: int, cases: List[dict]) -> None:
        self.validation_sets[trajectory_k] = cases

    def compute_score(self, rules: List[Rule], trajectory_k: int) -> float:
        """Compute quality score for rules on a trajectory's validation set.

        Score in [0, 1]: measures rule completeness and specificity.
        """
        if trajectory_k not in self.validation_sets:
            return self._structural_score(rules)

        val_cases = self.validation_sets[trajectory_k]
        if not val_cases:
            return self._structural_score(rules)

        # For each validation case, check trigger match and evaluate action quality
        scores = []
        for case in val_cases:
            case_score = self._evaluate_case(rules, case)
            scores.append(case_score)

        return float(np.mean(scores)) if scores else 0.0

    def compute_all_scores(self, rules: List[Rule]) -> Dict[int, float]:
        """Compute scores for all trajectory validation sets."""
        return {
            k: self.compute_score(rules, k)
            for k in self.validation_sets
        }

    def _evaluate_case(self, rules: List[Rule], case: dict) -> float:
        """Evaluate a single validation case against rules."""
        trigger_text = case.get("trigger", case.get("input", ""))
        if not trigger_text:
            return 0.5

        # Find matching rule by trigger keyword overlap
        best_match = None
        best_overlap = 0
        trigger_words = set(trigger_text.lower().split())

        for rule in rules:
            rule_words = set(rule.trigger.lower().split())
            if not rule_words:
                continue
            overlap = len(trigger_words & rule_words) / len(rule_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = rule

        if best_match is None or best_overlap < 0.2:
            return 0.3  # No matching rule

        # Score based on rule completeness
        score = 0.0
        if best_match.trigger:
            score += 0.2
        if best_match.precondition:
            score += 0.2
        if best_match.action:
            score += 0.3
        if best_match.postcondition:
            score += 0.2
        if best_match.tools:
            score += 0.1

        # Bonus for trigger match quality
        score *= (0.5 + 0.5 * best_overlap)

        return min(score, 1.0)

    def _structural_score(self, rules: List[Rule]) -> float:
        """Structural quality score: completeness and specificity of rules."""
        if not rules:
            return 0.0
        scores = []
        for r in rules:
            filled = sum(
                1 for f in [Field.TRIGGER, Field.PRECONDITION, Field.ACTION, Field.POSTCONDITION]
                if getattr(r, f.value, "").strip()
            )
            scores.append(filled / 4.0)
        return float(np.mean(scores))

    def compute_rawlsian_score(self, rules: List[Rule]) -> float:
        """maximin (Rawlsian) score: min across trajectories."""
        scores = self.compute_all_scores(rules)
        return min(scores.values()) if scores else 0.0
