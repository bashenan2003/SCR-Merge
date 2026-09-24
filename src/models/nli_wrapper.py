"""RoBERTa-large-mnli wrapper for entailment/contradiction scoring."""

from typing import List, Tuple


class NLIClassifier:
    """NLI classifier using RoBERTa-large-mnli for entailment and contradiction."""

    LABELS = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
    CONTRADICTION_IDX = 0
    NEUTRAL_IDX = 1
    ENTAILMENT_IDX = 2

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _predict(self, premise: str, hypothesis: str) -> dict:
        """Get probability scores for premise-hypothesis pair."""
        if not premise.strip() or not hypothesis.strip():
            return {"CONTRADICTION": 0.0, "NEUTRAL": 0.0, "ENTAILMENT": 0.0}
        result = self.pipeline(
            hypothesis,
            candidate_labels=self.LABELS,
            hypothesis_template="{}",
        )
        scores = {label: score for label, score in zip(result["labels"], result["scores"])}
        return scores

    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[dict]:
        """Batch predict for multiple premise-hypothesis pairs."""
        results = []
        for premise, hypothesis in pairs:
            results.append(self._predict(premise, hypothesis))
        return results

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """P(ENTAILMENT) in [0, 1]."""
        scores = self._predict(premise, hypothesis)
        return scores.get("ENTAILMENT", 0.0)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        """P(CONTRADICTION) in [0, 1]."""
        scores = self._predict(premise, hypothesis)
        return scores.get("CONTRADICTION", 0.0)

    def entail(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        """Binary entailment check."""
        return self.entailment_score(premise, hypothesis) > threshold

    def three_way_classify(self, premise: str, hypothesis: str) -> str:
        """Return the predicted NLI class label."""
        scores = self._predict(premise, hypothesis)
        return max(scores, key=scores.get)

    def entail_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Batch entailment scores."""
        results = self._predict_batch(pairs)
        return [r.get("ENTAILMENT", 0.0) for r in results]
