"""Sentence-BERT wrapper for rule field embeddings."""

import numpy as np
from typing import List
from ..core.types import Rule, Field
from ..utils.math_utils import cosine_similarity as cos_sim


class SBertEncoder:
    """Encodes rule fields into dense embeddings via Sentence-BERT."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def encode_field(self, text: str) -> np.ndarray:
        """Encode a single text field to embedding vector."""
        if not text or not text.strip():
            dim = self._get_dim()
            return np.zeros(dim, dtype=np.float32)
        emb = self.model.encode([text], show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float64)

    def encode_rule(self, rule: Rule) -> np.ndarray:
        """Encode all 5 fields of a rule into a concatenated embedding."""
        embeddings = []
        for f in [Field.TRIGGER, Field.PRECONDITION, Field.ACTION, Field.POSTCONDITION]:
            embeddings.append(self.encode_field(getattr(rule, f.value)))
        # Tools: mean embedding of tool names, or zero vector
        if rule.tools:
            tool_embs = [self.encode_field(t) for t in rule.tools]
            tool_mean = np.mean(tool_embs, axis=0)
        else:
            tool_mean = np.zeros(self._get_dim(), dtype=np.float64)
        embeddings.append(tool_mean)
        return np.concatenate(embeddings)

    def encode_rules(self, rules: List[Rule]) -> None:
        """Encode all rules in-place, populating rule.embedding."""
        for rule in rules:
            rule.embedding = self.encode_rule(rule)

    def cosine_sim(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return cos_sim(emb_a, emb_b)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch-encode a list of texts."""
        if not texts:
            return np.array([])
        embs = self.model.encode(texts, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float64)

    def _get_dim(self) -> int:
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()
