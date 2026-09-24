"""ModelLoader: unified local/Hub discovery with case-insensitive path matching."""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Any


class ModelLoader:
    """Discovers and loads local models with case-insensitive matching.
    Falls back to HuggingFace Hub if no local model found."""

    MODELS_BASE = Path("E:/资料备份/论文写作/第3篇/我的方案/代码/SkillSCR/models")

    @classmethod
    def _normalize(cls, name: str) -> str:
        """Normalize name for matching: lowercase, remove hyphens and underscores."""
        return name.lower().replace("-", "").replace("_", "")

    @classmethod
    def resolve_path(cls, model_name: str) -> Optional[Path]:
        """Case-insensitive directory lookup under MODELS_BASE."""
        if not cls.MODELS_BASE.exists():
            return None
        target = cls._normalize(model_name)
        for entry in cls.MODELS_BASE.iterdir():
            if entry.is_dir():
                if cls._normalize(entry.name) == target:
                    return entry
        return None

    @classmethod
    def _inject_sentence_transformers(cls) -> None:
        """Inject local sentence-transformers source into sys.path."""
        st_src = cls.MODELS_BASE / "sentence-transformers"
        if st_src.exists() and str(st_src.parent) not in sys.path:
            sys.path.insert(0, str(st_src.parent))
            # Also try as direct package
            if str(st_src) not in sys.path:
                sys.path.insert(0, str(st_src))

    @classmethod
    def _inject_gpytorch(cls) -> None:
        """Inject local GPyTorch source into sys.path."""
        gp_src = cls.MODELS_BASE / "GPyTorch"
        if gp_src.exists() and str(gp_src.parent) not in sys.path:
            sys.path.insert(0, str(gp_src.parent))

    @classmethod
    def load_t5(cls, path: Optional[Path] = None, device: str = "cpu") -> Tuple[Any, Any]:
        """Load T5ForConditionalGeneration + T5Tokenizer from local or Hub."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        if path is None:
            path = cls.resolve_path("t5-small")

        if path is not None and path.exists() and (path / "pytorch_model.bin").exists():
            model = T5ForConditionalGeneration.from_pretrained(str(path))
            tokenizer = T5Tokenizer.from_pretrained(str(path), legacy=False)
            model = model.to(device)
            return model, tokenizer

        # Fallback to HuggingFace Hub
        model_name = "t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = model.to(device)
        return model, tokenizer

    @classmethod
    def load_sentence_bert(cls, path: Optional[Path] = None, device: str = "cpu") -> Any:
        """Load SentenceTransformer from local model dir."""
        cls._inject_sentence_transformers()
        from sentence_transformers import SentenceTransformer

        if path is None:
            path = cls.resolve_path("sentence_bert")

        if path is not None and path.exists():
            model = SentenceTransformer(str(path), device=device)
            return model

        # Fallback to HuggingFace Hub
        return SentenceTransformer("all-MiniLM-L6-v2", device=device)

    @classmethod
    def load_nli(cls, path: Optional[Path] = None, device: str = "cpu") -> Any:
        """Load RoBERTa-large-mnli as zero-shot-classification pipeline."""
        from transformers import (
            RobertaForSequenceClassification,
            RobertaTokenizerFast,
            pipeline,
        )

        if path is None:
            path = cls.resolve_path("roberta-large-mnli")

        if path is not None and path.exists() and (path / "pytorch_model.bin").exists():
            model = RobertaForSequenceClassification.from_pretrained(str(path))
            tokenizer = RobertaTokenizerFast.from_pretrained(str(path))
            model = model.to(device)
        else:
            model_name = "roberta-large-mnli"
            model = RobertaForSequenceClassification.from_pretrained(model_name)
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            model = model.to(device)

        nli_pipeline = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1 if device == "cpu" else 0,
        )
        return nli_pipeline

    @classmethod
    def load_gpytorch(cls) -> Any:
        """Load GPyTorch from local source or pip install."""
        cls._inject_gpytorch()
        try:
            import gpytorch
            return gpytorch
        except ImportError:
            raise ImportError(
                "GPyTorch not found. Install with: pip install gpytorch"
            )

    @classmethod
    def verify_model(cls, model_name: str) -> bool:
        """Check if a model exists locally."""
        path = cls.resolve_path(model_name)
        if path is None:
            return False
        return path.exists() and (path / "pytorch_model.bin").exists()
