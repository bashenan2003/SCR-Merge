"""MainPipeline: sequential orchestrator for all 6 modules with checkpoint-based rollback."""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure project root is on path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.types import (
    EditSet, PipelineResult, Rule,
    Module1Output, Module2Output, Module3Output,
    Module4Output, Module5Output, Module6Output
)
from src.core.skill_document import SkillDocument
from src.models.loader import ModelLoader
from src.models.t5_wrapper import T5RuleParser
from src.models.sbert_wrapper import SBertEncoder
from src.models.nli_wrapper import NLIClassifier
from src.models.gpytorch_wrapper import GPyTorchSurrogate
from src.modules.module1_parser import SemanticParserAndAlignment
from src.modules.module2_conflict import ConflictDetector
from src.modules.module3_compromise import BSSPlusCompromise
from src.modules.module4_verification import VerificationGating
from src.modules.module5_incremental import IncrementalUpdateEngine
from src.modules.module6_renderer import NLRenderer
from src.pipeline.checkpointer import CheckpointManager
from src.pipeline.evaluator import ScoreComputer
from src.utils.config import Config
from src.utils.logging import PipelineLogger


class MainPipeline:
    """SCR-Merge Ultima: full 6-module pipeline with checkpoint-based rollback."""

    def __init__(self, config: Config = None, mode: str = "offline"):
        if config is None:
            config = Config()
        self.config = config
        self.mode = mode
        self.logger = PipelineLogger()
        self.checkpointer = CheckpointManager()

        # Models (lazy-loaded)
        self._t5_model = None
        self._t5_tokenizer = None
        self._sbert_model = None
        self._nli_pipeline = None
        self._surrogate = None

        # Wrappers
        self._t5_parser = None
        self._sbert_encoder = None
        self._nli_classifier = None

        # Modules
        self._m1 = None
        self._m2 = None
        self._m3 = None
        self._m4 = None
        self._m5 = None
        self._m6 = None
        self._evaluator = ScoreComputer()

    def _load_models(self) -> None:
        """Lazy-load all models."""
        device = self.config.device
        self.logger.info(f"Loading models on {device}...")

        # T5-small
        self._t5_model, self._t5_tokenizer = ModelLoader.load_t5(device=device)
        self._t5_parser = T5RuleParser(self._t5_model, self._t5_tokenizer, device=device)
        self.logger.info("  T5-small loaded")

        # Sentence-BERT
        self._sbert_model = ModelLoader.load_sentence_bert(device=device)
        self._sbert_encoder = SBertEncoder(self._sbert_model, device=device)
        self.logger.info("  Sentence-BERT loaded")

        # RoBERTa-large-mnli
        self._nli_pipeline = ModelLoader.load_nli(device=device)
        self._nli_classifier = NLIClassifier(self._nli_pipeline)
        self.logger.info("  RoBERTa-large-mnli loaded")

        self.logger.info("All models loaded successfully")

    def _init_modules(self) -> None:
        """Initialize all 6 modules."""
        self._m1 = SemanticParserAndAlignment(
            self._t5_parser, self._sbert_encoder, self._nli_classifier, self.config
        )
        self._m2 = ConflictDetector(
            self._sbert_encoder, self._nli_classifier, self.config
        )

        if self.mode == "online":
            self._surrogate = GPyTorchSurrogate(input_dim=1, num_tasks=1)

        self._m3 = BSSPlusCompromise(
            self._sbert_encoder, self.config, surrogate=self._surrogate
        )
        self._m4 = VerificationGating(
            self._nli_classifier, self.config
        )
        self._m5 = IncrementalUpdateEngine(
            self._sbert_encoder, self._nli_classifier, self.config
        )

        renderer_mode = self.config.get("renderer.mode", "agent_spec")
        self._m6 = NLRenderer(mode=renderer_mode, config=self.config)

    def run(self,
            S0_dict: dict,
            S_k_dicts: Dict[int, dict],
            edit_sets: Dict[int, EditSet] = None,
            validation_set: List[dict] = None) -> PipelineResult:
        """Execute the full 6-module pipeline.

        Args:
            S0_dict: Base skill document as dict.
            S_k_dicts: Trajectory-optimized documents as {traj_k: doc_dict}.
            edit_sets: Optional pre-computed edit sets.
            validation_set: Optional validation cases.

        Returns:
            PipelineResult with final document and all module outputs.
        """
        start_time = time.time()
        metrics = {}

        # Load models and init modules
        self._load_models()
        self._init_modules()

        # Parse input documents
        S0_doc = SkillDocument.from_dict(S0_dict, trajectory_k=0)
        S_k_docs = {k: SkillDocument.from_dict(d, trajectory_k=k)
                     for k, d in S_k_dicts.items()}

        # Build edit sets if not provided
        if edit_sets is None:
            edit_sets = {}
            for k, doc in S_k_docs.items():
                edit_sets[k] = doc.compute_edit_set(S0_doc, k)

        # Build raw text representations
        S0_raw = self._doc_to_text(S0_doc)
        S_k_raw = {k: self._doc_to_text(doc) for k, doc in S_k_docs.items()}

        # Checkpoint: START
        state = {
            "S0_dict": S0_dict,
            "S_k_dicts": S_k_dicts,
            "mode": self.mode,
        }
        self.checkpointer.save("START", state)
        self.logger.checkpoint("START")

        # === Module 1: Parse + Align ===
        self.logger.module_start("Module 1: Semantic Parser & Rule Alignment")
        t1 = time.time()
        m1_output = self._m1.run(S0_raw, S_k_raw, edit_sets)
        m1_elapsed = time.time() - t1
        self.logger.module_end("Module 1", m1_elapsed)
        metrics["m1_time"] = m1_elapsed
        metrics["n_rules"] = len(m1_output.base_rules)
        metrics["n_aligned"] = len(m1_output.alignment)

        self.checkpointer.save("M1_DONE", {"m1_output": m1_output, "edit_sets": edit_sets})
        self.logger.checkpoint("M1_DONE")

        # === Module 2: Conflict Detection ===
        self.logger.module_start("Module 2: Conflict Detection")
        t2 = time.time()
        m2_output = self._m2.run(m1_output, edit_sets)
        m2_elapsed = time.time() - t2
        self.logger.module_end("Module 2", m2_elapsed)
        metrics["m2_time"] = m2_elapsed
        metrics["n_conflicts"] = len(m2_output.conflicts)
        for c in m2_output.conflicts:
            self.logger.info(f"  {c}")

        self.checkpointer.save("M2_DONE", {
            "m1_output": m1_output,
            "m2_output": m2_output,
            "edit_sets": edit_sets,
        })
        self.logger.checkpoint("M2_DONE")

        # === Module 3: BSS+ Compromise Search ===
        self.logger.module_start("Module 3: BSS+ Compromise Search")
        t3 = time.time()
        max_rollback = self.config.get("verification.max_rollback_attempts", 3)
        rollback_attempt = 0
        m4_output = None

        while rollback_attempt < max_rollback:
            m3_output = self._m3.run(m1_output, m2_output, edit_sets)
            m3_elapsed = time.time() - t3
            self.logger.module_end("Module 3", m3_elapsed)
            metrics["m3_time"] = m3_elapsed
            metrics["n_disputed"] = len(m3_output.compromise_vector.disputed_fields)
            metrics["n_intermediate"] = len(m3_output.intermediate_states)
            self.logger.info(f"  Disputed fields: {metrics['n_disputed']}")
            self.logger.info(f"  Greedy iterations: {metrics['n_intermediate']}")

            self.checkpointer.save("M3_DONE", {
                "m1_output": m1_output,
                "m2_output": m2_output,
                "m3_output": m3_output,
                "edit_sets": edit_sets,
            })
            self.logger.checkpoint("M3_DONE")

            # === Module 4: Verification Gating ===
            self.logger.module_start("Module 4: Verification Gating")
            t4 = time.time()

            base_doc_score = self._evaluator.compute_all_scores(m1_output.base_rules)

            m4_output = self._m4.run(
                m3_output, m2_output,
                m1_output.base_rules,
                base_doc_score=base_doc_score,
            )
            m4_elapsed = time.time() - t4
            self.logger.module_end("Module 4", m4_elapsed)
            metrics["m4_time"] = m4_elapsed
            self.logger.info(f"  Gate results: {m4_output.gate_results}")
            self.logger.info(f"  Passed: {m4_output.passed}")

            if m4_output.passed:
                self.logger.info("  All verification gates passed")
                break
            else:
                rollback_attempt += 1
                if not m4_output.gate_results.get("gate1", True):
                    self.logger.warning(f"  Gate 1 failed, rollback attempt {rollback_attempt}")
                    # Restore M3 state and retry with different initialization
                    restored = self.checkpointer.rollback_to("M3_DONE")
                    m1_output = restored.get("m1_output", m1_output)
                    m2_output = restored.get("m2_output", m2_output)
                    edit_sets = restored.get("edit_sets", edit_sets)
                elif not m4_output.gate_results.get("gate2", True):
                    self.logger.warning(f"  Gate 2 failed, rollback attempt {rollback_attempt}")
                    restored = self.checkpointer.rollback_to("M2_DONE")
                    m1_output = restored.get("m1_output", m1_output)
                    m2_output = restored.get("m2_output", m2_output)
                    edit_sets = restored.get("edit_sets", edit_sets)
                elif not m4_output.gate_results.get("gate3", True):
                    self.logger.warning("  Gate 3 (Pareto) check failed, continuing with best-effort")
                    break

        if rollback_attempt >= max_rollback and not m4_output.passed:
            self.logger.warning(
                f"  Max rollback attempts ({max_rollback}) reached, returning best-effort merge"
            )

        # === Module 5: Incremental Update ===
        self.logger.module_start("Module 5: Incremental Update")
        t5 = time.time()

        resolved_conflicts = m2_output.conflicts[:len(m3_output.intermediate_states)] if m3_output.intermediate_states else []

        m5_output = self._m5.run(
            m3_output.merged_rules,
            m1_output.graph,
            m2_output.conflicts,
            m2_output.influence_sets,
            resolved_conflicts,
        )
        m5_elapsed = time.time() - t5
        self.logger.module_end("Module 5", m5_elapsed)
        metrics["m5_time"] = m5_elapsed

        self.checkpointer.save("M5_DONE", {
            "m1_output": m1_output,
            "m2_output": m2_output,
            "m3_output": m3_output,
            "m4_output": m4_output,
            "m5_output": m5_output,
        })
        self.logger.checkpoint("M5_DONE")

        # === Module 6: NL Renderer ===
        self.logger.module_start("Module 6: NL Renderer")
        t6 = time.time()
        self._m6.merge_meta = {
            "n_trajectories": len(edit_sets),
            "mode": self.mode,
        }
        m6_output = self._m6.run(m3_output.merged_rules, doc_id="S_merged")
        m6_elapsed = time.time() - t6
        self.logger.module_end("Module 6", m6_elapsed)
        metrics["m6_time"] = m6_elapsed

        total_elapsed = time.time() - start_time
        metrics["total_time"] = total_elapsed
        self.logger.info(f"Pipeline completed in {total_elapsed:.2f}s")

        self.checkpointer.save("COMPLETE", {
            "final_document": m6_output.final_text,
            "metrics": metrics,
        })
        self.logger.checkpoint("COMPLETE")

        # Compute final scores
        final_scores = self._evaluator.compute_all_scores(m3_output.merged_rules)
        metrics["final_scores"] = final_scores

        return PipelineResult(
            final_document=m6_output.final_text,
            module_outputs={
                "M1": m1_output,
                "M2": m2_output,
                "M3": m3_output,
                "M4": m4_output,
                "M5": m5_output,
                "M6": m6_output,
            },
            metrics=metrics,
            checkpoints=self.checkpointer.list_checkpoints(),
        )

    # ── File-based entry (auto-detect JSON / .md) ──────────────────────────

    def run_from_paths(
        self,
        s0_path: Union[str, Path],
        sk_paths: Dict[int, Union[str, Path]],
        output_path: Union[str, Path] = None,
        validation_set: List[dict] = None,
    ) -> PipelineResult:
        """Auto-detect input format (JSON or .md) and run the pipeline.

        Args:
            s0_path:     Path to S0 file (.json or .md)
            sk_paths:    {traj_k: path} for each trajectory file
            output_path: If given, write merged .md to this file
            validation_set: Optional validation cases

        Returns:
            PipelineResult with final_document and metrics.

        Supported formats:
          - .json:  Structured rules dict (same as demo/sample_docs/)
          - .md:    Natural-language skill doc (WHEN/IF/THEN/RESULTING IN/TOOLS)

        Mixed formats are OK - S0 could be .md while trajectories are .json.
        """
        # Load S0
        S0_dict = self._load_document(s0_path, doc_id="S0", trajectory_k=0)

        # Load trajectories
        S_k_dicts: Dict[int, dict] = {}
        edit_sets: Dict[int, EditSet] = {}
        for k, path in sk_paths.items():
            doc = self._load_document(path, doc_id=f"S{k}", trajectory_k=k)
            S_k_dicts[k] = doc

            # Extract edit_set if present (JSON only)
            if "edit_set" in doc:
                edit_sets[k] = EditSet.from_dict(doc["edit_set"])

        # Run existing pipeline
        result = self.run(
            S0_dict=S0_dict,
            S_k_dicts=S_k_dicts,
            edit_sets=edit_sets if edit_sets else None,
            validation_set=validation_set,
        )

        # Write output if path given
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.final_document, encoding="utf-8")
            self.logger.info(f"Merged document saved to: {out}")

        return result

    def _load_document(
        self, path: Union[str, Path], doc_id: str = "", trajectory_k: int = 0
    ) -> dict:
        """Load a document from file, auto-detecting JSON vs .md format.

        - .json: parse as dict, pass through unchanged.
        - .md:   parse with T5RuleParser into structured rules, build dict.
        - Other: raise ValueError.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))

        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
            rules_raw = self._parse_md_structured(text)
            rules = []
            for i, raw in enumerate(rules_raw):
                rid = raw.get("rule_id", f"r{i+1}")
                rules.append(Rule(
                    rule_id=rid,
                    trigger=raw.get("trigger", ""),
                    precondition=raw.get("precondition", ""),
                    action=raw.get("action", ""),
                    postcondition=raw.get("postcondition", ""),
                    tools=set(raw.get("tools", [])),
                ))
            self.logger.info(
                f"  Parsed {len(rules)} rules from {path.name} "
                f"(detected: {rules_raw[0].get('lang','?') if rules_raw else 'none'})"
            )
            return {
                "doc_id": doc_id or path.stem,
                "rules": [r.to_dict() for r in rules],
            }

        raise ValueError(
            f"Unsupported file format: {suffix!r} for {path}. "
            f"Expected .json or .md."
        )

    @staticmethod
    def _parse_md_structured(text: str) -> List[dict]:
        """Parse a bilingual structured .md skill document via regex patterns.

        Supports both English and Chinese rule formats.
        Does NOT use T5 — relies on the structured WHEN/IF/THEN or 当/如果/则 format.

        English pattern:
            ## Rule r1
            WHEN ..., IF ..., THEN ..., RESULTING IN ... TOOLS: a, b, c

        Chinese pattern:
            ## 规则 r1
            当 ..., 如果 ..., 则 ..., 结果 ... 工具: a, b, c
        """
        import re

        rules: List[dict] = []

        # Split into rule blocks by ## Rule / ## 规则 headers
        rule_blocks = re.split(
            r'\n(?=##\s+(?:Rule|规则)\s+)', text, flags=re.IGNORECASE
        )

        # If no split happened, try splitting by the header pattern directly
        if len(rule_blocks) <= 1:
            parts = re.split(
                r'##\s+(?:Rule\s+|规则\s*)(\S+)', text, flags=re.IGNORECASE
            )
            if len(parts) > 2:
                rule_blocks = []
                for j in range(1, len(parts), 2):
                    rule_blocks.append(f"## Rule {parts[j]}\n{parts[j+1] if j+1 < len(parts) else ''}")

        # Determine language from the first non-header content
        all_text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)

        for block in rule_blocks:
            block = block.strip()
            if not block:
                continue

            # Extract rule_id
            rid_match = re.match(
                r'##\s+(?:Rule\s+|规则\s*)(\S+)', block, flags=re.IGNORECASE
            )
            if not rid_match:
                continue
            rule_id = rid_match.group(1).rstrip('：:,.，。.')
            # Skip past rule title if present (e.g. "r1: Code Generation")
            content = block[rid_match.end():].strip()
            # Remove short title text after colon on the same header line only
            # (max 120 chars — long text means it's actual rule content, not a title)
            content = re.sub(r'^[^：:\n]{1,120}[：:][^\n]*\n?', '', content).strip()

            # Detect language for this block
            is_cn = bool(re.search(r'[一-鿿]', content))

            if is_cn:
                rule = MainPipeline._parse_cn_rule(rule_id, content)
            else:
                rule = MainPipeline._parse_en_rule(rule_id, content)

            if rule:
                rule["lang"] = "cn" if is_cn else "en"
                rules.append(rule)

        return rules

    @staticmethod
    def _parse_cn_rule(rule_id: str, content: str) -> dict | None:
        """Parse a single Chinese rule block."""
        import re

        trigger = ""
        precondition = ""
        action = ""
        postcondition = ""
        tools: List[str] = []

        # Match 当 ... 如果 ... 则 ... 结果 ... 工具: ...
        m = re.match(
            r'当\s*(.+?)，?\s*如果\s*(.+?)，?\s*则\s*(.+?)，?\s*结果\s*(.+?)。?\s*工具[：:]\s*(.+)',
            content, re.DOTALL
        )
        if m:
            trigger = m.group(1).strip()
            precondition = m.group(2).strip()
            action = m.group(3).strip()
            postcondition = m.group(4).strip()
            tools_str = m.group(5).strip()
            tools = [t.strip() for t in re.split(r'[,，、]', tools_str) if t.strip()]
            return {
                "rule_id": rule_id,
                "trigger": trigger,
                "precondition": precondition,
                "action": action,
                "postcondition": postcondition,
                "tools": tools,
            }

        # Loose match: find each field individually
        patterns = {
            "trigger": r'当\s*(.+?)(?:，|。|\n|如果|则|结果|工具)',
            "precondition": r'如果\s*(.+?)(?:，|。|\n|则|结果|工具)',
            "action": r'则\s*(.+?)(?:，|。|\n|结果|工具)',
            "postcondition": r'结果\s*(.+?)(?:。|\n|工具)',
            "tools": r'工具[：:]\s*(.+)',
        }
        for key, pat in patterns.items():
            m = re.search(pat, content, re.DOTALL)
            if m:
                val = m.group(1).strip().rstrip('。，')
                if key == "tools":
                    tools = [t.strip() for t in re.split(r'[,，、]', val) if t.strip()]
                elif key == "trigger":
                    trigger = val
                elif key == "precondition":
                    precondition = val
                elif key == "action":
                    action = val
                elif key == "postcondition":
                    postcondition = val

        if trigger or action:
            return {
                "rule_id": rule_id,
                "trigger": trigger,
                "precondition": precondition,
                "action": action,
                "postcondition": postcondition,
                "tools": tools,
            }
        return None

    @staticmethod
    def _parse_en_rule(rule_id: str, content: str) -> dict | None:
        """Parse a single English rule block."""
        import re

        trigger = ""
        precondition = ""
        action = ""
        postcondition = ""
        tools: List[str] = []

        # Match WHEN ... IF ... THEN ... RESULTING IN ... TOOLS: ...
        m = re.match(
            r'WHEN\s+(.+?),\s*IF\s+(.+?),\s*THEN\s+(.+?),\s*RESULTING\s+IN\s+(.+?)\.\s*TOOLS[：:]\s*(.+)',
            content, re.DOTALL | re.IGNORECASE
        )
        if m:
            trigger = m.group(1).strip()
            precondition = m.group(2).strip()
            action = m.group(3).strip()
            postcondition = m.group(4).strip()
            tools_str = m.group(5).strip()
            tools = [t.strip() for t in re.split(r'[,，、]', tools_str) if t.strip()]
            return {
                "rule_id": rule_id,
                "trigger": trigger,
                "precondition": precondition,
                "action": action,
                "postcondition": postcondition,
                "tools": tools,
            }

        # Loose match
        patterns = {
            "trigger": r'WHEN\s+(.+?)(?:,|\.|\n|IF|THEN|RESULTING|TOOLS)',
            "precondition": r'IF\s+(.+?)(?:,|\.|\n|THEN|RESULTING|TOOLS)',
            "action": r'THEN\s+(.+?)(?:,|\.|\n|RESULTING|TOOLS)',
            "postcondition": r'RESULTING\s+IN\s+(.+?)(?:\.|\n|TOOLS)',
            "tools": r'TOOLS[：:]\s*(.+)',
        }
        for key, pat in patterns.items():
            m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
            if m:
                val = m.group(1).strip().rstrip('.,')
                if key == "tools":
                    tools = [t.strip() for t in re.split(r'[,，、]', val) if t.strip()]
                elif key == "trigger":
                    trigger = val
                elif key == "precondition":
                    precondition = val
                elif key == "action":
                    action = val
                elif key == "postcondition":
                    postcondition = val

        if trigger or action:
            return {
                "rule_id": rule_id,
                "trigger": trigger,
                "precondition": precondition,
                "action": action,
                "postcondition": postcondition,
                "tools": tools,
            }
        return None

    @staticmethod
    def _doc_to_text(doc: SkillDocument) -> str:
        """Convert a SkillDocument to natural language text."""
        lines = []
        for rule in doc.rules:
            parts = []
            if rule.trigger:
                parts.append(f"WHEN {rule.trigger}")
            if rule.precondition:
                parts.append(f"IF {rule.precondition}")
            if rule.action:
                parts.append(f"THEN {rule.action}")
            if rule.postcondition:
                parts.append(f"RESULTING IN {rule.postcondition}")
            if rule.tools:
                parts.append(f"TOOLS: {', '.join(sorted(rule.tools))}")
            if parts:
                lines.append(", ".join(parts))
        return "\n\n".join(lines)
