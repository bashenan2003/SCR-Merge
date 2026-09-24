"""T5-small wrapper for structured rule parsing."""

from typing import List, Set
from ..core.types import Rule


class T5RuleParser:
    """Parses skill document text into structured 5-tuple rules using T5-small."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def parse_document(self, raw_text: str, trajectory_k: int = 0,
                       doc_id: str = "S0") -> List[Rule]:
        """Parse an entire skill document into rules using rule-based extraction.

        Since T5-small is used for rule parsing from structured skill documents,
        primary extraction relies on document structure patterns with T5 as fallback.
        """
        rules = []
        # Try to parse as structured text with rule markers
        if "rule_id" in raw_text.lower() or "WHEN" in raw_text:
            rules = self._parse_structured(raw_text, trajectory_k)
        else:
            rules = self._parse_with_t5(raw_text, trajectory_k)
        return rules

    def _parse_structured(self, text: str, trajectory_k: int) -> List[Rule]:
        """Parse from structured format: WHEN ... IF ... THEN ... RESULTING IN ... TOOLS: ..."""
        import re

        rules = []
        # Split by rule boundaries (double newline or explicit markers)
        blocks = re.split(r'\n\s*\n|(?=WHEN\s)', text)
        rule_idx = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            trigger = ""
            precondition = ""
            action = ""
            postcondition = ""
            tools: Set[str] = set()

            # Extract trigger: WHEN <text>
            trigger_m = re.search(r'WHEN\s+(.+?)(?:\s*[,.]?\s*(?:IF|THEN|RESULTING|TOOLS|$))', block, re.IGNORECASE)
            if trigger_m:
                trigger = trigger_m.group(1).strip().rstrip(",")

            # Extract precondition: IF <text>
            precond_m = re.search(r'IF\s+(.+?)(?:\s*[,.]?\s*(?:THEN|RESULTING|TOOLS|WHEN|$))', block, re.IGNORECASE)
            if precond_m:
                precondition = precond_m.group(1).strip().rstrip(",")

            # Extract action: THEN <text>
            action_m = re.search(r'THEN\s+(.+?)(?:\s*[,.]?\s*(?:RESULTING|TOOLS|WHEN|IF|$))', block, re.IGNORECASE)
            if action_m:
                action = action_m.group(1).strip().rstrip(",")

            # Extract postcondition: RESULTING IN <text>
            postcond_m = re.search(r'RESULTING\s+IN\s+(.+?)(?:\s*[,.]?\s*(?:TOOLS|WHEN|IF|THEN|$))', block, re.IGNORECASE)
            if postcond_m:
                postcondition = postcond_m.group(1).strip().rstrip(",")

            # Extract tools: TOOLS: <text>
            tools_m = re.search(r'TOOLS:\s*(.+?)(?:\s*[,.]?\s*(?:WHEN|IF|THEN|RESULTING|$))', block, re.IGNORECASE)
            if tools_m:
                tools_str = tools_m.group(1).strip()
                tools = {t.strip() for t in re.split(r'[,;]', tools_str) if t.strip()}

            if trigger or action:
                rule = Rule(
                    rule_id=f"r{rule_idx + 1}",
                    trigger=trigger,
                    precondition=precondition,
                    action=action,
                    postcondition=postcondition,
                    tools=tools,
                    trajectory_k=trajectory_k,
                )
                rules.append(rule)
                rule_idx += 1

        return rules

    def _parse_with_t5(self, text: str, trajectory_k: int) -> List[Rule]:
        """Use T5 to generate structured rule extraction. Fallback mode."""
        import re
        # Split into sentences/segments and parse
        segments = re.split(r'(?<=[.!?])\s+', text)
        rules = []
        rule_idx = 0

        for seg in segments:
            seg = seg.strip()
            if len(seg) < 10:
                continue

            prompt = (
                f"Extract the 5-tuple skill rule from: {seg}\n"
                f"Output format: trigger: <text>, precondition: <text>, action: <text>, "
                f"postcondition: <text>, tools: <list>"
            )
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.generate(**inputs, max_new_tokens=128)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                parsed = self._parse_t5_output(result)
                if parsed:
                    rule = Rule(
                        rule_id=f"r{rule_idx + 1}",
                        trigger=parsed.get("trigger", ""),
                        precondition=parsed.get("precondition", ""),
                        action=parsed.get("action", ""),
                        postcondition=parsed.get("postcondition", ""),
                        tools=parsed.get("tools", set()),
                        trajectory_k=trajectory_k,
                    )
                    rules.append(rule)
                    rule_idx += 1
            except Exception:
                continue

        return rules

    def _parse_t5_output(self, output: str) -> dict:
        """Parse T5-generated structured output."""
        import re
        result = {}
        for field in ["trigger", "precondition", "action", "postcondition"]:
            m = re.search(rf'{field}:\s*(.+?)(?:,\s*(?:precondition|action|postcondition|tools):|$)', output, re.IGNORECASE)
            if m:
                result[field] = m.group(1).strip()

        tools_m = re.search(r'tools:\s*(.+?)$', output, re.IGNORECASE)
        if tools_m:
            result["tools"] = {t.strip() for t in re.split(r'[,;]', tools_m.group(1)) if t.strip()}
        else:
            result["tools"] = set()

        return result

    def parse_single(self, text: str, rule_id: str = "r1", trajectory_k: int = 0) -> Rule:
        """Parse a single sentence/paragraph into one Rule."""
        rules = self.parse_document(text, trajectory_k=trajectory_k)
        if rules:
            rules[0].rule_id = rule_id
            return rules[0]
        return Rule(rule_id=rule_id, trajectory_k=trajectory_k)
