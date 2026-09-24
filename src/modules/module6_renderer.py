"""Module 6: Natural Language Renderer for merged skill documents.

Outputs Agent Skills Spec standard format:
  - YAML frontmatter (name, description, license, metadata)
  - When to Apply section
  - Multi-line WHEN/IF/THEN/RESULTING IN/TOOLS rules
"""

import datetime
from typing import List, Optional, Dict
from ..core.types import Rule, Module6Output
from ..utils.config import Config


class NLRenderer:
    """Renders merged edit vector into standard Agent Skills Spec .md document.

    Three modes:
    - agent_spec: Full YAML frontmatter + When to Apply + multi-line rules (default).
    - compact:    Compact single-line rules (legacy).
    - json:       Structured JSON for machine consumption.
    """

    MODE_AGENT_SPEC = "agent_spec"
    MODE_COMPACT = "compact"
    MODE_JSON = "json"

    def __init__(self, mode: str = "agent_spec", config: Config = None,
                 merge_meta: Optional[dict] = None):
        self.mode = mode
        self.config = config
        self.merge_meta = merge_meta or {}

    def run(self, merged_rules: List[Rule], doc_id: str = "S_merged") -> Module6Output:
        """Render the merged skill document."""
        if self.mode == self.MODE_AGENT_SPEC:
            final_text = self._render_agent_spec(merged_rules, doc_id)
        elif self.mode == self.MODE_COMPACT:
            final_text = self._render_compact(merged_rules, doc_id)
        elif self.mode == self.MODE_JSON:
            final_text = self._render_json(merged_rules, doc_id)
        else:
            final_text = self._render_agent_spec(merged_rules, doc_id)

        return Module6Output(
            final_text=final_text,
            template_used=self.mode,
        )

    # ── Agent Skills Spec format (default) ──────────────────────────────

    def _render_agent_spec(self, rules: List[Rule], doc_id: str = "S_merged") -> str:
        """Render in standard Agent Skills Spec format.

        Produces:
          ---
          name: ...
          description: >
            ...
          license: MIT
          metadata:
            version: "1.0.0"
            type: skill
            merged_trajectories: N
            generated_at: <iso timestamp>
          ---
          # Skill Document: S_merged

          ## When to Apply
          - <trigger 1>
          - <trigger 2>

          ## Rule r1
          WHEN ...
          IF ...
          THEN ...
          RESULTING IN ...
          TOOLS: ...
        """
        lines: List[str] = []

        # ── YAML frontmatter ──
        name = self._derive_name(rules)
        description = self._derive_description(rules)
        n_rules = len(rules)
        n_trajs = self.merge_meta.get("n_trajectories", "K")
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        lines.append("---")
        lines.append(f"name: {name}")
        if len(description) > 80:
            lines.append("description: >")
            remaining = description.strip()
            while len(remaining) > 80:
                cut = remaining.rfind(' ', 0, 80)
                if cut < 20:
                    cut = 80
                lines.append(f"  {remaining[:cut]}")
                remaining = remaining[cut:].strip()
            if remaining:
                lines.append(f"  {remaining}")
        else:
            lines.append(f"description: {description}")
        lines.append("license: MIT")
        lines.append("metadata:")
        lines.append(f'  version: "1.0.0"')
        lines.append(f'  type: skill')
        lines.append(f'  merged_trajectories: {n_trajs}')
        lines.append(f'  total_rules: {n_rules}')
        lines.append(f'  generated_at: {ts}')
        lines.append("---")
        lines.append("")

        # ── Document header ──
        lines.append(f"# Skill Document: {doc_id}")
        lines.append("")

        # ── When to Apply ──
        triggers = self._derive_triggers(rules)
        if triggers:
            lines.append("## When to Apply")
            for t in triggers:
                lines.append(f"- {t}")
            lines.append("")

        # ── Rules ──
        for rule in rules:
            lines.append(f"## Rule {rule.rule_id}")
            # Multi-line format for readability
            if rule.trigger:
                lines.append(f"WHEN {rule.trigger.strip()},")
            if rule.precondition:
                lines.append(f"IF {rule.precondition.strip()},")
            if rule.action:
                lines.append(f"THEN {rule.action.strip()},")
            if rule.postcondition:
                lines.append(f"RESULTING IN {rule.postcondition.strip()}.")
            if rule.tools:
                tools_str = ", ".join(sorted(rule.tools))
                lines.append(f"TOOLS: {tools_str}")
            lines.append("")

        return "\n".join(lines)

    # ── Legacy compact format ──────────────────────────────────────────

    def _render_compact(self, rules: List[Rule], doc_id: str = "S_merged") -> str:
        """Compact single-line rule format (legacy)."""
        lines = [f"# Skill Document: {doc_id}", ""]
        for rule in rules:
            tools_str = ", ".join(sorted(rule.tools)) if rule.tools else "none"
            rendered = (
                f"WHEN {rule.trigger}, "
                f"IF {rule.precondition}, "
                f"THEN {rule.action}, "
                f"RESULTING IN {rule.postcondition}. "
                f"TOOLS: {tools_str}"
            )
            lines.append(f"## Rule {rule.rule_id}")
            lines.append(rendered)
            lines.append("")
        return "\n".join(lines)

    # ── JSON format ────────────────────────────────────────────────────

    def _render_json(self, rules: List[Rule], doc_id: str = "S_merged") -> str:
        """Structured JSON for machine consumption."""
        import json
        doc = {
            "doc_id": doc_id,
            "generated_at": datetime.datetime.now().isoformat(),
            "metadata": {
                "version": "1.0.0",
                "type": "skill",
                "merged_trajectories": self.merge_meta.get("n_trajectories", "K"),
                "total_rules": len(rules),
            },
            "rules": [r.to_dict() for r in rules],
        }
        return json.dumps(doc, indent=2, ensure_ascii=False)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _derive_name(rules: List[Rule]) -> str:
        """Derive a skill name from merged rule content."""
        # Use the first action verb + domain hints
        domain_words: set = set()
        for r in rules:
            for word in r.trigger.lower().split():
                if len(word) > 4:
                    domain_words.add(word)
        # Build a slug from key action verbs
        action_verbs = []
        for r in rules:
            words = r.action.lower().split()
            for w in words[:3]:
                if len(w) > 3 and w not in action_verbs:
                    action_verbs.append(w)
        if action_verbs:
            slug = "-".join(action_verbs[:4])
            return f"merged-{slug}"[:64]
        return "merged-agent-skill"

    @staticmethod
    def _derive_description(rules: List[Rule]) -> str:
        """Derive a description from merged rules."""
        domains = set()
        for r in rules:
            for word in r.trigger.lower().split():
                if len(word) >= 5 and word not in {"agent", "needs", "answer", "using", "tasks"}:
                    domains.add(word)
        domain_hint = ", ".join(sorted(domains)[:5]) if domains else "multiple domains"
        return (
            f"Merged skill document combining optimized trajectories across "
            f"{domain_hint}. Use when the agent needs to handle diverse tasks "
            f"including data processing, error recovery, and output generation."
        )

    @staticmethod
    def _derive_triggers(rules: List[Rule]) -> List[str]:
        """Derive When to Apply triggers from rule triggers and actions."""
        triggers = []
        for r in rules[:6]:
            t = r.trigger.strip()
            # Truncate long triggers
            if len(t) > 120:
                t = t[:117] + "..."
            # Capitalize first letter
            if t and t[0].islower():
                t = t[0].upper() + t[1:]
            triggers.append(t)
        return triggers
