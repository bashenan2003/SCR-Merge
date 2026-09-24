"""Skill document representation and serialization."""

import json
from pathlib import Path
from typing import List, Dict, Set
from .types import Rule, Field, EditSet, AtomicEdit, EditCategory


class SkillDocument:
    """A skill document S comprising n procedural rules, each a 5-tuple."""

    def __init__(self, doc_id: str, rules: List[Rule] = None):
        self.doc_id = doc_id
        self.rules: List[Rule] = rules or []

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __getitem__(self, idx) -> Rule:
        return self.rules[idx]

    def get_rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(f"Rule {rule_id} not found")

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_fields(self) -> int:
        return self.n_rules * 5

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: dict, trajectory_k: int = 0) -> "SkillDocument":
        return cls(
            doc_id=d.get("doc_id", f"doc_{trajectory_k}"),
            rules=[Rule.from_dict(r, trajectory_k=trajectory_k) for r in d.get("rules", [])],
        )

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path, trajectory_k: int = 0) -> "SkillDocument":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), trajectory_k=trajectory_k)

    def compute_edit_set(self, base: "SkillDocument", trajectory_k: int) -> EditSet:
        """Compute Δ = S_k ⊖ S_base as field-level symmetric difference."""
        edits = []
        base_rules = {r.rule_id: r for r in base.rules}
        cur_rules = {r.rule_id: r for r in self.rules}

        for rid in set(base_rules.keys()) | set(cur_rules.keys()):
            br = base_rules.get(rid)
            cr = cur_rules.get(rid)
            if br is None:
                for f in Field:
                    val = getattr(cr, f.value)
                    edits.append(AtomicEdit(
                        rule_id=rid, field=f, v_old=None,
                        v_new=str(val) if f != Field.TOOLS else ",".join(sorted(val)),
                        trajectory_k=trajectory_k,
                        category=EditCategory.ACTION_ADDITION))
            elif cr is None:
                for f in Field:
                    val = getattr(br, f.value)
                    edits.append(AtomicEdit(
                        rule_id=rid, field=f,
                        v_old=str(val) if f != Field.TOOLS else ",".join(sorted(val)),
                        v_new=None, trajectory_k=trajectory_k,
                        category=EditCategory.ACTION_REMOVAL))
            else:
                for f in Field:
                    ov = getattr(br, f.value)
                    nv = getattr(cr, f.value)
                    if f == Field.TOOLS:
                        ov_s = set(ov) if isinstance(ov, (set, list)) else set()
                        nv_s = set(nv) if isinstance(nv, (set, list)) else set()
                    else:
                        ov_s = str(ov).strip()
                        nv_s = str(nv).strip()
                    if ov_s != nv_s:
                        cat = EditCategory.VALUE_REPLACEMENT
                        if f == Field.TRIGGER:
                            if ov_s and nv_s and ov_s in nv_s:
                                cat = EditCategory.TRIGGER_EXPANSION
                            elif ov_s and nv_s and nv_s in ov_s:
                                cat = EditCategory.TRIGGER_NARROWING
                        elif f == Field.ACTION:
                            if not ov_s and nv_s:
                                cat = EditCategory.ACTION_ADDITION
                            elif ov_s and not nv_s:
                                cat = EditCategory.ACTION_REMOVAL
                        elif f == Field.TOOLS:
                            cat = EditCategory.TOOL_SET_MODIFICATION
                        edits.append(AtomicEdit(
                            rule_id=rid, field=f,
                            v_old=str(ov_s) if f != Field.TOOLS else ",".join(sorted(ov_s)),
                            v_new=str(nv_s) if f != Field.TOOLS else ",".join(sorted(nv_s)),
                            trajectory_k=trajectory_k, category=cat))
        return EditSet(trajectory_k=trajectory_k, edits=edits)

    def apply_compromise_vector(self, cv: "CompromiseVector",
                                 trajectories: Dict[int, "SkillDocument"]) -> "SkillDocument":
        """Apply a compromise vector to produce merged rules."""
        from .types import CompromiseVector
        merged_rules = []
        for i, rule in enumerate(self.rules):
            new_rule = Rule(
                rule_id=rule.rule_id,
                trigger=rule.trigger,
                precondition=rule.precondition,
                action=rule.action,
                postcondition=rule.postcondition,
                tools=set(rule.tools),
                trajectory_k=rule.trajectory_k,
            )
            merged_rules.append(new_rule)

        for field_idx in range(cv.n_fields):
            if cv.vector[field_idx] == 0:
                continue
            rule_idx = field_idx // 5
            field_type = list(Field)[field_idx % 5]
            rule = merged_rules[rule_idx]

            # Find the trajectory that contributes this field's value
            best_traj = None
            best_score = -float("inf")
            for traj_k in trajectories:
                if field_idx in cv.field_to_trajectory and traj_k in cv.field_to_trajectory.get(field_idx, []):
                    best_traj = traj_k
                    break
            if best_traj is None:
                best_traj = min(cv.per_trajectory_score, key=lambda k: cv.per_trajectory_score[k], default=0) if cv.per_trajectory_score else 0
                if best_traj not in trajectories:
                    best_traj = list(trajectories.keys())[0] if trajectories else 0

            traj_doc = trajectories.get(best_traj)
            if traj_doc:
                try:
                    traj_rule = traj_doc.get_rule(rule.rule_id)
                    val = getattr(traj_rule, field_type.value)
                    setattr(merged_rules[rule_idx], field_type.value, val)
                except (KeyError, AttributeError):
                    pass

        return SkillDocument(doc_id=f"{self.doc_id}_merged", rules=merged_rules)
