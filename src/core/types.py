"""Core domain types for SCR-Merge Ultima."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Dict, Optional, Any
import numpy as np


class Field(Enum):
    TRIGGER = "trigger"
    PRECONDITION = "precondition"
    ACTION = "action"
    POSTCONDITION = "postcondition"
    TOOLS = "tools"


class EditCategory(Enum):
    VALUE_REPLACEMENT = "value_replacement"
    TRIGGER_EXPANSION = "trigger_expansion"
    TRIGGER_NARROWING = "trigger_narrowing"
    ACTION_ADDITION = "action_addition"
    ACTION_REMOVAL = "action_removal"
    TOOL_SET_MODIFICATION = "tool_set_modification"


class EdgeType(Enum):
    SEQUENTIAL = 1
    TRIGGER_OVERLAP = 2
    TOOL_CONFLICT = 3


@dataclass
class Rule:
    rule_id: str
    trigger: str = ""
    precondition: str = ""
    action: str = ""
    postcondition: str = ""
    tools: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    trajectory_k: int = 0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "trigger": self.trigger,
            "precondition": self.precondition,
            "action": self.action,
            "postcondition": self.postcondition,
            "tools": sorted(self.tools),
            "trajectory_k": self.trajectory_k,
        }

    @classmethod
    def from_dict(cls, d: dict, trajectory_k: int = 0) -> "Rule":
        return cls(
            rule_id=d["rule_id"],
            trigger=d.get("trigger", ""),
            precondition=d.get("precondition", ""),
            action=d.get("action", ""),
            postcondition=d.get("postcondition", ""),
            tools=set(d.get("tools", [])),
            trajectory_k=trajectory_k,
        )


@dataclass
class AtomicEdit:
    rule_id: str
    field: Field
    v_old: Optional[str] = None
    v_new: Optional[str] = None
    trajectory_k: int = 0
    category: EditCategory = EditCategory.VALUE_REPLACEMENT

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "field": self.field.value,
            "v_old": self.v_old,
            "v_new": self.v_new,
            "trajectory_k": self.trajectory_k,
            "category": self.category.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AtomicEdit":
        return cls(
            rule_id=d["rule_id"],
            field=Field(d["field"]),
            v_old=d.get("v_old"),
            v_new=d.get("v_new"),
            trajectory_k=d.get("trajectory_k", 0),
            category=EditCategory(d.get("category", "value_replacement")),
        )


@dataclass
class EditSet:
    trajectory_k: int
    edits: List[AtomicEdit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_k": self.trajectory_k,
            "edits": [e.to_dict() for e in self.edits],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditSet":
        return cls(
            trajectory_k=d["trajectory_k"],
            edits=[AtomicEdit.from_dict(e) for e in d.get("edits", [])],
        )


@dataclass
class Conflict:
    conflict_id: str
    type: str  # "I", "II", "III", "IV"
    involved_rules: List[str] = field(default_factory=list)
    involved_fields: List[Field] = field(default_factory=list)
    score: float = 0.0
    priority: float = 0.0
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Conflict({self.conflict_id}, type={self.type}, score={self.score:.3f}, rules={self.involved_rules})"


@dataclass
class CompromiseVector:
    vector: np.ndarray  # shape (F,), binary
    field_to_trajectory: Dict[int, List[int]] = field(default_factory=dict)
    n_rules: int = 0
    per_trajectory_score: Dict[int, float] = field(default_factory=dict)
    n_fields: int = 0
    disputed_fields: Set[int] = field(default_factory=set)

    def copy(self) -> "CompromiseVector":
        return CompromiseVector(
            vector=self.vector.copy(),
            field_to_trajectory=dict(self.field_to_trajectory),
            n_rules=self.n_rules,
            per_trajectory_score=dict(self.per_trajectory_score),
            n_fields=self.n_fields,
            disputed_fields=set(self.disputed_fields),
        )


@dataclass
class AlignmentResult:
    """Maps rule_ids across trajectories for the same semantic rule."""
    base_rule_id: str
    aligned_ids: Dict[int, str] = field(default_factory=dict)  # trajectory_k -> rule_id
    alignment_cost: float = 0.0


@dataclass
class Module1Output:
    rules_by_traj: Dict[int, List[Rule]] = field(default_factory=dict)
    alignment: Dict[str, AlignmentResult] = field(default_factory=dict)
    graph: Any = None  # SemanticDepGraph
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    base_rules: List[Rule] = field(default_factory=list)


@dataclass
class Module2Output:
    conflicts: List[Conflict] = field(default_factory=list)
    influence_sets: Dict[str, Set[str]] = field(default_factory=dict)
    conflict_matrix: Optional[np.ndarray] = None


@dataclass
class Module3Output:
    compromise_vector: CompromiseVector = None
    merged_rules: List[Rule] = field(default_factory=list)
    intermediate_states: List[dict] = field(default_factory=list)


@dataclass
class Module4Output:
    passed: bool = False
    gate_results: dict = field(default_factory=dict)
    degradation: Dict[int, float] = field(default_factory=dict)
    diagnostics: str = ""


@dataclass
class Module5Output:
    updated_graph: Any = None
    updated_conflicts: List[Conflict] = field(default_factory=list)
    updated_influence_sets: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class Module6Output:
    final_text: str = ""
    template_used: str = ""


@dataclass
class PipelineResult:
    final_document: str = ""
    module_outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
