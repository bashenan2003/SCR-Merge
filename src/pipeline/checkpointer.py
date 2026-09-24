"""Checkpoint manager for pipeline state save/restore/rollback."""

import pickle
import time
from pathlib import Path
from typing import Optional, List


class CheckpointManager:
    """Save and restore pipeline state at named checkpoints."""

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = str(Path(__file__).parent.parent.parent / "checkpoints")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_order: List[str] = []

    def save(self, name: str, state: dict) -> Path:
        """Save pipeline state to a checkpoint file."""
        timestamp = int(time.time() * 1000)
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{timestamp}.pkl"
        path = self.storage_dir / filename
        with open(path, "wb") as f:
            pickle.dump(state, f)
        self._checkpoint_order.append(str(path))
        return path

    def restore(self, name: str) -> Optional[dict]:
        """Restore most recent checkpoint matching name prefix."""
        matching = sorted(
            [p for p in self.storage_dir.glob(f"{name}_*.pkl")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matching:
            return None
        with open(matching[0], "rb") as f:
            return pickle.load(f)

    def restore_latest(self) -> Optional[dict]:
        """Restore the most recent checkpoint overall."""
        files = sorted(
            self.storage_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        with open(files[0], "rb") as f:
            return pickle.load(f)

    def rollback_to(self, name: str) -> dict:
        """Restore state and remove checkpoints after it."""
        state = self.restore(name)
        if state is None:
            raise FileNotFoundError(f"No checkpoint found for '{name}'")

        # Clean up later checkpoints
        target_path = None
        for p in sorted(
            self.storage_dir.glob(f"{name}_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            target_path = str(p)
            break

        if target_path and target_path in self._checkpoint_order:
            idx = self._checkpoint_order.index(target_path)
            for path in self._checkpoint_order[idx + 1:]:
                Path(path).unlink(missing_ok=True)
            self._checkpoint_order = self._checkpoint_order[:idx + 1]

        return state

    def list_checkpoints(self) -> List[str]:
        """List checkpoint names in order."""
        return self._checkpoint_order.copy()

    def cleanup(self, keep_last: int = 3) -> None:
        """Remove old checkpoints, keeping the most recent N."""
        files = sorted(
            self.storage_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[keep_last:]:
            f.unlink(missing_ok=True)
            if str(f) in self._checkpoint_order:
                self._checkpoint_order.remove(str(f))

    def clear(self) -> None:
        """Remove all checkpoints."""
        for f in self.storage_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
        self._checkpoint_order.clear()
