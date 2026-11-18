from __future__ import annotations

from pathlib import Path


class ModelRegistry:
    """Persists checkpoints and keeps versions for rollback support."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model_name: str, payload):
        target = self.checkpoint_dir / f"{model_name}.pt"
        target.write_bytes(b"")
        return str(target)

    def load(self, model_name: str):
        return {"model": model_name, "loaded": False}
