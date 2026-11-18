from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingRun:
    epoch: int = 0
    loss: float | None = None
    status: str = "idle"


class IncrementalTrainer:
    """Background training manager placeholder for future incremental updates."""

    def __init__(self):
        self.current_run = TrainingRun()

    def start(self):
        self.current_run.status = "running"
        self.current_run.epoch = 1
        self.current_run.loss = 0.0
        return self.current_run

    def status(self):
        return self.current_run
