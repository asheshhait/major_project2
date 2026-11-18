from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "samples.db"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
UPLOAD_DIR = DATA_DIR / "uploads"


@dataclass(frozen=True)
class AppConfig:
    """Centralizes threshold choices and model hyperparameters."""

    app_name: str = "bangla-ohr"
    app_version: str = "0.1.0"
    debug: bool = False
    database_url: str = f"sqlite:///{DB_PATH}"
    upload_dir: str = str(UPLOAD_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    pen_down_value: int = 1
    min_points_per_stroke: int = 3
    min_stroke_bbox_area: float = 8.0
    word_gap_sensitivity: float = 1.0


APP_CONFIG = AppConfig()
