from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


@dataclass
class TextConfig:
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 8000


@dataclass
class TrainConfig:
    alpha: float = 1.0
    test_size: float = 0.2
    random_state: int = 42
    min_samples_per_class: int = 5
    do_grid_search: bool = False


@dataclass
class MatchConfig:
    posterior_weight: float = 0.4
    similarity_weight: float = 0.6