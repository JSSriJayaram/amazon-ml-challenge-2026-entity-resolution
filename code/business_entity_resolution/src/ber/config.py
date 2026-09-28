"""Paths and global settings. Override the data/artifact roots with env vars
BER_DATA_DIR and BER_ARTIFACT_DIR so the package runs from any checkout."""
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]  # code/business_entity_resolution
REPO_ROOT = PKG_ROOT.parents[1]  # repository root

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "student_resource" / "dataset"))
ARTIFACT_DIR = Path(os.environ.get("BER_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_ROOT / "output"))

SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def source_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def ground_truth_path() -> Path:
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def artifact(*parts: str) -> Path:
    p = ARTIFACT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
