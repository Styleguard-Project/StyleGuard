from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
BATCH_DIR = DATA_DIR / "batch"

for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, BATCH_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_PER_CLASS = 2500
TEST_SIZE = 0.20

ARXIV_CATEGORIES = {
    "cs.CL": "Computer Science - Computation & Language",
    "cs.AI": "Computer Science - Artificial Intelligence",
    "physics.optics": "Physics - Optics",
    "q-bio.GN": "Biology - Genomics",
}

CATEGORY_MAP = {
    "cs.CL": "Natural Language Processing",
    "cs.AI": "Artificial Intelligence",
    "physics.optics": "Optics and Photonics",
    "q-bio.GN": "Genomics and Computational Biology",
}

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
