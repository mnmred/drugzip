from pathlib import Path

ROOT = Path(__file__).parent.resolve().parent

PROJECT_DIR = ROOT  # Use root directory instead of project_folder
DATA_DIR = ROOT / "datasets"
EMBEDDING_DIR = ROOT / "embeddings"
CHECKPOINT_DIR = ROOT / "checkpoints"
FIGURE_DIR = ROOT / "figures"
WB_DIR = ROOT / "wandb"
