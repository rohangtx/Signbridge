from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="vidit031/isl-isolated-40words")
args = parser.parse_args()

OUT.mkdir(parents=True, exist_ok=True)
print(f"Downloading {args.repo} ...")
snapshot_download(
    repo_id=args.repo,
    repo_type="dataset",
    local_dir=str(OUT),
    allow_patterns=["*.mp4", "*.avi", "*.mov", "*.mkv", "metadata.csv", "*.json", "*.txt", "README.md", "*/sources.txt"],
)
meta = OUT / "metadata.csv"
print(f"Dataset directory: {OUT}")
print(f"Metadata present: {meta.exists()}")
