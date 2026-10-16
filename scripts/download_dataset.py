"""SignBridge reliable, resumable dataset downloader and validator.
Supports:
  1. isolated-smoke: vidit031/isl-isolated-8words (quick smoke test, ~15 MB, 56 clips)
  2. isolated-40   : vidit031/isl-isolated-40words (baseline, ~196 MB, 642 clips, 40 glosses)
  3. sentence      : ISLTranslate / iSign (continuous ISL sentence pairs, ~31k pairs, gated)
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure UTF-8 output on Windows consoles if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]

# Dataset registry with official metadata and estimated sizes
DATASETS = {
    "isolated-smoke": {
        "repo_id": "vidit031/isl-isolated-8words",
        "repo_type": "dataset",
        "official_url": "https://huggingface.co/datasets/vidit031/isl-isolated-8words",
        "default_dir": ROOT / "data" / "raw" / "isolated_smoke",
        "symlink_target": ROOT / "data" / "raw",
        "estimated_size_mb": 25,
        "required_free_mb": 150,
        "expected_classes": 8,
        "expected_clips": 56,
        "is_gated": False,
        "description": "Small isolated-sign smoke test dataset (8 glosses, 56 clips) for pipeline verification.",
    },
    "isolated-40": {
        "repo_id": "vidit031/isl-isolated-40words",
        "repo_type": "dataset",
        "official_url": "https://huggingface.co/datasets/vidit031/isl-isolated-40words",
        "default_dir": ROOT / "data" / "raw" / "isolated_40words",
        "symlink_target": ROOT / "data" / "raw",
        "estimated_size_mb": 210,
        "required_free_mb": 600,
        "expected_classes": 40,
        "expected_clips": 642,
        "is_gated": False,
        "description": "Isolated-sign baseline dataset (40 glosses, 642 clips, ~196 MB) with mixed upstream licenses.",
    },
    "sentence": {
        "repo_id": "Exploration-Lab/iSign",
        "repo_type": "dataset",
        "official_url": "https://huggingface.co/datasets/Exploration-Lab/iSign",
        "github_url": "https://github.com/Exploration-Lab/ISLTranslate",
        "paper_url": "https://aclanthology.org/2023.findings-acl.665/",
        "default_dir": ROOT / "data" / "raw" / "sentence",
        "estimated_size_mb": 15000,
        "required_free_mb": 25000,
        "is_gated": True,
        "description": "ISLTranslate continuous ISL-English translation (~31k pairs, 15+ GB video archives). Non-commercial license.",
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def check_disk_space(target_dir: Path, required_mb: int) -> Tuple[bool, float]:
    """Check if the target drive has enough free disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        total, used, free = shutil.disk_usage(str(target_dir))
        free_mb = free / (1024 * 1024)
        return (free_mb >= required_mb), free_mb
    except Exception:
        return True, 0.0


def resolve_clip_path(dataset_dir: Path, raw_path: str, original_filename: str = "") -> Optional[Path]:
    """
    Resolve video path against the actual downloaded files:
    1. Normalizes Windows / POSIX slashes.
    2. Strips absolute machine-specific prefixes.
    3. Tries relative path within dataset_dir.
    4. Tries basename in gloss subdirectory or dataset_dir.
    5. Tries original_filename if unambiguous.
    """
    cleaned = str(raw_path).strip().replace("\\", "/")
    
    # Try direct relative path inside dataset_dir
    direct = dataset_dir / cleaned
    if direct.is_file():
        return direct
        
    # Strip machine-specific absolute path components (e.g. C:/Users/.../gloss/file.mp4)
    path_obj = Path(cleaned)
    basename = path_obj.name
    
    # If path has a parent directory (e.g. eat/eat__CISLR__...mp4)
    if len(path_obj.parts) >= 2:
        rel_candidate = dataset_dir / path_obj.parts[-2] / path_obj.parts[-1]
        if rel_candidate.is_file():
            return rel_candidate

    # Try searching for filename inside dataset_dir
    candidates = list(dataset_dir.rglob(basename))
    if len(candidates) == 1:
        return candidates[0]
        
    # Try original_filename if provided
    if original_filename:
        orig_clean = Path(str(original_filename).strip().replace("\\", "/")).name
        if orig_clean:
            candidates_orig = list(dataset_dir.rglob(orig_clean))
            if len(candidates_orig) == 1:
                return candidates_orig[0]

    return None


def validate_isolated_dataset(dataset_dir: Path, expected_classes: Optional[int] = None) -> Dict[str, Any]:
    """Validate an isolated-sign dataset directory."""
    import pandas as pd
    
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.is_file():
        return {"valid": False, "error": f"metadata.csv missing in {dataset_dir}"}
        
    try:
        df = pd.read_csv(meta_path)
    except Exception as exc:
        return {"valid": False, "error": f"Could not read metadata.csv: {exc}"}
        
    label_col = "normalized_word" if "normalized_word" in df.columns else ("word" if "word" in df.columns else None)
    if not label_col or "video_path" not in df.columns:
        return {"valid": False, "error": f"metadata.csv missing required columns: {label_col}, video_path"}
        
    total_rows = len(df)
    unique_classes = sorted(df[label_col].dropna().unique().tolist())
    
    # Resolve all videos
    resolved_count = 0
    missing_paths = []
    unresolved_rows = []
    
    for idx, row in df.iterrows():
        raw_vp = str(row.get("video_path", ""))
        orig_fn = str(row.get("original_filename", "")) if "original_filename" in df.columns else ""
        resolved = resolve_clip_path(dataset_dir, raw_vp, orig_fn)
        if resolved and resolved.is_file():
            resolved_count += 1
        else:
            missing_paths.append(raw_vp)
            unresolved_rows.append({"row": idx, "video_path": raw_vp, "original_filename": orig_fn})
            
    is_valid = (resolved_count > 0 and len(missing_paths) == 0)
    
    return {
        "valid": is_valid,
        "metadata_rows": total_rows,
        "resolved_clips": resolved_count,
        "missing_clips": len(missing_paths),
        "unresolved_examples": unresolved_rows[:5],
        "class_count": len(unique_classes),
        "classes": unique_classes,
        "has_signer_info": ("signer" in df.columns and df["signer"].notna().sum() > 0),
        "licenses_found": df["license"].dropna().unique().tolist() if "license" in df.columns else [],
    }


def download_isolated(dataset_key: str, out_dir: Path, max_retries: int = 3) -> bool:
    """Download and verify isolated sign dataset via huggingface_hub."""
    info = DATASETS[dataset_key]
    repo_id = info["repo_id"]
    required_mb = info["required_free_mb"]
    
    print(f"\nTarget Dataset: {dataset_key} ({repo_id})")
    print(f"Description: {info['description']}")
    print(f"Official URL: {info['official_url']}")
    print(f"Destination: {out_dir}")
    
    # 1. Disk check
    ok_space, free_mb = check_disk_space(out_dir, required_mb)
    print(f"Free disk space on target: {free_mb:.1f} MB (Required: ~{required_mb} MB)")
    if not ok_space:
        print(f"[FAIL] Insufficient disk space! Need at least {required_mb} MB free.")
        return False
        
    # 2. Download via huggingface_hub snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[FAIL] huggingface_hub is not installed! Run: pip install huggingface_hub")
        return False
        
    print(f"Beginning resumable snapshot download of {repo_id}...")
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(out_dir),
                allow_patterns=[
                    "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
                    "metadata.csv", "*.json", "*.txt", "README.md", "*/sources.txt"
                ],
                resume_download=True,
            )
            success = True
            break
        except Exception as exc:
            print(f"[WARN] Download attempt {attempt}/{max_retries} encountered an error: {exc}")
            if attempt < max_retries:
                time.sleep(3)
                
    if not success:
        print(f"[FAIL] Failed to download {repo_id} after {max_retries} attempts.")
        return False
        
    # 3. Validate
    print("\nValidating downloaded files and metadata...")
    val = validate_isolated_dataset(out_dir)
    if val["valid"]:
        print(f"[OK] Successfully validated {dataset_key}!")
        print(f"     Clips resolved: {val['resolved_clips']} / {val['metadata_rows']}")
        print(f"     Classes: {val['class_count']} {val['classes'][:6]}...")
        if val["licenses_found"]:
            print(f"     Licensing note: Upstream licenses reported: {val['licenses_found']}")
        return True
    else:
        print(f"[WARN] Dataset validation reported issues: {val.get('error', 'Incomplete clips')}")
        print(f"       Clips resolved: {val['resolved_clips']} / {val['metadata_rows']} (missing: {val['missing_clips']})")
        return False


def handle_sentence_dataset(out_dir: Path, check_only: bool, confirm: bool) -> bool:
    """Handle continuous sentence translation dataset (ISLTranslate / iSign)."""
    info = DATASETS["sentence"]
    print("\n" + "=" * 65)
    print("  ISLTranslate Continuous Sentence-Level Translation Dataset")
    print("=" * 65)
    print(f"Official Repository : {info['github_url']}")
    print(f"Official Paper      : {info['paper_url']}")
    print(f"Hugging Face Dataset: {info['official_url']}")
    print(f"Target Directory    : {out_dir}")
    print(f"Estimated Size      : ~15 GB to 40 GB across video archives")
    print(f"License Notice      : Non-commercial research use only.")
    print("=" * 65)
    
    # Check if files are already extracted in data/translation/ or out_dir
    manifest_paths = [
        ROOT / "data" / "translation" / "manifest.csv",
        out_dir / "iSign_v1.1.csv",
        out_dir / "ISLTranslate.csv",
        out_dir / "manifest.csv",
    ]
    
    found_manifest = None
    for mp in manifest_paths:
        if mp.is_file():
            found_manifest = mp
            break
            
    if check_only or not confirm:
        if found_manifest:
            print(f"\n[OK] Found existing manifest: {found_manifest}")
            import pandas as pd
            df = pd.read_csv(found_manifest)
            print(f"     Entries: {len(df)}")
            print(f"     Columns: {list(df.columns)}")
            return True
        else:
            print("\n[NOTE] Continuous sentence translation dataset is not downloaded yet.")
            print("\nTo obtain the ISLTranslate dataset:")
            print("1. Visit the official Hugging Face dataset card:")
            print("   https://huggingface.co/datasets/Exploration-Lab/iSign")
            print("2. Accept the dataset license agreement (non-commercial research).")
            print("3. Download iSign_v1.1.csv and video archives, or set your HF_TOKEN:")
            print("   set HF_TOKEN=your_token_here")
            print("   python scripts/download_dataset.py --dataset sentence --confirm")
            print("4. Alternatively, populate data/translation/manifest.csv with paired")
            print("   video paths and English sentence transcripts.")
            return False

    # If --confirm was passed
    print("\nChecking Hugging Face credentials for gated dataset...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("[FAIL] HF_TOKEN environment variable is not set.")
        print("Because Exploration-Lab/iSign is gated, you must provide an authorized token.")
        print("Example in PowerShell:")
        print("  $env:HF_TOKEN=\"hf_your_token\"")
        print("  python scripts/download_dataset.py --dataset sentence --confirm")
        return False
        
    ok_space, free_mb = check_disk_space(out_dir, info["required_free_mb"])
    print(f"Free disk space on target: {free_mb / 1024:.2f} GB (Required: ~{info['required_free_mb'] / 1024:.1f} GB)")
    if not ok_space:
        print(f"[FAIL] Insufficient disk space! Needs at least ~{info['required_free_mb'] / 1024:.1f} GB free.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        print(f"Attempting download of {info['repo_id']} with token...")
        snapshot_download(
            repo_id=info["repo_id"],
            repo_type="dataset",
            local_dir=str(out_dir),
            token=hf_token,
            allow_patterns=["*.csv", "*.md", "*.txt"],  # Start with metadata files
            resume_download=True,
        )
        print("[OK] Successfully downloaded sentence dataset metadata files!")
        return True
    except Exception as exc:
        print(f"[FAIL] Download failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SignBridge reliable dataset downloader.")
    parser.add_argument(
        "--dataset",
        choices=["isolated-smoke", "isolated-40", "sentence"],
        default="isolated-smoke",
        help="Dataset to download or check (default: isolated-smoke)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect whether expected files exist without downloading",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output directory",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm large download (required for sentence dataset)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of download retry attempts",
    )
    args = parser.parse_args()

    dataset_key = args.dataset
    info = DATASETS[dataset_key]
    out_dir = Path(args.output).resolve() if args.output else info["default_dir"]

    if dataset_key == "sentence":
        success = handle_sentence_dataset(out_dir, args.check_only, args.confirm)
        sys.exit(0 if success else 1)

    if args.check_only:
        print(f"Checking status of {dataset_key} in {out_dir}...")
        if not out_dir.is_dir():
            print(f"Directory {out_dir} does not exist.")
            print(f"Run without --check-only to download: python scripts/download_dataset.py --dataset {dataset_key}")
            sys.exit(1)
        val = validate_isolated_dataset(out_dir, info.get("expected_classes"))
        print(json.dumps(val, indent=2))
        sys.exit(0 if val["valid"] else 1)

    success = download_isolated(dataset_key, out_dir, args.max_retries)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
