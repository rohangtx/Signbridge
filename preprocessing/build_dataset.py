"""Build the V1 landmark dataset with per-video caching and atomic output.

Successful videos are cached individually under data/processed/landmark_cache.
If extraction stops due to low memory or interruption, rerunning this script
reuses completed videos instead of processing them from scratch.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.landmark_extractor import LandmarkExtractor, FEATURES, SEQUENCE_LENGTH

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CACHE = OUT / "landmark_cache"
TARGET = OUT / "landmarks.npz"
COMPLETE_MARKER = OUT / "landmarks_build_complete.json"


def normalize_label(value) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def cache_path_for(video: Path, row_number: int) -> Path:
    stat = video.stat()
    identity = f"{video.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return CACHE / f"{row_number:05d}_{digest}.npy"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # A marker proves that this run reached a successful, verified save.
    COMPLETE_MARKER.unlink(missing_ok=True)

    meta_path = RAW / "metadata.csv"
    if not meta_path.exists():
        raise SystemExit("metadata.csv not found. Run scripts\\download_isl40.py first.")

    df = pd.read_csv(meta_path)
    label_col = "normalized_word" if "normalized_word" in df.columns else "word"
    if label_col not in df.columns or "video_path" not in df.columns:
        raise SystemExit("metadata.csv must contain video_path and word/normalized_word columns.")
    signer_col = "signer" if "signer" in df.columns else None

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    signers: list[str] = []
    paths: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    cache_hits = 0

    extractor = LandmarkExtractor(max_dimension=640, reset_every_videos=35)
    try:
        for row_number, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Extracting landmarks")):
            relative = str(row["video_path"]).strip().replace("\\", "/")
            video = Path(relative).expanduser()
            if not video.is_absolute():
                video = RAW / video
            if not video.exists():
                missing.append(str(video))
                continue

            try:
                cache_file = cache_path_for(video, row_number)
                sequence = None
                if cache_file.exists():
                    try:
                        cached = np.load(cache_file, allow_pickle=False)
                        if cached.shape == (SEQUENCE_LENGTH, FEATURES) and cached.dtype == np.float32:
                            sequence = cached
                            cache_hits += 1
                        else:
                            cache_file.unlink(missing_ok=True)
                    except Exception:
                        cache_file.unlink(missing_ok=True)

                if sequence is None:
                    sequence = extractor.extract_video(video, n=SEQUENCE_LENGTH)
                    if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
                        raise ValueError(f"Expected {(SEQUENCE_LENGTH, FEATURES)}, got {sequence.shape}")
                    sequence = np.asarray(sequence, dtype=np.float32)
                    temp_cache = cache_file.with_name(cache_file.stem + ".tmp.npy")
                    np.save(temp_cache, sequence, allow_pickle=False)
                    os.replace(temp_cache, cache_file)

                sequences.append(sequence)
                labels.append(normalize_label(row[label_col]))
                signer_value = str(row[signer_col]).strip() if signer_col and pd.notna(row[signer_col]) else "unknown"
                if signer_value.lower() in {"", "nan", "none", "null", "unknown"}:
                    signer_value = "unknown"
                signers.append(signer_value)
                paths.append(relative)
                del sequence
            except Exception as exc:
                failed.append(f"{video}: {type(exc).__name__}: {exc}")
                print(f"Skipping {video}: {type(exc).__name__}: {exc}")
            finally:
                if row_number % 10 == 0:
                    gc.collect()
    finally:
        extractor.close()

    if failed:
        (OUT / "failed_videos.txt").write_text("\n".join(failed), encoding="utf-8")
    else:
        (OUT / "failed_videos.txt").unlink(missing_ok=True)
    if missing:
        (OUT / "missing_files.txt").write_text("\n".join(missing), encoding="utf-8")

    if len(sequences) < 10 or len(set(labels)) < 2:
        raise SystemExit(
            f"Not enough valid data to train: {len(sequences)} videos across {len(set(labels))} classes. "
            "Check failed_videos.txt, video codecs, free RAM, and data/raw/metadata.csv."
        )

    temp_target = OUT / "landmarks.building.npz"
    try:
        with temp_target.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=np.stack(sequences).astype(np.float32, copy=False),
                y=np.asarray(labels),
                signer=np.asarray(signers),
                path=np.asarray(paths),
            )
        # Read back the archive before making it the completed dataset.
        with np.load(temp_target, allow_pickle=False) as check:
            if check["X"].shape != (len(sequences), SEQUENCE_LENGTH, FEATURES):
                raise ValueError(f"Saved dataset has unexpected shape: {check['X'].shape}")
            sample_count = int(check["X"].shape[0])
            class_count = int(len(np.unique(check["y"])))
        os.replace(temp_target, TARGET)
    except Exception as exc:
        temp_target.unlink(missing_ok=True)
        raise SystemExit(f"Could not save/verify landmarks dataset: {type(exc).__name__}: {exc}") from exc

    class_counts = {label: int(labels.count(label)) for label in sorted(set(labels))}
    summary = {
        "processed_samples": sample_count,
        "classes": sorted(set(labels)),
        "class_count": class_count,
        "unique_signers": sorted(set(signers)),
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES,
        "cache_hits": cache_hits,
        "failed_videos": len(failed),
        "missing_files": len(missing),
        "class_counts": class_counts,
        "completed_unix_time": time.time(),
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    COMPLETE_MARKER.write_text(json.dumps({"ok": True, "samples": sample_count, "classes": class_count, "target": str(TARGET)}, indent=2), encoding="utf-8")
    print("\nVerified landmark dataset written to:", TARGET)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
