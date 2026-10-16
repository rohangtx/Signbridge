import mediapipe as mp
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocessing.landmark_extractor import LandmarkExtractor

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "translation" / "manifest.csv"
OUT = ROOT / "data" / "processed"
SEQ_LEN = 32


def resolve_video_path(raw_path: str) -> Path:
    p = Path(str(raw_path).strip().replace("\\", "/")).expanduser()
    if p.is_absolute():
        return p
    return (MANIFEST.parent / p).resolve()


def main():
    if not MANIFEST.exists():
        raise SystemExit(
            f"Missing {MANIFEST}. Copy manifest_template.csv to manifest.csv and add genuine paired videos/transcripts."
        )
    df = pd.read_csv(MANIFEST).fillna("")
    required = {"video_path", "translation"}
    if not required.issubset(df.columns):
        raise SystemExit("manifest.csv must contain columns: video_path, translation; optional: signer")
    df["video_path"] = df["video_path"].astype(str).str.strip()
    df["translation"] = df["translation"].astype(str).str.strip()
    df = df[(df.video_path != "") & (df.translation != "")]
    if not len(df):
        raise SystemExit("No non-empty video/transcript pairs in manifest.csv")

    extractor = LandmarkExtractor()
    X, texts, signers, paths, missing, failed = [], [], [], [], [], []
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sentence-video landmarks"):
            video = resolve_video_path(row["video_path"])
            if not video.is_file():
                missing.append(str(video))
                continue
            try:
                seq = extractor.extract_video(video, n=SEQ_LEN)
                if seq.shape != (SEQ_LEN, 258):
                    raise ValueError(f"Unexpected feature shape: {seq.shape}")
                X.append(seq)
                texts.append(row["translation"])
                signers.append(str(row.get("signer", "")).strip() or "unknown")
                paths.append(str(video))
            except Exception as exc:
                failed.append(f"{video}\t{type(exc).__name__}: {exc}")
    finally:
        extractor.close()

    OUT.mkdir(parents=True, exist_ok=True)
    if missing:
        (OUT / "translation_missing_files.txt").write_text("\n".join(missing), encoding="utf-8")
    if failed:
        (OUT / "translation_failed_files.txt").write_text("\n".join(failed), encoding="utf-8")
    if not X:
        raise SystemExit(
            f"No valid video/transcript pairs were processed (missing={len(missing)}, failed={len(failed)}). Check paths/codecs."
        )

    np.savez_compressed(
        OUT / "translation_sequences.npz",
        X=np.asarray(X, dtype=np.float32),
        text=np.asarray(texts, dtype=str),
        signer=np.asarray(signers, dtype=str),
        path=np.asarray(paths, dtype=str),
    )
    summary = {
        "valid_pairs": len(X),
        "skipped_missing": len(missing),
        "failed_extractions": len(failed),
        "unique_signers_known": len(set(s for s in signers if s != "unknown")),
        "sequence_length": SEQ_LEN,
        "features_per_frame": 258,
        "source_manifest": str(MANIFEST),
        "warning": "Training validity depends on transcript quality, signer diversity, data licenses and the split protocol.",
    }
    (OUT / "translation_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
