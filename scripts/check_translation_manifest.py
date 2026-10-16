"""Validate genuine paired ISL videos and English transcripts before V3/V4 training."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "translation" / "manifest.csv"
MIN_PAIRS = 10

def resolve_video(value):
    candidate = Path(str(value).strip().replace("\\", "/")).expanduser()
    return candidate if candidate.is_absolute() else (MANIFEST.parent / candidate).resolve()

def main():
    if not MANIFEST.is_file():
        print(f"[SKIP V3/V4] Missing {MANIFEST}")
        print("Add genuine paired clips/transcripts to data/translation/manifest.csv.")
        return 2
    try:
        with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"video_path", "translation"}.issubset(set(reader.fieldnames or [])):
                print("[SKIP V3/V4] Required columns are video_path and translation; signer is optional.")
                return 2
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        print(f"[SKIP V3/V4] Cannot read manifest: {exc}")
        return 2
    valid = set()
    missing = []
    for row in rows:
        video = str(row.get("video_path", "") or "").strip()
        translation = str(row.get("translation", "") or "").strip()
        if not video or not translation:
            continue
        path = resolve_video(video)
        if path.is_file():
            valid.add(str(path.resolve()))
        elif len(missing) < 5:
            missing.append(str(path))
    print(f"Manifest rows: {len(rows)}; valid distinct paired clips: {len(valid)}")
    if missing:
        print("Examples of missing paths:")
        for path in missing:
            print(" -", path)
    if len(valid) < MIN_PAIRS:
        print(f"[SKIP V3/V4] Need at least {MIN_PAIRS} valid distinct video/transcript pairs for this experimental pipeline.")
        print("That threshold is only a smoke-test minimum, not enough for research-quality claims.")
        return 2
    print("[OK] Manifest basic validation passed; feature extraction and training may still fail on bad clips.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
