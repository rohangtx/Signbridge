"""Rebuild manifest.csv using ALL 101 ISL-CSLRT sentence directories."""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_DIR = ROOT / "data" / "raw" / "sentence" / "ISL_CSLRT" / "Videos_Sentence_Level"
MANIFEST = ROOT / "data" / "translation" / "manifest.csv"

records = []
sentence_dirs = sorted([d for d in VIDEOS_DIR.iterdir() if d.is_dir()])
print(f"Found {len(sentence_dirs)} sentence directories.")

for s_dir in sentence_dirs:
    sentence_text = s_dir.name.strip()
    video_files = sorted([f for f in s_dir.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]])
    for idx, vf in enumerate(video_files):
        fname = vf.name.lower()
        if fname.startswith(sentence_text.lower()):
            signer = "native_signer_2" if "(2)" in fname else "native_signer_1"
        elif "mvi_" in fname:
            signer = f"volunteer_signer_{idx}"
        else:
            signer = f"volunteer_signer_{idx}"
        rel_path = f"../raw/sentence/ISL_CSLRT/Videos_Sentence_Level/{s_dir.name}/{vf.name}"
        records.append({"video_path": rel_path, "translation": sentence_text, "signer": signer})

df = pd.DataFrame(records)
n_sentences = df["translation"].nunique()
n_signers = df["signer"].nunique()
print(f"Total pairs: {len(df)}, unique sentences: {n_sentences}, unique signers: {n_signers}")

# Back up old manifest
if MANIFEST.exists():
    shutil.copy(MANIFEST, MANIFEST.with_name("manifest_backup_40.csv"))
    print(f"Backed up old manifest to manifest_backup_40.csv")

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(MANIFEST, index=False)
print(f"Saved expanded manifest to {MANIFEST}")

if __name__ == "__main__":
    pass
