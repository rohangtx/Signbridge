"""Extract ISL-CSLRT dataset videos and metadata from Downloads zip and build manifest.csv"""
import zipfile
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = Path(r"C:\Users\mrsen.ASPIRE_7\Downloads\ISL_CSLRT_Corpus.zip")
TARGET_DIR = ROOT / "data" / "raw" / "sentence" / "ISL_CSLRT"
MANIFEST_PATH = ROOT / "data" / "translation" / "manifest.csv"

def extract_corpus():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Opening {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = [
            m for m in z.namelist()
            if (m.startswith("ISL_CSLRT_Corpus/Videos_Sentence_Level/") or
                m.startswith("ISL_CSLRT_Corpus/corpus_csv_files/"))
            and not m.endswith("/")
        ]
        print(f"Extracting {len(members)} files to {TARGET_DIR}...")
        for m in tqdm(members, desc="Extracting ISL-CSLRT files"):
            rel_path = m.replace("ISL_CSLRT_Corpus/", "")
            out_file = TARGET_DIR / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if not out_file.exists() or out_file.stat().st_size == 0:
                with z.open(m) as src, open(out_file, "wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
    print("Extraction complete!")

def build_manifest():
    videos_dir = TARGET_DIR / "Videos_Sentence_Level"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    
    records = []
    sentence_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sentence_dirs)} sentence directories.")
    
    for s_dir in sentence_dirs:
        sentence_text = s_dir.name.strip()
        # Clean sentence punctuation/formatting if needed
        # Collect all mp4 / MP4 files
        video_files = sorted([f for f in s_dir.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov"]])
        
        # Categorize signers:
        # Native signers usually have the full sentence name as the video name
        # Volunteer signers usually have MVI_ or word (num)
        for idx, vf in enumerate(video_files):
            fname = vf.name.lower()
            if fname.startswith(sentence_text.lower()):
                signer = "native_signer_2" if "(2)" in fname or "_2" in fname else "native_signer_1"
            else:
                signer = f"volunteer_signer_{idx + 1}"
            
            # Relative path from MANIFEST_PATH.parent (data/translation)
            rel_video_path = f"../raw/sentence/ISL_CSLRT/Videos_Sentence_Level/{s_dir.name}/{vf.name}"
            records.append({
                "video_path": rel_video_path,
                "translation": sentence_text,
                "signer": signer
            })
            
    df = pd.DataFrame(records)
    print(f"Generated manifest with {len(df)} video-sentence pairs across {df['signer'].nunique()} signers.")
    print("Signer distribution:\n", df['signer'].value_counts())
    
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Saved manifest to {MANIFEST_PATH}")

if __name__ == "__main__":
    extract_corpus()
    build_manifest()
