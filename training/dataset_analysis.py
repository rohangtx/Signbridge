from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

meta = RAW / "metadata.csv"
if not meta.exists():
    raise SystemExit("metadata.csv not found. Run scripts\\download_isl40.py first.")

df = pd.read_csv(meta)
label_col = "normalized_word" if "normalized_word" in df.columns else "word"
if label_col not in df.columns:
    raise SystemExit("Could not find word/normalized_word in metadata.csv")

signer_col = "signer" if "signer" in df.columns else None
summary = {
    "rows": int(len(df)),
    "classes": int(df[label_col].nunique()),
    "class_names": sorted(df[label_col].dropna().astype(str).unique().tolist()),
    "signers": int(df[signer_col].nunique()) if signer_col else None,
    "missing_signer_rows": int(df[signer_col].isna().sum()) if signer_col else int(len(df)),
    "source_columns": df.columns.tolist(),
}
(REPORTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

counts = df[label_col].value_counts().sort_index()
counts.rename_axis("class").reset_index(name="samples").to_csv(REPORTS / "class_counts.csv", index=False)

if signer_col:
    signer_counts = df[signer_col].fillna("unknown").value_counts().sort_index()
    signer_counts.rename_axis("signer").reset_index(name="samples").to_csv(REPORTS / "signer_counts.csv", index=False)

lines = ["# SignBridge V1 Dataset Analysis", "", f"- Samples: {summary['rows']}", f"- Classes: {summary['classes']}", f"- Signers: {summary['signers']}", "", "## Class distribution", ""]
for label, n in counts.items():
    lines.append(f"- {label}: {n}")
(REPORTS / "dataset_analysis.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2))
