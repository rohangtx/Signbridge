"""Consolidate actual V1 model metrics and create a machine-readable evaluation summary."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]

def main():
    found = {}
    for key, rel in [("Random Forest", "models/baseline/metrics.json"), ("LSTM", "models/lstm/metrics.json")]:
        p = ROOT / rel
        if p.exists():
            found[key] = json.loads(p.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit("No trained V1 metrics found. Run training/train_baseline.py and training/train_lstm.py first.")
    result = {"models": found, "notice": "Metrics are read from actual training artifacts; this script does not create or estimate accuracy."}
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Wrote reports/evaluation_summary.json")

if __name__ == "__main__":
    main()
