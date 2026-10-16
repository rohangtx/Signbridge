from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
models = {}
for name in ["baseline", "lstm"]:
    p = ROOT / f"models/{name}/metrics.json"
    if p.exists(): models[name] = json.loads(p.read_text())
lines = ["# SignBridge V1 Evaluation Report", "", "## Model comparison", "", "| Model | Accuracy | Split | Train | Test |", "|---|---:|---|---:|---:|"]
for name, m in models.items():
    lines.append(f"| {name.title()} | {m['accuracy']:.4f} | {m['split_type']} | {m['train_samples']} | {m['test_samples']} |")
lines += ["", "## Interpretation", "", "Accuracy values above are generated from the actual local training run; no values are hard-coded.", "", "## Generated artifacts", "- dataset_analysis.md", "- class_counts.csv", "- signer_counts.csv (when signer metadata exists)", "- model_accuracy.png", "- baseline_confusion_matrix.png", "- lstm_confusion_matrix.png", "- lstm_accuracy_curve.png", "- lstm_loss_curve.png"]
(ROOT / "reports/evaluation.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote reports/evaluation.md")
