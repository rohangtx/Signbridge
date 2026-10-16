from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]; REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
accs = {}
for name in ["baseline", "lstm"]:
    p = ROOT / f"models/{name}/metrics.json"
    if not p.exists(): continue
    m = json.loads(p.read_text()); accs[name] = m["accuracy"]
    cm = confusion_matrix(m["y_true"], m["y_pred"], labels=list(range(len(m["classes"]))))
    fig, ax = plt.subplots(figsize=(12, 10)); ConfusionMatrixDisplay(cm, display_labels=m["classes"]).plot(ax=ax, xticks_rotation="vertical", colorbar=False); ax.set_title(f"{name.title()} Confusion Matrix"); fig.tight_layout(); fig.savefig(REPORTS / f"{name}_confusion_matrix.png", dpi=160); plt.close(fig)

if accs:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(accs.keys()), [float(v) for v in accs.values()], color=["#4f46e5", "#0e9f9a"][:len(accs)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("SignBridge Model Accuracy")
    fig.tight_layout()
    fig.savefig(REPORTS / "model_accuracy.png", dpi=160)
    plt.close(fig)

p = ROOT / "models/lstm/metrics.json"
if p.exists():
    h = json.loads(p.read_text()).get("history", {})
    if "accuracy" in h:
        fig, ax = plt.subplots(figsize=(7,4)); ax.plot(h["accuracy"], label="train"); ax.plot(h.get("val_accuracy", []), label="validation"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("LSTM Accuracy Curve"); ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "lstm_accuracy_curve.png", dpi=160); plt.close(fig)
    if "loss" in h:
        fig, ax = plt.subplots(figsize=(7,4)); ax.plot(h["loss"], label="train"); ax.plot(h.get("val_loss", []), label="validation"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("LSTM Loss Curve"); ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "lstm_loss_curve.png", dpi=160); plt.close(fig)
