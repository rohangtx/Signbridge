"""Compare real V3/V4 held-out text metrics, when both trainings have completed."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]

def main():
    models = []
    for version, title, p in [
        ("V3", "LSTM encoder-decoder", ROOT / "models" / "v3" / "metrics.json"),
        ("V4", "Temporal Transformer", ROOT / "models" / "v4" / "metrics.json"),
    ]:
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            tm = m.get("text_metrics", {})
            models.append((version, title, m.get("split_type", "unknown"), m.get("train_pairs", 0), m.get("test_pairs", 0), tm.get("exact_match"), tm.get("corpus_bleu")))
    if not models:
        raise SystemExit("No V3/V4 metrics exist yet. Train at least one translation model first.")
    lines = ["# SignBridge sentence-translation comparison", "", "| Version | Model | Split | Train pairs | Test pairs | Exact-match | Corpus BLEU |", "|---|---|---|---:|---:|---:|---:|"]
    for v, title, split, train_n, test_n, exact, bleu in models:
        exact_s = f"{exact:.4f}" if isinstance(exact, (int, float)) else "n/a"
        bleu_s = f"{bleu:.4f}" if isinstance(bleu, (int, float)) else "n/a"
        lines.append(f"| {v} | {title} | {split} | {train_n} | {test_n} | {exact_s} | {bleu_s} |")
    lines += ["", "## Interpretation", "", "- These values are generated from saved held-out predictions; no scores are hard-coded.", "- Compare models only when trained on the same paired dataset and evaluation split.", "- Automatic metrics are not a substitute for fluent-signer/human assessment of meaning, grammar, omissions and hallucinations.", "- Very small datasets or transcript noise can make the metrics unstable."]
    out = ROOT / "reports" / "translation_comparison.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
