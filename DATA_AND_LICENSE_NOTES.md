# SignBridge — Data and License Notes

This file documents the provenance, licensing constraints and download instructions
for all datasets used by the SignBridge project.  **Read this before redistributing
any data or trained model weights.**

---

## Dataset 1 — Isolated-sign smoke test (8 glosses)

| Field | Value |
|---|---|
| Hugging Face | `vidit031/isl-isolated-8words` |
| Official page | https://huggingface.co/datasets/vidit031/isl-isolated-8words |
| Glosses | 8 |
| Clips | ~56 |
| Approx. size | ~25 MB |
| Intended use | Quick end-to-end pipeline smoke test **only**. Not a final benchmark. |

**Licensing:** The dataset card on Hugging Face documents upstream sources (CISLR, ISL500, etc.)
each with their own licence terms.  Respect the per-clip `license` column in `metadata.csv`.
Do **not** claim the aggregate has a single permissive licence.

**Download command:**
```powershell
.\venv\Scripts\python.exe -m scripts.download_dataset --dataset isolated-smoke
```

---

## Dataset 2 — Isolated-sign baseline (40 glosses)

| Field | Value |
|---|---|
| Hugging Face | `vidit031/isl-isolated-40words` |
| Official page | https://huggingface.co/datasets/vidit031/isl-isolated-40words |
| Glosses | 40 |
| Clips | ~642 |
| Approx. size | ~196 MB |

**Licensing:** Mixed upstream licensing from multiple source datasets (CISLR, ISL500, etc.).
Check the `license` and `repository` columns in `metadata.csv` for each clip.
Do **not** assume the aggregate has a single permissive licence or that all clips
may be freely redistributed.

**Important — video\_path column:** Some rows contain original machine-specific absolute
Windows paths (`C:\Users\...`).  The downloader and preprocessing pipeline resolve paths
using relative locations within the downloaded dataset root and the `original_filename`
basename.  Absolute paths from another machine are **never** opened directly.

**Download command:**
```powershell
.\venv\Scripts\python.exe -m scripts.download_dataset --dataset isolated-40
```

---

## Dataset 3 — Continuous sentence-level translation (ISLTranslate / iSign)

| Field | Value |
|---|---|
| Repository | https://github.com/Exploration-Lab/ISLTranslate |
| Hugging Face | `Exploration-Lab/iSign` (gated — requires account and agreement) |
| Paper | https://aclanthology.org/2023.findings-acl.665/ |
| Sentence pairs | ~30 000 ISL-English pairs |
| Approx. size | ~15 GB (video archives) + extracted feature archives |
| Gated access | **Yes** — requires HF account + licence agreement |

### Licence — NON-COMMERCIAL RESTRICTION

The ISLTranslate / iSign dataset is released for **non-commercial research use only**.
This restriction applies to all video clips, pre-extracted features and annotation files.

- Do **not** include downloaded videos or features in any public code repository.
- Do **not** use for commercial products or services without permission from the authors.
- Preserve all attribution and licence notices from the original dataset card.
- Cite the paper if you use this data in research:

```
@inproceedings{joshi-etal-2023-isltranslate,
  title     = {ISLTranslate: Dataset for Translating Indian Sign Language},
  author    = {Joshi, Abhinav and Agrawal, Susmit and Modi, Ashutosh},
  booktitle = {Findings of ACL 2023},
  year      = {2023},
  url       = {https://aclanthology.org/2023.findings-acl.665/}
}
```

### Manual download steps (gated dataset)

1. Create a Hugging Face account at https://huggingface.co
2. Visit https://huggingface.co/datasets/Exploration-Lab/iSign
3. Accept the dataset licence agreement.
4. Set your token: `$env:HF_TOKEN = "hf_your_token_here"`
5. Run the downloader to fetch metadata CSV first:
   ```powershell
   .\venv\Scripts\python.exe -m scripts.download_dataset --dataset sentence --confirm
   ```
6. For large video archives, follow the manual download instructions on the dataset card.
7. Extract archives into `data\raw\sentence\` and populate `data\translation\manifest.csv`
   using the `iSign_v1.1.csv` file as a guide.

### Populating manifest.csv for V3/V4 sentence training

After downloading and inspecting the real ISLTranslate data, create `data\translation\manifest.csv`
with columns: `video_path, translation, signer` (signer is optional but enables group-aware
splitting). Use `data\translation\manifest_template.csv` as a starting point.

```powershell
# Once manifest.csv is populated with real paired data:
.\run_translation_v3.bat   # trains V3 BiLSTM encoder-decoder
.\run_translation_v4.bat   # trains V4 Transformer encoder-decoder
```

---

## What is and is not committed to the repository

| Item | Committed |
|---|---|
| Source code, configs, scripts | Yes |
| Vocabulary files (40 words) | Yes |
| Downloaded video clips | **No** — in `.gitignore` |
| Extracted landmark `.npz` files | **No** — in `.gitignore` |
| Trained model weights (`.joblib`, `.keras`) | **No** — in `.gitignore` |
| Prediction history (`.db`) | **No** — in `.gitignore` |
| Virtual environment (`venv/`) | **No** — in `.gitignore` |

Users who want to share trained weights should do so through a separate, clearly licensed
model card and should verify they have the right to redistribute based on the upstream
dataset licence terms.

---

*Last updated: October 2026*
