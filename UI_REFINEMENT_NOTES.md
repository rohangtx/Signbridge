# SignBridge frontend refinement — product workspace

This revision elevates the interface from a basic engineering dashboard into a clearer single-product workspace.

## Visual system
- Dark navy/indigo brand treatment with restrained teal status accents.
- Strong headline and typographic hierarchy, carefully spaced sections, consistent rounded surfaces, and clearer visual grouping.
- Custom sign-inspired SVG mark in the sidebar; no external image or network request is needed for the logo.
- Responsive hero layout and small-screen CSS adjustments.
- Styled sidebar navigation, upload regions, primary actions, metric cards, status messages, empty states, and history rows.

## User-facing structure
- **Translate a sentence** — primary workspace for continuous ISL video and English draft generation.
- **Recognize one sign** — clearly secondary, for one isolated sign from a still image, short clip, or camera snapshot.
- **Model health** — technical artifact inventory, metrics and recent local history.
- **About the project** — scope, roadmap and research limitations.

## Workflow improvements
- Translation uses a three-step flow: add video, review capture guidance, inspect output.
- An output preview area remains useful before training; it clearly says that the model must be trained rather than displaying a fabricated translation.
- Engine options appear only when the required model/vocabulary artifacts exist.
- Hindi sentence generation is not implied by the V1 Hindi word-label map.
- Single-sign input has its own capture checklist and model availability indicators.
- Stale sentence output is cleared when the selected upload changes.
- Prediction history is rendered as lightweight rows rather than `st.table`/`st.dataframe`, avoiding the Arrow conversion path involved in the reported error.
- `requirements.txt` explicitly pins PyArrow to the chosen NumPy 1.x dependency set; if local package resolution still reports a version conflict, follow the actual traceback and install in the environment that launches `run.bat`.

## Research honesty
The interface is a frontend and workflow refinement, not a claim that the sentence model has been trained. No model weights are bundled. Sentence translation still requires genuine continuous ISL video paired with reviewed English transcripts, then training and evaluation.

## Verification performed for this revision
- `python -m py_compile app/streamlit_app.py` — passed.
- `python -m pytest -q` — 5 passed.
- Streamlit browser preview/startup was not executed in this build environment because Streamlit is not installed here.
