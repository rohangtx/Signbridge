# V2 — Continuous camera gloss stream

Run `run_live_v2.bat` after V1 training. A rolling 32-frame buffer produces periodic LSTM classifications. A small temporal vote filters jitter and the app accumulates unique stable glosses.

Scope limitation: this stage does not infer grammatical relationships or translate the gloss buffer into a true natural-language sentence. It must be run locally with a working camera. It currently supports one camera stream per process.
