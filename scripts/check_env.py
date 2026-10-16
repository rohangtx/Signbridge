"""SignBridge environment diagnostic script.

Checks Python, package versions, NumPy/PyArrow compatibility, model artifacts,
and dataset presence.

KNOWN BEHAVIOUR: TensorFlow and MediaPipe each import successfully in isolation.
Importing 'streamlit' before 'tensorflow' in the same process can trigger a
TF DLL initialisation failure on Windows because Streamlit loads PyArrow, which
pre-loads certain MSVC runtime DLLs.  This script avoids that by NOT importing
streamlit at the top level and by checking TF in a subprocess.
"""
import sys
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_import_test(module_name: str) -> tuple:
    """Test whether a module can be imported in a clean subprocess."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"import {module_name}; "
         f"m=__import__('{module_name}'); "
         f"print(getattr(m,'__version__','installed'))"],
        capture_output=True, text=True, timeout=30,
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        version = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "installed"
        return True, version
    else:
        first_error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed"
        return False, first_error


def check_environment():
    print("=" * 65)
    print("  SignBridge Environment & Dependency Diagnostics")
    print("=" * 65)

    py_ver = sys.version.split()[0]
    is_64bit = sys.maxsize > 2 ** 32
    print(f"OS      : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python  : {py_ver} ({'64-bit' if is_64bit else '32-bit — WARNING: requires 64-bit'})")
    print(f"Venv    : {sys.executable}")

    if not (3, 10) <= sys.version_info < (3, 12):
        print("  NOTE: Tested Python version is 3.11.x; 3.10 is also supported.")

    # ------------------------------------------------------------------
    # Package checks — each in its own subprocess to avoid DLL conflicts
    # ------------------------------------------------------------------
    packages = [
        ("numpy",           "numpy"),
        ("pyarrow",         "pyarrow"),
        ("pandas",          "pandas"),
        ("streamlit",       "streamlit"),
        ("cv2",             "cv2"),
        ("tensorflow",      "tensorflow"),
        ("mediapipe",       "mediapipe"),
        ("sklearn",         "sklearn"),
        ("joblib",          "joblib"),
        ("huggingface_hub", "huggingface_hub"),
        ("tqdm",            "tqdm"),
        ("pytest",          "pytest"),
        ("nltk",            "nltk"),
    ]

    print("\nPackage versions (each tested in a clean subprocess):")
    installed: dict = {}
    status_ok = True

    for mod_name, pkg_label in packages:
        ok, info = _run_import_test(mod_name)
        if ok:
            installed[mod_name] = info
            print(f"  [OK]   {pkg_label:<18}: {info}")
        else:
            status_ok = False
            short = info[:90] if len(info) > 90 else info
            print(f"  [FAIL] {pkg_label:<18}: {short}")

    # ------------------------------------------------------------------
    # NumPy / PyArrow interop
    # ------------------------------------------------------------------
    print("\nCompatibility checks:")
    if "numpy" in installed and "pyarrow" in installed:
        np_ver = installed["numpy"]
        pa_ver = installed["pyarrow"]
        np_major = int(np_ver.split(".")[0])
        pa_major = int(pa_ver.split(".")[0])
        print(f"  NumPy {np_ver} + PyArrow {pa_ver}")
        test = subprocess.run(
            [sys.executable, "-c",
             "import pyarrow as pa, pandas as pd, numpy as np;"
             "arr=pa.Array.from_pandas(pd.Series([1.0,2.0])); arr.to_numpy();"
             "print('OK')"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT))
        if test.returncode == 0:
            print("  [OK]   NumPy + PyArrow conversion: compatible")
        else:
            status_ok = False
            print(f"  [FAIL] NumPy + PyArrow conversion failed")
            if np_major < 2 and pa_major >= 19:
                print("         FIX: pip install \"pyarrow>=16.0.0,<17.0.0\"")

    # MediaPipe Holistic
    if "mediapipe" in installed:
        test = subprocess.run(
            [sys.executable, "-c",
             "import mediapipe as mp;"
             "h=mp.solutions.holistic.Holistic(static_image_mode=True,model_complexity=0);"
             "h.close();print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT))
        if test.returncode == 0:
            print("  [OK]   MediaPipe Holistic pipeline: functional")
        else:
            print(f"  [NOTE] MediaPipe Holistic: {test.stderr.strip().splitlines()[-1][:100]}")

    # TensorFlow CPU/GPU
    if "tensorflow" in installed:
        test = subprocess.run(
            [sys.executable, "-c",
             "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3';"
             "import tensorflow as tf;"
             "gpus=tf.config.list_physical_devices('GPU');"
             "print('GPU:'+str(len(gpus)))"],
            capture_output=True, text=True, timeout=30, cwd=str(ROOT))
        if test.returncode == 0:
            gpu_count = int(test.stdout.strip().split("GPU:")[-1])
            if gpu_count:
                print(f"  [OK]   TensorFlow: {gpu_count} GPU(s) detected")
            else:
                print("  [OK]   TensorFlow: CPU-only mode (suitable for laptop training)")
        else:
            print(f"  [NOTE] TensorFlow GPU check: {test.stderr.strip().splitlines()[-1][:100]}")

    # ------------------------------------------------------------------
    # Model artifacts
    # ------------------------------------------------------------------
    print("\nModel artifacts:")
    models = {
        "Random Forest":  ROOT / "models" / "baseline" / "random_forest.joblib",
        "RF encoder":     ROOT / "models" / "baseline" / "label_encoder.joblib",
        "LSTM model":     ROOT / "models" / "lstm"     / "signbridge_lstm.keras",
        "LSTM encoder":   ROOT / "models" / "lstm"     / "label_encoder.joblib",
        "V3 sentence":    ROOT / "models" / "v3"       / "sentence_lstm.keras",
        "V4 transformer": ROOT / "models" / "v4"       / "sentence_transformer.keras",
    }
    any_model = False
    for name, p in models.items():
        if p.is_file():
            any_model = True
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [FOUND]  {name:<20}: {p.relative_to(ROOT)}  ({size_mb:.1f} MB)")
        else:
            print(f"  [ABSENT] {name:<20}: not trained yet")
    if not any_model:
        print("  --> Run .\\run_training_smoke.bat (quick) or .\\run_pipeline_v1.bat (full)")

    # ------------------------------------------------------------------
    # Dataset directories
    # ------------------------------------------------------------------
    print("\nDataset directories:")
    try:
        import pandas as pd
        _pd_ok = True
    except ImportError:
        _pd_ok = False

    dataset_dirs = {
        "isolated-smoke": ROOT / "data" / "raw" / "isolated_smoke",
        "isolated-40":    ROOT / "data" / "raw" / "isolated_40words",
        "raw (legacy)":   ROOT / "data" / "raw",
    }
    for label, d in dataset_dirs.items():
        meta = d / "metadata.csv"
        if meta.is_file():
            if _pd_ok:
                try:
                    import pandas as pd
                    df = pd.read_csv(meta)
                    print(f"  [FOUND]  {label}: metadata.csv ({len(df)} rows)")
                except Exception:
                    print(f"  [FOUND]  {label}: metadata.csv (could not parse)")
            else:
                print(f"  [FOUND]  {label}: metadata.csv")
        else:
            print(f"  [ABSENT] {label}: run download_dataset.py --dataset {label}")

    # ------------------------------------------------------------------
    # Disk space
    # ------------------------------------------------------------------
    print("\nDisk space:")
    try:
        _, _, free = shutil.disk_usage(str(ROOT))
        free_gb = free / (1024 ** 3)
        warn = "  [WARN: < 5 GB — may not fit full 40-word dataset]" if free_gb < 5 else ""
        print(f"  {ROOT.anchor}: {free_gb:.1f} GB free{warn}")
    except Exception as exc:
        print(f"  Could not check: {exc}")

    print("\n" + "=" * 65)
    if status_ok:
        print("  Status: ALL CORE DEPENDENCIES OK")
    else:
        print("  Status: SOME DEPENDENCIES HAVE ISSUES — see [FAIL] lines above")
        print("  Fix:    .\\setup.bat   (re-installs packages from requirements.txt)")
    print("=" * 65)
    return 0 if status_ok else 1


if __name__ == "__main__":
    sys.exit(check_environment())
