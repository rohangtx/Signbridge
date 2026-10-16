from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_all_requested_input_modes_are_wired_into_app():
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert '"Photo", "Video", "Real-time camera"' in app
    assert "sentence-translation-live-camera" in app
    assert "single-sign-live-camera" in app
    assert "translate_live_camera_button" in app

def test_live_camera_dependency_and_processor_exist():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    processor = (ROOT / "inference" / "live_stream.py").read_text(encoding="utf-8")
    assert "streamlit-webrtc" in requirements
    assert "class LiveSignProcessor" in processor
    assert "export_recording" in processor
