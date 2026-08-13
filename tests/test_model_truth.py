import hashlib
import json
from pathlib import Path

from retouch.model_fetch import model_status, verify_model


def test_bundled_nafnet_matches_manifest():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["models"]["denoise_nafnet"]
    path = Path("models") / entry["filename"]
    assert entry["availability"] == "bundled"
    assert path.is_file()
    assert path.stat().st_size == entry["size_bytes"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"]
    assert verify_model("denoise_nafnet") is True


def test_optional_models_are_explicitly_unavailable_and_not_downloadable():
    for name in ("lama_inpaint", "sr_real_esrgan"):
        status = model_status(name)
        assert status["availability"] == "unavailable"
        assert status["available"] is False
        assert status["downloadable"] is False
        assert not status.get("url")


def test_parked_neural_boosters_are_not_cli_exposed():
    from retouch.params import get_param

    assert get_param("neural_stray_hair_boost").cli_flag is None
    assert get_param("neural_defect_boost").cli_flag is None
