import hashlib
import json
from pathlib import Path

import pytest

from retouch.model_fetch import (
    ModelFetchError,
    ModelSizeMismatchError,
    model_status,
    verify_model,
)


def test_optional_onnx_models_are_not_claimed_as_wheel_bundled():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    for name in ("resnet18_bisenet", "retinaface_mv1", "denoise_nafnet"):
        entry = manifest["models"][name]
        assert entry["availability"] == "unavailable"
        assert not entry.get("url")
        path = Path("models") / entry["filename"]
        if path.is_file():
            assert path.stat().st_size == entry["size_bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            assert verify_model(name) is True


def test_optional_models_are_explicitly_unavailable_and_not_downloadable():
    for name in ("lama_inpaint", "sr_real_esrgan"):
        status = model_status(name)
        assert status["availability"] == "unavailable"
        assert status["available"] is False
        assert status["downloadable"] is False
        assert not status.get("url")


def test_downloadable_models_have_pinned_urls_and_integrity_metadata():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    for name in ("face_landmarker", "selfie_segmenter", "pose_landmarker_full"):
        entry = manifest["models"][name]
        assert entry["availability"] == "downloadable"
        assert entry["url"]
        assert "generation=" in entry["url"]
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0


def test_verify_model_rejects_missing_integrity_metadata(tmp_path, monkeypatch):
    import retouch.model_fetch as model_fetch

    monkeypatch.setattr(
        model_fetch,
        "load_manifest",
        lambda: {"models": {"broken": {"filename": "broken.bin", "sha256": "", "size_bytes": 3}}},
    )
    monkeypatch.setattr(model_fetch, "_MODELS_DIR", str(tmp_path))
    (tmp_path / "broken.bin").write_bytes(b"abc")
    with pytest.raises(ModelFetchError, match="sha256"):
        model_fetch.verify_model("broken")


def test_download_fails_closed_on_size_mismatch(tmp_path, monkeypatch):
    import retouch.model_fetch as model_fetch

    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        model_fetch,
        "load_manifest",
        lambda: {
            "models": {
                "broken": {
                    "filename": "broken.bin",
                    "url": "https://example.invalid/broken.bin",
                    "sha256": "0" * 64,
                    "size_bytes": 4,
                }
            }
        },
    )

    def write_wrong_size(_url, dest, _expected_size, _progress):
        Path(dest).write_bytes(b"abc")

    monkeypatch.setattr(model_fetch, "_stream_download", write_wrong_size)
    with pytest.raises(ModelSizeMismatchError):
        model_fetch.download_model("broken")
    assert not (tmp_path / "cache" / "models" / "broken.bin").exists()


def test_download_stores_verified_model_in_user_cache(tmp_path, monkeypatch):
    import retouch.model_fetch as model_fetch

    payload = b"verified model"
    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        model_fetch,
        "load_manifest",
        lambda: {
            "models": {
                "cached": {
                    "filename": "cached.bin",
                    "url": "https://example.invalid/cached.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            }
        },
    )

    def write_payload(_url, dest, _expected_size, _progress):
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(model_fetch, "_stream_download", write_payload)
    result = model_fetch.download_model("cached")
    assert Path(result) == tmp_path / "cache" / "models" / "cached.bin"
    assert Path(result).read_bytes() == payload


def test_parked_neural_boosters_are_not_cli_exposed():
    from retouch.params import get_param

    assert get_param("neural_stray_hair_boost").cli_flag is None
    assert get_param("neural_defect_boost").cli_flag is None
