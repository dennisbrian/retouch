"""Focused tests for the engine's truthful output-precision contract."""

import json

import numpy as np

from retouch.engine import (
    ProcessingResult,
    RetouchEngine,
    _CoreResult,
)


def _stub_no_face_engine(monkeypatch):
    """Build an engine shell that exercises process()'s output boundary."""
    engine = RetouchEngine.__new__(RetouchEngine)

    def fake_process_with_proxy(image, _ctx, _style_ref, _timings):
        return _CoreResult(
            result=image.copy(),
            acc_skin=None,
            acc_skin_hair=None,
            acc_lips=None,
            acc_sharpen=None,
            faces=[],
            person_mask=None,
            no_face=True,
            face_contexts=None,
            qa=[],
        )

    monkeypatch.setattr(engine, "_process_with_proxy", fake_process_with_proxy)
    return engine


def test_processing_result_reports_native_uint8_precision():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    result = ProcessingResult(image)

    assert result.precision.processed_precision == "uint8"
    assert result.precision.processed_bit_depth == 8
    assert result.precision.storage_dtype == "uint8"
    assert result.precision.storage_bit_depth == 8
    assert result.precision.precision_status == "native_8bit"
    assert result.precision.downgraded is False
    assert result.precision.is_true_16bit is False
    assert result.precision.supports_16bit_export is False
    assert result.precision_metadata == result.precision.to_dict()
    json.dumps(result.precision_metadata)


def test_float_source_reports_truthful_uint8_downgrade(monkeypatch):
    source = np.linspace(0.0, 255.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    engine = _stub_no_face_engine(monkeypatch)

    result = engine.process(source)

    assert result.dtype == np.uint8
    assert result.precision.source_dtype == "float32"
    assert result.precision.source_bit_depth is None
    assert result.precision.processed_precision == "uint8"
    assert result.precision.processed_bit_depth == 8
    assert result.precision.precision_status == "downgraded_to_uint8"
    assert result.precision.downgraded is True
    assert result.precision.is_true_16bit is False
    assert result.precision.supports_16bit_export is False
    assert "final delivery boundary" in result.precision.downgrade_reason


def test_uint8_engine_path_remains_byte_stable(monkeypatch):
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    engine = _stub_no_face_engine(monkeypatch)

    result = engine.process(source)

    np.testing.assert_array_equal(result, source)
    assert result.precision.precision_status == "native_8bit"
    assert result.precision.downgraded is False


def test_uint16_container_does_not_claim_true_png16_precision():
    # A container can be uint16 while carrying only 8-bit-style values.  A
    # producer must explicitly verify the effective precision before claiming
    # a true PNG-16 export; dtype alone is insufficient.
    image = np.array([[[0, 127, 255], [255, 127, 0]]], dtype=np.uint16)

    result = ProcessingResult(image)

    assert result.precision.storage_dtype == "uint16"
    assert result.precision.storage_bit_depth == 16
    assert result.precision.processed_bit_depth is None
    assert result.precision.precision_status == "unverified_container_precision"
    assert result.precision.is_true_16bit is False
    assert result.precision.supports_16bit_export is False
