"""Tests for signing retouched photos with Content Credentials (C2PA).

Certificates are generated per test session with ``cryptography`` (a
throwaway root CA and signing certificate), so nothing secret is checked in.
The signing tests skip when the optional ``c2pa-python`` package is missing.
"""

import datetime as dt
import json

import cv2
import numpy as np
import pytest

from retouch.content_credentials import (
    CredentialsError,
    SigningConfig,
    build_manifest,
    sniff_mime,
)
from retouch.edit_report import build_edit_report
from retouch.io import write_image_with_icc

c2pa = pytest.importorskip("c2pa", reason="optional c2pa-python not installed")
x509 = pytest.importorskip("cryptography.x509")

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

from retouch.content_credentials import read_credentials, sign_output  # noqa: E402


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Retouch Tests")])


@pytest.fixture(scope="module")
def signing_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("certs")
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name("Test Root CA")).issuer_name(_name("Test Root CA"))
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("Test Photographer")).issuer_name(ca.subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path = d / "chain.pem"
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM)
                          + ca.public_bytes(serialization.Encoding.PEM))
    key_path = d / "signing.key"
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


@pytest.fixture
def config(signing_files):
    return SigningConfig.from_paths(*signing_files)


def _photo(path, value=120):
    img = np.full((64, 96, 3), value, np.uint8)
    img[16:48, 24:72] = (40, 90, 200)
    cv2.imwrite(str(path), img)
    return img


def _retouch(src, out):
    img = cv2.imread(str(src))
    edited = cv2.GaussianBlur(img, (5, 5), 0)
    write_image_with_icc(out, edited)
    return img, edited


def _report(before, after, **kw):
    return build_edit_report({"smooth": 30, "slimming": 20}, recipe="natural", face_count=1,
                             before=before, after=after, signed=True,
                             include_output_hash=False, **kw)


def test_signed_output_verifies_and_carries_report(tmp_path, config):
    src, out = tmp_path / "src.jpg", tmp_path / "out.jpg"
    _photo(src)
    before, after = _retouch(src, out)
    result = sign_output(out, _report(before, after), config, source_path=src)
    assert result == {"signed": True, "parent": True,
                      "parent_note": "Source photo attached as the parent ingredient."}

    creds = read_credentials(out)
    assert creds["validation_state"] in ("Valid", "Trusted")
    # A self-made certificate is intact but not on a C2PA trust list.
    assert not [c for c in creds["codes"] if c != "signingCredential.untrusted"]
    assert creds["ingredients"] == ["src.jpg"]
    assert creds["report"]["shape_changed"] is True
    assert creds["report"]["edits"]["skin"]["settings"] == {"smooth": 30}


def test_camera_credentials_are_kept_as_history_not_copied(tmp_path, config):
    # A "camera" photo that already carries a manifest.
    camera = tmp_path / "camera.jpg"
    _photo(camera, 90)
    unsigned_copy = tmp_path / "plain.jpg"
    _photo(unsigned_copy, 90)
    sign_output(camera, _report(None, None), config, source_path=unsigned_copy)
    assert read_credentials(camera) is not None

    # Retouching alone must not copy the camera's manifest (it would fail
    # verification against the edited pixels).
    out = tmp_path / "out.jpg"
    before, after = _retouch(camera, out)
    assert read_credentials(out) is None

    # Signing records the camera manifest as the parent's history.
    sign_output(out, _report(before, after), config, source_path=camera)
    creds = read_credentials(out)
    assert creds["validation_state"] in ("Valid", "Trusted")
    assert len(creds["manifest"]["manifests"]) == 2


def test_extensionless_source_is_sniffed(tmp_path, config):
    src, out = tmp_path / "upload", tmp_path / "out.jpg"
    _photo(tmp_path / "tmp.jpg")
    (tmp_path / "tmp.jpg").rename(src)
    assert sniff_mime(src) == "image/jpeg"
    before, after = _retouch(src, out)
    assert sign_output(out, _report(before, after), config, source_path=src)["parent"] is True


def test_unreadable_source_uses_decoded_preview(tmp_path, config):
    src, out = tmp_path / "shot.raf", tmp_path / "out.jpg"
    src.write_bytes(b"FUJIFILMCCD-RAW 0201" + b"\0" * 64)
    before = np.full((64, 96, 3), 100, np.uint8)
    write_image_with_icc(out, before + 10)
    result = sign_output(out, _report(before, before + 10), config,
                         source_path=src, source_preview=before)
    assert result["parent"] is False
    assert "decoded preview" in result["parent_note"]
    creds = read_credentials(out)
    assert creds["validation_state"] in ("Valid", "Trusted")
    assert creds["ingredients"] == ["shot.raf (decoded preview)"]


def test_unreadable_source_without_preview_fails_cleanly(tmp_path, config):
    src, out = tmp_path / "shot.raf", tmp_path / "out.jpg"
    src.write_bytes(b"FUJIFILMCCD-RAW 0201")
    _photo(out)
    original = out.read_bytes()
    with pytest.raises(CredentialsError):
        sign_output(out, _report(None, None), config, source_path=src)
    assert out.read_bytes() == original


def test_png_output_can_be_signed(tmp_path, config):
    src, out = tmp_path / "src.jpg", tmp_path / "out.png"
    _photo(src)
    before, after = _retouch(src, out)
    sign_output(out, _report(before, after), config, source_path=src)
    assert read_credentials(out)["validation_state"] in ("Valid", "Trusted")


def test_unsupported_output_format(tmp_path, config):
    out = tmp_path / "out.exr"
    out.write_bytes(b"x")
    with pytest.raises(CredentialsError, match="can't be embedded"):
        sign_output(out, _report(None, None), config)


def test_tampering_after_signing_is_detected(tmp_path, config):
    src, out = tmp_path / "src.jpg", tmp_path / "out.jpg"
    _photo(src)
    before, after = _retouch(src, out)
    sign_output(out, _report(before, after), config, source_path=src)
    data = bytearray(out.read_bytes())
    data[-100] ^= 0xFF  # flip bits in the compressed image data
    out.write_bytes(bytes(data))
    assert read_credentials(out)["validation_state"] == "Invalid"


class TestSigningConfig:
    def test_missing_file(self, tmp_path, signing_files):
        with pytest.raises(CredentialsError, match="not found"):
            SigningConfig.from_paths(tmp_path / "nope.pem", signing_files[1])

    def test_key_passed_as_certificate(self, signing_files):
        with pytest.raises(CredentialsError, match="holds a private key"):
            SigningConfig.from_paths(signing_files[1], signing_files[1])

    def test_not_pem(self, tmp_path, signing_files):
        bad = tmp_path / "cert.der"
        bad.write_bytes(b"\x30\x82\x01")
        with pytest.raises(CredentialsError, match="not a PEM"):
            SigningConfig.from_paths(bad, signing_files[1])

    def test_traditional_ec_key_is_rejected_with_fix(self, tmp_path, signing_files):
        legacy = tmp_path / "ec.key"
        legacy.write_text("-----BEGIN EC PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----\n")
        with pytest.raises(CredentialsError, match="openssl pkcs8"):
            SigningConfig.from_paths(signing_files[0], legacy)

    def test_encrypted_key_is_rejected(self, tmp_path, signing_files):
        enc = tmp_path / "enc.key"
        enc.write_text("-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----\n")
        with pytest.raises(CredentialsError, match="password-protected"):
            SigningConfig.from_paths(signing_files[0], enc)

    def test_self_signed_certificate_fails_at_signing(self, tmp_path):
        key = ec.generate_private_key(ec.SECP256R1())
        now = dt.datetime.now(dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(_name("Me")).issuer_name(_name("Me"))
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
            .sign(key, hashes.SHA256())
        )
        cert_path, key_path = tmp_path / "self.pem", tmp_path / "self.key"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                               serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        src, out = tmp_path / "src.jpg", tmp_path / "out.jpg"
        _photo(src)
        _retouch(src, out)
        original = out.read_bytes()
        with pytest.raises(CredentialsError):
            sign_output(out, _report(None, None), SigningConfig.from_paths(cert_path, key_path),
                        source_path=src)
        assert out.read_bytes() == original

    def test_unknown_algorithm(self, signing_files):
        with pytest.raises(CredentialsError, match="Unknown signing algorithm"):
            SigningConfig.from_paths(*signing_files, alg="md5")

    def test_wrong_algorithm_for_key_fails_at_signing(self, tmp_path, signing_files):
        cfg = SigningConfig.from_paths(*signing_files, alg="ps256")
        src, out = tmp_path / "src.jpg", tmp_path / "out.jpg"
        _photo(src)
        _retouch(src, out)
        with pytest.raises(CredentialsError):
            sign_output(out, _report(None, None), cfg, source_path=src)


def test_manifest_actions_describe_the_edits():
    report = build_edit_report({"slimming": 30, "smooth": 20, "saturation": 5, "ai_denoise": 40},
                               face_count=1)
    actions = build_manifest(report, "x.jpg", "image/jpeg", parent_label="p")["assertions"][0]
    kinds = [a["action"] for a in actions["data"]["actions"]]
    assert kinds[0] == "c2pa.opened"
    assert actions["data"]["actions"][0]["parameters"] == {"ingredientIds": ["p"]}
    assert "c2pa.color_adjustments" in kinds and "c2pa.filtered" in kinds
    assert any(a.get("description") == "Face or body shape changed" for a in actions["data"]["actions"])
    json.dumps(actions)
