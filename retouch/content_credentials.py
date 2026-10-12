"""Sign retouched photos with Content Credentials (C2PA), using your own certificate.

A camera that signs its photos (Leica, Sony, Nikon, some Fujifilm bodies)
writes a C2PA manifest whose hash covers the exact pixels it captured. Once
the photo is retouched that hash can no longer match, so copying the camera's
manifest onto the retouched file makes verifiers report it as *invalid*
(``assertion.dataHash.mismatch``). :mod:`retouch.io` therefore no longer
copies it.

The right way to carry provenance forward is a *new* manifest, signed by
whoever did the edit, that lists what was changed and names the original
photo (with its camera manifest, if it had one) as its parent. That is what
:func:`sign_output` does. It needs:

* the optional ``c2pa-python`` package (``uv sync --extra credentials``), and
* a signing certificate chain and private key that **you** supply (PEM
  files). Nothing is bundled and no certificate is bought or fetched by the
  app. A certificate from a C2PA trust-list issuer shows as trusted in
  verifiers such as contentcredentials.org/verify; a self-made one still
  verifies as intact and unaltered but is flagged as an unknown signer.

The edit report from :mod:`retouch.edit_report` is embedded as the
``org.retouch.edit_report`` assertion, alongside standard ``c2pa.actions``.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)

REPORT_ASSERTION = "org.retouch.edit_report"
_PARENT_LABEL = "parent_photo"
ALGORITHMS = ("es256", "es384", "es512", "ps256", "ps384", "ps512", "ed25519")

# Output formats the C2PA SDK can embed a manifest in.
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jpe": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".dng": "image/x-adobe-dng",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
}


class CredentialsError(RuntimeError):
    """Signing could not be set up or failed; the message says why in plain terms."""


def _c2pa():
    try:
        import c2pa  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised when not installed
        raise CredentialsError(
            "Signing needs the optional c2pa-python package. Install it with "
            "`uv sync --extra desktop --extra credentials` (or "
            "`pip install c2pa-python`)."
        ) from exc
    return c2pa


def signing_available() -> bool:
    """True when the C2PA SDK is installed."""
    try:
        _c2pa()
    except CredentialsError:
        return False
    return True


def mime_for(path: Union[str, Path]) -> Optional[str]:
    """Media type from the file extension, or ``None`` if the SDK can't embed in it."""
    return _MIME.get(Path(path).suffix.lower())


def sniff_mime(path: Union[str, Path]) -> Optional[str]:
    """Media type from the file's first bytes, falling back to its extension.

    Sources are often extensionless or misnamed (camera-card copies, chat
    uploads). RAW files return ``None``: the SDK can't read them.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"FUJIFILMCCD-RAW"):
        return None
    if head[4:8] == b"ftyp":
        return mime_for(path)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        # Most camera RAWs are TIFF containers; trust only a TIFF extension.
        return mime_for(path) if Path(path).suffix.lower() in (".tif", ".tiff") else None
    return None


@dataclass(frozen=True)
class SigningConfig:
    """Where the user's certificate chain and private key live.

    ``cert_path`` is a PEM file holding the signing certificate followed by
    any intermediates (not the root). ``key_path`` is the matching PEM private
    key. ``tsa_url`` is an optional RFC 3161 timestamp server, which keeps a
    signature valid after the certificate expires.
    """

    cert_path: Path
    key_path: Path
    alg: str = "es256"
    tsa_url: Optional[str] = None

    @classmethod
    def from_paths(cls, cert: Union[str, Path], key: Union[str, Path],
                   alg: str = "es256", tsa_url: Optional[str] = None) -> "SigningConfig":
        cfg = cls(Path(cert).expanduser(), Path(key).expanduser(), alg.lower(), tsa_url or None)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.alg not in ALGORITHMS:
            raise CredentialsError(
                f"Unknown signing algorithm {self.alg!r}; use one of {', '.join(ALGORITHMS)}"
            )
        for label, path in (("certificate", self.cert_path), ("private key", self.key_path)):
            if not path.is_file():
                raise CredentialsError(f"Signing {label} file not found: {path}")
            head = path.read_bytes()[:4096]
            if b"-----BEGIN" not in head:
                raise CredentialsError(f"Signing {label} {path} is not a PEM file")
        if b"PRIVATE KEY" in self.cert_path.read_bytes():
            raise CredentialsError(
                f"{self.cert_path} holds a private key; pass the certificate chain as the "
                "certificate and the key separately"
            )
        key = self.key_path.read_bytes()
        if b"-----BEGIN ENCRYPTED PRIVATE KEY" in key:
            raise CredentialsError(
                f"{self.key_path} is password-protected; signing needs an unencrypted key "
                f"(openssl pkcs8 -topk8 -nocrypt -in {self.key_path.name} -out signing.key)"
            )
        if b"-----BEGIN PRIVATE KEY" not in key:
            raise CredentialsError(
                f"{self.key_path} is not a PKCS#8 private key; convert it with "
                f"openssl pkcs8 -topk8 -nocrypt -in {self.key_path.name} -out signing.key"
            )

    def signer(self):
        """Build a ``c2pa.Signer``. The key bytes are never logged."""
        c2pa = _c2pa()
        alg = getattr(c2pa.C2paSigningAlg, self.alg.upper())
        info = c2pa.C2paSignerInfo(
            alg,
            self.cert_path.read_bytes(),
            self.key_path.read_bytes(),
            self.tsa_url.encode() if self.tsa_url else None,
        )
        try:
            return c2pa.Signer.from_info(info)
        except Exception as exc:
            raise CredentialsError(f"Could not load the signing certificate or key: {exc}") from exc


def _preview_u8(img: Any):
    import numpy as np

    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        arr = arr / 257.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _actions(report: Mapping[str, Any], parent_label: Optional[str] = None) -> List[Dict[str, Any]]:
    edits = report.get("edits", {}) or {}
    actions: List[Dict[str, Any]] = []
    # C2PA requires the first action to be "created" or "opened"; a retouch
    # always opens an existing photo.
    opened: Dict[str, Any] = {"action": "c2pa.opened"}
    if parent_label:
        opened["parameters"] = {"ingredientIds": [parent_label]}
    actions.append(opened)

    def add(action: str, description: str, cats: List[str]) -> None:
        settings = {}
        for cat in cats:
            settings.update((edits.get(cat) or {}).get("settings", {}))
        entry: Dict[str, Any] = {
            "action": action,
            "softwareAgent": {"name": "retouch", "version": str(report.get("app", {}).get("version", ""))},
            "description": description,
        }
        if settings:
            entry["parameters"] = {"settings": settings}
        actions.append(entry)

    if "shape" in edits:
        add("c2pa.edited", "Face or body shape changed", ["shape"])
    retouch_cats = [c for c in ("skin", "makeup", "eyes_teeth", "hair_costume", "light", "background") if c in edits]
    if retouch_cats:
        add("c2pa.edited", "Portrait retouching: " + ", ".join(
            edits[c]["label"].lower() for c in retouch_cats), retouch_cats)
    if "colour" in edits:
        add("c2pa.color_adjustments", "Colour, tone and finish", ["colour"])
    if "ai" in edits:
        add("c2pa.filtered", "AI-assisted, non-generative: " + "; ".join(report.get("ai_used", [])), ["ai"])
    if len(actions) == 1:
        add("c2pa.edited", "Processed with no active retouch settings", [])
    return actions


def build_manifest(report: Mapping[str, Any], title: str, mime: str,
                   parent_label: Optional[str] = None) -> Dict[str, Any]:
    """The manifest definition handed to the C2PA builder."""
    app = report.get("app", {})
    return {
        "claim_generator_info": [{"name": "retouch", "version": str(app.get("version", ""))}],
        "title": title,
        "format": mime,
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": _actions(report, parent_label)}},
            {"label": REPORT_ASSERTION, "data": dict(report)},
        ],
    }


def sign_output(
    output_path: Union[str, Path],
    report: Mapping[str, Any],
    config: SigningConfig,
    source_path: Optional[Union[str, Path]] = None,
    source_preview: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sign *output_path* in place with a new manifest describing *report*.

    The source photo is attached as the ``parentOf`` ingredient when its
    format is one the SDK reads; a camera manifest inside it is then kept as
    history. Otherwise *source_preview* (the decoded pre-retouch pixels, BGR)
    is attached in its place. Returns ``{"signed": True, "parent": bool, "parent_note": str}``.
    Raises :class:`CredentialsError` on any failure, leaving the unsigned
    output untouched.
    """
    c2pa = _c2pa()
    out = Path(output_path)
    mime = mime_for(out)
    if mime is None:
        raise CredentialsError(f"Content Credentials can't be embedded in {out.suffix} files")
    builder = c2pa.Builder(json.dumps(
        build_manifest(report, out.name, mime, parent_label=_PARENT_LABEL)
    ))
    parent = False
    parent_note = "No source photo given."
    ingredient = {"relationship": "parentOf", "label": _PARENT_LABEL}
    src = Path(source_path) if source_path is not None else None
    src_mime = sniff_mime(src) if src is not None else None
    if src is not None and src_mime is not None:
        try:
            with open(src, "rb") as fh:
                builder.add_ingredient(json.dumps({**ingredient, "title": src.name}), src_mime, fh)
            parent = True
            parent_note = "Source photo attached as the parent ingredient."
        except Exception as exc:
            parent_note = f"Source photo could not be read as a C2PA ingredient ({exc})"
            logger.warning("C2PA: %s", parent_note)
    elif src is not None:
        parent_note = f"Source format of {src.name} is not readable by the C2PA SDK"
    if not parent:
        # C2PA needs the "opened" action to name an ingredient. When the
        # original can't be read (RAW, HEIC), attach the decoded pre-retouch
        # pixels instead, and say so.
        if source_preview is None:
            raise CredentialsError(f"{parent_note}; nothing to record as the opened photo")
        import cv2

        ok, buf = cv2.imencode(".jpg", _preview_u8(source_preview), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise CredentialsError("Could not encode the source preview for signing")
        title = src.name if src is not None else "source"
        try:
            builder.add_ingredient(
                json.dumps({**ingredient, "title": f"{title} (decoded preview)"}),
                "image/jpeg", io.BytesIO(buf.tobytes()),
            )
        except Exception as exc:
            raise CredentialsError(f"Could not attach the source preview: {exc}") from exc
        parent_note += "; a decoded preview of it is the parent instead."

    signer = config.signer()
    tmp = out.with_name(f".{out.stem}.signing{out.suffix}")
    try:
        with open(out, "rb") as src_fh, open(tmp, "wb") as dst_fh:
            builder.sign(signer, mime, src_fh, dst_fh)
        os.replace(tmp, out)
    except CredentialsError:
        raise
    except Exception as exc:
        raise CredentialsError(f"Signing {out.name} failed: {exc}") from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return {"signed": True, "parent": parent, "parent_note": parent_note}


def read_credentials(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read and validate the Content Credentials in *path*.

    Returns ``None`` when the file carries none. Otherwise a dict with
    ``validation_state`` (``Valid``/``Trusted``/``Invalid``), the failure
    ``codes``, and the embedded edit ``report`` if there is one.
    """
    c2pa = _c2pa()
    try:
        reader = c2pa.Reader(str(path))
    except Exception as exc:
        text = str(exc)
        if "ManifestNotFound" in text or "JumbfNotFound" in text or "not found" in text.lower():
            return None
        raise CredentialsError(f"Could not read Content Credentials from {path}: {exc}") from exc
    data = json.loads(reader.json())
    active = data.get("manifests", {}).get(data.get("active_manifest", ""), {})
    report = None
    for assertion in active.get("assertions", []):
        if assertion.get("label") == REPORT_ASSERTION:
            report = assertion.get("data")
    codes = [s.get("code") for s in data.get("validation_status", []) or []]
    return {
        "validation_state": data.get("validation_state"),
        "codes": codes,
        "report": report,
        "ingredients": [i.get("title") for i in active.get("ingredients", [])],
        "manifest": data,
    }
