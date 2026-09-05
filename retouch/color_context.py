"""Explicit color-management contract for image ingest and delivery.

The Retouch engine works in display-referred sRGB.  A :class:`ColorContext`
keeps that working-space fact separate from the profile that described the
source pixels, so an export can never accidentally attach the source profile
to pixels that have already been converted to sRGB.

The raw ICC bytes are retained for an explicit source-profile export, while
``to_dict()`` deliberately emits only compact, non-pixel provenance suitable
for manifests or session diagnostics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


COLOR_CONTEXT_SCHEMA = "retouch.color-context.v1"
WORKING_SPACE_SRGB = "sRGB"

SOURCE_EMBEDDED_ICC = "embedded-icc"
SOURCE_ASSUMED_SRGB = "assumed-srgb"
SOURCE_RAW_SRGB = "raw-srgb"

_SOURCE_KINDS = {
    SOURCE_EMBEDDED_ICC,
    SOURCE_ASSUMED_SRGB,
    SOURCE_RAW_SRGB,
}


@dataclass(frozen=True)
class ColorContext:
    """Describe the source and working color spaces for one image.

    Attributes:
        working_space: Engine working space. Retouch currently requires sRGB.
        working_profile: ICC bytes for the working space when LittleCMS is
            available. It is optional so an untagged/diagnostic-only context
            can still be represented in a reduced Pillow installation.
        source_profile: Embedded source ICC bytes, if the input was tagged.
            ``None`` means there is no source profile to restore.
        source_kind: ``embedded-icc`` for tagged input, ``assumed-srgb`` for
            untagged non-RAW input, or ``raw-srgb`` for a RAW decoder that
            explicitly requested an sRGB rendition.
        conversion_applied: Whether ingest actually transformed source pixels
            into the working profile. An embedded sRGB profile may be tagged
            without requiring a non-identity transform.
        source_profile_name: Best-effort human-readable ICC description.
    """

    working_space: str = WORKING_SPACE_SRGB
    working_profile: Optional[bytes] = None
    source_profile: Optional[bytes] = None
    source_kind: str = SOURCE_ASSUMED_SRGB
    conversion_applied: bool = False
    source_profile_name: Optional[str] = None
    source_bit_depth: Optional[int] = None
    working_bit_depth: Optional[int] = None
    transform_intent: Optional[str] = None
    black_point_compensation: Optional[bool] = None
    alpha_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.working_space != WORKING_SPACE_SRGB:
            raise ValueError(
                "Retouch ColorContext currently supports only the sRGB working space"
            )
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(
                f"Unsupported ColorContext source_kind: {self.source_kind!r}"
            )
        if self.source_kind == SOURCE_EMBEDDED_ICC and not self.source_profile:
            raise ValueError("embedded-icc ColorContext requires source_profile bytes")
        if self.source_kind != SOURCE_EMBEDDED_ICC and self.source_profile is not None:
            raise ValueError(
                "Only embedded-icc ColorContext values may carry a source profile"
            )
        if self.working_profile is not None:
            object.__setattr__(self, "working_profile", bytes(self.working_profile))
        if self.source_profile is not None:
            object.__setattr__(self, "source_profile", bytes(self.source_profile))

    @property
    def assumed_srgb(self) -> bool:
        """Whether the source had no embedded profile and was assumed sRGB."""
        return self.source_kind == SOURCE_ASSUMED_SRGB

    @property
    def is_tagged(self) -> bool:
        """Whether the source carried an embedded ICC profile."""
        return self.source_kind == SOURCE_EMBEDDED_ICC

    @property
    def source_profile_sha256(self) -> Optional[str]:
        """SHA-256 of the source ICC bytes, or ``None`` when untagged."""
        if self.source_profile is None:
            return None
        return hashlib.sha256(self.source_profile).hexdigest()

    @property
    def working_profile_sha256(self) -> Optional[str]:
        """SHA-256 of the working ICC bytes when available."""
        if self.working_profile is None:
            return None
        return hashlib.sha256(self.working_profile).hexdigest()

    def profile_for_export(self, preserve_source_profile: bool = False) -> Optional[bytes]:
        """Return the ICC profile that should describe an exported image.

        The default always describes the engine's sRGB working pixels.  The
        source profile is selected only when ``preserve_source_profile=True``
        and an actual embedded source profile exists.
        """
        if preserve_source_profile and self.source_profile is not None:
            return self.source_profile
        return self.working_profile

    def to_dict(self) -> Dict[str, Any]:
        """Return compact provenance without embedding raw ICC payloads."""
        return {
            "schema": COLOR_CONTEXT_SCHEMA,
            "working_space": self.working_space,
            "source_kind": self.source_kind,
            "assumed_srgb": self.assumed_srgb,
            "is_tagged": self.is_tagged,
            "conversion_applied": bool(self.conversion_applied),
            "source_profile_name": self.source_profile_name,
            "source_profile_sha256": self.source_profile_sha256,
            "working_profile_sha256": self.working_profile_sha256,
            "source_bit_depth": self.source_bit_depth,
            "working_bit_depth": self.working_bit_depth,
            "transform_intent": self.transform_intent,
            "black_point_compensation": self.black_point_compensation,
            "alpha_mode": self.alpha_mode,
        }

    @classmethod
    def from_embedded_profile(
        cls,
        source_profile: bytes,
        working_profile: Optional[bytes],
        *,
        conversion_applied: bool,
        source_profile_name: Optional[str] = None,
    ) -> "ColorContext":
        """Build a context for a source image carrying an embedded ICC."""
        return cls(
            working_profile=working_profile,
            source_profile=bytes(source_profile),
            source_kind=SOURCE_EMBEDDED_ICC,
            conversion_applied=conversion_applied,
            source_profile_name=source_profile_name,
        )

    @classmethod
    def assumed_srgb_context(
        cls,
        working_profile: Optional[bytes],
    ) -> "ColorContext":
        """Build a context for an untagged image treated as sRGB."""
        return cls(
            working_profile=working_profile,
            source_kind=SOURCE_ASSUMED_SRGB,
            conversion_applied=False,
        )

    @classmethod
    def raw_srgb_context(
        cls,
        working_profile: Optional[bytes],
    ) -> "ColorContext":
        """Build a context for a RAW decoder that emits sRGB pixels."""
        return cls(
            working_profile=working_profile,
            source_kind=SOURCE_RAW_SRGB,
            conversion_applied=False,
        )
