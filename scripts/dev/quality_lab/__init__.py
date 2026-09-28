"""Autonomous Image Quality & Regression Lab.

Tests verify software correctness; this lab answers the other question:
"did the photo actually get better / stay the same?" It renders a tagged
benchmark corpus through recipes, measures region-aware deltas against a
frozen baseline, and triages each case PASS / REVIEW / REGRESSION so the
human owner inspects only the images that need it.

See docs/QUALITY_LAB.md.
"""
