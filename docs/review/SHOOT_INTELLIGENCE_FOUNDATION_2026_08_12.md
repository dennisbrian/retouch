# Shoot Intelligence Foundation — 2026-08-12

Implemented `retouch/shoot_intelligence.py` as a non-destructive foundation:

- `ProjectGraph` stores ingest, cull, grouping, processing, and export-style
  dependencies with pending/ready/running/succeeded/failed/blocked/skipped
  states.
- Missing dependencies and cycles fail closed.
- `inspect_asset()` captures image dimensions, EXIF camera/lens facts, capture
  time, and a small perceptual fingerprint.
- `group_bursts()` groups only close-timestamp, camera/lens-matching,
  perceptually similar assets. Missing timestamps never create a burst solely
  from filename order.
- Every burst is marked `review_required`; no source is deleted, rejected, or
  modified.
- `rank_burst_candidates()` provides a capture-quality-only recommendation
  using sharpness, exposure, clipping, and resolution evidence. It does not
  inspect identity, expression, eyes, hands, or hair.
- `WatchFolder` provides polling ingestion with two-scan file stability,
  atomic resumable state, retryable failures, and an optional processing
  callback. It never processes a file while its size/mtime is still changing.

Remaining work is the product integration: assisted culling UI, watch-folder
GUI, subject-linked project profiles, Look Board, and the later aligned burst
fusion stage with temporal QA. Candidate ranking remains a recommendation
until the human/corpus acceptance gates are complete.
