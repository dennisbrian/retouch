# P2 Safe Auto Contract — 2026-08-12

`retouch/safe_auto.py` adds a processor-agnostic decision contract:

- `apply` keeps the candidate output;
- `dampen` blends a conservative strength into the source;
- `review` returns the unchanged source and records the reason;
- `skip` returns a byte-identical copy of the source.

The decision records stage, confidence, reason, and evidence. Mask stages can
combine mask coverage, landmark stability, model confidence, and occlusion.
The module does not pretend that one confidence threshold fits every stage;
callers provide stage-specific thresholds.

The remaining P2 work is to wire this contract into each automatic facial,
blemish, glasses, body, and optional-model stage, then collect risk-versus-
coverage evidence on the certification corpus.
