# Shoot Intelligence Culling and Watch-Folder Slice — 2026-08-12

## Assisted culling

`rank_burst_candidates()` ranks frames only by capture-quality evidence:
sharpness, exposure, clipping, and resolution. It reports the scoring policy
and each component score. It does not inspect identity, facial expression,
eyes, hands, hair, or likeness, so every result remains `review_required`.

No source is deleted or automatically rejected.

## Watch-folder ingestion

`WatchFolder` is a polling service with:

- two-scan size/mtime stability before processing;
- persistent atomic JSON state;
- resumable `pending`, `processing`, `done`, and `failed` records;
- retryable callback failures and attempt counts;
- recursive or non-recursive scanning and a processing limit.

It intentionally does not start a daemon or mutate source files. The GUI now
exposes scan/watch operations in the Shoot Intelligence tab. Candidate ranking
remains a recommendation until the human/corpus acceptance gates are complete.

Remaining work is the corpus-backed review experience and the later aligned
burst fusion stage with temporal QA.
