# P4 Personal Style Instrumentation — 2026-08-12

`Session.record_style_event()` stores only scalar/string/bool metadata:

- stage name;
- suggested parameters;
- final user parameters;
- accepted, rejected, skipped, or review outcome;
- optional scene-analysis features.

Image arrays and face pixels are not accepted into the serialized event. The
events round-trip with sessions and remain separate from the future learner.
The next P4 step is an inspectable delta model with consent and retention
controls; it must not generate or store a new face representation.
