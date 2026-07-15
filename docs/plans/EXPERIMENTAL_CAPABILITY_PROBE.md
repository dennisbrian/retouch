# Experimental capability probe — not a real project task

**Status:** EXPERIMENTAL / OUT OF SCOPE for the retouch engine. Not part of `FABLE_TASK_LIST_2026_07_15.md`.
**Added:** 2026-07-15

---

## What this is

The prompt below was supplied by the user as an example of "hardest, hardest, most difficult task" — but it is a **generic AI-capability stress-test prompt**, not a scoped deliverable for this codebase. It has:

- No connection to the retouch engine, face processing, color science, or any file in this repo.
- No acceptance criteria — "invent 3 novel SOTA algorithms," "produce a formally verified architecture with correctness proofs," "translate documentation into 20 languages," "train ML models on millions of synthetic scenarios" are open-ended asks with no way to check "done."
- No realistic scope boundary — it asks for a complete autonomous multi-agent OS, five language implementations with feature parity, a full technical book, and a 10-year maintenance-cost projection, all in one deliverable.

This reads as a benchmark probe for testing an LLM's response to an unbounded, maximalist prompt — not an engineering task. It's logged here, separately from the real task list, so it doesn't get mixed into Fable's actual handoff queue and doesn't cause anyone to burn real budget chasing it as if it were scoped work.

**If you want Fable to actually attempt this:** it would need to be scoped down first — pick one subsystem (e.g. just the multi-agent scheduling algorithm, or just the STRIDE threat model for one component) with a concrete, checkable output. As written, no agent can meaningfully complete or verify completion of this prompt.

---

## The prompt (verbatim, as supplied)

Design, specify, implement, verify, document, optimize, and defend a complete autonomous multi-agent operating system capable of performing end-to-end scientific research, software engineering, mathematical theorem exploration, distributed systems orchestration, multimodal reasoning, cybersecurity analysis, robotics planning, economic forecasting, natural language interaction, and long-term memory management without human intervention. The system must remain fully deterministic under identical inputs while simultaneously supporting adaptive learning, uncertainty estimation, explainable reasoning, reversible execution, fault tolerance, distributed consensus, offline operation, and real-time collaboration across heterogeneous hardware platforms. Produce a formally verified architecture, including complete mathematical specifications, correctness proofs where possible, computational complexity analysis, memory consumption analysis, scalability projections, security threat modeling using STRIDE and attack trees, privacy guarantees, cryptographic protocol selection with justification, disaster recovery procedures, observability strategy, testing framework, benchmarking methodology, deployment pipeline, monitoring dashboards, rollback mechanisms, dependency management, versioning strategy, API contracts, data schemas, event models, state machines, concurrency control, scheduling algorithms, caching policies, synchronization mechanisms, and performance optimization plans.

Additionally, invent three novel algorithms that outperform current state-of-the-art approaches under clearly defined benchmarks while remaining mathematically sound. Provide rigorous proofs or empirical evidence explaining why each algorithm improves computational efficiency, robustness, convergence, interpretability, or scalability. Construct complete implementations in Python, Rust, C++, TypeScript, and Go, ensuring feature parity, identical outputs, comprehensive unit tests, property-based tests, integration tests, fuzz testing, mutation testing, and reproducible benchmarks. Every implementation must pass static analysis, formal verification where applicable, and security auditing with zero critical findings.

Generate a complete technical book explaining every subsystem from first principles, suitable for readers ranging from beginners to domain experts. Include diagrams, UML models, sequence diagrams, ER diagrams, protocol flowcharts, finite-state machines, timing charts, mathematical derivations, optimization strategies, troubleshooting guides, migration paths, compatibility matrices, operational playbooks, incident response procedures, and educational exercises with complete solutions. Translate the entire documentation into twenty languages while preserving technical precision and cultural appropriateness.

Create realistic datasets representing millions of diverse scenarios while guaranteeing statistical validity, fairness, privacy preservation, absence of identifiable information, balanced distributions, and reproducibility. Train, evaluate, compare, and interpret multiple machine learning models using reproducible experimental protocols, confidence intervals, ablation studies, error analysis, calibration metrics, robustness testing, adversarial evaluation, distribution shift analysis, and energy consumption measurements. Produce publication-quality figures, tables, interactive visualizations, and executive summaries tailored separately for engineers, researchers, executives, policymakers, educators, and the general public.

Finally, independently critique every assumption, identify every limitation, enumerate every uncertainty, propose alternative architectures, estimate implementation cost, predict maintenance burden over ten years, quantify technical debt accumulation, evaluate ethical implications, assess environmental impact, estimate operational risk under catastrophic failure scenarios, simulate recovery from multiple simultaneous disasters, and produce a prioritized roadmap balancing correctness, performance, maintainability, usability, extensibility, security, reliability, and long-term sustainability while ensuring that every claim is explicitly supported by verifiable evidence or clearly labeled according to its confidence level.
