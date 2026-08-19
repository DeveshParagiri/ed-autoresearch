# Meeting intake

This directory receives research feedback supplied as a transcript, Granola export, or ordinary notes. A meeting is context for research decisions. It is not an experiment, graph node, run, benchmark, or substitute for the project’s authoritative files.

## Private storage

For each meeting, create `research/meetings/private/meeting.YYYYMMDD.<slug>/`. Preserve the supplied material without rewriting it as `source.<ext>`, then write a structured `record.md` beside it. The entire `private/` directory is ignored by Git. Raw meeting material and private records must not be committed, pasted into a public issue, or exposed by the viewer.

Use this front matter in the private record:

```yaml
---
schema: autoresearch-meeting/v1
id: meeting.YYYYMMDD.slug
date: YYYY-MM-DD
source_type: transcript
related_experiments: []
---
```

Use `granola`, `transcript`, or `notes` for `source_type`. The body has Context, Explicit feedback, Decisions, Actions, Open questions, and Canonical updates sections. Keep statements made in the meeting separate from later interpretation. Record uncertainty instead of filling gaps in the source.

## Applying feedback

After the private record is checked, update the file that actually owns each consequence. Project-wide scientific rules belong in `research.md`; the active handoff belongs in `memory.md`; a branch-level position belongs in an existing framing when one applies; an experiment’s rationale, result, interpretation, decision, or revisit condition belongs in its `experiment.md`; and a changed evaluation rule requires a reviewed contract version.

The graph continues to contain experiments only. An experiment may say that researcher feedback motivated it, but its Rationale must remain understandable without access to a private meeting record. Nothing from a meeting becomes public merely because it was supplied to the autoresearch agent.
