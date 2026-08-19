from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.experiments import load_experiments, load_framings, validate_experiments


FRAMING = """---
schema: autoresearch-framing/v1
id: framing.test
title: Test framing
status: active
created_at: 2026-08-18T00:00:00Z
---

# Question

What research question groups these experiments?

# Scope

Only the test experiments.

# Current position

The framing is active.

# Revisit when

Revisit when the question changes.
"""


EXPERIMENT_BODY = """# Question

What happens?

# Rationale

Explain why this is the next decision-relevant experiment.

# Change

Change one thing.

# Prediction

State the expected result.

# Plan

State the evidence, failure modes, cost, and stopping rule.

# Result

No run yet.

# Evidence

No recorded evidence yet.

# Interpretation

No interpretation is available before a result exists.

# Decision

Keep proposed.

# Revisit when

Run when the adapter is ready.
"""


def write_framing(root: Path) -> None:
    framing_root = root / "research" / "framings"
    framing_root.mkdir(parents=True)
    (framing_root / "framing.test.md").write_text(FRAMING)


class ExperimentValidationTests(unittest.TestCase):
    def test_valid_experiment_dag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "research" / "experiments" / "experiment.baseline"
            directory.mkdir(parents=True)
            (root / "research.md").write_text("# Project\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "run_model.py").write_text("print('run')\n")
            (directory / "experiment.md").write_text(
                """---
schema: autoresearch-experiment/v2
id: experiment.baseline
title: Baseline
kind: baseline
status: proposed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs: []
contract: evals/contracts/baseline-eval-v1.json
execution:
  mode: simulation
  tool: direct
  adapter: scripts/run_model.py
  argv: ["{python}", "scripts/run_model.py"]
search: null
---

""" + EXPERIMENT_BODY
            )

            result = validate_experiments(root)
            experiments, issues = load_experiments(root)
            framings, framing_issues = load_framings(root)

            self.assertTrue(result.ok, result.issues)
            self.assertEqual(result.framing_count, 0)
            self.assertEqual(result.experiment_count, 1)
            self.assertEqual(issues, [])
            self.assertEqual(framing_issues, [])
            self.assertEqual(framings, {})
            self.assertIn("experiment.baseline", experiments)

    def test_optuna_requires_declared_search_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "research" / "experiments" / "experiment.search"
            directory.mkdir(parents=True)
            (root / "research.md").write_text("# Project\n")
            write_framing(root)
            (root / "scripts").mkdir()
            (root / "scripts" / "optimize.py").write_text("print('search')\n")
            (directory / "experiment.md").write_text(
                """---
schema: autoresearch-experiment/v2
id: experiment.search
title: Parameter search
kind: parameterization
status: proposed
created_at: 2026-08-18T00:00:00Z
framing: framing.test
parents: []
inputs: []
contract: evals/contracts/baseline-eval-v1.json
execution:
  mode: mechanistic
  tool: optuna
  adapter: scripts/optimize.py
  argv: ["{python}", "scripts/optimize.py"]
search: null
---

""" + EXPERIMENT_BODY
            )

            result = validate_experiments(root)

            self.assertFalse(result.ok)
            self.assertIn("invalid-optuna-search", {issue.code for issue in result.issues})

    def test_completed_experiment_requires_selected_run_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "research" / "experiments" / "experiment.completed"
            run = directory / "runs" / "run.20260818T120000Z.example"
            (run / "figures").mkdir(parents=True)
            (run / "figures" / "summary.png").write_bytes(b"png")
            (run / "metrics.json").write_text("{}\n")
            (root / "research.md").write_text("# Project\n")
            write_framing(root)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_model.py").write_text("print('run')\n")
            body = EXPERIMENT_BODY.replace(
                "No recorded evidence yet.",
                """![Summary](runs/run.20260818T120000Z.example/figures/summary.png)

[Metrics](runs/run.20260818T120000Z.example/metrics.json)""",
            )
            (directory / "experiment.md").write_text(
                """---
schema: autoresearch-experiment/v2
id: experiment.completed
title: Completed experiment
kind: baseline
status: completed
created_at: 2026-08-18T00:00:00Z
framing: framing.test
parents: []
inputs: []
contract: evals/contracts/baseline-eval-v1.json
execution:
  mode: mechanistic
  tool: direct
  adapter: scripts/run_model.py
  argv: ["{python}", "scripts/run_model.py"]
search: null
selected_run: run.20260818T120000Z.example
---

"""
                + body
            )

            result = validate_experiments(root)

            self.assertTrue(result.ok, result.issues)

    def test_completed_experiment_rejects_unlinked_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "research" / "experiments" / "experiment.completed"
            run = directory / "runs" / "run.20260818T120000Z.example"
            run.mkdir(parents=True)
            (root / "research.md").write_text("# Project\n")
            write_framing(root)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_model.py").write_text("print('run')\n")
            (directory / "experiment.md").write_text(
                """---
schema: autoresearch-experiment/v2
id: experiment.completed
title: Completed experiment
kind: baseline
status: completed
created_at: 2026-08-18T00:00:00Z
framing: framing.test
parents: []
inputs: []
contract: evals/contracts/baseline-eval-v1.json
execution:
  mode: mechanistic
  tool: direct
  adapter: scripts/run_model.py
  argv: ["{python}", "scripts/run_model.py"]
search: null
selected_run: run.20260818T120000Z.example
---

"""
                + EXPERIMENT_BODY
            )

            result = validate_experiments(root)
            codes = {issue.code for issue in result.issues}

            self.assertFalse(result.ok)
            self.assertIn("missing-evidence-figure", codes)
            self.assertIn("missing-evidence-result", codes)


if __name__ == "__main__":
    unittest.main()
