from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.experiments import load_experiments, validate_experiments


EXPERIMENT_BODY = """# Question

What happens?

# Change

Change one thing.

# Prediction

State the expected result.

# Result

No run yet.

# Decision

Keep proposed.

# Revisit when

Run when the adapter is ready.
"""


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
schema: autoresearch-experiment/v1
id: experiment.baseline
title: Baseline
kind: baseline
status: proposed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs: []
contract: evals/contracts/baseline-v1.json
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

            self.assertTrue(result.ok, result.issues)
            self.assertEqual(result.experiment_count, 1)
            self.assertEqual(issues, [])
            self.assertIn("experiment.baseline", experiments)

    def test_optuna_requires_declared_search_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = root / "research" / "experiments" / "experiment.search"
            directory.mkdir(parents=True)
            (root / "research.md").write_text("# Project\n")
            (root / "scripts").mkdir()
            (root / "scripts" / "optimize.py").write_text("print('search')\n")
            (directory / "experiment.md").write_text(
                """---
schema: autoresearch-experiment/v1
id: experiment.search
title: Parameter search
kind: parameterization
status: proposed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs: []
contract: evals/contracts/baseline-v1.json
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


if __name__ == "__main__":
    unittest.main()
