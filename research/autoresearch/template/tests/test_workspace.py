from __future__ import annotations

import unittest
from pathlib import Path

from autoresearch.validation import load_experiments, validate_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_EXPERIMENT = "experiment.baseline"


class GeneratedWorkspaceTests(unittest.TestCase):
    def test_zero_run_scaffold_is_valid(self) -> None:
        report = validate_workspace(PROJECT_ROOT)

        self.assertTrue(report.ok, report.as_dict())
        self.assertEqual(report.dataset_count, 0)
        self.assertEqual(report.experiment_count, 1)
        self.assertEqual(report.run_count, 0)

    def test_initial_experiment_is_the_only_root_node(self) -> None:
        experiments, issues = load_experiments(PROJECT_ROOT)

        self.assertFalse([issue for issue in issues if issue.severity == "error"], issues)
        self.assertEqual(set(experiments), {ROOT_EXPERIMENT})
        self.assertEqual(experiments[ROOT_EXPERIMENT].metadata["parents"], [])


if __name__ == "__main__":
    unittest.main()
