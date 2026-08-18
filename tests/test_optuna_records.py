from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autoresearch.optuna_records import export_study, run_local_storage


class Named:
    def __init__(self, name: str) -> None:
        self.name = name


class Trial:
    def __init__(self, number: int, value: float) -> None:
        self.number = number
        self.state = Named("COMPLETE")
        self.values = [value]
        self.params = {"rate": value}
        self.user_attrs = {"fold": 1}
        self.intermediate_values = {0: value + 1}
        self.datetime_start = datetime(2026, 8, 18, tzinfo=timezone.utc)
        self.datetime_complete = datetime(2026, 8, 18, 0, 1, tzinfo=timezone.utc)
        self.duration = timedelta(minutes=1)


class FakeStudy:
    study_name = "fixture"
    directions = [Named("MINIMIZE")]
    sampler = type("TPESampler", (), {})()
    pruner = type("MedianPruner", (), {})()
    user_attrs = {"purpose": "test"}

    def get_trials(self, deepcopy: bool = False) -> list[Trial]:
        return [Trial(0, 0.8), Trial(1, 0.4)]


class OptunaRecordTests(unittest.TestCase):
    def test_export_preserves_study_trials_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            study, trials, selected = export_study(
                FakeStudy(),
                run_root,
                selected_trial_number=1,
                selection_rule="lowest development loss",
            )

            self.assertEqual(json.loads(study.read_text())["trial_count"], 2)
            self.assertEqual(len(trials.read_text().splitlines()), 2)
            self.assertEqual(json.loads(selected.read_text())["parameters"], {"rate": 0.4})
            self.assertTrue(run_local_storage(run_root).endswith("/artifacts/optuna-study.db"))


if __name__ == "__main__":
    unittest.main()
