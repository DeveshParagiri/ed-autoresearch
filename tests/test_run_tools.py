from __future__ import annotations

import unittest
from pathlib import Path

from autoresearch.run_tools import build_command, metric, numeric_metrics, shapley_values


class RunToolTests(unittest.TestCase):
    def test_shapley_values_split_an_interaction(self) -> None:
        values = {
            frozenset(): 0.0,
            frozenset({"fuel"}): 1.0,
            frozenset({"human"}): 2.0,
            frozenset({"fuel", "human"}): 4.0,
        }
        result = shapley_values(["fuel", "human"], values)
        self.assertEqual(result, {"fuel": 1.5, "human": 2.5})
        self.assertEqual(sum(result.values()), 4.0)

    def test_build_command_replaces_paths_without_a_shell(self) -> None:
        result = build_command(
            ["python", "builder.py", "--params={parameters}", "{candidate}"],
            values={"parameters": Path("p.json"), "candidate": Path("candidate.nc")},
        )
        self.assertEqual(
            result,
            ["python", "builder.py", "--params=p.json", "candidate.nc"],
        )

    def test_metric_helpers_read_evaluation_output(self) -> None:
        values = {"candidate": {"gfed5": {"overall_score": 0.7}}, "valid": True}
        self.assertEqual(
            numeric_metrics(values),
            {"candidate.gfed5.overall_score": 0.7},
        )
        self.assertEqual(metric(values, "candidate.gfed5.overall_score"), 0.7)


if __name__ == "__main__":
    unittest.main()
