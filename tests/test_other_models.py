from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from scripts.burned_area_figures import (
    BASIER_REGULAR,
    BASIER_SEMIBOLD,
    FIGURE_NAMES,
    FONT_FAMILY,
    FONT_SHA256,
)


class HistoricalModelArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.archive = cls.root / "model" / "other-models"
        cls.registry = tomllib.loads(
            (cls.archive / "registry.toml").read_text()
        )

    def test_registry_names_every_final_historical_variant(self) -> None:
        model_ids = [model["id"] for model in self.registry["models"]]

        self.assertEqual(
            model_ids,
            [
                "A-legacy",
                "B-legacy",
                "C-legacy",
                "C",
                "D",
                "E",
                "F",
                "G",
                "G6",
                "G7",
                "H",
                "I",
                "Ibest",
            ],
        )

    def test_parameter_inventory_covers_every_retained_json(self) -> None:
        parameter_root = self.archive / "parameters"
        files = sorted(parameter_root.rglob("*.json"))
        with (self.archive / "parameter-inventory.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_path = {row["path"]: row for row in rows}

        self.assertEqual(len(files), 303)
        self.assertEqual(len(rows), len(files))
        for path in files:
            relative = path.relative_to(self.root).as_posix()
            self.assertIn(relative, by_path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(by_path[relative]["sha256"], digest)

    def test_every_registry_parameter_reference_resolves(self) -> None:
        for model in self.registry["models"]:
            for path in model.get("parameter_files", []):
                self.assertTrue((self.root / path).is_file(), path)
            patterns = []
            if "parameter_pattern" in model:
                patterns.append(model["parameter_pattern"])
            patterns.extend(model.get("parameter_patterns", []))
            for pattern in patterns:
                self.assertTrue(list(self.root.glob(pattern)), pattern)

    def test_standardized_figures_match_the_locked_contract(self) -> None:
        contract = json.loads(
            (self.root / "evals" / "contracts" / "burned-area-eval-v2.json").read_text()
        )
        names = tuple(figure["filename"] for figure in contract["figures"])
        dimensions = {
            (figure["dimensions"]["width"], figure["dimensions"]["height"])
            for figure in contract["figures"]
        }
        project = tomllib.loads((self.root / "pyproject.toml").read_text())

        self.assertEqual(names, FIGURE_NAMES)
        self.assertEqual(dimensions, {(1800, 800), (1800, 1200)})
        self.assertEqual(
            contract["evaluation"]["plot_scales"],
            {
                "annual_percent_max": 80.0,
                "residual_percent_abs": 60.0,
                "seasonal_percent_max_by_region": {
                    "global": 1.2,
                    "africa": 3.0,
                    "south_america": 2.4,
                    "north_america": 1.2,
                    "boreal_eurasia": 1.2,
                    "tropical_se_asia": 3.0,
                    "australia": 1.5,
                    "europe": 2.1,
                },
                "scatter_percent_max": 80.0,
                "regional_bias_percent_abs": 25.0,
                "score_difference_abs": 0.15,
                "global_mean_percent_max": 1.0,
            },
        )
        self.assertEqual(
            {figure["generator"] for figure in contract["figures"]},
            {"scripts/evaluate_burned_area_v2.py"},
        )
        self.assertIn(
            "SciencePlots==2.2.2",
            project["project"]["optional-dependencies"]["historical"],
        )

    def test_figure_fonts_are_bundled_for_fresh_clones(self) -> None:
        self.assertEqual(FONT_FAMILY, "Basier Square")
        for path in (BASIER_REGULAR, BASIER_SEMIBOLD):
            self.assertTrue(path.is_relative_to(self.root), path)
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.read_bytes()[:4], b"OTTO")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                FONT_SHA256[path.name],
            )

    def test_git_bundle_is_complete_and_valid(self) -> None:
        bundle = self.archive / "upstream" / "ed-autoresearch.bundle"
        completed = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("complete history", completed.stdout + completed.stderr)

    def test_commit_history_archive_matches_every_audited_ref(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/sync_other_model_history.py"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Historical model artifact coverage: PASS", completed.stdout)

        artifact_root = self.archive / "commit-artifacts"
        coverage = json.loads((artifact_root / "coverage.json").read_text())
        self.assertTrue(coverage["coverage_complete"])
        self.assertEqual(coverage["commit_count"], 115)
        self.assertEqual(coverage["source_path_count"], 381)
        self.assertEqual(coverage["path_blob_version_count"], 406)
        self.assertEqual(coverage["unique_blob_count"], 373)
        self.assertEqual(coverage["materialized_file_count"], 406)
        self.assertEqual(coverage["json_attempt_record_count"], 366)
        self.assertEqual(
            {
                name: entry["tip"]
                for name, entry in coverage["audited_refs"].items()
            },
            {
                "main": "222b8569268a65566de0073a5f84dcbb2028da12",
                "coupled-refit-gfed5": (
                    "11ee71418e597e977a4d49f6fda166e20c098e9f"
                ),
                "modelD-paper-params": (
                    "32d283ff595d6653dfd84c076f418a82074b0d26"
                ),
            },
        )

    def test_commit_manifest_resolves_every_path_blob_version(self) -> None:
        artifact_root = self.archive / "commit-artifacts"
        with (artifact_root / "manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with (artifact_root / "model-attempts.csv").open(newline="") as handle:
            attempts = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 406)
        self.assertEqual(len(attempts), 366)
        self.assertEqual(
            len({(row["source_path"], row["git_blob"]) for row in rows}),
            len(rows),
        )
        self.assertEqual(
            {row["model_family"] for row in rows},
            {
                "A",
                "A-B-C-attribution",
                "B",
                "C-development",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "combustion",
                "coupled-E",
                "coupled-handoff",
                "curing",
                "initial-fire-model",
            },
        )

        source_versions: dict[str, int] = {}
        for row in rows:
            path = self.root / row["archive_path"]
            self.assertTrue(path.is_file(), row["archive_path"])
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            git_header = f"blob {len(data)}\0".encode()
            self.assertEqual(
                hashlib.sha1(git_header + data).hexdigest(), row["git_blob"]
            )
            source_versions[row["source_path"]] = (
                source_versions.get(row["source_path"], 0) + 1
            )

        self.assertEqual(source_versions["models/A/params.json"], 2)
        self.assertEqual(source_versions["models/B/params.json"], 2)
        self.assertEqual(source_versions["models/C/params.json"], 6)
        self.assertEqual(source_versions["models/paper/D.json"], 2)
        self.assertEqual(source_versions["patches/fire_modelC.cc"], 3)
        self.assertIn("data_human/coupling_inputs/gdp_gamma.nc", source_versions)
        self.assertIn("patches/belowgrnd.cc.patch", source_versions)


if __name__ == "__main__":
    unittest.main()
