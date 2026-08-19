from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.catalog import validate_catalog, validate_sources


class CatalogValidationTests(unittest.TestCase):
    def test_workspace_source_manifest_is_complete_and_canonical(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        self.assertEqual(validate_sources(project_root), ())

    def test_required_and_optional_datasets_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            (root / "data" / "present.txt").write_text("evidence")
            (root / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "required.present"
category = "inputs"
role = "baseline-input"
required_for = ["offline-baseline"]
path = "data/present.txt"
kind = "file"
required = true

[[datasets]]
id = "optional.missing"
category = "benchmarks"
role = "candidate-input"
required_for = ["candidate-experiment"]
path = "data/missing.txt"
kind = "file"
required = false
""".strip()
            )

            result = validate_catalog(root)

            self.assertTrue(result.ok)
            self.assertEqual(result.present_count, 1)
            self.assertEqual(result.optional_missing_count, 1)
            self.assertEqual(result.warning_count, 1)

    def test_file_size_check_rejects_partial_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data" / "downloads").mkdir(parents=True)
            (root / "data" / "downloads" / "large.nc").write_bytes(b"partial")
            (root / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "required.download"
category = "inputs"
role = "reconstruction-source"
required_for = ["reference-reconstruction"]
path = "data/downloads"
kind = "directory"
required = true
checks = [{ type = "file_size", path = "large.nc", bytes = 100 }]
""".strip()
            )

            result = validate_catalog(root)

            self.assertFalse(result.ok)
            self.assertEqual(result.error_count, 1)
            self.assertEqual(result.issues[0].code, "size-mismatch")

    def test_catalog_rejects_a_path_outside_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            (root / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "required.escape"
category = "inputs"
role = "baseline-input"
required_for = ["baseline"]
path = "../outside.nc"
kind = "file"
required = true
description = "Invalid external input."
""".strip()
            )

            result = validate_catalog(root)

            self.assertFalse(result.ok)
            self.assertEqual(result.issues[0].code, "invalid-path")

    def test_source_manifest_rejects_an_external_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            (root / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "input.test"
category = "inputs"
role = "baseline-input"
required_for = ["baseline"]
path = "data/inputs/test"
kind = "file"
required = true
description = "Test input."
""".strip()
            )
            (root / "data" / "sources.toml").write_text(
                """
schema = "data-sources/v1"

[store]
environment_variable = "TEST_DATA_ROOT"
default_relative_to_project = "../data-store"

[[sources]]
id = "input.test"
source_path = "payload/test"
acquisition = "lab-seed"
source_name = "Test input"
version = "1"
retrieval = "Place the input in the project."
time_coverage = "Test period"
spatial_coverage = "Test grid"
units = "Dimensionless"
preprocessing = "None"
limitations = "Fixture only"
license = "Test only"
integrity = "Fixture identity"
experiments = []
""".strip()
            )

            result = validate_sources(root)
            codes = {issue.code for issue in result}

            self.assertIn("noncanonical-data-root", codes)
            self.assertIn("external-data-root", codes)
            self.assertIn("noncanonical-source-path", codes)


if __name__ == "__main__":
    unittest.main()
