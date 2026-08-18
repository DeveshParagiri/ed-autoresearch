from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.data_install import install_links, load_sources, validate_manifest_coverage


class DataInstallTests(unittest.TestCase):
    def test_workspace_source_manifest_covers_catalog(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        validate_manifest_coverage(project_root)

    def test_installer_creates_idempotent_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "project"
            source = root / "source"
            (project / "data").mkdir(parents=True)
            (source / "payload").mkdir(parents=True)
            (source / "payload" / "value.txt").write_text("evidence")
            (project / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "input.test"
category = "inputs"
role = "baseline-input"
required_for = ["offline-baseline"]
path = "data/inputs/test"
kind = "directory"
required = true
""".strip()
            )
            (project / "data" / "sources.toml").write_text(
                """
schema = "data-sources/v1"

[store]
environment_variable = "TEST_DATA_SOURCE_ROOT"
default_relative_to_project = "../source"

[[sources]]
id = "input.test"
source_path = "payload"
acquisition = "lab-seed"
source_name = "Test fixture"
version = "1"
retrieval = "Supply the fixture source."
time_coverage = "test period"
spatial_coverage = "test grid"
units = "dimensionless"
preprocessing = "none"
limitations = "test fixture only"
license = "test only"
integrity = "fixture contents"
experiments = []
""".strip()
            )

            first = install_links(project, source)
            second = install_links(project, source)

            destination = project / "data" / "inputs" / "test"
            self.assertTrue(destination.is_symlink())
            self.assertEqual(first[0].status, "linked")
            self.assertEqual(second[0].status, "ready")

    def test_installer_does_not_replace_regular_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "project"
            source = root / "source"
            (project / "data" / "inputs" / "test").mkdir(parents=True)
            (source / "payload").mkdir(parents=True)
            (project / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "input.test"
category = "inputs"
role = "baseline-input"
required_for = ["offline-baseline"]
path = "data/inputs/test"
kind = "directory"
required = true
""".strip()
            )
            (project / "data" / "sources.toml").write_text(
                """
schema = "data-sources/v1"

[store]
environment_variable = "TEST_DATA_SOURCE_ROOT"
default_relative_to_project = "../source"

[[sources]]
id = "input.test"
source_path = "payload"
acquisition = "lab-seed"
source_name = "Test fixture"
version = "1"
retrieval = "Supply the fixture source."
time_coverage = "test period"
spatial_coverage = "test grid"
units = "dimensionless"
preprocessing = "none"
limitations = "test fixture only"
license = "test only"
integrity = "fixture contents"
experiments = []
""".strip()
            )

            result = install_links(project, source)

            self.assertEqual(result[0].status, "conflict")
            self.assertFalse((project / "data" / "inputs" / "test").is_symlink())

    def test_installer_reports_optional_missing_without_required_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "project"
            source = root / "source"
            (project / "data").mkdir(parents=True)
            source.mkdir()
            (project / "data" / "catalog.toml").write_text(
                """
schema = "data-catalog/v1"

[[datasets]]
id = "input.optional"
category = "inputs"
role = "candidate-input"
required_for = ["candidate-experiment"]
path = "data/inputs/optional"
kind = "directory"
required = false
""".strip()
            )
            (project / "data" / "sources.toml").write_text(
                """
schema = "data-sources/v1"

[store]
environment_variable = "TEST_DATA_SOURCE_ROOT"
default_relative_to_project = "../source"

[[sources]]
id = "input.optional"
source_path = "missing-payload"
acquisition = "lab-seed"
source_name = "Optional test fixture"
version = "1"
retrieval = "Supply the optional fixture source."
time_coverage = "test period"
spatial_coverage = "test grid"
units = "dimensionless"
preprocessing = "none"
limitations = "test fixture only"
license = "test only"
integrity = "fixture contents"
experiments = []
""".strip()
            )

            result = install_links(project, source)

            self.assertEqual(result[0].status, "optional-missing")

    def test_source_manifest_rejects_incomplete_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data").mkdir()
            (root / "data" / "sources.toml").write_text(
                """
schema = "data-sources/v1"

[store]
environment_variable = "TEST_DATA_SOURCE_ROOT"
default_relative_to_project = "../source"

[[sources]]
id = "input.incomplete"
source_path = "payload"
acquisition = "lab-seed"
retrieval = "Supply the fixture."
""".strip()
            )

            with self.assertRaisesRegex(ValueError, "source_name"):
                load_sources(root)


if __name__ == "__main__":
    unittest.main()
