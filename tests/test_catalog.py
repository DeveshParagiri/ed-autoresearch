from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.catalog import validate_catalog


class CatalogValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
