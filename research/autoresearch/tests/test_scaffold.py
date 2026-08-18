from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCAFFOLD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCAFFOLD_ROOT))

import scaffold  # noqa: E402


class ScaffoldTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _script(
        self,
        destination: Path,
        environment: dict[str, str],
        script: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(destination / "scripts" / script), *arguments],
            cwd=destination,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def _prepare_workspace(self, destination: Path, candidate_source: str) -> dict[str, str]:
        scaffold.create_workspace(destination, name="Runnable Research")
        candidate = destination / "scripts" / "candidate.py"
        candidate.write_text(candidate_source)
        experiment = destination / "research" / "experiments" / "experiment.baseline" / "experiment.md"
        experiment.write_text(
            """---
schema: autoresearch-experiment/v1
id: experiment.baseline
title: First controlled candidate
kind: baseline
status: proposed
created_at: 2026-08-18T00:00:00Z
parents: []
inputs: []
contract: evals/contracts/baseline-eval-v1.json
execution:
  mode: simulation
  tool: direct
  adapter: scripts/candidate.py
  argv: ["{python}", "scripts/candidate.py"]
search: null
---

# Question

Can one candidate complete the protected evaluation?

# Change

Run one fixture candidate.

# Prediction

The trusted evaluator will replace candidate-written official evidence.

# Plan

Expect trusted metrics and one fixed figure; stop after one terminal attempt.

# Result

No run yet.

# Decision

Keep proposed.

# Revisit when

Run after the fixture contract is active.
"""
        )
        benchmark = destination / "data" / "benchmarks" / "locked.txt"
        benchmark.parent.mkdir(parents=True, exist_ok=True)
        benchmark.write_text("locked benchmark\n")
        evaluator = destination / "scripts" / "trusted_eval.py"
        evaluator.write_text(
            """import base64
import json
import os
from pathlib import Path

run_root = Path(os.environ["AUTORESEARCH_RUN_ROOT"])
candidate = json.loads((run_root / "work" / "candidate.json").read_text())
(run_root / "metrics.json").write_text(json.dumps({"score": candidate["value"]}) + "\\n")
pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=")
(run_root / "figures" / "comparison.png").write_bytes(pixel)
"""
        )
        contract = {
            "schema": "autoresearch-contract/v1",
            "id": "contract.runnable.v1",
            "status": "active",
            "owner": "test-owner",
            "approved_by": "test-reviewer",
            "approved_at": "2026-08-18T00:00:00Z",
            "protected_files": [
                {"path": "data/benchmarks/locked.txt", "sha256": self._sha256(benchmark)}
            ],
            "evaluator": {
                "path": "scripts/trusted_eval.py",
                "sha256": self._sha256(evaluator),
                "argv": ["{python}", "scripts/trusted_eval.py"],
            },
            "development_evaluation": {"definition": "The locked local benchmark."},
            "promotion_evaluation": {"definition": "A separate sealed evaluation."},
            "metrics": [{"id": "score", "direction": "higher", "required": True}],
            "aggregation": "No aggregation is needed for this test.",
            "comparison_baselines": ["baseline.test"],
            "figures": [
                {
                    "filename": "comparison.png",
                    "description": "The fixed test comparison.",
                    "generator": "scripts/trusted_eval.py",
                    "format": "png",
                    "dimensions": {"width": 1, "height": 1},
                    "panels": ["candidate"],
                    "scope": "The locked test case.",
                    "scale": "Fixed.",
                    "labels": ["candidate"],
                    "units": "score",
                    "required": True,
                }
            ],
        }
        self._write_json(
            destination / "evals" / "contracts" / "baseline-eval-v1.json",
            contract,
        )
        subprocess.run(["git", "init", "-q"], cwd=destination, check=True)
        subprocess.run(
            ["git", "config", "user.email", "scaffold-test@example.invalid"],
            cwd=destination,
            check=True,
        )
        subprocess.run(["git", "config", "user.name", "Scaffold Test"], cwd=destination, check=True)
        subprocess.run(["git", "add", "."], cwd=destination, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "Initialize test workspace"], cwd=destination, check=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(destination / "src")
        return environment

    def test_template_is_complete(self) -> None:
        self.assertEqual(scaffold.check_template(), ())

    def test_document_responsibilities_do_not_overlap(self) -> None:
        research = (scaffold.TEMPLATE_ROOT / "research.md").read_text()
        memory = (scaffold.TEMPLATE_ROOT / "memory.md").read_text()
        self.assertIn("## Research sequence", research)
        self.assertNotIn("## Current frontier", research)
        self.assertIn("current frontier", memory.lower())
        experiment = (
            scaffold.TEMPLATE_ROOT
            / "research"
            / "experiments"
            / "experiment.baseline"
            / "experiment.md"
        ).read_text()
        self.assertIn("# Plan", experiment)

    def test_ed_fire_and_template_share_the_same_core_modules(self) -> None:
        repository_root = SCAFFOLD_ROOT.parents[1]
        for filename in ("runner.py", "optuna_records.py"):
            project_text = (repository_root / "src" / "autoresearch" / filename).read_text()
            template_text = (scaffold.TEMPLATE_ROOT / "src" / "autoresearch" / filename).read_text()
            self.assertEqual(project_text, template_text, filename)

    def test_slugify_is_stable(self) -> None:
        self.assertEqual(scaffold.slugify("A Research Problem 2.0"), "a-research-problem-2-0")

    def test_dry_run_does_not_create_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "planned"
            planned = scaffold.create_workspace(destination, name="Planned Research", dry_run=True)
            self.assertFalse(destination.exists())
            self.assertIn(destination.resolve() / "research.md", planned)

    def test_create_refuses_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                scaffold.create_workspace(destination, name="Existing")

    def test_create_sets_each_supported_baseline_mode(self) -> None:
        for mode in ("mechanistic", "simulation", "hybrid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / mode
                scaffold.create_workspace(
                    destination,
                    name=f"{mode.title()} Research",
                    mode=mode,
                )
                experiment = (
                    destination
                    / "research"
                    / "experiments"
                    / "experiment.baseline"
                    / "experiment.md"
                ).read_text()
                self.assertIn(f"  mode: {mode}\n", experiment)

    def test_generated_workspace_has_only_the_three_public_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "generated"
            scaffold.create_workspace(destination, name="Generated Research")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(destination / "src")
            status = self._script(destination, environment, "check_workspace.py", "--json")
            self.assertEqual(status.returncode, 0, status.stdout + status.stderr)
            payload = json.loads(status.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["experiments"], 1)
            self.assertEqual(
                {path.name for path in (destination / "scripts").iterdir()},
                {"check_workspace.py", "install_data.py", "run_experiment.py"},
            )

            workspace_tests = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=destination,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(workspace_tests.returncode, 0, workspace_tests.stdout + workspace_tests.stderr)
            self.assertFalse((destination / "src" / "autoresearch" / "cli.py").exists())
            self.assertFalse(any(destination.rglob(".gitkeep")))

            template_skill = (scaffold.TEMPLATE_ROOT / "skills" / "autoresearch" / "SKILL.md").read_text()
            generated_skill = (destination / "skills" / "autoresearch" / "SKILL.md").read_text()
            self.assertEqual(generated_skill, template_skill)

    def test_generated_installer_accepts_a_member_through_a_parent_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generated"
            source = root / "source"
            scaffold.create_workspace(destination, name="Linked Data Research")
            (source / "collection").mkdir(parents=True)
            (source / "collection" / "value.txt").write_text("evidence")
            (destination / "data" / "inputs").mkdir()
            (destination / "data" / "inputs" / "collection").symlink_to(
                source / "collection"
            )
            (destination / "data" / "catalog.toml").write_text(
                """schema = "data-catalog/v1"
datasets = [{ id = "input.member", path = "data/inputs/collection/value.txt", required = true }]
"""
            )
            (destination / "data" / "sources.toml").write_text(
                """schema = "data-sources/v1"
[store]
environment_variable = "LINKED_DATA_ROOT"
default_relative_to_project = "../source"

[[sources]]
id = "input.member"
source_path = "collection/value.txt"
acquisition = "lab-seed"
"""
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(destination / "src")

            result = self._script(
                destination,
                environment,
                "install_data.py",
                "plan",
                "--source-root",
                str(source),
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("READY", result.stdout)

    def test_run_is_transactional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "model-run"
            source = """import json, os
from pathlib import Path
root = Path(os.environ['AUTORESEARCH_RUN_ROOT'])
(root / 'work' / 'candidate.json').write_text(json.dumps({'value': 0.75}))
(root / 'metrics.json').write_text(json.dumps({'score': 999}))
(root / 'figures' / 'comparison.png').write_bytes(b'candidate-figure')
"""
            environment = self._prepare_workspace(destination, source)
            readiness = self._script(destination, environment, "check_workspace.py", "--json")
            self.assertEqual(json.loads(readiness.stdout)["runnable_experiments"], ["experiment.baseline"])
            result = self._script(destination, environment, "run_experiment.py", "experiment.baseline", "--json")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["status"], "completed")
            self.assertEqual([item["role"] for item in record["commands"]], ["candidate", "evaluator"])
            protected_paths = [item["path"] for item in record["protected_files"]]
            self.assertEqual(len(protected_paths), len(set(protected_paths)))
            run_root = destination / "research" / "experiments" / "experiment.baseline" / "runs" / record["id"]
            self.assertTrue((run_root / "contract.json").is_file())
            self.assertTrue((run_root / "events.jsonl").is_file())
            self.assertEqual(json.loads((run_root / "metrics.json").read_text()), {"score": 0.75})
            artifacts = json.loads((run_root / "artifacts.json").read_text())["artifacts"]
            self.assertEqual([item["name"] for item in artifacts], ["comparison.png"])
            status = self._script(destination, environment, "check_workspace.py", "--json")
            self.assertEqual(json.loads(status.stdout)["runs"], 1)

    def test_protected_benchmark_drift_invalidates_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "drift-run"
            source = "from pathlib import Path\nPath('data/benchmarks/locked.txt').write_text('changed\\n')\n"
            environment = self._prepare_workspace(destination, source)
            result = self._script(destination, environment, "run_experiment.py", "experiment.baseline", "--json")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["status"], "invalid")
            self.assertEqual([item["role"] for item in record["commands"]], ["candidate"])

    def test_failed_candidate_keeps_record_events_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "failed-run"
            source = "import sys\nprint('candidate output')\nprint('candidate error', file=sys.stderr)\nraise SystemExit(7)\n"
            environment = self._prepare_workspace(destination, source)
            result = self._script(destination, environment, "run_experiment.py", "experiment.baseline", "--json")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["status"], "failed")
            run_root = destination / "research" / "experiments" / "experiment.baseline" / "runs" / record["id"]
            self.assertIn("candidate output", (run_root / "logs" / "candidate.stdout.log").read_text())
            self.assertIn("candidate error", (run_root / "logs" / "candidate.stderr.log").read_text())
            self.assertIn("run-finished", (run_root / "events.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
