from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCAFFOLD_ROOT = PROJECT_ROOT / "research" / "autoresearch"
sys.path.insert(0, str(SCAFFOLD_ROOT))

import scaffold  # noqa: E402


SKILL_PATHS = (
    PROJECT_ROOT / "skills" / "autoresearch" / "SKILL.md",
    PROJECT_ROOT
    / "research"
    / "autoresearch"
    / "template"
    / "skills"
    / "autoresearch"
    / "SKILL.md",
)


def parse_skill(path: Path) -> tuple[dict, str]:
    content = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not match:
        raise AssertionError(f"invalid skill frontmatter: {path}")
    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise AssertionError(f"skill metadata is not a mapping: {path}")
    return metadata, match.group(2)


class SkillQualityTests(unittest.TestCase):
    def test_skills_have_clean_routing_metadata(self) -> None:
        for path in SKILL_PATHS:
            metadata, _ = parse_skill(path)
            description = metadata.get("description")

            self.assertEqual(metadata.get("name"), "autoresearch", path)
            self.assertIsInstance(description, str, path)
            self.assertGreaterEqual(len(description), 30, path)
            self.assertLessEqual(len(description), 300, path)
            self.assertIn("Use when", description, path)

    def test_skills_are_focused_three_module_workflows(self) -> None:
        expected = [
            "## Module 1: Recover state and choose the next question",
            "## Module 2: Build and execute a controlled experiment",
            "## Module 3: Interpret evidence and preserve the result",
        ]
        for path in SKILL_PATHS:
            _, body = parse_skill(path)
            modules = [line for line in body.splitlines() if line.startswith("## ")]

            self.assertEqual(modules, expected, path)
            self.assertLessEqual(len(body.splitlines()), 500, path)
            self.assertNotIn("TODO", body, path)
            self.assertNotIn("/Users/", body, path)

    def test_generic_skill_has_no_legacy_protocol_dependency(self) -> None:
        generic = SKILL_PATHS[1].read_text().casefold()

        self.assertNotIn("research/protocols", generic)
        self.assertNotIn("applicable protocols", generic)

    def test_agent_metadata_matches_skill_interface(self) -> None:
        for skill_path in SKILL_PATHS:
            agent_path = skill_path.parent / "agents" / "openai.yaml"
            text = agent_path.read_text()
            if "__DEFAULT_PROMPT_YAML__" in text:
                values = scaffold.token_values("Skill Audit", "skill-audit", "Skill audit.")
                text = scaffold.render_text(text, values)
            record = yaml.safe_load(text)
            interface = record.get("interface", {})
            short_description = interface.get("short_description", "")
            prompt = interface.get("default_prompt", "")

            self.assertEqual(interface.get("display_name"), "Autoresearch", agent_path)
            self.assertGreaterEqual(len(short_description), 25, agent_path)
            self.assertLessEqual(len(short_description), 64, agent_path)
            self.assertIn("$autoresearch", prompt, agent_path)


if __name__ == "__main__":
    unittest.main()
