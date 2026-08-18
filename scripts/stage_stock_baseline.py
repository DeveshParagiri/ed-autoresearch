#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected a nonempty project-relative path, got {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path


def main() -> int:
    project_root = Path(os.environ["AUTORESEARCH_PROJECT_ROOT"]).resolve()
    run_root = Path(os.environ["AUTORESEARCH_RUN_ROOT"]).resolve()
    contract = json.loads((run_root / "contract.json").read_text())

    stock = contract["evaluation"]["stock"]
    source_relative = safe_relative(stock["path"])
    output_relative = safe_relative(contract["candidate_output"]["path"])
    source = project_root / source_relative
    output = run_root / output_relative

    if not source.is_file():
        raise FileNotFoundError(f"stock ED output is missing: {source_relative}")
    source_hash = sha256(source)
    if source_hash != stock["sha256"]:
        raise ValueError(
            f"stock ED output hash changed: expected {stock['sha256']}, got {source_hash}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    record = {
        "schema": "ed-fire-candidate/v1",
        "kind": "stock-output-staging",
        "source": str(source_relative),
        "source_sha256": source_hash,
        "output": str(output_relative),
        "output_sha256": sha256(output),
        "bytes": output.stat().st_size,
    }
    (run_root / "work" / "candidate.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(f"staged {source_relative} -> {output_relative}")
    print(f"sha256={source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
