#!/usr/bin/env python3
"""Download and checksum the official LUH2-GCB2026 NetCDF files."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests


RECORD = "https://zenodo.org/records/20932027"


@dataclass(frozen=True)
class FileSpec:
    size: int
    md5: str
    url: str


FILES = {
    "states.nc": FileSpec(
        6_080_810_028,
        "8a74fa6273d38307a14780bc654e0028",
        "https://zenodo.org/api/records/20932027/files/states.nc/content",
    ),
    "transitions.nc": FileSpec(
        16_842_976_839,
        "d1a6c7d5fae587beb186cb1af1cfa032",
        "https://zenodo.org/api/records/20932027/files/transitions.nc/content",
    ),
    "management.nc": FileSpec(
        1_534_118_739,
        "d20b4dfd9813384203aef57593c5609c",
        "https://zenodo.org/api/records/20932027/files/management.nc/content",
    ),
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranges(size: int, workers: int) -> list[tuple[int, int]]:
    chunk = (size + workers - 1) // workers
    return [(start, min(size - 1, start + chunk - 1)) for start in range(0, size, chunk)]


def download_range(
    spec: FileSpec, destination: Path, start: int, end: int, index: int
) -> Path:
    expected = end - start + 1
    current = destination.stat().st_size if destination.exists() else 0
    if current > expected:
        raise RuntimeError(f"oversized partial file: {destination}")
    if current == expected:
        print(f"part {index:02d} already complete")
        return destination

    for attempt in range(1, 6):
        offset = start + current
        try:
            with requests.get(
                spec.url,
                headers={"Range": f"bytes={offset}-{end}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"server did not honor byte range for part {index}: "
                        f"HTTP {response.status_code}"
                    )
                with destination.open("ab") as handle:
                    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if block:
                            handle.write(block)
                            current += len(block)
            if current != expected:
                raise RuntimeError(
                    f"short part {index}: expected {expected} bytes, found {current}"
                )
            print(f"part {index:02d} complete: {expected} bytes")
            return destination
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 5:
                raise
            print(f"part {index:02d} retry {attempt}: {error}")
            time.sleep(2**attempt)
            current = destination.stat().st_size if destination.exists() else 0
    raise AssertionError("unreachable")


def download_file(name: str, spec: FileSpec, output_dir: Path, workers: int) -> None:
    target = output_dir / name
    if target.is_file() and target.stat().st_size == spec.size:
        digest = md5sum(target)
        if digest != spec.md5:
            raise RuntimeError(f"checksum mismatch for existing {target}: {digest}")
        print(f"verified existing {name}: md5:{digest}")
        return

    byte_ranges = ranges(spec.size, workers)
    parts = [output_dir / f".{name}.part{index:02d}" for index in range(len(byte_ranges))]
    if target.exists():
        size = target.stat().st_size
        first_size = byte_ranges[0][1] - byte_ranges[0][0] + 1
        if parts[0].exists():
            raise RuntimeError(f"both target and first partial exist for {name}")
        if size > first_size:
            raise RuntimeError(f"cannot reuse {size}-byte prefix for {name}")
        target.replace(parts[0])
        print(f"reused {size} downloaded bytes for {name}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_range, spec, part, start, end, index)
            for index, (part, (start, end)) in enumerate(zip(parts, byte_ranges))
        ]
        for future in as_completed(futures):
            future.result()

    assembling = output_dir / f".{name}.assembling"
    digest = hashlib.md5()
    with assembling.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
    actual = digest.hexdigest()
    assembled_size = assembling.stat().st_size
    if assembled_size != spec.size or actual != spec.md5:
        assembling.unlink(missing_ok=True)
        raise RuntimeError(
            f"validation failed for {name}: size={assembled_size}, md5={actual}"
        )
    os.replace(assembling, target)
    for part in parts:
        part.unlink()
    print(f"verified {name}: {spec.size} bytes, md5:{actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("files", nargs="*", choices=sorted(FILES), default=sorted(FILES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"LUH2-GCB2026 source: {RECORD}")
    for name in args.files:
        download_file(name, FILES[name], args.output_dir, args.workers)


if __name__ == "__main__":
    main()
