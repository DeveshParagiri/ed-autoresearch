#!/usr/bin/env python3
"""Download and unpack the official ED v3.0 release into this project."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/records/6901510"
CONTENT_ROOT = "https://zenodo.org/api/records/6901510/files"
SIMULATION_MEMBER = "EDv3_simulation/EDv3_global_simulation_1981_2016.nc"
SIMULATION_SHA256 = "15f34eaee68bc4e5b27ca768dc11e4e02370f53ac9230ba7236b801838655eb3"


@dataclass(frozen=True)
class ReleaseFile:
    name: str
    bytes: int
    md5: str
    root: str | None = None
    destination: str | None = None
    flatten_root: bool = False


FILES = {
    "source": ReleaseFile(
        "EDv3_code.zip",
        186_932,
        "0dc5c1d26d9f2fcd965812af44df6c83",
    ),
    "dependencies": ReleaseFile(
        "EDv3_dependencies.zip",
        38_591_525,
        "d583a858c8742bfcb11c8ad402182ea4",
        root="EDv3_dependencies",
        destination="model/stock-ed/external_apps/source_code",
        flatten_root=True,
    ),
    "inputs": ReleaseFile(
        "EDv3_inputs.zip",
        15_299_859_096,
        "40e96db270117fe1e2933b0bd49fc003",
        root="EDv3_inputs",
        destination="data/inputs/stock-ed",
    ),
    "environment": ReleaseFile(
        "EDv3_env_setup.pdf",
        108_439,
        "eccfa1ec5cc877a013d0d1a7e88f373b",
        destination="data/reference/ed-v3-release/EDv3_env_setup.pdf",
    ),
    "evaluation": ReleaseFile(
        "EDv3_evaluation.zip",
        16_607,
        "65637a0c56ebb332f6aeed23e4335d8d",
        root="EDv3_evaluation",
        destination="data/reference/ed-v3-release/evaluation",
        flatten_root=True,
    ),
    "simulation": ReleaseFile(
        "EDv3_simulation.zip",
        1_576_948_578,
        "5d02c8a2827063ebf109e6df5130ea88",
        root="EDv3_simulation",
        destination="data/inputs/stock-ed",
    ),
}


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stream_digest(handle, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def verified(path: Path, spec: ReleaseFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec.bytes
        and digest(path) == spec.md5
    )


def download(archive_root: Path, spec: ReleaseFile) -> Path:
    destination = archive_root / spec.name
    if verified(destination, spec):
        print(f"verified {destination}")
        return destination
    if destination.exists():
        raise RuntimeError(f"refusing unverified release file: {destination}")

    partial = archive_root / f".{spec.name}.partial"
    if partial.exists() and partial.stat().st_size > spec.bytes:
        raise RuntimeError(f"partial file exceeds publisher size: {partial}")
    archive_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "10",
            "-C",
            "-",
            "-o",
            str(partial),
            f"{CONTENT_ROOT}/{spec.name}/content",
        ],
        check=True,
    )
    if not verified(partial, spec):
        raise RuntimeError(
            f"download failed verification: {partial} "
            f"bytes={partial.stat().st_size} md5={digest(partial)}"
        )
    os.replace(partial, destination)
    print(f"verified {destination}")
    return destination


def safe_relative(member: zipfile.ZipInfo, root: str, flatten_root: bool) -> Path | None:
    value = PurePosixPath(member.filename)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(f"unsafe archive member: {member.filename}")
    if not value.parts or value.parts[0] in {"__MACOSX", ".DS_Store"}:
        return None
    if value.parts[0] != root:
        raise RuntimeError(f"unexpected archive root for {member.filename}; expected {root}")
    remainder = value.parts[1:]
    if not remainder or any(part == ".DS_Store" or part.startswith("._") for part in remainder):
        return None
    return Path(*remainder) if flatten_root else Path(root, *remainder)


def extract(archive: Path, project_root: Path, spec: ReleaseFile) -> None:
    if not spec.root or not spec.destination:
        return
    destination = project_root / spec.destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        broken = bundle.testzip()
        if broken:
            raise RuntimeError(f"archive CRC failure at {broken}: {archive}")
        extracted = 0
        for member in bundle.infolist():
            relative = safe_relative(member, spec.root, spec.flatten_root)
            if relative is None:
                continue
            output = destination / relative
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if output.is_file() and output.stat().st_size == member.file_size:
                extracted += 1
                continue
            if output.exists():
                raise RuntimeError(f"refusing to replace existing path: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(f".{output.name}.partial")
            with bundle.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise RuntimeError(f"short extraction: {output}")
            os.replace(partial, output)
            extracted += 1
    print(f"verified {extracted} extracted files under {destination}")


def install_source_tree(archive: Path, project_root: Path) -> None:
    destination = project_root / "model" / "stock-ed"
    destination.mkdir(parents=True, exist_ok=True)
    verified_files = 0
    with zipfile.ZipFile(archive) as bundle:
        broken = bundle.testzip()
        if broken:
            raise RuntimeError(f"archive CRC failure at {broken}: {archive}")
        for member in bundle.infolist():
            relative = safe_relative(member, "EDv3_code", True)
            if relative is None:
                continue
            output = destination / relative
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            with bundle.open(member) as source:
                if output.is_file():
                    archive_sha256 = stream_digest(source)
                    if (
                        output.stat().st_size != member.file_size
                        or digest(output, "sha256") != archive_sha256
                    ):
                        raise RuntimeError(
                            f"stock source differs from publisher archive: {output}"
                        )
                else:
                    if output.exists() or output.is_symlink():
                        raise RuntimeError(f"refusing to replace stock source path: {output}")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    partial = output.with_name(f".{output.name}.partial")
                    with partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                    if partial.stat().st_size != member.file_size:
                        raise RuntimeError(f"short stock source extraction: {output}")
                    os.replace(partial, output)
            verified_files += 1
    if verified_files != 50:
        raise RuntimeError(f"expected 50 stock source files, verified {verified_files}")
    print(f"verified {verified_files} publisher source files under {destination}")


def install_environment(archive: Path, project_root: Path, spec: ReleaseFile) -> None:
    if not spec.destination:
        return
    destination = project_root / spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and digest(destination) == spec.md5:
        print(f"verified {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    shutil.copy2(archive, destination)
    print(f"verified {destination}")


def install_input_link(project_root: Path) -> None:
    source = project_root / "data" / "inputs" / "stock-ed" / "EDv3_inputs"
    destination = project_root / "model" / "stock-ed" / "EDv3_inputs"
    if not source.is_dir():
        return
    expected = Path("..") / ".." / "data" / "inputs" / "stock-ed" / "EDv3_inputs"
    if destination.is_symlink() and Path(os.readlink(destination)) == expected:
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace stock input path: {destination}")
    destination.symlink_to(expected)
    print(f"linked {destination} -> {expected}")


def install_simulation(archive: Path, project_root: Path) -> None:
    canonical = project_root / "data" / "inputs" / "ecosystem" / "ed-simulation.nc"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        broken = bundle.testzip()
        if broken:
            raise RuntimeError(f"archive CRC failure at {broken}: {archive}")
        member = bundle.getinfo(SIMULATION_MEMBER)
        if canonical.is_file():
            if canonical.stat().st_size != member.file_size:
                raise RuntimeError(f"existing ED simulation has the wrong size: {canonical}")
            if digest(canonical, "sha256") != SIMULATION_SHA256:
                raise RuntimeError(f"existing ED simulation has the wrong digest: {canonical}")
        else:
            if canonical.exists() or canonical.is_symlink():
                raise RuntimeError(f"refusing to replace existing path: {canonical}")
            partial = canonical.with_name(f".{canonical.name}.partial")
            with bundle.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            if (
                partial.stat().st_size != member.file_size
                or digest(partial, "sha256") != SIMULATION_SHA256
            ):
                raise RuntimeError(f"extracted ED simulation failed validation: {partial}")
            os.replace(partial, canonical)

    release_directory = (
        project_root / "data" / "inputs" / "stock-ed" / "EDv3_simulation"
    )
    release_directory.mkdir(parents=True, exist_ok=True)
    release_path = release_directory / "EDv3_global_simulation_1981_2016.nc"
    expected = Path("..") / ".." / "ecosystem" / "ed-simulation.nc"
    if release_path.is_symlink() and Path(os.readlink(release_path)) == expected:
        print(f"verified {canonical}")
        return
    if release_path.exists() or release_path.is_symlink():
        raise RuntimeError(f"refusing to replace ED simulation path: {release_path}")
    release_path.symlink_to(expected)
    print(f"verified {canonical}")
    print(f"linked {release_path} -> {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Install the official ED v3.0 release from {RECORD}."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--only", choices=("all", *FILES), default="all")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="verify archives without extracting runtime files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    archive_root = project_root / "data" / "reference" / "ed-v3-release" / "archives"
    selected = FILES if args.only == "all" else {args.only: FILES[args.only]}
    for name, spec in selected.items():
        print(f"installing stock ED {name}")
        archive = download(archive_root, spec)
        if args.download_only:
            continue
        if name == "source":
            install_source_tree(archive, project_root)
        elif name == "environment":
            install_environment(archive, project_root, spec)
        elif name == "simulation":
            install_simulation(archive, project_root)
        else:
            extract(archive, project_root, spec)
    if not args.download_only and (args.only in {"all", "inputs"}):
        install_input_link(project_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
