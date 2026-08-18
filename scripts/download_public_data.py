#!/usr/bin/env python3
"""Download the public-source datasets used by the ED-Fire data catalog."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


GFED4_ROOT = "https://www.geo.vu.nl/~gwerf/GFED/GFED4"
GFED5_EMISSIONS_URL = (
    "https://zenodo.org/api/records/16794692/files/GFED5.1_monthly.zip/content"
)
GFED5_BURNED_AREA_ROOT = "https://zenodo.org/api/records/7668424/files"
GFED4_ILAMB_URL = (
    "https://www.ilamb.org/ILAMB-Data/DATA/burntArea/GFED4.1S/burntArea.nc"
)
ILAMB_DATA_REPOSITORY = "https://github.com/rubisco-sfa/ILAMB-Data.git"
ILAMB_DATA_COMMIT = "ccfa789f187ebc2a72386d65df65151c52f4c928"


@dataclass(frozen=True)
class DigestSpec:
    algorithm: str
    value: str


GFED4_FILES = {
    2001: (30_765_559, "1b684bf0b348e92a5d63ea660564f01439f69c4eb88eacd46280237d51ce5815"),
    2002: (31_398_196, "dcf624961512dbb93759248bc2b75d404b3be68f1f6fdcb01f0c7dc7f11a517a"),
    2003: (73_101_743, "91d61b67d04b4a32d534f5d68ae1de7929f7ea75bb9d25d3273c4d5d75bda4d3"),
    2004: (73_132_456, "931e063f796bf1f7d391d3f03342d2dd2ad1b234cb317f826adfab201003f4cd"),
    2005: (73_394_752, "159e7704d14089496d051546c20b644a443308eeb7d79bf338226af2b4bdc2b7"),
    2006: (72_125_825, "a69d5bf6b8fa3324c2922aac07306ec6e488a850ca4f42d09a397cee30eebd4c"),
    2007: (73_249_418, "1d7f77e6f7b13cc2a8ef9d26ecb9ea3d18e70cfeb8a47e7ecb26f9613888f937"),
    2008: (72_239_399, "bd3771b9b3032d459a79c0da449fdb497cd3400e0e07a0da6b41e930fc5d3e14"),
    2009: (71_577_995, "36ea9b6036cd0ff3672502c3c04180bd209ddb192f86a2e791a2b896308bc5ff"),
    2010: (71_969_657, "5b2d30b5ddc3e20c38c7971faf6791b313b1bbff22e8bc2b14ca7ea9079aa12c"),
    2011: (72_056_341, "fb19c001bef26ca23d07dd8978fd998f4692bdecdec5eb86b91d4b1ffb4a9aa7"),
    2012: (72_794_603, "08033c90295bbc208fac426e01809b68cef62997668085b1e096d8a61ab43e9b"),
    2013: (71_385_388, "cf5249811af4b7099f886e61125dcd15c1127b6125392fe8358d3f0bf8ddb064"),
    2014: (71_585_045, "a293b4c6e03898a0dc184a082a37435673916a02ff02c06668152dcc4d4b8405"),
    2015: (72_926_611, "8eae810c38e667e296bde84e288b4dc365816c6c3f4eb9dd569ba1d4c5bc3c41"),
    2016: (70_379_149, "2f3b54ff5698ba7f7aa2bb1d4b5e5f95124c0e0db32830ed94aa04bea2cbc2a6"),
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verified(path: Path, expected_size: int | None, expected: DigestSpec | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return expected is None or digest(path, expected.algorithm) == expected.value


def download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected: DigestSpec | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified(destination, expected_size, expected):
        print(f"verified existing {destination}")
        return
    partial = destination.with_name(f".{destination.name}.partial")
    partial.unlink(missing_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ed-fire-data-installer/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    if not verified(partial, expected_size, expected):
        actual_size = partial.stat().st_size
        actual_digest = digest(partial, expected.algorithm) if expected else "not checked"
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"download validation failed for {destination}: "
            f"size={actual_size}, digest={actual_digest}"
        )
    os.replace(partial, destination)
    print(f"verified {destination}")


def extract_selected(archive: Path, destination: Path, pattern: re.Pattern[str]) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        selected = [member for member in bundle.infolist() if pattern.fullmatch(Path(member.filename).name)]
        if not selected:
            raise RuntimeError(f"no expected files found in {archive}")
        for member in selected:
            name = Path(member.filename).name
            output = destination / name
            if output.is_file() and output.stat().st_size == member.file_size:
                count += 1
                continue
            partial = destination / f".{name}.partial"
            with bundle.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if partial.stat().st_size != member.file_size:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"short extraction for {name}")
            os.replace(partial, output)
            count += 1
    print(f"verified {count} files in {destination}")
    return count


def install_gfed4_source(source_root: Path) -> None:
    output = source_root / "data" / "gfed" / "4.1"
    for year, (size, sha256) in GFED4_FILES.items():
        name = f"GFED4.1s_{year}.hdf5"
        download(
            f"{GFED4_ROOT}/{name}",
            output / name,
            expected_size=size,
            expected=DigestSpec("sha256", sha256),
        )


def install_gfed5_emissions(source_root: Path) -> None:
    output = source_root / "data" / "gfed" / "5"
    expected_files = [output / f"GFED5.1_monthly_{year}.nc" for year in range(2000, 2017)]
    if all(path.is_file() and path.stat().st_size > 1_000_000 for path in expected_files):
        print(f"verified existing 17-file GFED5.1 subset in {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gfed5-emissions-", dir=output.parent) as temporary:
        archive = Path(temporary) / "GFED5.1_monthly.zip"
        download(
            GFED5_EMISSIONS_URL,
            archive,
            expected=DigestSpec("md5", "c1aeb51bc629a08345af1b91b5c5236d"),
        )
        years = "|".join(str(year) for year in range(2000, 2017))
        count = extract_selected(
            archive,
            output,
            re.compile(rf"GFED5\.1_monthly_({years})\.nc"),
        )
    if count != 17:
        raise RuntimeError(f"expected 17 GFED5.1 emissions files, found {count}")


def install_gfed5_burned_area(source_root: Path) -> None:
    output = source_root / "data" / "gfed5"
    existing = list(output.glob("BA??????.nc"))
    if len(existing) == 288 and all(path.stat().st_size > 100_000 for path in existing):
        print(f"verified existing 288-file GFED5 burned-area set in {output}")
    else:
        archive = output / "BA.zip"
        download(
            f"{GFED5_BURNED_AREA_ROOT}/BA.zip/content",
            archive,
            expected=DigestSpec("md5", "2fe4a5ced7e19823f78eafb73616d9f1"),
        )
        count = extract_selected(archive, output, re.compile(r"BA\d{6}\.nc"))
        if count != 288:
            raise RuntimeError(f"expected 288 GFED5 burned-area files, found {count}")
    download(
        f"{GFED5_BURNED_AREA_ROOT}/BurnableArea.nc/content",
        output / "BurnableArea.nc",
        expected=DigestSpec("md5", "7f0df5c21bd568c5358b03690f068650"),
    )
    download(
        f"{GFED5_BURNED_AREA_ROOT}/BurnableArea_preMOD.nc/content",
        output / "BurnableArea_preMOD.nc",
        expected=DigestSpec("md5", "0960868e0258655cda720bdac3dcbde4"),
    )


def install_gfed4_ilamb(source_root: Path) -> None:
    download(
        GFED4_ILAMB_URL,
        source_root / "ilamb" / "DATA" / "burntArea" / "GFED4.1s" / "burntArea.nc",
        expected_size=497_712_890,
        expected=DigestSpec(
            "sha256", "1dec2035daf0048be74161e95995202e68138a156e9edba8da8b84b0d05f4095"
        ),
    )


def install_ilamb_tools(source_root: Path) -> None:
    destination = source_root / "data" / "observations" / "ILAMB-Data"
    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git path: {destination}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", ILAMB_DATA_REPOSITORY, str(destination)],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual == ILAMB_DATA_COMMIT:
        print(f"verified {destination} at {actual}")
        return
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", ILAMB_DATA_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", ILAMB_DATA_COMMIT],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != ILAMB_DATA_COMMIT:
        raise RuntimeError(f"ILAMB-Data commit mismatch: {actual}")
    print(f"verified {destination} at {actual}")


INSTALLERS = {
    "gfed4-source": install_gfed4_source,
    "gfed5-emissions": install_gfed5_emissions,
    "gfed5-burned-area": install_gfed5_burned_area,
    "gfed4-ilamb": install_gfed4_ilamb,
    "ilamb-tools": install_ilamb_tools,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=("all", *INSTALLERS),
        default="all",
        help="public dataset group to install",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    selected = INSTALLERS if args.only == "all" else {args.only: INSTALLERS[args.only]}
    for name, installer in selected.items():
        print(f"installing {name}")
        installer(source_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
