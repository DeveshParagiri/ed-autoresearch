"""Check that every file in CHECKSUMS.txt is present and matches its pinned SHA256.

Runs in 3 tiers:
    [EXISTS?] Is the file on disk?
    [HASH?]   Does its SHA256 match the pinned value?
    [PIPELINE] (with --pipeline) Re-run reproduce_modelC.py end-to-end and
               SHA256 the produced burntArea.nc.

Exits 0 if every pinned file is present AND matches.
Exits non-zero if anything is missing or mismatched.

Usage:
    python scripts/verify.py
    python scripts/verify.py --pipeline   # also regenerate burntArea.nc and check
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKSUMS = REPO / "CHECKSUMS.txt"


def parse_checksums(p: Path):
    """Yield (sha256, size, relpath) tuples, skipping comments/blank lines."""
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        h = parts[0]
        try:
            sz = int(parts[1])
        except ValueError:
            continue
        rel = " ".join(parts[2:])
        yield h, sz, rel


def sha256_file(p: Path, block=1024*1024):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", action="store_true",
                    help="Also regenerate burntArea.nc via reproduce_modelC.py and check its hash.")
    args = ap.parse_args()

    if not CHECKSUMS.exists():
        print(f"[FATAL] {CHECKSUMS} not found")
        sys.exit(2)

    exist_ok, exist_missing = 0, 0
    hash_ok, hash_mismatch = 0, 0
    print(f"{'status':<12} {'file':<60} {'note'}")
    print("-" * 110)

    missing_paths = []
    mismatch_paths = []

    for sha, size, rel in parse_checksums(CHECKSUMS):
        p = REPO / rel
        if not p.exists():
            print(f"{'[MISSING]':<12} {rel:<60}")
            exist_missing += 1
            missing_paths.append(rel)
            continue
        exist_ok += 1
        actual_size = p.stat().st_size
        if actual_size != size:
            print(f"{'[SIZEDIFF]':<12} {rel:<60} expected {size}, got {actual_size}")
            hash_mismatch += 1
            mismatch_paths.append(rel)
            continue
        actual = sha256_file(p)
        if actual != sha:
            print(f"{'[MISMATCH]':<12} {rel:<60} expected {sha[:16]}..., got {actual[:16]}...")
            hash_mismatch += 1
            mismatch_paths.append(rel)
        else:
            print(f"{'[OK]':<12} {rel:<60} {sha[:16]}...")
            hash_ok += 1

    print()
    print(f"Present: {exist_ok}   Missing: {exist_missing}")
    print(f"Hash OK: {hash_ok}    Mismatch: {hash_mismatch}")

    if missing_paths:
        print()
        print("Missing files — either:")
        print("  (a) unzip the Drive bundle and drop its contents under data/")
        print("  (b) run `ED_RAW_DATA=/path/to/raw python scripts/prep_monthly_inputs.py` to regenerate")

    if mismatch_paths:
        print()
        print("Hash mismatches — your files differ from the pinned reference.")
        print("  If you regenerated .npy files yourself: this is usually a numpy/xarray")
        print("  version drift. Array contents may still be bit-identical; compare via")
        print("    np.array_equal(np.load(...), np.load(pinned_file))")

    # --pipeline: re-run reproduce and re-hash
    if args.pipeline:
        print()
        print("=== PIPELINE CHECK ===")
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "reproduce_modelC.py")],
            cwd=REPO, capture_output=True, text=True,
        )
        print(r.stdout[-500:] if r.stdout else "")
        if r.returncode != 0:
            print("[FAIL] reproduce_modelC.py exited non-zero")
            print(r.stderr[-500:])
            sys.exit(3)

        produced = REPO / "ilamb" / "MODELS" / "ED-ModelC-final" / "burntArea.nc"
        if not produced.exists():
            print(f"[FAIL] produced file not found: {produced}")
            sys.exit(3)
        produced_hash = sha256_file(produced)
        pinned_hash = None
        for sha, _, rel in parse_checksums(CHECKSUMS):
            if rel == "ilamb/MODELS/ED-ModelC-final/burntArea.nc":
                pinned_hash = sha
                break
        if pinned_hash is None:
            print("[WARN] no pinned hash for produced burntArea.nc")
        elif produced_hash == pinned_hash:
            print(f"[OK] produced burntArea.nc hash matches pinned ({produced_hash[:16]}...)")
        else:
            print(f"[FAIL] produced burntArea.nc hash differs:")
            print(f"    expected {pinned_hash}")
            print(f"    got      {produced_hash}")
            sys.exit(3)

    if exist_missing == 0 and hash_mismatch == 0:
        print("\nPASS")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
