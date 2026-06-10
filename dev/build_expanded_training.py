#!/usr/bin/env python3
"""Build a large deduplicated URL training CSV while reserving eval sets."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
URL_ML_DIR = BASE / "url_ml"
if str(URL_ML_DIR) not in sys.path:
    sys.path.insert(0, str(URL_ML_DIR))

from train_url_ml import canonical_url_key


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def read_reserved(paths: list[Path]) -> set[str]:
    reserved: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                if key:
                    reserved.add(key)
    return reserved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--reserve", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-label-conflicts",
        action="store_true",
        help="Write non-conflicting rows even if conflicting labels were found.",
    )
    args = parser.parse_args(argv)

    input_paths = list(args.input)
    for pattern in args.input_glob:
        input_paths.extend(
            str(Path(path).relative_to(BASE))
            for path in glob.glob(str(BASE / pattern), recursive=True)
            if Path(path).is_file()
        )
    input_paths = sorted(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("no inputs")

    out = BASE / args.out
    reserved = read_reserved([BASE / path for path in args.reserve])
    seen: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    conflicts = 0
    conflict_examples: list[str] = []
    skipped_reserved = 0

    for raw_path in input_paths:
        path = BASE / raw_path
        if path.resolve() == out.resolve():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            if "url" not in sample.splitlines()[0].lower() or "label" not in sample.splitlines()[0].lower():
                continue
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key:
                    continue
                if key in reserved:
                    skipped_reserved += 1
                    continue
                previous = seen.get(key)
                if previous is not None:
                    if previous != label:
                        conflicts += 1
                        if len(conflict_examples) < 20:
                            conflict_examples.append(f"{key} labels={previous},{label} input={raw_path}")
                    continue
                seen[key] = label
                rows.append(
                    {
                        "url": url,
                        "label": label,
                        "source": (row.get("source") or path.stem).strip() or path.stem,
                        "canonical_key": key,
                    }
                )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in rows)
    print(
        f"wrote={display_path(out)} rows={len(rows)} "
        f"benign={counts.get('0', 0)} malicious={counts.get('1', 0)} "
        f"reserved_keys={len(reserved)} skipped_reserved={skipped_reserved} conflicts={conflicts}"
    )
    for example in conflict_examples:
        print(f"  LABEL_CONFLICT {example}")
    if conflicts and not args.allow_label_conflicts:
        print("FAIL label conflicts found; rerun with --allow-label-conflicts only for intentional experiments")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
