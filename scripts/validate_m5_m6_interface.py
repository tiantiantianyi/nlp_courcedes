#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.m6.interface_validation import validate_interface_file
from anima_search.m6.path_safety import reject_output_path_aliases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate an m5-to-m6-v1.0 JSONL delivery.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--m5-config-snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reject_output_path_aliases(
        read_only={
            "input": args.input,
            "project_root": args.project_root,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "index_manifest": args.index_manifest,
            "m5_config_snapshot": args.m5_config_snapshot,
        },
        outputs={"report": args.report},
    )
    _, report = validate_interface_file(
        input_path=args.input,
        project_root=args.project_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        index_manifest_path=args.index_manifest,
        config_snapshot_path=args.m5_config_snapshot,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"queries={report.query_count} "
        f"candidates={report.candidate_count} "
        f"errors={len(report.issues)}"
    )
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
