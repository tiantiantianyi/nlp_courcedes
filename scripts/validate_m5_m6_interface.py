from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.delivery.m5_validation import validate_m5_m6_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate the M5-to-M6 v1.0 JSONL interface.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-dir", type=Path, default=Path("../Train"))
    parser.add_argument("--val-dir", type=Path, default=Path("../Val"))
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--config-snapshot", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = validate_m5_m6_file(
        args.input,
        project_root=args.project_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        index_manifest=args.index_manifest,
        config_snapshot=args.config_snapshot,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
