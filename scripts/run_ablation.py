from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anima_search.evaluation.ablation import ablation_matrix


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    output = Path("artifacts/evaluation/ablation_plan.csv"); output.parent.mkdir(parents=True, exist_ok=True)
    rows = ablation_matrix()
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    if args.dry_run:
        print(json.dumps(rows, ensure_ascii=False, indent=2)); return
    print(f"Ablation matrix written to {output}. Build the matching prompt/model indexes, then evaluate each row.")


if __name__ == "__main__": main()
