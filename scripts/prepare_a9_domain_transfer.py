from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.evaluation.domain_transfer import (
    MEDICAL_CATEGORIES,
    WIKIART_STYLES,
    balanced_sample,
    build_manual_review_rows,
    build_queries,
    build_relevance,
    dataset_summary,
    medical_labels,
    read_csv_text,
    write_jsonl,
)


def archive_text(archive: Path, member: str, seven_zip: str) -> str:
    result = subprocess.run(
        [seven_zip, "x", "-so", str(archive), member],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8-sig")


def extract_members(
    archive: Path,
    members: list[str],
    destination: Path,
    seven_zip: str,
) -> None:
    subprocess.run(
        [seven_zip, "x", "-y", str(archive), f"-o{destination}", *members],
        check=True,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the optional A9 domain-transfer subset."
    )
    parser.add_argument("--archive", type=Path, default=Path("../data.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../domain_transfer_data/a9_subset"),
    )
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--seven-zip", default="7z")
    args = parser.parse_args()

    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    if not archive.with_suffix(".z01").is_file():
        raise FileNotFoundError(
            f"split volume does not exist: {archive.with_suffix('.z01')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    art_rows = read_csv_text(
        archive_text(archive, "data/wikiart/metadata.csv", args.seven_zip)
    )
    medical_rows = read_csv_text(
        archive_text(
            archive,
            "data/MIMIC-CXR/test/reports.csv",
            args.seven_zip,
        )
    )
    selected_art, art_groups = balanced_sample(
        art_rows,
        WIKIART_STYLES,
        lambda row: {row["style_name"]},
        args.per_category,
        args.seed,
    )
    selected_medical, medical_groups = balanced_sample(
        medical_rows,
        MEDICAL_CATEGORIES,
        lambda row: medical_labels(row["report"]),
        args.per_category,
        args.seed + 1,
    )

    members = [f"data/wikiart/{row['file']}" for row in selected_art]
    members += [
        f"data/MIMIC-CXR/test/{row['image']}" for row in selected_medical
    ]
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="a9-extract-") as temporary:
        temporary_root = Path(temporary)
        extract_members(archive, members, temporary_root, args.seven_zip)
        for row in selected_art:
            image_id = f"a9-art-{int(row['index']):06d}"
            relative_path = Path("images/wikiart") / f"{image_id}.jpg"
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                temporary_root / "data/wikiart" / row["file"],
                destination,
            )
            records.append(
                {
                    "image_id": image_id,
                    "domain": "wikiart",
                    "relative_path": relative_path.as_posix(),
                    "source_member": f"data/wikiart/{row['file']}",
                    "labels": [row["style_name"]],
                    "artist_name": row["artist_name"],
                    "genre_name": row["genre_name"],
                    "style_name": row["style_name"],
                    "report": "",
                }
            )
        for row in selected_medical:
            image_id = f"a9-medical-{int(row['index']):06d}"
            relative_path = Path("images/mimic_cxr") / f"{image_id}.jpg"
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                temporary_root / "data/MIMIC-CXR/test" / row["image"],
                destination,
            )
            records.append(
                {
                    "image_id": image_id,
                    "domain": "mimic_cxr",
                    "relative_path": relative_path.as_posix(),
                    "source_member": (
                        f"data/MIMIC-CXR/test/{row['image']}"
                    ),
                    "labels": sorted(medical_labels(row["report"])),
                    "artist_name": "",
                    "genre_name": "",
                    "style_name": "",
                    "report": row["report"],
                }
            )

    queries = build_queries()
    relevance = build_relevance(queries, records)
    qrel_rows = [
        {
            "query_id": query_id,
            "image_id": image_id,
            "relevance": grade,
        }
        for query_id, image_grades in relevance.items()
        for image_id, grade in image_grades.items()
    ]
    review_rows = build_manual_review_rows(
        queries,
        records,
        relevance,
        seed=args.seed + 2,
    )
    write_jsonl(output_dir / "metadata.jsonl", records)
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_csv(output_dir / "auto_relevance.csv", qrel_rows)
    write_csv(output_dir / "manual_review_50.csv", review_rows)
    summary = {
        "schema_version": "a9-domain-transfer-subset-v1.0",
        "source_archive": str(archive),
        "seed": args.seed,
        "per_category": args.per_category,
        "medical_notice": (
            "仅用于模型跨领域行为研究，不用于临床诊断或医疗决策。"
        ),
        "art_selection": {
            key: len(value) for key, value in art_groups.items()
        },
        "medical_selection": {
            key: len(value) for key, value in medical_groups.items()
        },
        **dataset_summary(records),
        "query_count": len(queries),
        "manual_review_count": len(review_rows),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
