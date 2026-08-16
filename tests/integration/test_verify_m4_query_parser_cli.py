from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verify_m4_query_parser_reads_jsonl_and_writes_auditable_rows(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            [
                json.dumps(
                    {"query_id": "q-query", "query": "雨夜城市"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"query_id": "q-text", "text": "无人机拍摄的城市"},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "m4.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "verify_m4_query_parser.py"),
            "--config",
            str(repository / "configs" / "default.yaml"),
            "--backend",
            "rules",
            "--queries-file",
            str(queries),
            "--output",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["query_id"] for row in rows] == ["q-query", "q-text"]
    assert [row["query"] for row in rows] == ["雨夜城市", "无人机拍摄的城市"]
    assert all(row["requested_backend"] == "rules" for row in rows)
    assert all(row["effective_backend"] == "rules" for row in rows)
    assert all(row["fallback_error"] is None for row in rows)
    assert all(row["elapsed_seconds"] >= 0 for row in rows)
    assert all(isinstance(row["parsed"], dict) for row in rows)


def test_verify_m4_query_parser_requires_query_input(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "verify_m4_query_parser.py"),
            "--config",
            str(repository / "configs" / "default.yaml"),
            "--backend",
            "rules",
            "--output",
            str(tmp_path / "m4.json"),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--query or --queries-file" in completed.stderr
