#!/usr/bin/env python3
"""Dependency-free local web server for the M1 three-model blind rating."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from blind_rating_common import (
    RATING_VERSION,
    blind_source_order,
    canonical_json,
    read_jsonl,
    reviewer_name,
    validate_rating,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rating-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "web",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RatingStore:
    def __init__(self, rating_dir: Path, workspace_root: Path) -> None:
        self.rating_dir = rating_dir.resolve()
        self.workspace_root = workspace_root.resolve()
        self.tasks_path = self.rating_dir / "rating_tasks.jsonl"
        self.manifest_path = self.rating_dir / "rating_manifest.json"
        self.db_path = self.rating_dir / "reviews.sqlite3"
        self.tasks = sorted(
            read_jsonl(self.tasks_path), key=lambda item: int(item["sample_index"])
        )
        self.tasks_by_id = {str(item["image_id"]): item for item in self.tasks}
        if len(self.tasks_by_id) != len(self.tasks):
            raise ValueError("rating tasks contain duplicate image IDs")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.blind_seed = int(manifest["blind_seed"])
        self._write_lock = threading.Lock()
        self._verify_images()
        self._init_database()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_database(self) -> None:
        self.rating_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    reviewer TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    task_sha256 TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK (phase IN ('draft', 'submitted')),
                    rating_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    submitted_at_utc TEXT,
                    PRIMARY KEY (reviewer, image_id)
                )
                """
            )

    def image_path(self, image_id: str) -> Path:
        task = self.tasks_by_id.get(image_id)
        if task is None:
            raise KeyError(image_id)
        path = (self.workspace_root / str(task["processed_path"])).resolve()
        if not path.is_relative_to(self.workspace_root):
            raise ValueError("image path escapes workspace root")
        return path

    def _verify_images(self) -> None:
        for task in self.tasks:
            path = self.image_path(str(task["image_id"]))
            if not path.is_file():
                raise FileNotFoundError(path)

    def row(self, reviewer: str, image_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE reviewer = ? AND image_id = ?",
                (reviewer, image_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "reviewer": row["reviewer"],
            "image_id": row["image_id"],
            "task_sha256": row["task_sha256"],
            "phase": row["phase"],
            "rating": json.loads(row["rating_json"]),
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "submitted_at_utc": row["submitted_at_utc"],
        }

    def summary(self, reviewer: str) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT image_id, phase FROM reviews WHERE reviewer = ?", (reviewer,)
            ).fetchall()
        statuses = {str(row["image_id"]): str(row["phase"]) for row in rows}
        counts = {"not_started": 0, "draft": 0, "submitted": 0}
        tasks = []
        for task in self.tasks:
            image_id = str(task["image_id"])
            phase = statuses.get(image_id, "not_started")
            counts[phase] += 1
            tasks.append(
                {
                    "image_id": image_id,
                    "sample_index": int(task["sample_index"]),
                    "phase": phase,
                }
            )
        return {
            "rating_version": RATING_VERSION,
            "reviewer": reviewer,
            "sample_size": len(self.tasks),
            "counts": counts,
            "tasks": tasks,
        }

    def public_task(self, image_id: str, reviewer: str) -> dict[str, Any]:
        task = self.tasks_by_id[image_id]
        candidates = {item["source_id"]: item for item in task["candidates"]}
        order = blind_source_order(image_id, reviewer, self.blind_seed)
        return {
            "rating_version": RATING_VERSION,
            "image_id": image_id,
            "sample_index": int(task["sample_index"]),
            "sample_size": len(self.tasks),
            "width": int(task["width"]),
            "height": int(task["height"]),
            "image_url": f"/api/image/{image_id}",
            "candidates": [
                {
                    "slot": f"candidate_{index + 1}",
                    "annotation": candidates[source_id]["annotation"],
                }
                for index, source_id in enumerate(order)
            ],
            "review": self.row(reviewer, image_id),
        }

    def save(
        self, image_id: str, reviewer: str, payload: Any, *, submit: bool
    ) -> dict[str, Any]:
        task = self.tasks_by_id[image_id]
        current = self.row(reviewer, image_id)
        if current and current["phase"] == "submitted":
            raise ValueError("该图片已经提交；如需修改，请先重新打开")
        rating = validate_rating(payload, require_complete=submit)
        phase = "submitted" if submit else "draft"
        now = utc_now()
        created = current.get("created_at_utc", now) if current else now
        submitted_at = now if submit else None
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO reviews (
                    reviewer, image_id, task_sha256, phase, rating_json,
                    created_at_utc, updated_at_utc, submitted_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reviewer, image_id) DO UPDATE SET
                    task_sha256 = excluded.task_sha256,
                    phase = excluded.phase,
                    rating_json = excluded.rating_json,
                    updated_at_utc = excluded.updated_at_utc,
                    submitted_at_utc = excluded.submitted_at_utc
                """,
                (
                    reviewer,
                    image_id,
                    task["task_sha256"],
                    phase,
                    canonical_json(rating),
                    created,
                    now,
                    submitted_at,
                ),
            )
        return self.row(reviewer, image_id) or {}

    def reopen(self, image_id: str, reviewer: str) -> dict[str, Any]:
        current = self.row(reviewer, image_id)
        if current is None:
            raise ValueError("该图片还没有保存记录")
        if current["phase"] != "submitted":
            return current
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "UPDATE reviews SET phase = 'draft', updated_at_utc = ?, "
                "submitted_at_utc = NULL WHERE reviewer = ? AND image_id = ?",
                (utc_now(), reviewer, image_id),
            )
        return self.row(reviewer, image_id) or {}


class RatingHandler(BaseHTTPRequestHandler):
    server_version = "M1BlindRating/1.0"
    store: RatingStore
    static_dir: Path

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {format_string % args}")

    def json_response(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def error_response(self, status: HTTPStatus, message: str) -> None:
        self.json_response({"error": message}, status)

    def parsed(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return unquote(parsed.path), parse_qs(parsed.query)

    def reviewer_from_query(self, query: dict[str, list[str]]) -> str:
        return reviewer_name((query.get("reviewer") or [""])[0])

    def request_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 200_000:
            raise ValueError("请求正文为空或过大")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, query = self.parsed()
            if path == "/api/health":
                self.json_response({"status": "ok", "rating_version": RATING_VERSION})
                return
            if path == "/api/summary":
                self.json_response(self.store.summary(self.reviewer_from_query(query)))
                return
            match = re.fullmatch(r"/api/task/([^/]+)", path)
            if match:
                image_id = match.group(1)
                if image_id not in self.store.tasks_by_id:
                    self.error_response(HTTPStatus.NOT_FOUND, "未知图片")
                    return
                self.json_response(
                    self.store.public_task(image_id, self.reviewer_from_query(query))
                )
                return
            match = re.fullmatch(r"/api/image/([^/]+)", path)
            if match:
                self.serve_image(match.group(1))
                return
            self.serve_static(path)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.error_response(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # pragma: no cover
            self.error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务器错误：{error}")

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path, query = self.parsed()
            reviewer = self.reviewer_from_query(query)
            match = re.fullmatch(r"/api/review/([^/]+)/(rating|reopen)", path)
            if not match:
                self.error_response(HTTPStatus.NOT_FOUND, "未知接口")
                return
            image_id, action = match.groups()
            if image_id not in self.store.tasks_by_id:
                self.error_response(HTTPStatus.NOT_FOUND, "未知图片")
                return
            if action == "reopen":
                result = self.store.reopen(image_id, reviewer)
            else:
                submit = (query.get("submit") or ["false"])[0].lower() == "true"
                result = self.store.save(
                    image_id, reviewer, self.request_json(), submit=submit
                )
            self.json_response({"review": result})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.error_response(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # pragma: no cover
            self.error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f"服务器错误：{error}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, PUT, OPTIONS")
        self.end_headers()

    def serve_image(self, image_id: str) -> None:
        if image_id not in self.store.tasks_by_id:
            self.error_response(HTTPStatus.NOT_FOUND, "未知图片")
            return
        path = self.store.image_path(image_id)
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (self.static_dir / relative).resolve()
        if not target.is_relative_to(self.static_dir) or not target.is_file():
            self.error_response(HTTPStatus.NOT_FOUND, "页面不存在")
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    args = parse_args()
    store = RatingStore(args.rating_dir, args.workspace_root)
    static_dir = args.static_dir.resolve()
    if not (static_dir / "index.html").is_file():
        raise FileNotFoundError(static_dir / "index.html")
    handler = type(
        "ConfiguredRatingHandler",
        (RatingHandler,),
        {"store": store, "static_dir": static_dir},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"M1 blind rating UI: http://{args.host}:{args.port}")
    print(f"Rating directory: {args.rating_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
