from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.app.ui import APP_CSS, build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the AskAlbum search interface.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    service = create_service(args.config, args.split)
    app = build_app(service)
    project_root = Path(service.config["project_root"]).resolve()
    data_dir = (
        project_root / service.config["data"][f"{args.split}_dir"]
    ).resolve()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=APP_CSS,
        allowed_paths=[str(data_dir)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
