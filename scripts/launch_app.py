from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anima_search.app.factory import create_service
from anima_search.app.ui import APP_CSS, build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Anima search interface.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(create_service(args.config, args.split))
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
