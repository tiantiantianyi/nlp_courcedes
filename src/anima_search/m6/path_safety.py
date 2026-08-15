from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def reject_output_path_aliases(
    *,
    read_only: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> None:
    """Reject direct and symlink aliases before any output is opened."""
    resolved_inputs = {
        name: path.resolve() for name, path in read_only.items()
    }
    resolved_outputs = {
        name: path.resolve() for name, path in outputs.items()
    }

    for output_name, output_path in resolved_outputs.items():
        for input_name, input_path in resolved_inputs.items():
            if output_path == input_path:
                raise ValueError(
                    "output path alias: "
                    f"{output_name} resolves to read-only {input_name}"
                )

    output_items = list(resolved_outputs.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if left_path == right_path:
                raise ValueError(
                    "output path alias: "
                    f"{left_name} and {right_name} resolve to the same path"
                )
