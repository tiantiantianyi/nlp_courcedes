# syntax=docker/dockerfile:1.7

ARG PIXI_VERSION=0.55.0
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.pixi/envs/default/bin:${PATH}"

# Keep dependency resolution cacheable and require the committed lock file.
COPY pixi.toml pixi.lock pyproject.toml ./
RUN pixi install --locked

# Runtime data, indexes, model weights, generated images, and secrets are mounts
# or environment variables. They are intentionally absent from image layers.
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY run.py README.md ./

# configs/default.yaml resolves ../Val to /Val. Keep that repository-relative
# contract while Compose mounts the host dataset at the explicit /data/Val path.
RUN mkdir -p /data /app/artifacts/generated && ln -s /data/Val /Val

EXPOSE 7860

CMD ["python", "scripts/launch_app.py", "--config", "configs/default.yaml", "--split", "val", "--host", "0.0.0.0", "--port", "7860"]
