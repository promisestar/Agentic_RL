#!/usr/bin/env python3
"""Download sentence-transformers/all-MiniLM-L6-v2 into the local Hugging Face cache.

Usage (from repo root):
    uv run python tests/download_embedding_model.py
    uv run python tests/download_embedding_model.py --force

Default cache (Windows):
    %USERPROFILE%\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\\

After a successful download, you can run the offline smoke test:
    uv run python tests/test_embedding_model.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("download_embedding_model")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _hub_cache_dir() -> Path:
    """Resolve the HF hub cache root (respects HF_HOME / HUGGINGFACE_HUB_CACHE)."""
    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(cache)
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cache_dir(model_id: str = MODEL_ID) -> Path:
    # Hub layout: models--org--name
    folder = "models--" + model_id.replace("/", "--")
    return _hub_cache_dir() / folder


def download(model_id: str, force: bool = False) -> Path:
    """Download the full model snapshot into the HF hub cache and return local path."""
    # Ensure we are online for this script
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from huggingface_hub import snapshot_download

    logger.info("Downloading %s into HF hub cache ...", model_id)
    logger.info("Cache root: %s", _hub_cache_dir())

    local_dir = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        force_download=force,
        resume_download=True,
    )
    local_path = Path(local_dir)
    logger.info("Snapshot path: %s", local_path)

    # Quick sanity check: expected weight / config files
    expected = ["config.json", "tokenizer.json"]
    weight_candidates = ["model.safetensors", "pytorch_model.bin"]
    missing = [name for name in expected if not (local_path / name).exists()]
    if not any((local_path / name).exists() for name in weight_candidates):
        missing.append("model.safetensors|pytorch_model.bin")
    if missing:
        raise FileNotFoundError(
            f"Download finished but required files missing under {local_path}: {missing}"
        )

    files = sorted(p.name for p in local_path.iterdir() if p.is_file() or p.is_symlink())
    logger.info("Files in snapshot (%d): %s", len(files), ", ".join(files))
    return local_path


def verify_load(model_id: str) -> None:
    """Load via SentenceTransformer from cache to confirm the download is usable."""
    from sentence_transformers import SentenceTransformer

    logger.info("Verifying load with SentenceTransformer (local_files_only=True) ...")
    model = SentenceTransformer(model_id, local_files_only=True)
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    logger.info("OK device=%s dim=%s", model.device, dim)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download {MODEL_ID} into the local Hugging Face cache",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"HF model id (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist in cache",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SentenceTransformer load check after download",
    )
    args = parser.parse_args()

    try:
        snapshot = download(args.model, force=args.force)
        print(f"Downloaded to: {snapshot}")
        print(f"Model cache dir: {_model_cache_dir(args.model)}")
        if not args.skip_verify:
            verify_load(args.model)
    except Exception as exc:
        logger.exception("Download failed: %s", exc)
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
