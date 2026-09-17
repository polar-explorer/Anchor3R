from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "checkpoints" / "release.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    *, repo_id: str, revision: str, filename: str, sha256: str, output_dir: Path,
) -> Path:
    if not repo_id or not revision:
        raise ValueError("Set the Hugging Face repo_id and immutable revision in checkpoints/release.json.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be the full 40-character Hugging Face commit hash, not main or a tag.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256 or ""):
        raise ValueError("Provide the published 64-character SHA256 checksum.")
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("filename must be a single file name, not a path.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    if destination.is_file() and sha256_file(destination) == sha256.lower():
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('Install download support first: pip install -e ".[hub]"') from exc
    with tempfile.TemporaryDirectory(prefix=".download-", dir=output_dir) as directory:
        downloaded = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, local_dir=directory,
        ))
        actual = sha256_file(downloaded)
        if actual != sha256.lower():
            raise ValueError(f"SHA256 mismatch: expected {sha256.lower()}, received {actual}. File not installed.")
        os.replace(downloaded, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser("Download and verify the Anchor3R inference checkpoint")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--sha256")
    parser.add_argument("--filename")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        destination = download_checkpoint(
            repo_id=args.repo_id or manifest.get("repo_id"),
            revision=args.revision or manifest.get("revision"),
            sha256=args.sha256 or manifest.get("sha256"),
            filename=args.filename or manifest.get("filename", "Anchor3R.pt"),
            output_dir=args.output_dir,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Verified checkpoint: {destination}")


if __name__ == "__main__":
    main()
