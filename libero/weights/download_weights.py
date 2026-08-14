#!/usr/bin/env python3
"""Fetch the released PSG-JEPA LIBERO-Goal checkpoints and verify them against `manifest.json`.

    python weights/download_weights.py                     # encoder + head (~0.33 GiB)
    python weights/download_weights.py --kind encoder      # encoder only (~0.07 GiB)
    python weights/download_weights.py --verify-only       # re-check what is on disk

Files land under `--dest` (default `weights/checkpoints/`) in the manifest's layout:

    <dest>/<variant>/seed<S>/encoder_epoch40.ckpt
    <dest>/<variant>/seed<S>/oft_head_epoch30.ckpt

Every file is checked against the sha256 recorded in the manifest; an existing file with a
matching hash is left alone, so re-running resumes rather than re-downloading. A file that is
present but hash-mismatched is reported and re-fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
DEFAULT_REPO = "Haodong082399/PSG-JEPA-LIBERO"
CHUNK = 8 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def select(files: list[dict], args: argparse.Namespace) -> list[dict]:
    out = files
    if args.kind:
        out = [f for f in out if f["kind"] == args.kind]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="Hugging Face model repo")
    parser.add_argument("--kind", choices=("encoder", "head"))
    parser.add_argument("--verify-only", action="store_true", help="no download, just re-hash")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    wanted = select(manifest["files"], args)
    if not wanted:
        raise SystemExit("no files matched the given filters")
    total = sum(f["bytes"] for f in wanted)
    print(f"{len(wanted)} files, {total / 2**30:.2f} GiB -> {args.dest}")

    hf_download = None
    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit("pip install huggingface_hub, or pass --verify-only")
        hf_download = hf_hub_download

    bad = []
    for i, entry in enumerate(wanted, 1):
        dst = args.dest / entry["path"]
        label = f"[{i:2d}/{len(wanted)}] {entry['path']}"

        if dst.exists() and sha256(dst) == entry["sha256"]:
            print(f"{label}: ok")
            continue
        if args.verify_only:
            print(f"{label}: {'MISMATCH' if dst.exists() else 'missing'}")
            bad.append(entry["path"])
            continue
        if dst.exists():
            print(f"{label}: hash mismatch, re-downloading")
            dst.unlink()

        dst.parent.mkdir(parents=True, exist_ok=True)
        fetched = hf_download(repo_id=args.repo_id, filename=entry["path"],
                              local_dir=args.dest, repo_type="model")
        if sha256(Path(fetched)) != entry["sha256"]:
            print(f"{label}: FAILED sha256 after download")
            bad.append(entry["path"])
        else:
            print(f"{label}: downloaded")

    if bad:
        print(f"\n{len(bad)} file(s) failed verification:")
        for path in bad:
            print("   ", path)
        sys.exit(1)
    print("\nall files verified")


if __name__ == "__main__":
    main()
