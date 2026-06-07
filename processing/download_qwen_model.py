from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUTPUT_DIR = Path("models") / "Qwen2.5-1.5B-Instruct"

FILES = {
    ".gitattributes": 1519,
    "LICENSE": 11343,
    "README.md": 4917,
    "config.json": 660,
    "generation_config.json": 242,
    "merges.txt": 1671839,
    "model.safetensors": 3087467144,
    "tokenizer.json": 7031645,
    "tokenizer_config.json": 7305,
    "vocab.json": 2776833,
}


def download_file(endpoint: str, repo: str, filename: str, expected_size: int, output_dir: Path) -> None:
    target = output_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        print(f"[skip] {filename} already complete")
        return

    url = f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}"
    print(f"[download] {filename} -> {target}")
    command = [
        "curl.exe",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "30",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--noproxy",
        "*",
        "--output",
        str(target),
        url,
    ]
    env = dict(os.environ)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    subprocess.run(command, check=True, env=env)
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"{filename} size mismatch: expected {expected_size}, got {actual_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-1.5B-Instruct from a mirror with curl resume.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", nargs="*", help="Optional file names to download.")
    args = parser.parse_args()

    selected = FILES
    if args.only:
        unknown = sorted(set(args.only) - set(FILES))
        if unknown:
            raise SystemExit(f"Unknown files: {unknown}")
        selected = {name: FILES[name] for name in args.only}

    for filename, expected_size in selected.items():
        download_file(args.endpoint, args.repo, filename, expected_size, args.output_dir)
    print(f"[done] model files are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
