"""Upload top_models to Google Drive through an authenticated rclone remote."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="Configured rclone remote, e.g. gdrive")
    parser.add_argument("--folder", default="Placas/top_models")
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "top_models")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"No existe {args.source}; ejecuta random_search.py primero")
    subprocess.run(["rclone", "copy", str(args.source), f"{args.remote}:{args.folder}", "--progress"], check=True)
    print(f"Subidos modelos y configuraciones a {args.remote}:{args.folder}")


if __name__ == "__main__":
    main()
