"""Download and extract Caltech-256 (~1.2 GB) from CaltechDATA.

    python scripts/download_caltech256.py --out data
"""
from __future__ import annotations

import argparse
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"


def download(tar_fp: Path, max_attempts: int = 50) -> None:
    """Resumable download: on a stall or dropped connection, continue from the bytes on disk."""
    part = tar_fp.with_suffix(".part")
    total = None
    for attempt in range(max_attempts):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            # read timeout: abort a connection that stops delivering data, then resume
            with requests.get(URL, stream=True, timeout=(30, 60), headers=headers) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the range; start over
                    have = 0
                    part.unlink(missing_ok=True)
                total = have + int(r.headers.get("content-length", 0))
                with open(part, "ab") as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                                  desc=f"download (try {attempt + 1})") as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
        except requests.RequestException as e:
            print(f"connection dropped ({e.__class__.__name__}); resuming")
            time.sleep(min(30, 2 ** attempt))
    if total is None or part.stat().st_size < total:
        raise SystemExit("download did not complete; re-run to resume")
    part.replace(tar_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tar_fp = out / "256_ObjectCategories.tar"

    if not tar_fp.exists():
        download(tar_fp)

    if not (out / "256_ObjectCategories").is_dir():
        with tarfile.open(tar_fp) as tar:
            tar.extractall(out, filter="data")
    n = sum(1 for _ in (out / "256_ObjectCategories").rglob("*.jpg"))
    print(f"Caltech-256 ready: {out / '256_ObjectCategories'} ({n} images)")


if __name__ == "__main__":
    main()
