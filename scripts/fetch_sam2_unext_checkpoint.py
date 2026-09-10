from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


OFFICIAL_DRIVE_ID = "1F9B1fSJF3c2YM68KMvrsPwZr4ZTFaRgf"


def _download(file_id: str, output: Path) -> Path:
    try:
        import gdown
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint download requires gdown. Install requirements-sam2-unext.txt first."
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(output), quiet=False)
    if not result:
        raise RuntimeError("Google Drive checkpoint download failed")
    return output


def _extract_checkpoint(archive: Path, destination: Path) -> Path:
    if not zipfile.is_zipfile(archive):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archive), str(destination))
        return destination

    with zipfile.ZipFile(archive, "r") as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith((".pth", ".pt", ".ckpt"))]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one checkpoint in archive, found {len(candidates)}: {candidates[:10]}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(candidates[0], "r") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    archive.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the official SAM2-UNeXT DIS5K checkpoint")
    parser.add_argument("--drive-id", default=OFFICIAL_DRIVE_ID)
    parser.add_argument("--output", default="weights/sam2_unext_dis5k.pth")
    args = parser.parse_args()

    destination = Path(args.output).expanduser().resolve()
    temp = destination.with_suffix(destination.suffix + ".download")
    downloaded = _download(args.drive_id, temp)
    checkpoint = _extract_checkpoint(downloaded, destination)
    size = checkpoint.stat().st_size
    if size < 100_000_000:
        raise RuntimeError(f"Checkpoint looks unexpectedly small: {size} bytes")
    print(checkpoint)


if __name__ == "__main__":
    main()
