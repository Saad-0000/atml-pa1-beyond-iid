import argparse
import os
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from shared.pacs import PACS_DOMAINS, PACS_CLASSES

# Public PACS archive
PACS_DOWNLOAD_URL = "https://huggingface.co/datasets/Azeez577/PACS/resolve/main/PACS.zip?download=true"


def _missing_pacs_dirs(data_root):
    return [
        str(data_root / domain / class_name)
        for domain in PACS_DOMAINS
        for class_name in PACS_CLASSES
        if not (data_root / domain / class_name).is_dir()
    ]


def _validate_pacs_root(data_root):
    missing_dirs = _missing_pacs_dirs(data_root)
    if missing_dirs:
        preview = ", ".join(missing_dirs[:3])
        raise FileNotFoundError(
            f"PACS dataset is incomplete or missing under {data_root}. "
            f"Expected domain/class directories such as: {preview}"
        )


def _safe_extract(archive, destination):
    """Extract a zip only if every member stays inside ``destination``."""
    destination = destination.resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if member_path != destination and destination not in member_path.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def download_pacs(data_root="data/PACS", url=PACS_DOWNLOAD_URL):
    """Download and unpack PACS into ``data_root`` without overwriting data.

    The archive's top-level directory differs between PACS mirrors, so this
    function locates the directory containing all four domain folders and
    moves that directory to the requested dataset root.
    """
    data_root = Path(data_root)
    if not _missing_pacs_dirs(data_root):
        print(f"PACS is already available at {data_root}.")
        return data_root
    if data_root.exists():
        raise FileExistsError(
            f"{data_root} already exists but is not a complete PACS dataset. "
            "Move or remove it manually before downloading."
        )

    data_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PACS to {data_root}...")

    with tempfile.TemporaryDirectory(
        prefix="pacs-download-", dir=data_root.parent
    ) as temp_dir:
        temp_dir = Path(temp_dir)
        archive_path = temp_dir / "PACS.zip"
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        print("Downloading from:", url)
        urllib.request.urlretrieve(url, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extract_dir)

        candidates = [extract_dir]
        candidates.extend(
            path for path in extract_dir.rglob("*") if path.is_dir()
        )

        source_root = next(
            (
                path
                for path in candidates
                if all(
                    (path / domain).is_dir()
                    for domain in PACS_DOMAINS
                )
            ),
            None,
        )

        if source_root is None:
            raise ValueError(
                "The downloaded archive does not contain the expected PACS domain folders."
            )

        shutil.move(str(source_root), str(data_root))

    _validate_pacs_root(data_root)
    print(f"PACS is ready at {data_root}.")
    return data_root


def create_or_load_pacs_splits(
    data_root,
    split_json_path="splits/pacs_sketch_seed6304.json",
    seed=6304
):
    data_root = Path(data_root)
    _validate_pacs_root(data_root)

    split_path = Path(split_json_path)
    if split_path.exists():
        with open(split_path, "r") as f:
            return json.load(f)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    splits = {
        "sources": {"photo": {}, "art_painting": {}, "cartoon": {}},
        "target": {"sketch": []}
    }

    # Collect source splits (stratified 80/20 per domain)
    for domain in ["photo", "art_painting", "cartoon"]:
        splits["sources"][domain]["train"] = []
        splits["sources"][domain]["val"] = []

        for label_idx, cname in enumerate(PACS_CLASSES):
            class_dir = data_root / domain / cname

            imgs = sorted([
                str(p.resolve())
                for p in class_dir.glob("*.*")
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ])

            rng.shuffle(imgs)

            n_val = int(round(len(imgs) * 0.20))

            splits["sources"][domain]["val"].extend(
                [(p, label_idx, domain) for p in imgs[:n_val]]
            )

            splits["sources"][domain]["train"].extend(
                [(p, label_idx, domain) for p in imgs[n_val:]]
            )

    # Collect target (all Sketch images)
    for label_idx, cname in enumerate(PACS_CLASSES):
        class_dir = data_root / "sketch" / cname

        imgs = sorted([
            str(p.resolve())
            for p in class_dir.glob("*.*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ])

        splits["target"]["sketch"].extend(
            [(p, label_idx, "sketch") for p in imgs]
        )

    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PACS into the layout used by Task 2."
    )
    parser.add_argument("--data_root", default="data/PACS")
    args = parser.parse_args()

    download_pacs(args.data_root)