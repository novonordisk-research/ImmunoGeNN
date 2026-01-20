from __future__ import annotations

import os
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

DATA_RECORD_URL = "https://github.com/esben-kok/ImmunoGeNN/releases/download/v0.1.0/data_record.zip"

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_RECORD_DIR = PACKAGE_ROOT / "data_record"
DATA_RECORD_MARKER = DATA_RECORD_DIR / "human_references_9mers.pkl.lz4"
DATA_RECORD_ARCHIVE = PACKAGE_ROOT / "data_record.zip"


def ensure_data_record(force: bool = False) -> Path:
    if DATA_RECORD_MARKER.exists() and not force:
        return DATA_RECORD_DIR

    DATA_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    if force and DATA_RECORD_ARCHIVE.exists():
        DATA_RECORD_ARCHIVE.unlink()

    if not DATA_RECORD_ARCHIVE.exists():
        urllib.request.urlretrieve(DATA_RECORD_URL, DATA_RECORD_ARCHIVE)

    with zipfile.ZipFile(DATA_RECORD_ARCHIVE, "r") as archive:
        archive.extractall(PACKAGE_ROOT)

    if DATA_RECORD_ARCHIVE.exists():
        DATA_RECORD_ARCHIVE.unlink()

    return DATA_RECORD_DIR


@contextmanager
def package_working_directory():
    previous = Path.cwd()
    os.chdir(PACKAGE_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


__all__ = ["ensure_data_record", "package_working_directory", "PACKAGE_ROOT"]
