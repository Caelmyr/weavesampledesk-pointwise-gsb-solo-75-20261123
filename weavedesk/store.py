"""Local immutable evidence blobs with revision-checked print run metadata."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from image_lab.validation import Conflict, Invalid

from .model import identifier


def identity():
    return uuid4().hex[:24]


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode_doc(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="write-", delete=False) as handle:
        scratch = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(scratch, path)
        finally:
            scratch.unlink(missing_ok=True)


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.records = self.root / "readings"
        self.blobs = self.root / "blobs"
        self.guard = RLock()
        self.runs.mkdir(parents=True, exist_ok=True)

    def read(self, path):
        try:
            return json.loads(path.read_bytes())
        except FileNotFoundError as error:
            raise Invalid("This sample record is no longer available.") from error

    def run(self, key):
        return self.read(self.runs / (identifier(key) + ".json"))

    def list(self):
        documents = (self.read(path) for path in self.runs.glob("*.json"))
        return sorted(
            ({"id": run["id"], "title": run["title"], "stock": run["stock"], "updated": run["updated"]}
             for run in documents),
            key=lambda run: run["updated"], reverse=True,
        )

    def new_run(self, fields, photos=None, marks=None):
        with self.guard:
            current = stamp()
            run = {
                **fields, "id": identity(), "revision": 1, "created": current, "updated": current,
                "photos": photos or [], "marks": marks or [], "readings": [], "reference": None,
            }
            atomic_write(self.runs / (run["id"] + ".json"), encode_doc(run))
            return run

    def change(self, key, revision, apply):
        with self.guard:
            run = self.run(key)
            if type(revision) is not int or run["revision"] != revision:
                raise Conflict("The run changed. Reload it before saving.")
            apply(run)
            run["revision"] += 1
            run["updated"] = stamp()
            atomic_write(self.runs / (key + ".json"), encode_doc(run))
            return run

    def blob(self, digest):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Invalid("The evidence reference is invalid.")
        path = self.blobs / digest[:2] / digest
        if not path.is_file():
            raise Invalid("The stored image is unavailable.")
        return path

    def keep(self, data):
        digest = hashlib.sha256(data).hexdigest()
        path = self.blobs / digest[:2] / digest
        if not path.is_file():
            atomic_write(path, data)
        return digest

    def reading(self, run, reading_key):
        identifier(reading_key)
        detail = next((row for row in run["readings"] if row["id"] == reading_key), None)
        if detail is None:
            raise Invalid("That reading is outside this sample run.")
        return self.read(self.records / (reading_key + ".json"))

    def save_reading(self, key, revision, result):
        reading_key = identity()
        row = {**result, "id": reading_key, "recorded": stamp()}
        path = self.records / (reading_key + ".json")

        def append(run):
            if len(run["readings"]) >= 160:
                raise Invalid("This run already holds 160 saved readings.")
            atomic_write(path, encode_doc(row))
            run["readings"].append({k: row[k] for k in ("id", "mode", "title", "recorded")})

        try:
            updated = self.change(key, revision, append)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return updated, row
