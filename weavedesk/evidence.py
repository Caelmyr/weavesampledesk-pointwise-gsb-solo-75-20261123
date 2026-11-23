"""Create readable, self-contained local proof evidence downloads."""
import csv
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

from .store import encode_doc


def field(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def rows_csv(reading):
    rows = reading["rows"]
    output = io.StringIO(newline="")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if keys:
        report = csv.DictWriter(output, fieldnames=keys)
        report.writeheader()
        report.writerows({key: field(value) for key, value in row.items()} for row in rows)
    return output.getvalue().encode("utf-8")


def packet(library, reading):
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("reading.json", encode_doc(reading))
        archive.writestr("measured-values.csv", rows_csv(reading))
        archive.writestr("proof-overlay.png", library.blob(reading["overlay"]).read_bytes())
        for index, image in enumerate(reading["photos"], 1):
            archive.writestr(f"source/{index}-{image['filename']}", library.blob(image["blob"]).read_bytes())
    return output.getvalue()
