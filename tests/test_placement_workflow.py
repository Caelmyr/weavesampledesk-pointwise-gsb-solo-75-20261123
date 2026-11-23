"""End-to-end checks for placement-match and seam-coverage readings.

Run from the repository root with:  python3 -I tests/test_placement_workflow.py
The script creates an isolated demo run through the real Flask routes and never
touches the on-disk sample library used by the service.
"""
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
# -I/isolated mode drops the ambient PYTHONPATH, so add third-party packages explicitly.
for candidate in ("/opt/pointwise-env/site-packages", "/usr/local/lib/python3.11/dist-packages",
                  "/usr/lib/python3/dist-packages", "/usr/lib/python3.11/lib-dynload"):
    if Path(candidate).is_dir():
        sys.path.append(candidate)
sys.path.insert(0, str(REPOSITORY))

import numpy as np  # noqa: E402

from image_lab import create_app  # noqa: E402

from weavedesk import analysis  # noqa: E402
from weavedesk.photos import Frame  # noqa: E402
from weavedesk.web import install  # noqa: E402


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def make_frames():
    """Two same-size calibrated frames; the comparison support edge crosses the seam window so that a
    translation changes how much of the window is jointly visible."""
    height, width = 200, 240
    step = (1.0, 1.0)  # 1 pixel == 1 mm for transparent math
    rgb = np.full((height, width, 3), 0.5, dtype=np.float32)
    support_a = np.ones((height, width), dtype=bool)          # reference covers the whole plane
    support_b = np.zeros((height, width), dtype=bool)
    support_b[:, :180] = True                                 # comparison ends at column 179
    return Frame(rgb.copy(), support_a, step), Frame(rgb.copy(), support_b, step)


def make_seam():
    # The seam window straddles the comparison support edge (column 179).
    return [(170, 40), (200, 40), (200, 160), (170, 160)]


def register_marks(points_a, points_b, seam=None):
    marks = {
        "reference": {"points": [[x / 239, y / 199] for x, y in points_a]},
        "comparison": {"points": [[x / 239, y / 199] for x, y in points_b]},
    }
    if seam is not None:
        marks["seam"] = {"points": [[x / 239, y / 199] for x, y in seam]}
    return marks


def test_selection_rules():
    reference, comparison = make_frames()
    centres = [(60, 60), (120, 140)]
    seam = make_seam()

    # Coverage tie-break: code maps moved = comparison + shift, residual = reference - (comparison + shift).
    # With comparison centres equal to reference, opposite +/-2 shifts give both candidates a worst-centre
    # residual of 2 mm. The +2 shift extends the comparison support edge (column 179) to 181, covering more
    # of the seam window jointly; -2 pulls it back to 177. Accuracy ties, so joint seam coverage decides.
    marks = register_marks(centres, centres, seam)
    summary = analysis.placement_match(reference, comparison, marks, {
        "candidate_a_dx": -2, "candidate_a_dy": 0,
        "candidate_b_dx": 2, "candidate_b_dy": 0, "tolerance_mm": 5,
    })[0]
    print("coverage-rule summary:", {k: summary[k] for k in
          ("chosen_candidate", "chosen_max_residual_mm", "chosen_seam_both_pct", "selection_reason")})
    assert summary["chosen_max_residual_mm"] == 2.0
    assert summary["chosen_candidate"] == "B"
    assert "seam visible in both photos" in summary["selection_reason"]

    # Accuracy rule (no seam window): the exact-correcting candidate wins on the smaller worst-centre residual.
    marks = register_marks(centres, [(62, 60), (122, 140)])  # comparison centres 2 mm right of reference
    summary = analysis.placement_match(reference, comparison, marks, {
        "candidate_a_dx": -2, "candidate_a_dy": 0,       # moves comparison back -> residual 0
        "candidate_b_dx": 0, "candidate_b_dy": 0,        # leaves the 2 mm offset
        "tolerance_mm": 5,
    })[0]
    assert summary["chosen_candidate"] == "A"
    assert summary["chosen_max_residual_mm"] == 0.0

    # Deterministic full tie (identical candidates): label A is chosen, no crash.
    summary = analysis.placement_match(reference, comparison, register_marks(centres, centres), {
        "candidate_a_dx": 0, "candidate_a_dy": 0, "candidate_b_dx": 0, "candidate_b_dy": 0,
        "tolerance_mm": 1,
    })[0]
    assert summary["chosen_candidate"] == "A"

    # Rule 3: neither candidate within tolerance -> Invalid, message names both maxima.
    try:
        analysis.placement_match(
            reference, comparison, register_marks(centres, centres),
            {"candidate_a_dx": 10, "candidate_a_dy": 0, "candidate_b_dx": -10,
             "candidate_b_dy": 0, "tolerance_mm": 1},
        )
    except Exception as error:  # noqa: BLE001
        assert "tolerance" in str(error).lower()
    else:
        raise AssertionError("placement_match should reject candidates that miss the tolerance")


def main():
    test_selection_rules()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        app = create_app(root / "preparation")
        install(app, root / "samples")
        client = app.test_client()

        run = client.post("/weave/demo").get_json()
        key = run["id"]
        face = next(p for p in run["photos"] if p["role"] == "face")
        rake = next(p for p in run["photos"] if p["role"] == "rake")
        ref_mark = next(m for m in run["marks"] if m["photo"] == face["id"] and m["purpose"] == "register")
        cmp_mark = next(m for m in run["marks"] if m["photo"] == rake["id"] and m["purpose"] == "register")

        # First discover the true translation between the two calibrated photos.
        offset_resp = client.post(f"/weave/runs/{key}/readings", json={
            "revision": run["revision"], "mode": "motif-offset", "photo": face["id"], "other": rake["id"],
            "marks": {"reference": ref_mark["id"], "comparison": cmp_mark["id"]}, "options": {},
        })
        assert offset_resp.status_code == 201, offset_resp.get_json()
        offset = offset_resp.get_json()
        run = offset["run"]
        suggested = offset["reading"]["summary"]["suggested_shift_mm"]
        sx, sy = suggested
        print("motif suggested shift (mm):", suggested)

        # Mark a seam window on the reference (face) photograph.
        mark_resp = client.post(f"/weave/runs/{key}/marks", json={
            "revision": run["revision"], "photo": face["id"], "title": "Join seam window",
            "purpose": "seam", "points": box(0.55, 0.55, 0.8, 0.8),
        })
        assert mark_resp.status_code == 201, mark_resp.get_json()
        run = mark_resp.get_json()
        seam_mark = run["marks"][-1]
        assert seam_mark["purpose"] == "seam"

        def placement(a, b, tolerance, use_seam):
            marks = {"reference": ref_mark["id"], "comparison": cmp_mark["id"]}
            if use_seam:
                marks["seam"] = seam_mark["id"]
            return client.post(f"/weave/runs/{key}/readings", json={
                "revision": run["revision"], "mode": "placement-match", "photo": face["id"], "other": rake["id"],
                "marks": marks,
                "options": {"candidate_a_dx": a[0], "candidate_a_dy": a[1],
                            "candidate_b_dx": b[0], "candidate_b_dy": b[1], "tolerance_mm": tolerance},
            })

        # 1) Candidate A matches the true translation; B is deliberately worse. Both within 3 mm.
        good = (round(sx, 4), round(sy, 4))
        worse = (round(sx + 1.5, 4), round(sy - 1.0, 4))
        response = placement(good, worse, 3.0, True)
        assert response.status_code == 201, response.get_json()
        reading = response.get_json()["reading"]
        run = response.get_json()["run"]
        summary = reading["summary"]
        print("placement summary:", summary)
        assert summary["chosen_candidate"] == "A"
        assert set(summary["passing_candidates"]) == {"A", "B"}
        assert isinstance(summary["chosen_max_residual_mm"], (int, float))
        assert summary["chosen_seam_both_pct"] is not None and summary["chosen_seam_both_pct"] >= 0
        assert "Both placements passed" in summary["selection_reason"]
        placement_id = reading["id"]

        # The candidate rows expose per-candidate pass/fail and the seam metric.
        candidate_rows = [r for r in reading["rows"] if "candidate" in r and "shift_x_mm" in r]
        assert {r["candidate"] for r in candidate_rows} == {"A", "B"}
        assert all(isinstance(r["seam_both_visible_pct"], (int, float)) for r in candidate_rows)
        centre_rows = [r for r in reading["rows"] if "residual_mm" in r and "point" in r]
        assert len(centre_rows) == len(ref_mark["points"])

        # 2) Tight tolerance: the worse candidate must fail and only A remain valid.
        response = placement(good, worse, 0.6, True)
        assert response.status_code == 201
        tight = response.get_json()["reading"]["summary"]
        print("tight summary:", tight)
        assert tight["passing_candidates"] == ["A"]
        assert tight["chosen_candidate"] == "A"
        run = response.get_json()["run"]

        # 3) Tolerance so strict that neither passes -> explicit error, no reading saved.
        before = len(run["readings"])
        response = placement((round(sx + 0.4, 4), sy), (round(sx - 0.4, 4), sy), 0.05, True)
        assert response.status_code == 400
        assert "tolerance" in response.get_json()["error"].lower()
        run = client.get(f"/weave/runs/{key}").get_json()
        assert len(run["readings"]) == before

        # 4) Seam coverage under the saved placement.
        seam_response = client.post(f"/weave/runs/{key}/readings", json={
            "revision": run["revision"], "mode": "seam-coverage", "photo": face["id"], "other": rake["id"],
            "marks": {"seam": seam_mark["id"]}, "options": {},
            "placement": {"reading": placement_id},
        })
        assert seam_response.status_code == 201, seam_response.get_json()
        seam_reading = seam_response.get_json()["reading"]
        run = seam_response.get_json()["run"]
        print("seam summary:", seam_reading["summary"])
        labels = {r["observation"]: r["percent"] for r in seam_reading["rows"]}
        assert set(labels) == {"Both photos", "Reference only", "Comparison only", "Neither photo"}
        assert abs(sum(labels.values()) - 100.0) < 0.01
        # The saved placement snapshot must be carried into the new reading.
        assert seam_reading["options"]["placement"]["source_reading"] == placement_id
        assert seam_reading["summary"]["placement_shift_mm"] == reading["summary"]["adopted_shift_mm"]

        # 5) A later ordinary motif-offset reading must not mutate the saved placement or its photo blobs.
        original_blob = reading["photos"][0]["blob"]
        original_shift = reading["summary"]["adopted_shift_mm"]
        later = client.post(f"/weave/runs/{key}/readings", json={
            "revision": run["revision"], "mode": "motif-offset", "photo": face["id"], "other": rake["id"],
            "marks": {"reference": ref_mark["id"], "comparison": cmp_mark["id"]}, "options": {},
        })
        assert later.status_code == 201
        reopened = client.get(f"/weave/runs/{key}/readings/{placement_id}").get_json()
        assert reopened["summary"]["adopted_shift_mm"] == original_shift
        assert reopened["photos"][0]["blob"] == original_blob
        assert reopened["mode"] == "placement-match"

        # 6) seam-coverage rejects a non-placement reading id.
        bad = client.post(f"/weave/runs/{key}/readings", json={
            "revision": run["revision"] + 1, "mode": "seam-coverage", "photo": face["id"], "other": rake["id"],
            "marks": {"seam": seam_mark["id"]}, "options": {},
            "placement": {"reading": later.get_json()["reading"]["id"]},
        })
        assert bad.status_code in (400, 409)

        print("\nAll placement workflow checks passed.")


if __name__ == "__main__":
    main()
