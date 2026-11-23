"""Tests for raised-fiber comparison and saved-reading point inspection."""
import tempfile
import unittest

import numpy as np

from weavedesk.__main__ import application
from weavedesk import analysis, photos


def demo(client):
    run = client.post("/weave/demo").get_json()
    face = next(item for item in run["photos"] if item["role"] == "face")
    rake = next(item for item in run["photos"] if item["role"] == "rake")
    marks = {mark["purpose"]: mark["id"] for mark in run["marks"] if mark["photo"] == face["id"]}
    return run, face, rake, marks


def reading(client, run, face, rake, marks, **options):
    body = {
        "revision": run["revision"], "mode": "raised-fiber", "photo": face["id"], "other": rake["id"],
        "marks": {"region": marks["inspect"], "confirmed": marks["fuzz"], "sheen": marks["sheen"]},
        "options": {"minimum_mm2": .15, **options},
    }
    response = client.post(f"/weave/runs/{run['id']}/readings", json=body)
    return response, response.get_json()


class RaisedFiberTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.client = application(self.root.name).test_client()

    def test_reading_measures_both_reference_patches(self):
        run, face, rake, marks = demo(self.client)
        response, data = reading(self.client, run, face, rake, marks)
        self.assertEqual(response.status_code, 201, data)
        summary = data["reading"]["summary"]
        self.assertIn(summary["adopted_rule"], {"both", "extra"})
        self.assertGreater(summary["retained_pct"], 90)
        self.assertLess(summary["false_mark_pct"], 5)
        self.assertEqual([row["rule"] for row in data["reading"]["rows"]], ["Both lights", "Raking excess"])
        self.assertEqual(sum(row["adopted"] == "Yes" for row in data["reading"]["rows"]), 1)

    def test_gate_choice_prefers_retention_when_both_qualify(self):
        run, face, rake, marks = demo(self.client)
        _response, data = reading(self.client, run, face, rake, marks, keep_min=.3, false_max=.5)
        summary = data["reading"]["summary"]
        eligible = summary["passing_rules"]
        self.assertTrue(eligible)
        scores = {row["rule"]: row["retained_pct"] for row in data["reading"]["rows"]}
        best = max(eligible, key=lambda rule: scores["Both lights" if rule == "both" else "Raking excess"])
        self.assertEqual(summary["adopted_rule"], best)

    def test_impossible_gates_still_adopt_one_rule(self):
        run, face, rake, marks = demo(self.client)
        _response, data = reading(self.client, run, face, rake, marks, keep_min=1, false_max=0)
        summary = data["reading"]["summary"]
        self.assertEqual(summary["passing_rules"], [])
        self.assertIn(summary["adopted_rule"], {"both", "extra"})

    def test_reference_marks_must_be_clear_in_both_photographs(self):
        run, face, rake, marks = demo(self.client)
        body = {
            "revision": run["revision"], "mode": "raised-fiber", "photo": face["id"], "other": rake["id"],
            "marks": {"region": marks["inspect"], "confirmed": marks["fuzz"], "sheen": marks["sheen"]},
            "options": {},
        }
        # The raking photograph has an unlit gap inside the face-light inspection patch.
        body["marks"]["confirmed"] = self._mark_over_gap(run, face)
        body["revision"] = self.updated["revision"]
        response = self.client.post(f"/weave/runs/{run['id']}/readings", json=body)
        self.assertEqual(response.status_code, 400)
        self.assertIn("clear in both", response.get_json()["error"])

    def _mark_over_gap(self, run, face):
        # A fuzz reference on the face photograph that sits over the unlit raking gap.
        points = [[.54, .534], [.555, .534], [.555, .55], [.54, .55]]
        response = self.client.post(f"/weave/runs/{run['id']}/marks", json={
            "revision": run["revision"], "photo": face["id"], "title": "Gap", "purpose": "fuzz", "points": points,
        })
        self.assertEqual(response.status_code, 201)
        self.updated = response.get_json()
        return self.updated["marks"][-1]["id"]


class PointInspectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.client = application(self.root.name).test_client()
        run, face, rake, marks = demo(self.client)
        response, data = reading(self.client, run, face, rake, marks)
        self.assertEqual(response.status_code, 201)
        self.run_id = data["run"]["id"]
        self.reading_id = data["reading"]["id"]
        self.face = face
        self.rake = rake

    def point(self, x, y):
        response = self.client.get(f"/weave/runs/{self.run_id}/readings/{self.reading_id}/point?x={x}&y={y}")
        return response, response.get_json()

    def test_both_photographs_report_position_and_brightness(self):
        response, data = self.point(.48, .6)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["role"] for item in data["photos"]], ["face", "rake"])
        for item in data["photos"]:
            self.assertEqual(item["view"]["status"], "Clear")
            self.assertIsNotNone(item["view"]["luma"])
            self.assertEqual(len(item["location"]["pixel"]), 2)
        self.assertEqual(data["missing"], [])

    def test_unclear_photograph_is_named(self):
        _response, data = self.point(.6888, .7155)
        statuses = {item["role"]: item["view"]["status"] for item in data["photos"]}
        self.assertNotEqual(statuses["face"], "Clear")
        self.assertEqual(statuses["rake"], "Clear")
        self.assertEqual(data["missing"], [self.face["title"]])

    def test_original_photograph_can_be_opened_at_the_point(self):
        response = self.client.get(
            f"/weave/runs/{self.run_id}/readings/{self.reading_id}/point/{self.rake['id']}/image?x=.48&y=.6")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")

    def test_rejects_positions_outside_the_report(self):
        response, _data = self.point(1.4, .5)
        self.assertEqual(response.status_code, 400)


class GeometryTests(unittest.TestCase):
    def test_plane_position_round_trips_to_source(self):
        run = {"width_mm": 100, "height_mm": 80}
        shape = (400, 500, 4)
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        width, height = photos.calibrated_size(run)
        located = photos.original_point(corners, run, shape, .25, .75)
        self.assertAlmostEqual(located["x"], .25, places=3)
        self.assertAlmostEqual(located["y"], .75, places=3)
        self.assertEqual(located["width"], 500)

    def test_sample_flags_transparent_and_unlit_pixels(self):
        pixels = np.full((40, 40, 4), 200, dtype=np.uint8)
        pixels[:, :, 3] = 255
        self.assertEqual(photos.sample(pixels, .5, .5)["status"], "Clear")
        self.assertAlmostEqual(photos.sample(pixels, .5, .5)["luma"], 200 / 255, places=3)
        # A 9x9 sample patch centred on row 14 stays inside the 10x10 unlit block.
        pixels[10:20, 10:20, :3] = 0
        self.assertEqual(photos.sample(pixels, 14 / 39, 14 / 39)["status"], "Not captured (unlit)")
        pixels[10:20, 10:20, 3] = 0
        self.assertEqual(photos.sample(pixels, 14 / 39, 14 / 39)["status"], "Transparent area")
        self.assertEqual(photos.sample(pixels, 1.4, .5)["status"], "Outside photograph")
        self.assertIsNone(photos.sample(pixels, 1.4, .5)["luma"])


if __name__ == "__main__":
    unittest.main()
