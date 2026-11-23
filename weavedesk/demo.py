"""A synthetic woven swatch for trying the calibrated sample measurements."""

import math

import cv2 as cv
import numpy as np

from .photos import png
from .store import identity, stamp

WIDTH, HEIGHT = 1050, 660
CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]
FIDUCIALS = [(112, 113), (925, 112), (101, 540), (943, 548)]


def points(pixels):
    return [[x / (WIDTH - 1), y / (HEIGHT - 1)] for x, y in pixels]


def box(x0, y0, x1, y1):
    return points([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def sample_photograph(role, shift, seed):
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    local_x, local_y = x - shift[0], y - shift[1]
    fiber = 3.2 * np.cos(local_x * (2 * math.pi / 7.5)) + 4.5 * np.cos(local_y * (2 * math.pi / 8.6))
    grain = np.random.default_rng(seed).normal(0, 1.5, (HEIGHT, WIDTH))
    weave = 4 * np.cos((local_x + local_y) * (2 * math.pi / 21))
    base = np.full((HEIGHT, WIDTH, 3), (222, 178, 138), dtype=np.float32)
    field = (local_x > 86) & (local_x < 972) & (local_y > 81) & (local_y < 575)
    base[field] = [181, 127, 100]
    base[field & ((local_x % 190) < 91)] = [105, 89, 110]
    base[field & ((local_y % 158) < 49)] = [.87, .83, .84] * base[field & ((local_y % 158) < 49)]
    left = np.clip((local_x - 810) / 18, 0, 1)
    right = np.clip((local_x - 887) / 16, 0, 1)
    band = np.minimum(left, 1 - right) * (local_y > 286) * (local_y < 465)
    base = base * (1 - band[:, :, None]) + np.array([69, 50, 68]) * band[:, :, None]
    areas = [
        ((142, 148, 277, 240), [213, 182, 142]),
        ((319, 149, 444, 241), [106, 52, 78]),
        ((475, 148, 579, 241), [61, 100, 122]),
        ((605, 148, 730, 243), [157, 86, 78]),
        ((250, 333, 434, 490), [61, 68, 78]),
    ]
    for (x0, y0, x1, y1), rgb in areas:
        inside = (local_x >= x0) & (local_x <= x1) & (local_y >= y0) & (local_y <= y1)
        base[inside] = rgb
    texture_strength = 1.5 if role == "rake" else 1.0
    base += (fiber * texture_strength + weave + grain)[:, :, None]
    if role == "rake":
        base += np.array([4.0, 1.0, -1.5])
        slope = (x - 500) / 450
        base += (np.clip(slope, -.9, .9) * 9)[:, :, None]
    rgb = np.clip(base, 0, 255).astype(np.uint8)
    pixels = cv.cvtColor(rgb, cv.COLOR_RGB2RGBA)
    for cx, cy, radius in ((279, 373, 5), (313, 390, 8), (344, 414, 7), (379, 450, 10), (405, 438, 4)):
        cv.circle(pixels, (cx + shift[0], cy + shift[1]), radius, (204, 193, 167, 255), -1, cv.LINE_AA)
    for cx, cy, radius in ((533, 358, 7), (573, 403, 10), (660, 448, 8), (682, 349, 4)):
        color = (229, 196, 147, 255) if radius % 2 == 0 else (64, 39, 47, 255)
        cv.circle(pixels, (cx + shift[0], cy + shift[1]), radius, color, -1, cv.LINE_AA)
    for i, (fx, fy) in enumerate(FIDUCIALS):
        center = (fx + shift[0] + (i % 2), fy + shift[1])
        cv.drawMarker(pixels, center, (59, 48, 45, 255), cv.MARKER_CROSS, 15, 2)
    if role == "face":
        pixels[456:487, 698:747] = 0
    else:
        pixels[332:382, 548:600] = 0
    cv.putText(pixels, "WEAVE SAMPLE / SYNTHETIC", (108, 625), cv.FONT_HERSHEY_SIMPLEX, .73, (66, 53, 66, 255), 2)
    return pixels


def create(library):
    photos, marks = [], []
    sample_config = (
        ("Face light / sample A", "face", (0, 0), 29),
        ("Raking light / sample B", "rake", (4, -2), 62),
        ("Transmitted light / sample A", "backlight", (0, 0), 19),
    )
    for title, role, shift, seed in sample_config:
        photo = {
            "id": identity(), "title": title, "role": role, "filename": role + "-sample.png",
            "blob": library.keep(png(sample_photograph(role, shift, seed))), "corners": CORNERS,
            "synthetic": True, "uploaded": stamp(),
        }
        photos.append(photo)
        items = [
            ("Undyed yarn", "undyed", box(151, 159, 266, 229)),
            ("Plum dyed yarn", "dyed", box(329, 159, 432, 230)),
            ("Indigo warp", "warp", box(487, 160, 566, 230)),
            ("Rust weft", "weft", box(615, 158, 714, 231)),
            ("Mixed pattern", "woven", box(460, 252, 722, 313)),
            ("Plain repeat panel", "repeat", box(493, 170, 563, 222)),
            ("Transmitted pores", "transmit", box(259, 347, 422, 479)),
            ("Streak and spot check", "inspect", box(506, 327, 750, 501)),
            ("Across selvedge edge", "stroke", points([(780, 395), (875, 395)])),
            ("Corresponding motif marks", "register", points([
                (fx + shift[0] + (i % 2), fy + shift[1]) for i, (fx, fy) in enumerate(FIDUCIALS)
            ])),
        ]
        for mark_title, purpose, vertices in items:
            marks.append({"id": identity(), "title": mark_title, "purpose": purpose, "points": vertices,
                          "photo": photo["id"]})
    return library.new_run({"title": "Plum and indigo sample", "stock": "Synthetic woven swatch",
                            "width_mm": 350, "height_mm": 220}, photos, marks)
