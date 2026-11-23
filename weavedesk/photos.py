"""Turn sample photographs into a common, calibrated textile plane."""
import io
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from image_lab.validation import Invalid

from .model import points

MAX_PAYLOAD = 20 * 1024 * 1024
MAX_PIXELS = 14_000_000


@dataclass
class Frame:
    rgb: np.ndarray
    support: np.ndarray
    step: tuple

    @property
    def luma(self):
        return self.rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)

    @property
    def useful(self):
        return self.support & (self.rgb.max(axis=2) >= .018) & (self.rgb.min(axis=2) < .993)


def basename(filename):
    return Path((filename or "sample.png").replace("\\", "/")).name[:110]


def decode(raw):
    try:
        with Image.open(io.BytesIO(raw)) as raw_image:
            if raw_image.format not in {"JPEG", "PNG", "TIFF"}:
                raise Invalid("Upload a JPEG, PNG or TIFF photograph.")
            if raw_image.width * raw_image.height > MAX_PIXELS or min(raw_image.size) < 32:
                raise Invalid("Use photographs of at most 14 megapixels and at least 32 pixels per side.")
            image = ImageOps.exif_transpose(raw_image)
            return np.asarray(image.convert("RGBA"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise Invalid("The sample photograph could not be opened.") from error


def corners(value):
    selection = points(value, 4, 4, "sample corners")
    polygon = np.asarray(selection, dtype=np.float32)
    if not cv.isContourConvex(polygon) or cv.contourArea(polygon, oriented=True) < .025:
        raise Invalid("Select the sample corners clockwise, starting at upper left.")
    return selection


def coordinates(values, shape):
    height, width = shape[:2]
    return np.asarray(values, dtype=float) * (width - 1, height - 1)


def calibrated_size(run):
    scale = min(4, 1300 / max(run["width_mm"], run["height_mm"]))
    width = max(32, round(run["width_mm"] * scale))
    height = max(32, round(run["height_mm"] * scale))
    return width, height


def mapping(corners, run, shape):
    width, height = calibrated_size(run)
    origin = coordinates(corners, shape).astype(np.float32)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    step = (run["width_mm"] / (width - 1), run["height_mm"] / (height - 1))
    return cv.getPerspectiveTransform(origin, target), (width, height), step


def plane(capture, run):
    rgba = capture["pixels"]
    calibrated = capture["corners"]
    if not calibrated:
        raise Invalid("Mark the four sample corners before taking readings.")
    transform, (width, height), step = mapping(calibrated, run, rgba.shape)
    warped = cv.warpPerspective(rgba.astype(np.float32) / 255, transform, (width, height))
    return Frame(warped[:, :, :3], warped[:, :, 3] >= .999, step)


def original_point(corners, run, shape, x, y):
    """Map a calibrated-plane position back to the source photograph."""
    transform, (width, height), _step = mapping(corners, run, shape)
    inverse = np.linalg.inv(transform)
    mapped = inverse @ np.array([x * (width - 1), y * (height - 1), 1.0])
    if abs(mapped[2]) < 1e-12:
        raise Invalid("This position does not fall within the source photograph.")
    column, row = (mapped[:2] / mapped[2]).tolist()
    height_px, width_px = shape[:2]
    return {
        "pixel": [round(column, 2), round(row, 2)],
        "x": column / (width_px - 1), "y": row / (height_px - 1),
        "width": width_px, "height": height_px,
    }


def region(frame, mark):
    bitmap = np.zeros(frame.support.shape, dtype=np.uint8)
    vertices = np.rint(coordinates(mark["points"], bitmap.shape)).astype(np.int32)
    cv.fillPoly(bitmap, [vertices], 1)
    return bitmap.astype(bool)


def image(frame):
    shown = frame.rgb.copy()
    yy, xx = np.indices(frame.support.shape)
    paper = np.where((xx + yy) % 16 > 8, .38, .29)
    shown[~frame.support] = paper[~frame.support, None]
    return shown


def sample(pixels, x, y, reach=4):
    """Read the source photograph at a normalized position, with a clarity verdict."""
    height, width = pixels.shape[:2]
    inside = 0 <= x <= 1 and 0 <= y <= 1
    column = min(max(x, 0), 1) * (width - 1)
    row = min(max(y, 0), 1) * (height - 1)
    left, right = max(0, int(round(column)) - reach), min(width, int(round(column)) + reach + 1)
    top, bottom = max(0, int(round(row)) - reach), min(height, int(round(row)) + reach + 1)
    patch = pixels[top:bottom, left:right].astype(np.float32) / 255
    if not patch.size:
        return {"inside": inside, "pixel": [round(column, 2), round(row, 2)], "luma": None,
                "usable_fraction": 0.0, "status": "Outside photograph"}
    alpha = patch[:, :, 3]
    red, green, blue = (patch[:, :, index] for index in range(3))
    visible = alpha >= .999
    luma = .2126 * red + .7152 * green + .0722 * blue
    lit = visible & (np.maximum(np.maximum(red, green), blue) >= .018)
    clear = lit & (np.maximum(np.maximum(red, green), blue) < .993)
    visible_fraction = float(visible.mean())
    clear_fraction = float(clear.mean())
    value = round(float(np.median(luma[clear])), 4) if clear.any() else None
    if not inside:
        status = "Outside photograph"
    elif visible_fraction < .5:
        status = "Transparent area"
    elif clear_fraction >= .9:
        status = "Clear"
    elif float(np.median(np.maximum(np.maximum(red, green), blue)[visible])) < .018:
        status = "Not captured (unlit)"
    elif not clear_fraction:
        status = "Clipped highlight"
    else:
        status = "Partly obscured"
    return {"inside": inside, "pixel": [round(column, 2), round(row, 2)],
            "luma": value if inside else None,
            "usable_fraction": round(clear_fraction, 4) if inside else 0.0, "status": status}


def marked(pixels, x, y):
    """Copy the source photograph with a ring around one normalized position."""
    shown = pixels.astype(np.float32) / 255
    height, width = shown.shape[:2]
    if 0 <= x <= 1 and 0 <= y <= 1:
        center = (int(round(x * (width - 1))), int(round(y * (height - 1))))
        radius = max(12, int(min(width, height) * .035))
        cv.circle(shown, center, radius, (1, .32, .14, 1), 2, cv.LINE_AA)
        cv.drawMarker(shown, center, (.16, 1, .72, 1), cv.MARKER_CROSS, radius * 2, 2, cv.LINE_AA)
    return shown


def png(pixels):
    file = io.BytesIO()
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    picture = Image.fromarray(pixels)
    picture.thumbnail((1500, 1500))
    picture.save(file, "PNG")
    return file.getvalue()
