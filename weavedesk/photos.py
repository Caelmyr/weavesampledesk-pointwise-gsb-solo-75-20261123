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


def plane(capture, run):
    rgba = capture["pixels"]
    calibrated = capture["corners"]
    if not calibrated:
        raise Invalid("Mark the four sample corners before taking readings.")
    scale = min(4, 1300 / max(run["width_mm"], run["height_mm"]))
    width = max(32, round(run["width_mm"] * scale))
    height = max(32, round(run["height_mm"] * scale))
    origin = coordinates(calibrated, rgba.shape).astype(np.float32)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv.getPerspectiveTransform(origin, target)
    warped = cv.warpPerspective(rgba.astype(np.float32) / 255, transform, (width, height))
    step = (run["width_mm"] / (width - 1), run["height_mm"] / (height - 1))
    return Frame(warped[:, :, :3], warped[:, :, 3] >= .999, step)


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


def png(pixels):
    file = io.BytesIO()
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    picture = Image.fromarray(pixels)
    picture.thumbnail((1500, 1500))
    picture.save(file, "PNG")
    return file.getvalue()
