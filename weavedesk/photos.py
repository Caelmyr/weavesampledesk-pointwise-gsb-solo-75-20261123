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
    unseen: np.ndarray = None

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
    support = warped[:, :, 3] >= .999
    # Cloth this shot saw but left without pixels: outside its own frame or transparently cut away.
    # The PNG surround is outside the calibrated sheet, so it is not counted.
    photographed = cv.warpPerspective(np.full(rgba.shape[:2], 255, np.uint8), transform, (width, height)) >= 128
    present = cv.warpPerspective((rgba[:, :, 3] > 0).astype(np.uint8), transform, (width, height),
                                 flags=cv.INTER_NEAREST) >= .5
    unseen = photographed & ~present
    return Frame(warped[:, :, :3], support, step=(run["width_mm"] / (width - 1), run["height_mm"] / (height - 1)),
                 unseen=unseen)


def region(frame, mark):
    bitmap = np.zeros(frame.support.shape, dtype=np.uint8)
    vertices = np.rint(coordinates(mark["points"], bitmap.shape)).astype(np.int32)
    cv.fillPoly(bitmap, [vertices], 1)
    return bitmap.astype(bool)


def straight_line(knots):
    """Least-squares straight line through the marked cut points, in physical mm."""
    values = np.asarray(knots, dtype=float)
    if len(values) < 2 or np.linalg.matrix_rank(values - values[0], tol=1e-9) < 1:
        return None
    center = values.mean(axis=0)
    _values, _vectors = np.linalg.eigh((values - center).T @ (values - center))
    direction = _vectors[:, -1]
    if direction @ (values[-1] - values[0]) < 0:
        direction = -direction
    direction = direction / np.linalg.norm(direction)
    return {"origin": center, "direction": direction, "normal": np.array([direction[1], -direction[0]])}


def along(line, points):
    return (np.asarray(points, dtype=float) - line["origin"]) @ line["direction"]


def distance(line, points):
    """Signed perpendicular distance in mm to a fitted line."""
    return (np.asarray(points, dtype=float) - line["origin"]) @ line["normal"]


def side_mask(frame, line, side):
    """Pixels on the kept side of a cut line, using the image's own left/right/above/below axes."""
    height, width = frame.support.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    x_mm = xx * frame.step[0]
    y_mm = yy * frame.step[1]
    origin_x, origin_y = line["origin"]
    run, rise = line["direction"]
    if abs(rise) >= abs(run):
        position = origin_x + run / rise * (y_mm - origin_y)
        offset = x_mm - position
        return offset <= 0 if side == "left" else offset >= 0
    position = origin_y + rise / run * (x_mm - origin_x)
    offset = y_mm - position
    return offset <= 0 if side == "above" else offset >= 0


def band(frame, line, width_mm):
    """Pixels within the given perpendicular distance of a cut line."""
    return segment_band(frame, line, (-np.inf, np.inf), width_mm)


def segment_band(frame, line, extent, width_mm):
    """Band around the marked cut segment only, so beyond its ends the line is not assessed."""
    height, w = frame.support.shape
    yy, xx = np.indices((height, w), dtype=np.float32)
    shift = np.stack((xx * frame.step[0] - line["origin"][0], yy * frame.step[1] - line["origin"][1]), axis=-1)
    signed = shift @ line["normal"]
    reach = shift @ line["direction"]
    low, high = extent
    return (np.abs(signed) <= width_mm) & (reach >= low - width_mm) & (reach <= high + width_mm)


def kept(frame, line, side):
    return side_mask(frame, line, side) & frame.support


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


def mask_png(mask):
    file = io.BytesIO()
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").save(file, "PNG")
    return file.getvalue()


def read_mask(data):
    try:
        with Image.open(io.BytesIO(data)) as picture:
            return np.asarray(picture.convert("L")) >= 128
    except (UnidentifiedImageError, OSError) as error:
        raise Invalid("The adopted cut mask is unavailable.") from error
