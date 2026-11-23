"""Validation and document shapes for textile runs, registered photographs and marks."""
import math
import re

from image_lab.validation import Invalid

ROLES = {
    "face": "Face light", "backlight": "Transmitted light", "rake": "Raking light",
}
MARKS = {
    "undyed": "Undyed reference", "dyed": "Solid dyed reference", "repeat": "Weave repeat patch",
    "transmit": "Transmitted-light patch", "inspect": "Inspection patch",
    "warp": "Warp reference", "weft": "Weft reference", "woven": "Mixed weave patch",
    "register": "Matched motif points", "stroke": "Selvedge transect",
    "fuzz": "Confirmed fuzz reference", "sheen": "Weave-sheen reference",
}
READINGS = {
    "cloth-tone": ("Cloth tone", "Measure color and variation in a cloth patch."),
    "dye-contrast": ("Dye contrast", "Compare solid dyed yarn with an undyed control on the same sample."),
    "dyed-coverage": ("Dyed area", "Measure the proportion of a marked design close to the dyed yarn reference."),
    "weave-period": ("Weave periods", "Read dominant spatial periods and fiber directions from a repeat patch."),
    "light-openings": ("Transmitted openings", "Locate and measure visible pore or tear openings against the dark weave."),
    "selvedge": ("Selvedge transition", "Measure the 10–90% tone change along a line across the woven edge."),
    "motif-offset": ("Motif offsets", "Compare corresponding motif points in two registered sample photographs."),
    "yarn-balance": ("Warp and weft balance", "Compare the mixed weave tone with an equal-area warp and weft blend."),
    "paired-coverage": ("Paired sample coverage", "Locate observations shared by two calibrated textile photographs."),
    "surface-marks": ("Surface spot map", "Find localized mottle and raised-fiber highlights on a raking-light patch."),
    "color-drift": ("Sample color difference", "Read registered color differences in a selected area of two cloth photographs."),
    "raised-fiber": ("Raised-fiber comparison", "Compare face and raking light in one patch to separate raised fuzz from weave sheen."),
}
POINT_MARKS = {"register", "stroke"}
IDENTITY = re.compile(r"^[0-9a-f]{24}$")


def identifier(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise Invalid("The item reference is invalid.")
    return value


def locate(items, value, label):
    wanted = identifier(value)
    for item in items:
        if item["id"] == wanted:
            return item
    raise Invalid(f"{label} is no longer available.")


def obj(value):
    if not isinstance(value, dict):
        raise Invalid("Expected a set of named fields.")
    return value


def text(value, label, limit=100, optional=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not optional and not value.strip()):
        raise Invalid(f"Enter {label} using at most {limit} characters.")
    return value.strip()


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Invalid(f"Enter a valid {label}.")
    if not minimum <= value <= maximum:
        raise Invalid(f"{label.capitalize()} must be between {minimum:g} and {maximum:g}.")
    return value


def choice(value, values, label):
    if not isinstance(value, str) or value not in values:
        raise Invalid(f"Choose a valid {label}.")
    return value


def points(raw, minimum, maximum, label):
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise Invalid(f"{label.capitalize()} needs between {minimum} and {maximum} ordered points.")
    result = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise Invalid(f"{label.capitalize()} needs coordinate pairs.")
        result.append([number(point[0], "position", 0, 1), number(point[1], "position", 0, 1)])
    return result


def run_fields(raw):
    data = obj(raw)
    return {
        "title": text(data.get("title"), "a run title"),
        "stock": text(data.get("stock", ""), "the stock description", 100, optional=True),
        "width_mm": number(data.get("width_mm"), "sample width (mm)", 20, 1500),
        "height_mm": number(data.get("height_mm"), "sample height (mm)", 20, 1500),
    }


def mark_fields(raw):
    data = obj(raw)
    purpose = choice(data.get("purpose"), MARKS, "mark purpose")
    if purpose == "stroke":
        minimum, maximum = 2, 2
    elif purpose == "register":
        minimum, maximum = 2, 16
    else:
        minimum, maximum = 3, 32
    return {
        "title": text(data.get("title"), "a mark title"),
        "purpose": purpose,
        "points": points(data.get("points"), minimum, maximum, MARKS[purpose]),
    }
