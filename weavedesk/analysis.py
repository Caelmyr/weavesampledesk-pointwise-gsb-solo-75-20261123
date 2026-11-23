"""Physical-plane measurements for registered textile photographs."""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .model import number
from .photos import coordinates, image, region


def tint(shown, mask, color, amount=.58):
    shown[mask] = shown[mask] * (1 - amount) + np.asarray(color) * amount


def selection(frame, mark, minimum=32):
    mask = region(frame, mark) & frame.useful
    if np.count_nonzero(mask) < minimum:
        raise Invalid("The marked patch needs more usable image pixels.")
    return mask


def rounded(values, digits=4):
    return np.round(values, digits).tolist()


def lab(pixels):
    # Convert standard RGB-encoded input to display-referred Lab.
    return cv.cvtColor(np.asarray(pixels, np.float32).reshape(-1, 1, 3), cv.COLOR_RGB2LAB).reshape(-1, 3)


def cloth_tone(frame, marks, _opts):
    mask = selection(frame, marks["region"])
    lab_pixels = lab(frame.rgb[mask])
    low, center, high = np.percentile(lab_pixels, [10, 50, 90], axis=0)
    rows = [
        {"channel": name, "p10": round(float(low[i]), 4), "median": round(float(center[i]), 4),
         "p90": round(float(high[i]), 4)}
        for i, name in enumerate(("L*", "a*", "b*"))
    ]
    shown = image(frame)
    tint(shown, mask, (.38, .78, .86), .28)
    return {"pixels": int(mask.sum()), "lab": rounded(center), "lightness_spread": round(float(high[0] - low[0]), 4)}, rows, shown


def dye_contrast(frame, marks, _opts):
    masks = {key: selection(frame, marks[key]) for key in ("undyed", "dyed")}
    values = {key: np.median(lab(frame.rgb[area]), axis=0) for key, area in masks.items()}
    lightness_drop = float(values["undyed"][0] - values["dyed"][0])
    distance = float(np.linalg.norm(values["undyed"] - values["dyed"]))
    shown = image(frame)
    tint(shown, masks["undyed"], (.84, .75, .32))
    tint(shown, masks["dyed"], (.39, .68, .96))
    rows = [{"mark": marks[key]["title"], "lab": rounded(value), "pixels": int(masks[key].sum())}
            for key, value in values.items()]
    return {"lightness_drop": round(lightness_drop, 4), "delta_e76": round(distance, 4)}, rows, shown


def dye_projection(frame, undyed, dyed):
    blank_rgb = np.median(frame.rgb[selection(frame, undyed)], axis=0)
    dyed_rgb = np.median(frame.rgb[selection(frame, dyed)], axis=0)
    difference = dyed_rgb - blank_rgb
    leverage = float(difference @ difference)
    if leverage < .014:
        raise Invalid("Choose two yarn controls with stronger color separation.")
    return (frame.rgb - blank_rgb) @ difference / leverage


def dyed_coverage(frame, marks, opts):
    area = selection(frame, marks["region"])
    fraction = dye_projection(frame, marks["undyed"], marks["dyed"])
    threshold = number(opts.get("threshold", .5), "dyed-yarn threshold", .1, .9)
    dyed = area & (fraction >= threshold)
    border = area & (np.abs(fraction - threshold) < .08)
    shown = image(frame)
    tint(shown, area & ~dyed, (.36, .71, .95), .2)
    tint(shown, dyed, (.27, .79, .51))
    tint(shown, border, (.94, .67, .29), .34)
    rows = [
        {"class": key, "pixels": int(mask.sum())}
        for key, mask in (("Dyed", dyed), ("Undyed", area & ~dyed), ("Near threshold", border))
    ]
    return {"dyed_area_pct": round(100 * dyed.sum() / area.sum(), 3),
            "threshold": threshold, "pixels": int(area.sum())}, rows, shown


def weave_period(frame, marks, _opts):
    mask = selection(frame, marks["region"], 250)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if min(y1 - y0, x1 - x0) < 24:
        raise Invalid("Use at least 24 calibrated pixels along both repeat patch axes.")
    luma = frame.luma[y0:y1, x0:x1]
    present = mask[y0:y1, x0:x1]
    window = np.outer(np.hanning(luma.shape[0]), np.hanning(luma.shape[1]))
    centered = np.where(present, luma - np.mean(luma[present]), 0)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    freq_x = np.fft.fftshift(np.fft.fftfreq(power.shape[1], d=frame.step[0]))
    freq_y = np.fft.fftshift(np.fft.fftfreq(power.shape[0], d=frame.step[1]))
    xx, yy = np.meshgrid(freq_x, freq_y)
    possible = ((yy > 0) | ((yy == 0) & (xx > 0))) & (np.hypot(xx, yy) > .18)
    rows = []
    candidate = np.where(possible, power, 0)
    peak = float(candidate.max())
    if peak <= 1e-6:
        raise Invalid("This patch does not contain enough periodic tonal variation.")
    for i in range(1, 5):
        iy, ix = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        strength = float(candidate[iy, ix]) / peak
        if strength < .075:
            break
        rows.append({"peak": i, "spacing_mm": round(float(1 / np.hypot(xx[iy, ix], yy[iy, ix])), 4),
                     "fiber_direction_deg": round((math.degrees(math.atan2(yy[iy, ix], xx[iy, ix])) + 90) % 180, 2),
                     "relative_power": round(strength, 4)})
        candidate[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4] = 0
    shown = image(frame)
    tint(shown, mask, (.52, .79, .91), .22)
    return {"dominant_spacing_mm": rows[0]["spacing_mm"], "dominant_fiber_deg": rows[0]["fiber_direction_deg"],
            "peaks": len(rows)}, rows, shown


def light_openings(frame, marks, opts):
    area = selection(frame, marks["region"])
    cutoff = number(opts.get("luma", .62), "transmitted-light threshold", .05, .95)
    openings = area & (frame.luma >= cutoff)
    count, _labels, stats, centroids = cv.connectedComponentsWithStats(openings.astype(np.uint8), connectivity=8)
    pixel_area = frame.step[0] * frame.step[1]
    ordering = sorted(range(1, count), key=lambda key: stats[key, cv.CC_STAT_AREA], reverse=True)
    rows = [
        {"opening": index, "area_mm2": round(float(stats[label, cv.CC_STAT_AREA] * pixel_area), 4),
         "center_mm": rounded(centroids[label] * frame.step)}
        for index, label in enumerate(ordering[:30], 1)
    ]
    shown = image(frame)
    tint(shown, area & ~openings, (.25, .78, .73), .22)
    tint(shown, openings, (.94, .54, .29))
    return {"opening_pct": round(100 * openings.sum() / area.sum(), 3),
            "open_mm2": round(float(openings.sum() * pixel_area), 4), "openings": count - 1}, rows, shown


def selvedge(frame, marks, opts):
    knots = coordinates(marks["line"]["points"], frame.rgb.shape)
    length = float(np.linalg.norm((knots[1] - knots[0]) * frame.step))
    if length < 1:
        raise Invalid("Use a transect at least 1 mm long.")
    count = min(220, max(24, int(length * 5)))
    locations = np.linspace(knots[0], knots[1], count, dtype=np.float32)
    xs, ys = locations.T
    luma = cv.remap(frame.luma, xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_LINEAR)[0]
    valid = cv.remap(frame.useful.astype(np.float32), xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_LINEAR)[0] >= .999
    if valid.sum() < count * .85:
        raise Invalid("More of the edge transect must be visible in the photograph.")
    strip = max(3, count // 7)
    lead, tail = float(np.median(luma[:strip])), float(np.median(luma[-strip:]))
    if abs(tail - lead) < .07:
        raise Invalid("The two ends should fall on visibly different woven tones.")
    relative = np.convolve(np.pad((luma - lead) / (tail - lead), 2, mode="edge"), np.full(5, .2), mode="valid")
    position = np.linspace(0, length, count)
    limits = [np.flatnonzero(valid & (relative >= level)) for level in (.1, .9)]
    if any(len(indices) == 0 for indices in limits):
        raise Invalid("The transect does not bracket the selvedge transition.")
    width = float(position[limits[1][0]] - position[limits[0][0]])
    shown = image(frame)
    cv.line(shown, tuple(np.rint(knots[0]).astype(int)), tuple(np.rint(knots[1]).astype(int)), (.96, .52, .27), 2)
    rows = [{"position_mm": round(float(x), 3), "luma": round(float(y), 4) if on else None}
            for x, y, on in zip(position, luma, valid)]
    return {"edge_width_mm": round(max(0, width), 4), "length_mm": round(length, 4), "usable_samples": int(valid.sum())}, rows, shown


def motif_offsets(frame, other, marks, _opts):
    fixed = coordinates(marks["reference"]["points"], frame.rgb.shape)
    moved = coordinates(marks["comparison"]["points"], other.rgb.shape)
    if fixed.shape != moved.shape:
        raise Invalid("Both motif marks need the same number of centres in corresponding order.")
    difference = (fixed - moved) * frame.step
    shift = np.mean(difference, axis=0)
    offset = difference - shift
    shown = image(frame)
    rows = []
    for i, (fxy, mxy, dxy, residual) in enumerate(zip(fixed, moved, difference, offset), 1):
        start, end = tuple(np.rint(fxy).astype(int)), tuple(np.rint(mxy).astype(int))
        cv.drawMarker(shown, start, (.32, .89, .63), cv.MARKER_CROSS, 16, 2)
        cv.drawMarker(shown, end, (.98, .56, .31), cv.MARKER_TILTED_CROSS, 16, 2)
        cv.line(shown, start, end, (.76, .71, .34), 1)
        rows.append({"point": i, "reference_mm": rounded(fxy * frame.step), "comparison_mm": rounded(mxy * frame.step),
                     "displacement_mm": rounded(dxy), "residual_mm": round(float(np.linalg.norm(residual)), 4)})
    rms = np.sqrt(np.sum(offset ** 2) / len(offset))
    return {"suggested_shift_mm": rounded(shift), "rms_after_shift_mm": round(float(rms), 4)}, rows, shown


def yarn_balance(frame, marks, _opts):
    masks = {key: selection(frame, marks[key]) for key in ("warp", "weft", "woven")}
    sampled = {key: np.median(frame.rgb[mask], axis=0) for key, mask in masks.items()}

    def linear(rgb):
        return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

    blend = (linear(sampled["warp"]) + linear(sampled["weft"])) / 2
    expected = np.where(blend <= .0031308, blend * 12.92, 1.055 * blend ** (1 / 2.4) - .055)
    comparison = lab(np.array([expected, sampled["woven"]]))
    difference = float(np.linalg.norm(comparison[0] - comparison[1]))
    shown = image(frame)
    tint(shown, masks["warp"], (.91, .49, .39), .38)
    tint(shown, masks["weft"], (.40, .75, .96), .38)
    tint(shown, masks["woven"], (.87, .66, .97), .44)
    rows = [{"patch": key, "rgb": rounded(value)} for key, value in (*sampled.items(), ("Balanced blend", expected))]
    return {"delta_e76": round(difference, 4), "expected_rgb": rounded(expected),
            "woven_rgb": rounded(sampled["woven"])}, rows, shown


def paired_coverage(frame, other, marks, _opts):
    domain = region(frame, marks["region"])
    valid_a = domain & frame.useful
    valid_b = domain & other.useful
    both = valid_a & valid_b
    groups = [
        ("Both photos", both, (.27, .85, .59)),
        ("Reference only", valid_a & ~valid_b, (.38, .67, .99)),
        ("Comparison only", valid_b & ~valid_a, (.73, .51, .91)),
        ("Neither photo", domain & ~(valid_a | valid_b), (.94, .48, .23)),
    ]
    if not np.any(domain):
        raise Invalid("Mark a larger comparison area.")
    shown = image(frame)
    total = int(domain.sum())
    rows = []
    for label, mask, color in groups:
        tint(shown, mask, color)
        rows.append({"observation": label, "pixels": int(mask.sum()), "percent": round(100 * mask.sum() / total, 3)})
    return {"paired_percent": rows[0]["percent"], "marked_pixels": total}, rows, shown


def surface_marks(frame, marks, opts):
    area = selection(frame, marks["region"])
    contrast = number(opts.get("contrast", .075), "contrast", .02, .35)
    minimum = number(opts.get("minimum_mm2", .15), "minimum spot area", .03, 40)
    luma = frame.luma
    filled = np.where(frame.useful, luma, float(np.median(luma[area])))
    reference = cv.GaussianBlur(filled.astype(np.float32), (0, 0), 8)
    residual = filled - reference
    accumulated = np.zeros(area.shape, dtype=bool)
    shown = image(frame)
    rows = []
    for name, detected, color in (
        ("Dark", area & (residual < -contrast), (.93, .45, .27)),
        ("Light", area & (residual > contrast), (.34, .72, .95)),
    ):
        count, component, stats, centroids = cv.connectedComponentsWithStats(detected.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            size = stats[label, cv.CC_STAT_AREA] * frame.step[0] * frame.step[1]
            if size < minimum:
                continue
            selected = component == label
            accumulated |= selected
            tint(shown, selected, color)
            rows.append({"direction": name, "size_mm2": round(float(size), 4),
                         "center_mm": rounded(centroids[label] * frame.step),
                         "peak_contrast": round(float(np.abs(residual[selected]).max()), 4)})
    rows.sort(key=lambda row: row["size_mm2"], reverse=True)
    return {"spots": len(rows), "spot_fraction_pct": round(100 * accumulated.sum() / area.sum(), 3)}, rows[:50], shown


def raised_fiber(frame, other, marks, opts):
    domain = region(frame, marks["region"])
    both = domain & frame.useful & other.useful
    if both.sum() < 250:
        raise Invalid("Both photographs need useful pixels in this inspection patch.")
    minimum = number(opts.get("minimum_mm2", .15), "minimum spot area", .03, 40)
    keep = number(opts.get("keep_min", .9), "minimum confirmed-fuzz retention", .3, 1)
    falsely = number(opts.get("false_max", .08), "maximum sheen false-mark rate", 0, .5)

    def relief(luma, useful):
        # A 2.5 mm core against a 6 mm surround, in units of the patch's robust spread.
        pixel = float(np.hypot(*frame.step))
        core, surround = 2.5 / pixel, 6.0 / pixel
        filled = np.where(useful, luma, float(np.median(luma[both])))
        residual = (cv.GaussianBlur(filled.astype(np.float32), (0, 0), core)
                    - cv.GaussianBlur(filled.astype(np.float32), (0, 0), surround))
        measured = residual[both]
        scale = 1.4826 * float(np.median(np.abs(measured - np.median(measured))))
        return residual / max(scale, 1e-4)

    face = relief(frame.luma, frame.useful)
    rake = relief(other.luma, other.useful)
    confirmed = region(frame, marks["confirmed"]) & both
    sheen = region(frame, marks["sheen"]) & both
    if not confirmed.any() or not sheen.any():
        raise Invalid("Both reference patches need pixels that are clear in both photographs.")

    # Rule A marks only what both lights lift above the local weave; rule B marks what the
    # raking light lifts further than the face light. One rule is adopted per reading.
    conditions = {
        "both": both & (face >= 1.5) & (rake >= 1.5),
        "extra": both & ((rake - face) >= 1.5),
    }
    pixel_area = frame.step[0] * frame.step[1]

    def components(condition):
        count, component = cv.connectedComponents(condition.astype(np.uint8), connectivity=8)
        kept = np.zeros(domain.shape, dtype=bool)
        found = []
        for label in range(1, count):
            selected = component == label
            size = float(selected.sum() * pixel_area)
            if size < minimum:
                continue
            kept |= selected
            ys, xs = np.nonzero(selected)
            found.append({"size_mm2": round(size, 4),
                          "center_mm": rounded(np.array([xs.mean() * frame.step[0], ys.mean() * frame.step[1]]), 3)})
        return kept, found

    evaluated = {}
    for rule, condition in conditions.items():
        kept, found = components(condition)
        evaluated[rule] = {
            "rule": rule,
            "retained_pct": round(100 * np.count_nonzero(confirmed & kept) / confirmed.sum(), 3),
            "false_mark_pct": round(100 * np.count_nonzero(sheen & kept) / sheen.sum(), 3),
            "spots": len(found), "spot_fraction_pct": round(100 * kept.sum() / both.sum(), 3),
            "components": found,
        }
    for report in evaluated.values():
        report["retained_ok"] = report["retained_pct"] >= 100 * keep
        report["false_ok"] = report["false_mark_pct"] <= 100 * falsely
        report["eligible"] = report["retained_ok"] and report["false_ok"]

    order = ("both", "extra")
    eligible = [rule for rule in order if evaluated[rule]["eligible"]]
    if eligible:
        adopted = sorted(eligible, key=lambda rule: (-evaluated[rule]["retained_pct"],
                                                     evaluated[rule]["false_mark_pct"], order.index(rule)))[0]
        reason = "Both gates passed; the agreed order keeps the rule that retains more confirmed fuzz."
    else:
        def shortfall(rule):
            report = evaluated[rule]
            return (min(report["retained_pct"] - 100 * keep, 0)
                    + min(100 * falsely - report["false_mark_pct"], 0))

        adopted = sorted(order, key=lambda rule: (-shortfall(rule), order.index(rule)))[0]
        reason = "Neither rule met both gates; the agreed order keeps the rule closest to both limits."

    shown = image(frame)
    tint(shown, conditions[adopted], (.30, .86, .55), .44)
    for mark, color in ((marks["confirmed"], (.96, .58, .24)), (marks["sheen"], (.34, .72, .95))):
        vertices = np.rint(coordinates(mark["points"], shown.shape)).astype(np.int32)
        cv.polylines(shown, [vertices], True, color, 2)
    winner = evaluated[adopted]
    summary = {
        "adopted_rule": adopted, "rule_reason": reason, "passing_rules": eligible,
        "retained_pct": winner["retained_pct"], "false_mark_pct": winner["false_mark_pct"],
        "spots": winner["spots"], "spot_fraction_pct": winner["spot_fraction_pct"],
        "keep_min_pct": round(100 * keep, 3), "false_max_pct": round(100 * falsely, 3),
    }
    rows = [{
        "rule": "Both lights" if rule == "both" else "Raking excess",
        "retained_pct": evaluated[rule]["retained_pct"], "false_mark_pct": evaluated[rule]["false_mark_pct"],
        "spots": evaluated[rule]["spots"], "spot_fraction_pct": evaluated[rule]["spot_fraction_pct"],
        "gate": ("Eligible" if evaluated[rule]["eligible"] else
                 "Retention short" if not evaluated[rule]["retained_ok"] and evaluated[rule]["false_ok"] else
                 "False marks over" if evaluated[rule]["retained_ok"] else "Both gates missed"),
        "adopted": "Yes" if rule == adopted else "No",
    } for rule in order]
    return summary, rows, shown


def color_drift(frame, other, marks, opts):
    domain = region(frame, marks["region"])
    area = domain & frame.useful & other.useful
    if area.sum() < 32:
        raise Invalid("Both photographs need usable pixels in this patch.")
    tolerance = number(opts.get("delta_e", 5), "color difference threshold", .5, 40)
    first = lab(frame.rgb[area])
    second = lab(other.rgb[area])
    delta = np.linalg.norm(first - second, axis=1)
    above = np.zeros(area.shape, dtype=bool)
    above[area] = delta > tolerance
    shown = image(frame)
    tint(shown, area & ~above, (.23, .81, .67), .32)
    tint(shown, above, (.96, .51, .35))
    rows = [
        {"statistic": "Median", "delta_e76": round(float(np.median(delta)), 4)},
        {"statistic": "90th percentile", "delta_e76": round(float(np.percentile(delta, 90)), 4)},
    ]
    return {"median_delta_e76": rows[0]["delta_e76"], "above_tolerance_pct": round(100 * above.sum() / area.sum(), 3),
            "paired_pixels": int(area.sum())}, rows, shown
