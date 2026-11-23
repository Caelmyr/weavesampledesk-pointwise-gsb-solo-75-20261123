"""Physical-plane measurements for registered textile photographs."""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .model import number
from .photos import Frame, coordinates, image, region


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


def translated(frame, shift_mm):
    """Rigidly move a calibrated frame by shift_mm = (dx, dy); only translation is allowed."""
    delta = np.asarray(shift_mm, dtype=np.float64) / np.asarray(frame.step, dtype=np.float64)
    matrix = np.array([[1, 0, delta[0]], [0, 1, delta[1]]], dtype=np.float32)
    height, width = frame.support.shape
    warped_rgb = cv.warpAffine(frame.rgb, matrix, (width, height), flags=cv.INTER_LINEAR, borderValue=(0, 0, 0))
    warped_support = cv.warpAffine(frame.support.astype(np.uint8), matrix, (width, height),
                                   flags=cv.INTER_NEAREST, borderValue=0).astype(bool)
    return Frame(warped_rgb, warped_support, frame.step)


def seam_both_coverage(reference, moved, seam):
    """Percent of the seam window visible in both photographs once the comparison frame is moved."""
    if seam is None:
        return None
    domain = region(reference, seam)
    total = int(domain.sum())
    if total == 0:
        return None
    both = domain & reference.useful & moved.useful
    return round(100.0 * int(both.sum()) / total, 3)


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


def shift_pair(options, key, bound_mm):
    x = number(options.get(key + "_dx"), key + " horizontal shift (mm)", -bound_mm, bound_mm)
    y = number(options.get(key + "_dy"), key + " vertical shift (mm)", -bound_mm, bound_mm)
    return [round(x, 4), round(y, 4)]


def placement_match(frame, other, marks, opts):
    """Choose one of two translation-only placements whose motif centres all stay within tolerance."""
    fixed = coordinates(marks["reference"]["points"], frame.rgb.shape)
    moved_points = coordinates(marks["comparison"]["points"], other.rgb.shape)
    if fixed.shape != moved_points.shape:
        raise Invalid("Both motif marks need the same number of centres in corresponding order.")
    bound = max(frame.rgb.shape[1] * frame.step[0], frame.rgb.shape[0] * frame.step[1])
    candidates_in = [
        ("A", shift_pair(opts, "candidate_a", bound)),
        ("B", shift_pair(opts, "candidate_b", bound)),
    ]
    tolerance = number(opts.get("tolerance_mm"), "per-centre tolerance (mm)", .01, bound)
    seam = marks.get("seam")
    fixed_mm = fixed * frame.step

    candidates = []
    for label, shift in candidates_in:
        moved_plane = (moved_points * frame.step) + np.asarray(shift, dtype=float)
        delta = fixed_mm - moved_plane
        residuals = np.linalg.norm(delta, axis=1)
        coverage = seam_both_coverage(frame, translated(other, shift), seam)
        candidates.append({
            "candidate": label, "shift_mm": rounded(shift),
            "residuals": [round(float(value), 4) for value in residuals],
            "max_residual_mm": round(float(residuals.max()), 4),
            "rms_residual_mm": round(float(np.sqrt(np.mean(residuals ** 2))), 4),
            "seam_both_pct": coverage,
            "passes": bool(residuals.max() <= tolerance + 1e-9),
        })

    passing = [item for item in candidates if item["passes"]]
    if not passing:
        detail = "; ".join(f"{item['candidate']}: max {item['max_residual_mm']} mm" for item in candidates)
        raise Invalid("Neither placement keeps every motif centre within tolerance (" + detail + ").")
    chosen = sorted(
        passing,
        key=lambda item: (item["max_residual_mm"],
                          -(item["seam_both_pct"] if item["seam_both_pct"] is not None else -1.0),
                          item["rms_residual_mm"], item["candidate"]),
    )[0]

    reason = "Only this placement kept every motif centre within the tolerance."
    if len(passing) > 1:
        runner_up = next(item for item in passing if item["candidate"] != chosen["candidate"])
        if chosen["max_residual_mm"] != runner_up["max_residual_mm"]:
            reason = ("Both placements passed; " + chosen["candidate"] +
                      " aligns the centres more closely (smaller worst-centre residual "
                      f"{chosen['max_residual_mm']} mm vs {runner_up['max_residual_mm']} mm).")
        elif chosen["seam_both_pct"] is not None and runner_up["seam_both_pct"] is not None and \
                chosen["seam_both_pct"] != runner_up["seam_both_pct"]:
            reason = ("Both placements passed with the same worst-centre residual; " + chosen["candidate"] +
                      " keeps more of the marked seam visible in both photos ("
                      f"{chosen['seam_both_pct']}% vs {runner_up['seam_both_pct']}%).")
        else:
            reason = ("Both placements passed with the same worst-centre residual and seam coverage; "
                      + chosen["candidate"] + " has the lower RMS residual.")

    moved_frame = translated(other, chosen["shift_mm"])
    shown = image(frame)
    blend = shown * .62 + moved_frame.rgb * .38
    if seam is not None:
        domain = region(frame, seam)
        tint(shown, domain, (.93, .78, .35), .18)
    shown[moved_frame.support] = blend[moved_frame.support]
    for index, (fxy, mxy) in enumerate(zip(fixed, moved_points), 1):
        aligned = (mxy * frame.step + np.asarray(chosen["shift_mm"])) / frame.step
        start = tuple(np.rint(fxy).astype(int))
        end = tuple(np.rint(aligned).astype(int))
        cv.drawMarker(shown, start, (.32, .89, .63), cv.MARKER_CROSS, 16, 2)
        cv.drawMarker(shown, end, (.98, .56, .31), cv.MARKER_TILTED_CROSS, 16, 2)
        cv.line(shown, start, end, (.96, .93, .62), 1)

    rows = []
    for item in candidates:
        rows.append({
            "candidate": item["candidate"], "shift_x_mm": item["shift_mm"][0], "shift_y_mm": item["shift_mm"][1],
            "max_residual_mm": item["max_residual_mm"], "rms_residual_mm": item["rms_residual_mm"],
            "seam_both_visible_pct": item["seam_both_pct"], "passes_tolerance": item["passes"],
        })
    centre_rows = [
        {"point": index, "reference_mm": rounded(fxy),
         "comparison_aligned_mm": rounded(mxy * frame.step + np.asarray(chosen["shift_mm"])),
         "residual_mm": round(float(value), 4)}
        for index, (fxy, value) in enumerate(zip(fixed_mm, chosen["residuals"]), 1)
    ]
    summary = {
        "chosen_candidate": chosen["candidate"], "adopted_shift_mm": chosen["shift_mm"],
        "tolerance_mm": round(float(tolerance), 4), "passing_candidates": [item["candidate"] for item in passing],
        "chosen_max_residual_mm": chosen["max_residual_mm"], "chosen_rms_residual_mm": chosen["rms_residual_mm"],
        "chosen_seam_both_pct": chosen["seam_both_pct"], "selection_reason": reason,
    }
    return summary, rows + centre_rows, shown


def seam_coverage(frame, other, marks, opts):
    """Classify a seam window under a saved translation-only placement."""
    placement = opts.get("placement")
    if not isinstance(placement, dict) or not isinstance(placement.get("adopted_shift_mm"), list) \
            or len(placement["adopted_shift_mm"]) != 2:
        raise Invalid("Choose a saved placement reading.")
    shift = placement["adopted_shift_mm"]
    moved = translated(other, shift)
    domain = region(frame, marks["seam"])
    if not np.any(domain):
        raise Invalid("Mark a larger seam window.")
    valid_a = domain & frame.useful
    valid_b = domain & moved.useful
    groups = [
        ("Both photos", valid_a & valid_b, (.27, .85, .59)),
        ("Reference only", valid_a & ~valid_b, (.38, .67, .99)),
        ("Comparison only", valid_b & ~valid_a, (.73, .51, .91)),
        ("Neither photo", domain & ~(valid_a | valid_b), (.94, .48, .23)),
    ]
    shown = image(frame)
    total = int(domain.sum())
    rows = []
    for label, mask, color in groups:
        tint(shown, mask, color)
        rows.append({"observation": label, "pixels": int(mask.sum()),
                     "percent": round(100 * mask.sum() / total, 3)})
    summary = {
        "placement_shift_mm": rounded(shift), "placement_candidate": placement.get("chosen_candidate"),
        "both_visible_pct": rows[0]["percent"], "seam_pixels": total,
    }
    return summary, rows, shown
