"""Routes for the woven sample desk and archived reading evidence."""
from io import BytesIO

from flask import Blueprint, Response, jsonify, render_template, request, send_file

from image_lab.validation import Conflict, Invalid

from . import analysis, demo, evidence, photos
from .model import MARKS, READINGS, ROLES, choice, identifier, locate, mark_fields, obj, run_fields, text
from .store import Library, identity, stamp

MEASUREMENTS = {
    "cloth-tone": (analysis.cloth_tone, {"region": "inspect"}, {}),
    "dye-contrast": (analysis.dye_contrast, {"undyed": "undyed", "dyed": "dyed"}, {}),
    "dyed-coverage": (analysis.dyed_coverage, {"region": "woven", "undyed": "undyed", "dyed": "dyed"}, {}),
    "weave-period": (analysis.weave_period, {"region": "repeat"}, {}),
    "light-openings": (analysis.light_openings, {"region": "transmit"}, {}),
    "selvedge": (analysis.selvedge, {"line": "stroke"}, {}),
    "motif-offset": (analysis.motif_offsets, {"reference": "register"}, {"comparison": "register"}),
    "placement-match": (analysis.placement_match, {"reference": "register"}, {"comparison": "register"}),
    "seam-coverage": (analysis.seam_coverage, {"seam": "seam"}, {}),
    "yarn-balance": (analysis.yarn_balance, {"warp": "warp", "weft": "weft", "woven": "woven"}, {}),
    "paired-coverage": (analysis.paired_coverage, {"region": "inspect"}, {}),
    "surface-marks": (analysis.surface_marks, {"region": "inspect"}, {}),
    "color-drift": (analysis.color_drift, {"region": "inspect"}, {}),
}
PAIRED = {"motif-offset", "placement-match", "seam-coverage", "paired-coverage", "color-drift"}
OPTIONAL_SLOTS = {"placement-match": {"seam": "seam"}}
OPTIONS = {
    "dyed-coverage": {"threshold": ["Dyed yarn threshold", .5, .01]},
    "light-openings": {"luma": ["Transmitted light threshold", .62, .01]},
    "surface-marks": {"contrast": ["Spot contrast", .075, .005], "minimum_mm2": ["Minimum area (mm²)", .15, .01]},
    "color-drift": {"delta_e": ["Color difference threshold", 5, .1]},
    "placement-match": {
        "candidate_a_dx": ["Candidate A horizontal shift (mm)", 0, .1],
        "candidate_a_dy": ["Candidate A vertical shift (mm)", 0, .1],
        "candidate_b_dx": ["Candidate B horizontal shift (mm)", 0, .1],
        "candidate_b_dy": ["Candidate B vertical shift (mm)", 0, .1],
        "tolerance_mm": ["Per-centre tolerance (mm)", 2, .1],
    },
}


def input_json():
    return obj(request.get_json(silent=True))


def sample_frame(library, run, photo):
    return photos.plane({**photo, "pixels": photos.decode(library.blob(photo["blob"]).read_bytes())}, run)


def measure(library, run, data):
    mode = choice(data.get("mode"), MEASUREMENTS, "reading")
    work, source_roles, other_roles = MEASUREMENTS[mode]
    source = locate(run["photos"], data.get("photo"), "The source photograph")
    source_frame = sample_frame(library, run, source)
    options = obj(data.get("options", {}))
    marked = obj(data.get("marks", {}))
    selected = {}

    def read_marks(assigned, capture, optional=None):
        merged = {**assigned, **(optional or {})}
        for slot, purpose in merged.items():
            mark_id = marked.get(slot)
            if not mark_id:
                if slot in (optional or {}):
                    continue
                raise Invalid(f"Select a {MARKS[purpose].lower()} mark.")
            mark = locate(run["marks"], mark_id, MARKS[purpose])
            if mark["photo"] != capture["id"] or mark["purpose"] != purpose:
                raise Invalid(f"Select a {MARKS[purpose].lower()} mark from the specified photograph.")
            selected[slot] = mark

    read_marks(source_roles, source, OPTIONAL_SLOTS.get(mode, {}))
    used = [source]
    if mode in PAIRED:
        other = locate(run["photos"], data.get("other"), "The comparison photograph")
        if other["id"] == source["id"]:
            raise Invalid("Choose a different comparison photograph.")
        other_frame = sample_frame(library, run, other)
        read_marks(other_roles, other)
        used.append(other)
        if mode == "seam-coverage":
            raw_placement = data.get("placement")
            placement_id = raw_placement.get("reading") if isinstance(raw_placement, dict) else raw_placement
            saved = library.reading(run, identifier(placement_id))
            if saved["mode"] != "placement-match":
                raise Invalid("Choose a saved placement match reading.")
            options = {**options, "placement": {
                "adopted_shift_mm": saved["summary"]["adopted_shift_mm"],
                "chosen_candidate": saved["summary"]["chosen_candidate"],
                "source_reading": saved["id"],
            }}
        summary, rows, pixels = work(source_frame, other_frame, selected, options)
    else:
        summary, rows, pixels = work(source_frame, selected, options)
    return {
        "mode": mode, "title": READINGS[mode][0], "summary": summary, "rows": rows, "photos": used,
        "options": options, "marks": selected, "sheet_mm": [run["width_mm"], run["height_mm"]],
        "source_revision": run["revision"], "overlay": library.keep(photos.png(pixels)),
    }


def install(app, root):
    library = Library(root)
    api = Blueprint("weavedesk", __name__, url_prefix="/weave")

    @app.get("/")
    def weave_desk():
        return render_template("weave.html")

    @api.get("/health")
    def health():
        return jsonify(service="weavedesk", status="ready")

    @api.get("/config")
    def config():
        definitions = {}
        for mode, (title, help_text) in READINGS.items():
            _, source_roles, other_roles = MEASUREMENTS[mode]
            definitions[mode] = {"title": title, "help": help_text, "reference": source_roles, "comparison": other_roles,
                                 "paired": mode in PAIRED, "options": OPTIONS.get(mode, {}),
                                 "optional": OPTIONAL_SLOTS.get(mode, {}),
                                 "requires_placement": mode == "seam-coverage"}
        return jsonify(readings=definitions, marks=MARKS, roles=ROLES)

    @api.get("/runs")
    def runs():
        return jsonify(library.list())

    @api.post("/runs")
    def new_run():
        return jsonify(library.new_run(run_fields(input_json()))), 201

    @api.post("/demo")
    def new_demo():
        return jsonify(demo.create(library)), 201

    @api.get("/runs/<key>")
    def run(key):
        return jsonify(library.run(key))

    @api.post("/runs/<key>/photos")
    def add_photo(key):
        incoming = request.files.get("file")
        if incoming is None:
            raise Invalid("Choose a photograph to upload.")
        data = incoming.stream.read(photos.MAX_PAYLOAD + 1)
        if not data or len(data) > photos.MAX_PAYLOAD:
            raise Invalid("Use a nonempty photograph of at most 20 MiB.")
        photos.decode(data)
        title = text(request.form.get("title"), "a photograph title")
        role = choice(request.form.get("role"), ROLES, "photograph role")
        try:
            revision = int(request.form.get("revision"))
        except (TypeError, ValueError) as error:
            raise Invalid("The run revision is required.") from error

        def append(run):
            if len(run["photos"]) >= 12:
                raise Invalid("This run already holds 12 sample photographs.")
            run["photos"].append({"id": identity(), "title": title, "name": title, "role": role,
                                  "filename": photos.basename(incoming.filename), "blob": library.keep(data),
                                  "corners": None, "synthetic": False, "uploaded": stamp()})

        return jsonify(library.change(key, revision, append)), 201

    @api.put("/runs/<key>/photos/<photo_key>/calibration")
    def calibrate(key, photo_key):
        data = input_json()
        corners = photos.corners(data.get("corners"))

        def update(run):
            locate(run["photos"], photo_key, "The photograph")["corners"] = corners

        return jsonify(library.change(key, data.get("revision"), update))

    @api.get("/runs/<key>/photos/<photo_key>/image")
    def image_preview(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        if request.args.get("view") == "source" or not photo["corners"]:
            pixels = photos.decode(library.blob(photo["blob"]).read_bytes())
        else:
            pixels = photos.image(sample_frame(library, run, photo))
        return Response(photos.png(pixels), mimetype="image/png")

    @api.get("/runs/<key>/photos/<photo_key>/source")
    def original(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        return send_file(library.blob(photo["blob"]), download_name=photo["filename"], as_attachment=True)

    @api.post("/runs/<key>/marks")
    def create_mark(key):
        data = input_json()
        mark = mark_fields(data)

        def change(run):
            photo = locate(run["photos"], data.get("photo"), "The photograph")
            if not photo["corners"]:
                raise Invalid("Register the sheet before marking its calibrated plane.")
            if len(run["marks"]) >= 100:
                raise Invalid("This run already holds 100 marks.")
            run["marks"].append({**mark, "id": identity(), "photo": photo["id"]})

        return jsonify(library.change(key, data.get("revision"), change)), 201

    @api.delete("/runs/<key>/marks/<mark_key>")
    def delete_mark(key, mark_key):
        data = input_json()

        def change(run):
            run["marks"].remove(locate(run["marks"], mark_key, "The mark"))

        return jsonify(library.change(key, data.get("revision"), change))

    @api.post("/runs/<key>/readings")
    def create_reading(key):
        data = input_json()
        run = library.run(key)
        if run["revision"] != data.get("revision"):
            raise Conflict("The run changed. Reload it before saving.")
        result = measure(library, run, data)
        updated, reading = library.save_reading(key, data["revision"], result)
        return jsonify(run=updated, reading=reading), 201

    @api.get("/runs/<key>/readings/<reading_id>")
    def reading_detail(key, reading_id):
        return jsonify(library.reading(library.run(key), reading_id))

    @api.get("/runs/<key>/readings/<reading_id>/overlay")
    def reading_overlay(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(library.blob(result["overlay"]), mimetype="image/png")

    @api.get("/runs/<key>/readings/<reading_id>/csv")
    def reading_csv(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.rows_csv(result)), download_name="measured-values.csv",
                         as_attachment=True, mimetype="text/csv")

    @api.get("/runs/<key>/readings/<reading_id>/packet")
    def reading_packet(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.packet(library, result)), download_name="sample-reading.zip",
                         as_attachment=True, mimetype="application/zip")

    app.register_blueprint(api)
    app.extensions["proof_library"] = library
