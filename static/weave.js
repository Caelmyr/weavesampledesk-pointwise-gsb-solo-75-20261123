"use strict";

const byId = (id) => document.getElementById(id);
const state = { config: null, runs: [], run: null, photo: null, marked: null };
const dialog = byId("dialog");

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function activePhoto() {
  return state.run?.photos.find((item) => item.id === state.photo);
}

function runURL() {
  return "/weave/runs/" + state.run.id;
}

function photoURL(photo) {
  return runURL() + "/photos/" + photo.id;
}

function displayName(value) {
  return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function display(value) {
  return value == null ? "—" : Array.isArray(value) ? value.join(" · ") : String(value);
}

async function api(url, method = "GET", payload) {
  const options = { method };
  if (payload instanceof FormData) {
    options.body = payload;
  } else if (payload !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "The request failed.");
  return value;
}

function guarded(operation) {
  return async (...arguments_) => {
    byId("dialog-message").textContent = "";
    byId("notice").textContent = "";
    try {
      await operation(...arguments_);
    } catch (error) {
      const destination = dialog.open ? "dialog-message" : "notice";
      byId(destination).textContent = error.message;
    }
  };
}

function button(label, click, className) {
  const control = element("button", label);
  control.type = "button";
  if (className) control.className = className;
  if (click) control.addEventListener("click", guarded(click));
  return control;
}

function entry(title, comment, click, selected) {
  const control = button(title, click);
  if (comment) control.append(element("small", comment));
  if (selected) control.setAttribute("aria-current", "true");
  return control;
}

function link(label, url) {
  const node = element("a", label);
  node.href = url;
  return node;
}

function replaceRun(run) {
  state.run = run;
  if (!run.photos.some((item) => item.id === state.photo)) state.photo = run.photos[0]?.id ?? null;
  if (!run.marks.some((item) => item.id === state.marked)) state.marked = null;
  renderDesk();
}

async function pickRun(id) {
  state.photo = null;
  state.marked = null;
  replaceRun(await api("/weave/runs/" + id));
  history.replaceState(null, "", "#" + id);
}

async function openNewRun(run) {
  state.runs = await api("/weave/runs");
  state.photo = null;
  state.marked = null;
  replaceRun(run);
  history.replaceState(null, "", "#" + run.id);
}

function renderDesk() {
  byId("runs").replaceChildren(...state.runs.map((run) => entry(
    run.title, run.stock, () => pickRun(run.id), state.run?.id === run.id,
  )));
  const run = state.run;
  byId("welcome").hidden = Boolean(run);
  byId("work").hidden = !run;
  if (!run) return;
  byId("run-title").textContent = run.title;
  byId("stock").textContent = [run.stock, run.width_mm + " × " + run.height_mm + " mm"].filter(Boolean).join(" / ");
  byId("photos").replaceChildren(...run.photos.map((photo) => entry(
    photo.title, photo.corners ? state.config.roles[photo.role] : "Needs sample calibration",
    () => {
      state.photo = photo.id;
      state.marked = null;
      renderDesk();
    }, photo.id === state.photo,
  )));
  const photo = activePhoto();
  byId("active-photo").hidden = !photo;
  byId("take-reading").disabled = !photo?.corners;
  const marks = run.marks.filter((mark) => mark.photo === state.photo);
  byId("marks").replaceChildren(...marks.map((mark) => entry(
    mark.title, state.config.marks[mark.purpose], () => showMark(mark), mark.id === state.marked,
  )));
  byId("readings").replaceChildren(...[...run.readings].reverse().map((saved) => entry(
    saved.title, new Date(saved.recorded).toLocaleString(),
    async () => openReading(await api(runURL() + "/readings/" + saved.id)),
  )));
  if (!photo) return;
  byId("photo-title").textContent = photo.title;
  byId("photo-info").textContent = [photo.synthetic ? "Synthetic sample" : "Uploaded sample",
    photo.corners ? "Calibrated sample" : "Unregistered photograph"].join(" / ");
  byId("new-mark").disabled = !photo.corners;
  byId("download-original").href = photoURL(photo) + "/source";
  const proof = byId("sample-image");
  proof.src = photoURL(photo) + "/image?r=" + run.revision;
  if (proof.complete) drawCurrentMarks();
}

function field(label, control) {
  const wrap = element("label", label);
  wrap.append(control);
  return wrap;
}

function input(name, type, value, config = {}) {
  const widget = element("input");
  widget.name = name;
  widget.type = type;
  if (value !== undefined) widget.value = value;
  return Object.assign(widget, config);
}

function select(name, options) {
  const widget = element("select");
  widget.name = name;
  for (const [value, label] of options) {
    const option = element("option", label);
    option.value = value;
    widget.append(option);
  }
  return widget;
}

function grid(...fields) {
  const container = element("div");
  container.className = "field-grid";
  container.append(...fields);
  return container;
}

function form(label, submit, ...children) {
  const content = element("form");
  const commit = button(label, null, "primary");
  commit.type = "submit";
  content.append(...children, commit);
  content.addEventListener("submit", guarded(async (event) => {
    event.preventDefault();
    if (commit.disabled) return;
    commit.disabled = true;
    try {
      await submit(content);
    } finally {
      commit.disabled = false;
    }
  }));
  return content;
}

function open(title, body) {
  byId("dialog-title").textContent = title;
  byId("dialog-body").replaceChildren(body);
  byId("dialog-message").textContent = "";
  if (!dialog.open) dialog.showModal();
}

function newRun() {
  const title = input("title", "text", "", { required: true, maxLength: 100 });
  const stock = input("stock", "text", "", { maxLength: 100 });
  const dimensions = { required: true, min: 20, max: 1500, step: .1 };
  const width = input("width_mm", "number", 350, dimensions);
  const height = input("height_mm", "number", 220, dimensions);
  open("New sample run", form("Create run", async () => {
    const run = await api("/weave/runs", "POST", {
      title: title.value, stock: stock.value, width_mm: width.valueAsNumber, height_mm: height.valueAsNumber,
    });
    await openNewRun(run);
    dialog.close();
  }, grid(field("Run title", title), field("Stock", stock)),
  grid(field("Sheet width (mm)", width), field("Sheet height (mm)", height))));
}

function addPhoto() {
  const title = input("title", "text", "", { required: true, maxLength: 100 });
  const upload = input("file", "file", undefined, { accept: "image/jpeg,image/png,image/tiff", required: true });
  const role = select("role", Object.entries(state.config.roles));
  const contents = form("Upload photograph", async (current) => {
    const values = new FormData(current);
    values.set("revision", String(state.run.revision));
    const run = await api(runURL() + "/photos", "POST", values);
    state.photo = run.photos.at(-1).id;
    replaceRun(run);
    dialog.close();
  }, grid(field("Photograph title", title), field("Photo role", role)), field("Photo file", upload));
  open("Add photograph", contents);
}

function fittedImage(img) {
  const width = img.clientWidth;
  const height = img.clientHeight;
  if (!(img.naturalWidth && img.naturalHeight && width && height)) return null;
  const scale = Math.min(width / img.naturalWidth, height / img.naturalHeight);
  const drawWidth = img.naturalWidth * scale;
  const drawHeight = img.naturalHeight * scale;
  return { x: (width - drawWidth) / 2, y: (height - drawHeight) / 2, width: drawWidth, height: drawHeight };
}

function drawPoints(img, overlay, shapes) {
  const fit = fittedImage(img);
  if (!fit) return;
  const density = window.devicePixelRatio || 1;
  overlay.width = img.clientWidth * density;
  overlay.height = img.clientHeight * density;
  const brush = overlay.getContext("2d");
  brush.scale(density, density);
  for (const { points, color, closed } of shapes) {
    const vertices = points.map(([x, y]) => [fit.x + x * fit.width, fit.y + y * fit.height]);
    if (!vertices.length) continue;
    brush.beginPath();
    brush.moveTo(...vertices[0]);
    for (const [x, y] of vertices.slice(1)) brush.lineTo(x, y);
    if (closed && vertices.length > 2) brush.closePath();
    brush.lineWidth = 2.3;
    brush.strokeStyle = color;
    brush.stroke();
    brush.fillStyle = color;
    for (const [i, [x, y]] of vertices.entries()) {
      brush.beginPath();
      brush.arc(x, y, 4.5, 0, Math.PI * 2);
      brush.fill();
      if (shapes.length === 1) {
        brush.font = "bold 12px sans-serif";
        brush.fillText(String(i + 1), x + 6, y - 6);
      }
    }
  }
}

function drawCurrentMarks() {
  if (!activePhoto()?.corners) {
    const overlay = byId("sample-overlay");
    overlay.width = 0;
    overlay.height = 0;
    return;
  }
  const marks = state.run.marks.filter((mark) => mark.photo === state.photo);
  drawPoints(byId("sample-image"), byId("sample-overlay"), marks.map((mark) => ({
    points: mark.points, color: state.marked === mark.id ? "#eb9637" : "#1c9a8f",
    closed: !["register", "stroke"].includes(mark.purpose),
  })));
}

function pointEditor(url, update, maximum) {
  const node = element("div");
  node.className = "image-stage dialog-stage";
  const img = element("img");
  img.alt = "Photograph for placing points";
  img.src = url;
  const overlay = element("canvas");
  const points = [];
  const paint = () => drawPoints(img, overlay, [{ points, color: "#f48a3b", closed: false }]);
  img.addEventListener("load", paint);
  node.addEventListener("click", (event) => {
    const fit = fittedImage(img);
    if (!fit || points.length >= maximum()) return;
    const rect = img.getBoundingClientRect();
    const x = (event.clientX - rect.left - fit.x) / fit.width;
    const y = (event.clientY - rect.top - fit.y) / fit.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    points.push([x, y]);
    update(points);
    paint();
  });
  const clear = () => {
    points.pop();
    update(points);
    paint();
  };
  node.append(img, overlay);
  return { node, clear, points };
}

function calibration() {
  const photo = activePhoto();
  const hint = element("p", "Click the outer sample corners clockwise from the upper left.");
  const editor = pointEditor(photoURL(photo) + "/image?view=source", (points) => {
    hint.textContent = points.length + " of 4 corners placed.";
  }, () => 4);
  open("Calibrate sample", form("Save calibration", async () => {
    if (editor.points.length !== 4) throw new Error("Place four sample corners.");
    const run = await api(photoURL(photo) + "/calibration", "PUT", { revision: state.run.revision, corners: editor.points });
    replaceRun(run);
    dialog.close();
  }, hint, editor.node, button("Undo point", editor.clear)));
}

function newMark() {
  const photo = activePhoto();
  const purpose = select("purpose", Object.entries(state.config.marks));
  const title = input("title", "text", "", { required: true, maxLength: 100 });
  const indicator = element("p");
  const range = () => {
    if (purpose.value === "stroke") return [2, 2];
    if (purpose.value === "register") return [2, 16];
    return [3, 32];
  };
  const update = (points) => {
    const [minimum, maximum] = range();
    const instruction = minimum === maximum ? minimum + " points required" : minimum + "–" + maximum + " points";
    indicator.textContent = instruction + " / " + points.length + " placed";
  };
  const editor = pointEditor(photoURL(photo) + "/image", update, () => range()[1]);
  update([]);
  purpose.addEventListener("change", () => update(editor.points));
  open("Mark area", form("Save mark", async () => {
    const [minimum, maximum] = range();
    if (editor.points.length < minimum || editor.points.length > maximum) {
      throw new Error("The selected mark type needs a different number of points.");
    }
    const run = await api(runURL() + "/marks", "POST", {
      revision: state.run.revision, photo: photo.id, title: title.value, purpose: purpose.value, points: editor.points,
    });
    state.marked = run.marks.at(-1).id;
    replaceRun(run);
    dialog.close();
  }, grid(field("Mark name", title), field("Purpose", purpose)), indicator, editor.node, button("Undo point", editor.clear)));
}

function showMark(mark) {
  state.marked = mark.id;
  renderDesk();
  const view = element("div");
  view.append(element("p", state.config.marks[mark.purpose] + " / " + mark.points.length + " marked points"));
  view.append(button("Remove mark", async () => {
    const result = await api(runURL() + "/marks/" + mark.id, "DELETE", { revision: state.run.revision });
    state.marked = null;
    replaceRun(result);
    dialog.close();
  }));
  open(mark.title, view);
}

function measurement() {
  const available = state.config.readings;
  const operation = select("mode", Object.entries(available).map(([key, value]) => [key, value.title]));
  const fields = grid();
  const hint = element("p");
  const source = activePhoto();
  const comparison = select("other", state.run.photos.filter((item) => item.id !== source.id && item.corners)
    .map((item) => [item.id, item.title]));
  let controls = {};
  let options = {};

  function rebuild() {
    const spec = available[operation.value];
    controls = {};
    options = {};
    hint.textContent = spec.help;
    fields.replaceChildren();
    if (spec.paired) fields.append(field("Comparison photograph", comparison));
    function addMarks(specification, photo) {
      for (const [key, type] of Object.entries(specification)) {
        const entries = state.run.marks.filter((mark) => mark.purpose === type && mark.photo === photo);
        const editor = select(key, entries.map((mark) => [mark.id, mark.title]));
        editor.required = true;
        controls[key] = editor;
        fields.append(field(displayName(key) + " / " + state.config.marks[type], editor));
      }
    }
    addMarks(spec.reference, source.id);
    addMarks(spec.comparison, comparison.value);
    for (const [key, [name, initial, step]] of Object.entries(spec.options)) {
      const value = input(key, "number", initial, { step, required: true });
      options[key] = value;
      fields.append(field(name, value));
    }
  }

  operation.addEventListener("change", rebuild);
  comparison.addEventListener("change", rebuild);
  rebuild();
  open("Take reading", form("Save reading", async () => {
    const readingType = available[operation.value];
    if (readingType.paired && !comparison.value) throw new Error("Add a second calibrated photograph first.");
    if (Object.values(controls).some((control) => !control.value)) throw new Error("Mark the required references first.");
    const result = await api(runURL() + "/readings", "POST", {
      revision: state.run.revision, mode: operation.value, photo: source.id,
      other: comparison.value,
      marks: Object.fromEntries(Object.entries(controls).map(([key, control]) => [key, control.value])),
      options: Object.fromEntries(Object.entries(options).map(([key, control]) => [key, control.valueAsNumber])),
    });
    replaceRun(result.run);
    openReading(result.reading);
  }, field("Reading", operation), hint, fields));
}

function table(rows) {
  const wrapper = element("div");
  wrapper.className = "table-wrap";
  const grid = element("table");
  const keys = [...new Set(rows.flatMap(Object.keys))];
  const header = element("thead");
  const top = element("tr");
  top.append(...keys.map((name) => element("th", displayName(name))));
  header.append(top);
  const body = element("tbody");
  for (const row of rows) {
    const record = element("tr");
    record.append(...keys.map((name) => element("td", display(row[name]))));
    body.append(record);
  }
  grid.append(header, body);
  wrapper.append(grid);
  return wrapper;
}

function openReading(reading) {
  const url = runURL() + "/readings/" + reading.id;
  const pane = element("div");
  const hero = element("div");
  hero.className = "result-hero";
  const snapshot = element("img");
  snapshot.alt = "Archived measurement overlay";
  snapshot.src = url + "/overlay";
  const summary = element("dl");
  for (const [key, value] of Object.entries(reading.summary)) {
    summary.append(element("dt", displayName(key)), element("dd", display(value)));
  }
  hero.append(snapshot, summary);
  const links = element("div");
  links.className = "result-actions";
  links.append(link("Download CSV", url + "/csv"), link("Download evidence packet", url + "/packet"));
  pane.append(hero);
  if (reading.rows.length) pane.append(table(reading.rows));
  pane.append(links);
  open(reading.title, pane);
}

async function init() {
  for (const [id, operation] of Object.entries({
    "new-run": newRun, "add-photo": addPhoto, "new-mark": newMark, "calibrate": calibration,
    "take-reading": measurement, "open-demo": async () => openNewRun(await api("/weave/demo", "POST")),
  })) {
    byId(id).addEventListener("click", guarded(operation));
  }
  byId("dialog-close").addEventListener("click", () => dialog.close());
  byId("sample-image").addEventListener("load", drawCurrentMarks);
  window.addEventListener("resize", drawCurrentMarks);
  [state.config, state.runs] = await Promise.all([api("/weave/config"), api("/weave/runs")]);
  const start = location.hash.slice(1);
  if (state.runs.some((run) => run.id === start)) await pickRun(start);
  else renderDesk();
}

init().catch((error) => { byId("welcome").append(element("p", error.message)); });
