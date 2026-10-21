#!/usr/bin/env python
"""Crash-recovery harness for the Layer Editor tab (source app or packaged app).

Drives the Gradio app with gradio_client: open a photo, render a recipe layer,
regenerate that recipe layer with overrides, crop the canvas, then SIGKILL the
server, restart it, and restore the autosaved document. Asserts the restored
composite preview is pixel-identical to the pre-crash preview.

Endpoint discovery is label-based (raw /config is parsed, trigger-component
labels like "Open photo" are matched), so /edit_N id ordering churn cannot
silently retarget the harness.

Usage:
    python frozen_recovery_harness.py                     # launch source app
    python frozen_recovery_harness.py --url http://127.0.0.1:7860
    python frozen_recovery_harness.py --app "/path/Pro Max Retouch Studio.app"
Exit codes: 0 pass, 1 assertion failure, 2 harness/setup error.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import gui  # noqa: F401 - installs the gradio_client schema patch (bool schemas)
from gradio_client import Client, handle_file

CANVAS_EMPTY = {"background": None, "layers": [], "composite": None}


# ---------------------------------------------------------------- setup utils
def synthetic_photo(path: Path) -> Path:
    """Small deterministic test image (no face needed: recipes still run)."""
    y, x = np.mgrid[0:240, 0:320]
    rgb = np.stack([x * 255 // 320, y * 255 // 240,
                    ((x + y) * 255 // 560)], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return path


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/config", timeout=5)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"app not reachable at {url}")


class App:
    """Server under test: external URL or a subprocess we may SIGKILL."""

    def __init__(self, args, recovery_dir: Path):
        self.url = args.url
        self.proc = None
        self.launched = False
        self.args = args
        self.recovery_dir = recovery_dir

    def start(self):
        # --url mode connects once; a post-sigkill start() must relaunch.
        if self.url and not self.launched:
            wait_http(self.url, 10)
            return
        # A stale server from an earlier run would answer wait_http and get
        # driven instead of the freshly launched one; refuse to start on it.
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.args.port}/config",
                                   timeout=2)
            raise RuntimeError(
                f"port {self.args.port} already serves an app; "
                "kill it or pass --port")
        except Exception:
            pass
        env = dict(os.environ,
                   RETOUCH_EDITOR_RECOVERY_DIR=str(self.recovery_dir),
                   RETOUCH_OFFLINE="1",  # update check would stall startup offline
                   GRADIO_ANALYTICS_ENABLED="False")
        # ``gui.py`` __main__ hardcodes port 7860; launch the module's Blocks
        # app directly so the harness owns the port.
        boot = ("import gui; "
                f"gui.app.queue(default_concurrency_limit=1)"
                f".launch(server_name='127.0.0.1', server_port={self.args.port})")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", boot], cwd=str(REPO), env=env,
            stdout=open(self.recovery_dir / "server.log", "ab"),
            stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.args.port}"
        self.launched = True
        wait_http(self.url, self.args.startup_timeout)

    def sigkill(self):
        assert self.proc is not None, "--url mode cannot be crash-tested"
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()
        self.proc = None

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ------------------------------------------------------- endpoint discovery
class Api:
    """Map trigger-component labels to dependency ids via raw /config."""

    def __init__(self, client: Client, url: str):
        self.url = url
        cfg = json.load(urllib.request.urlopen(f"{url}/config"))
        self.comp = {c["id"]: c for c in cfg["components"]}
        self.deps = cfg["dependencies"]

    def _label(self, cid):
        """Trigger text: Buttons carry their caption in props.value."""
        if cid is None or cid not in self.comp:
            return ""
        props = self.comp[cid].get("props") or {}
        return props.get("value") if props.get("value") else props.get("label") or ""

    def by_trigger(self, label: str) -> dict:
        hits = [d for d in self.deps
                if any(self._label(t[0]) == label and t[1] == "click"
                       for t in d.get("targets", []))]
        assert len(hits) == 1, f"trigger {label!r}: {len(hits)} matches"
        return hits[0]

    def by_input(self, label: str) -> dict:
        hits = [d for d in self.deps
                if any(t[1] == "input" and self._label(t[0]) == label
                       for t in d.get("targets", []) if t)]
        assert len(hits) == 1, f"input {label!r}: {len(hits)} matches"
        return hits[0]

    def chain_after(self, dep: dict) -> list:
        """Deps chained via .success/.then (render click -> rendering -> finish).

        Gradio only fires these from the browser JS; a raw API predict runs
        the trigger alone, so the harness must run the chain itself.
        """
        by_pred = {d.get("trigger_after"): d for d in self.deps
                   if d.get("trigger_after") is not None}
        chain, cur = [], dep["id"]
        while cur in by_pred:
            cur_dep = by_pred[cur]
            chain.append(cur_dep)
            cur = cur_dep["id"]
        return chain

    def inputs(self, dep) -> list:
        return [self.comp[i] for i in dep["inputs"] if i in self.comp]

    def sendable(self, dep) -> list:
        """Inputs the client must supply (gr.State is session-injected)."""
        return [c for c in self.inputs(dep) if c["type"] != "state"]


def call(client: Client, api: Api, dep: dict, values: dict):
    """Predict on a dependency, values keyed by input component label."""
    args = [values.get((c.get("props") or {}).get("label"),
                       values.get(f"id:{c['id']}"))
            for c in api.sendable(dep)]
    assert all(a is not None for a in args), \
        f"missing arg for {[ (c.get('props') or {}).get('label') for c in api.sendable(dep)]}"
    res = client.predict(*args, fn_index=dep["id"])
    return res[0] if isinstance(res, tuple) and len(res) == 1 else res


def view_fields(res) -> dict:
    """Unnamed endpoints return a flat list: align by position with outputs."""
    return {"raw": res}


# ------------------------------------------------------------------- steps
def find_status_text(res):
    for v in (res if isinstance(res, (list, tuple)) else [res]):
        if isinstance(v, str) and "layers" in v and "·" in v:
            return v
    return None


def doc_dims(res):
    m = re.search(r"· (\d+) × (\d+) · (\d+) layers", find_status_text(res))
    assert m, f"no status line in view: {find_status_text(res)}"
    return int(m[1]), int(m[2]), int(m[3])


def layer_ids(res):
    for v in (res if isinstance(res, (list, tuple)) else [res]):
        if isinstance(v, dict) and "choices" in v and v["choices"]:
            return [c[1] if isinstance(c, (list, tuple)) else c
                    for c in v["choices"]]
    return None


def preview_pixels(res):
    """Composite preview: ndarray via client, or gradio tmp filepath str."""
    for v in (res if isinstance(res, (list, tuple)) else [res]):
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
            return v
    for v in (res if isinstance(res, (list, tuple)) else [res]):
        if isinstance(v, str) and "gradio" in v and Path(v).is_file():
            return np.asarray(Image.open(v).convert("RGBA"))
    return None


def poll(client, api, select_dep, layer_id, want, timeout, what):
    """Poll the layers dropdown until `want(layer_ids)` is truthy."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        res = call(client, api, select_dep,
                   {"Layers (top first)": layer_id,
                    "Paint selected layer mask": CANVAS_EMPTY})
        last = res
        ids = layer_ids(res)
        if ids and want(ids):
            return res
        time.sleep(1)
    raise AssertionError(f"timeout waiting for {what}; last ids={layer_ids(last)}")


def call_chain(client, api, dep, values):
    """Run dep then its .success/.then chain (server does not chain for API)."""
    res = call(client, api, dep, values)
    for chained in api.chain_after(dep):
        res = call(client, api, chained, values)
    return res


def run(args) -> int:
    recovery_dir = Path(args.recovery_dir).resolve()
    recovery_dir.mkdir(parents=True, exist_ok=True)
    leftovers = list(recovery_dir.glob("*.json"))
    assert not leftovers, f"recovery dir not clean: {leftovers}"

    photo = Path(args.photo) if args.photo else synthetic_photo(
        recovery_dir / "harness_photo.png")

    app = App(args, recovery_dir)
    app.start()
    client = Client(app.url)
    api = Api(client, app.url)
    open_dep = api.by_trigger("Open photo")
    select_dep = api.by_input("Layers (top first)")
    render_dep = api.by_trigger("Render recipe")
    crop_dep = api.by_trigger("Crop canvas")
    refresh_dep = api.by_trigger("Refresh list")
    restore_dep = api.by_trigger("Restore")

    # 1. Open photo.
    res = call(client, api, open_dep, {
        "Photo": handle_file(str(photo)),
        "Project folder": "",
        "Discard unsaved edits and unapplied strokes when opening": False,
        "Paint selected layer mask": CANVAS_EMPTY})
    ids = layer_ids(res)
    assert ids and len(ids) == 1, f"photo layer missing: {res}"
    photo_id = ids[0]
    w, h, nlayers = doc_dims(res)
    print(f"[1] opened photo {photo.name} {w}x{h} layer={photo_id}")

    # 2. Render recipe (new masked layer).
    from retouch.editor.recipe_state import EXTRA_RECIPE_CONTROLS, RECIPE_OVERRIDE_SPECS
    extra_defaults = {}
    for control_name, label in EXTRA_RECIPE_CONTROLS.items():
        spec = RECIPE_OVERRIDE_SPECS[control_name]
        extra_defaults[f"Override {label.lower()}"] = False
        extra_defaults[label] = spec.default
    res = call_chain(client, api, render_dep, {
        "Retouch recipe": args.recipe, "Result layer strength": 100,
        "Paint selected layer mask": CANVAS_EMPTY,
        "Recipe render target": "New layer",
        "Override smoothing and whitening": False,
        "Recipe skin smoothing": 30, "Recipe skin whitening": 0,
        **extra_defaults})
    ids = layer_ids(res)
    assert ids and len(ids) == 2, f"recipe layer missing after render: {res}"
    recipe_id = [i for i in ids if i != photo_id][0]
    print(f"[2] recipe layer rendered id={recipe_id}")

    # 3. Regenerate that recipe layer with overrides (new-feature edit).
    call(client, api, select_dep,
         {"Layers (top first)": recipe_id,
          "Paint selected layer mask": CANVAS_EMPTY})
    before = preview_pixels(call(client, api, select_dep,
                                 {"Layers (top first)": recipe_id,
                                  "Paint selected layer mask": CANVAS_EMPTY}))
    res = call_chain(client, api, render_dep, {
        "Retouch recipe": args.recipe, "Result layer strength": 100,
        "Paint selected layer mask": CANVAS_EMPTY,
        "Recipe render target": "Regenerate selected recipe layer",
        "Override smoothing and whitening": True,
        "Recipe skin smoothing": 55, "Recipe skin whitening": 25,
        **extra_defaults})
    after = preview_pixels(res)
    assert after is not None, "no preview pixels after regenerate"
    if after.shape != before.shape or np.array_equal(after, before):
        after = preview_pixels(call(client, api, select_dep,
                                    {"Layers (top first)": recipe_id,
                                     "Paint selected layer mask": CANVAS_EMPTY}))
    assert after is not None and before is not None and not np.array_equal(after, before), \
        "regenerated recipe layer preview never changed"
    print(f"[3] recipe layer regenerated (override smooth=55 whiten=25); "
          f"max diff {int(np.abs(after.astype(int)-before.astype(int)).max())}")

    # 4. Crop the canvas (second edit; dirties the document -> autosave).
    res = call(client, api, crop_dep, {
        "Layers (top first)": recipe_id,
        "Paint selected layer mask": CANVAS_EMPTY,
        "Crop X": 0, "Crop Y": 0,
        "Crop width (pixels)": max(1, w - 16), "Crop height (pixels)": max(1, h - 16)})
    w2, h2, _ = doc_dims(res)
    assert (w2, h2) == (max(1, w - 16), max(1, h - 16)), f"crop failed: {(w2, h2)}"
    print(f"[4] cropped canvas to {w2}x{h2}")

    pre = preview_pixels(call(client, api, select_dep,
                              {"Layers (top first)": recipe_id,
                               "Paint selected layer mask": CANVAS_EMPTY}))
    assert pre is not None, "no preview pixels pre-crash"
    pre_layers = layer_ids(call(client, api, select_dep,
                                {"Layers (top first)": recipe_id,
                                 "Paint selected layer mask": CANVAS_EMPTY}))
    assert len(pre_layers) == 2

    # 5. SIGKILL, restart.
    print("[5] SIGKILL server; restarting")
    app.sigkill()
    app.start()
    client = Client(app.url)
    api = Api(client, app.url)

    # 6. Restore the autosave.
    res = call(client, api, refresh_dep, {})
    entries = res if isinstance(res, list) else [res]
    entry = None
    for v in entries:
        if isinstance(v, dict) and v.get("choices"):
            assert len(v["choices"]) == 1, f"expected 1 recovery, got {v['choices']}"
            choice = v["choices"][0]
            entry = choice[1] if isinstance(choice, (list, tuple)) else choice
    assert entry, f"no recovery entry offered: {res}"
    print(f"[6] recovery entry {entry}")

    res = call(client, api, restore_dep, {
        "Autosaved documents": entry,
        "Discard unsaved edits and unapplied strokes when opening": False,
        "Paint selected layer mask": CANVAS_EMPTY})
    post = preview_pixels(res)
    assert post is not None, "no preview pixels post-restore"
    assert post.shape == pre.shape, f"shape {post.shape} != {pre.shape}"
    diff = int(np.abs(post.astype(int) - pre.astype(int)).max())
    assert diff == 0, f"restored preview differs: max pixel diff {diff}"
    w3, h3, n3 = doc_dims(res)
    assert (w3, h3, n3) == (w2, h2, 2), f"restored doc {w3}x{h3} n{n3}"
    print(f"[6] restored identical pixels ({w3}x{h3}, {n3} layers)")

    # 7. Cleanup: the restored entry is now live in the editor session, so
    # "Discard autosave" refuses it by design ("open in another editor
    # window"); the harness owns the recovery dir and clears it directly.
    app.stop()
    for stale in recovery_dir.glob("*.json"):
        stale.unlink()
    for stale in recovery_dir.glob("*.comp"):
        shutil.rmtree(stale, ignore_errors=True)
    print("[7] recovery dir cleared")
    print("PASS")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", help="connect to a running app instead of launching")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--recovery-dir",
                   default=str(Path(__file__).parent / "recovery"))
    p.add_argument("--photo", help="image to open (default: synthetic)")
    p.add_argument("--recipe", default="natural")
    p.add_argument("--startup-timeout", type=float, default=300)
    args = p.parse_args()
    try:
        sys.exit(run(args))
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    except Exception as exc:  # harness/infra error
        print(f"ERROR: {type(exc).__name__}: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
