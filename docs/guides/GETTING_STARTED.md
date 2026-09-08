# Retouch Getting Started

This is the shortest path to retouching one photo on a desktop. Retouch runs
locally; the GUI is opened in a browser, and the original photo remains your
source file.

## 1. Start the local GUI

From the repository root:

```bash
cd /Applications/htdocs/retouch
python3 gui.py
```

Open <http://127.0.0.1:7860/> in Safari, Chrome, or another browser. Keep the
terminal running while using the GUI. Press `Ctrl+C` in that terminal to stop
it.

For an offline run, use `RETOUCH_OFFLINE=1` before the command. Install the
dependencies and verify the required model files first; see the
[README installation section](../../README.md#install) and
[runtime recovery guide](RUNTIME_RECOVERY.md).

## 2. Retouch one photo

1. Open **Single Photo Editor** and upload a photo.
2. Choose a recipe. **Natural** is a good starting point for a restrained
   result; use the other recipes when a different look is intended.
3. Leave **Full (native face crops)** selected for a final image. Use
   **Draft (proxy, fast)** only for quick iteration or contact-sheet work.
4. Click **Render Preview** while choosing a recipe or adjusting sliders. Use
   the compare view to check the result against the source.
5. When the settings look right, click **Export Full Quality**. For a normal
   delivery, keep **Export Resolution** at **Original** and review the
   downloaded file before sharing it.

`Render Preview` is a fast preview and is not the final delivery file.
`Export Full Quality` processes one photo at the full quality tier. For a
folder, use **Export All → Batch** instead.

## 3. Use the command line instead

The CLI is useful when the output folder, recipe, or repeatability matters:

```bash
python3 cli.py /path/to/photo.jpg \
  -o /path/to/output \
  --recipe natural
```

To compare several recipes on one photo, use the recipe-sweep runner:

```bash
./executable/recipe_sweep /path/to/photo.jpg \
  /path/to/recipe-check \
  --recipes natural,portrait,cosplay \
  --compare
```

The sweep writes recipe renders, comparison images, a contact sheet, and a
machine-readable `manifest.json` in the output folder. Keep output in a
separate folder so the source photo is easy to identify and preserve.

## 4. Check a face-aware result

For a real face retouch, confirm the manifest reports:

- `global_only: false`
- `face_aware_run: true`
- a positive `face_count`
- output dimensions matching the selected delivery resolution

Inspect the source and output side by side, then zoom to the eyes, skin
texture, hairline, and face edges. Automated QA flags are review signals, not
a replacement for this visual check. Use `--global-only` only when you
intentionally want global colour/impact processing without face detection.

## 5. If something goes wrong

- **The page does not open:** confirm the terminal process is still running
  and try `http://127.0.0.1:7860/` again.
- **The render is slow:** previews use the fast path; full-resolution export
  takes longer, especially on CPU.
- **No face is detected:** check the image framing, lighting, and runtime
  status. Do not use global-only mode as proof that face retouching worked.
- **The result looks too strong:** try the `natural` recipe, reduce the
  relevant sliders, and compare at 100% before exporting again.

For deeper GUI details, see [GUI.md](GUI.md). For batch options, see
[BATCH_GUIDE.md](BATCH_GUIDE.md). For output colour and export contracts, see
[COLOR_DELIVERY_TRUTH.md](COLOR_DELIVERY_TRUTH.md).
