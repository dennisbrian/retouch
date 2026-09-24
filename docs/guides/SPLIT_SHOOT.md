# Split a Shoot by Capture Time

A convention or event card dump is one long run of files. `./run split` sorts it
into one folder per set, using each photo's EXIF capture time, so each cosplayer
or location can be batch-retouched with its own recipe.

```bash
./run split ~/shoots/2026-09-20                 # preview, changes nothing
./run split ~/shoots/2026-09-20 --move          # create the set folders
./run split --undo ~/shoots/2026-09-20/split-manifest.json
```

Without `./run`, use `python3 split_shoot.py` with the same options. It needs
only the Python standard library, so it works before `./setup` has finished.

## What a preview looks like

```
30 shot(s), 44 file(s) → 4 set(s) by gaps over 15m
  Cameras: FinePix S5Pro, NIKON D70 (use --camera-offset if their clocks differ)
  01_2026-09-20_1402/            12 shot(s)  2026-09-20 14:02–14:09
  02_2026-09-20_1431/            12 shot(s)  2026-09-20 14:31–14:43
  03_2026-09-20_1530/             5 shot(s)  2026-09-20 15:30–15:32
  no-capture-time/                1 shot(s)  no capture time
  Leaving 1 other file(s) where they are
Preview only. Re-run with --move or --copy to create the folders.
```

Folder names start with the set number and the time of its first shot, so they
sort in shooting order.

## How files are grouped

- **One shot, one folder.** Every file sharing a stem moves together:
  `DSCF1234.RAF`, `DSCF1234.JPG`, `DSCF1234.RAF.xmp` and `DSCF1234.xmp`.
  Stems match regardless of case.
- **Capture time** comes from EXIF `DateTimeOriginal` (with sub-seconds), then
  `DateTimeDigitized`, then `DateTime`. It is read straight from the file header
  without decoding pixels, for JPEG, TIFF, Fuji RAF, DNG, NEF, NRW, CR2, CR3,
  ARW, ORF, RW2, PEF, SRW, PNG and WebP.
- **Photos with no capture time** go to `no-capture-time/`. Pass
  `--mtime-fallback` to place them by file modified time instead (unreliable
  after a copy between drives).
- **Other files** (notes, hidden files) are left where they are.

## Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--by gap` | on | Start a new set when the pause since the previous shot is longer than `--gap`. |
| `--gap` | `15m` | Pause that starts a new set: `90s`, `10m`, `1h`, `1h30m` or `00:10:00`. A pause exactly equal to the gap stays in the same set. |
| `--by hour` / `--by day` | | Fixed buckets instead, named `2026-09-20_14h` or `2026-09-20`. |
| `--move` / `--copy` | preview | Act on the plan. Without either, nothing changes on disk. |
| `-o`, `--output` | inside the folder | Create the set folders somewhere else. |
| `-r`, `--recursive` | off | Include subfolders. Two files with the same name landing in one set stop the run before anything moves. |
| `--camera-offset MODEL=OFFSET` | none | Shift one camera's clock before grouping, for example `"X-T5=+00:03:20"` or `"NIKON D70=-2m"`. `MODEL` is the EXIF model name the preview lists. Repeatable. |
| `--mtime-fallback` | off | See above. |
| `--json` | off | Print the plan as JSON. |
| `--undo MANIFEST` | | Put every moved file back and remove the emptied set folders. |

## Safety

- The default run is a preview.
- `--move` and `--copy` check every destination first and change nothing if
  any file would be overwritten or two files would collide.
- Every move or copy writes `split-manifest.json` in the output folder, updated
  as files move, so `--undo` also recovers an interrupted run. A second split
  into the same output folder is refused while that manifest exists.
- `--undo` only works for `--move`. For `--copy`, delete the set folders.

## Then retouch each set

```bash
./run batch ~/shoots/2026-09-20/01_2026-09-20_1402 -o ~/out/set01 --recipe cosplay_clear_v1
./run batch ~/shoots/2026-09-20/02_2026-09-20_1431 -o ~/out/set02 --recipe cosplay_portrait_polish_v1
```

See [BATCH_GUIDE.md](BATCH_GUIDE.md) for the batch options.
