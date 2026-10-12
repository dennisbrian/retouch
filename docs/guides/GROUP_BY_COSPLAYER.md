# Group a Shoot by Cosplayer

At a convention the subjects rotate: one cosplayer for ten minutes, the next
for five, the first one again after lunch. `./run people` sorts a shoot into
one group per cosplayer so each person's photos can be found, retouched with
their own recipe, and handed over together.

```bash
./run people ~/shoots/con-day1                  # preview + report, changes nothing
./run people ~/shoots/con-day1 --move           # by-person/person-01, person-02, ...
./run people --undo ~/shoots/con-day1/by-person/people-manifest.json
```

Without `./run`, use `python -m retouch.cosplayer_groups` with the same options.

## What a preview looks like

```
26 shot(s) → 11 cosplayer group(s)
  person-01/        9 shot(s)  Sun 10:00, Sun 14:00
  person-02/        2 shot(s)  Sun 10:05
  ...
  person-11/        2 shot(s)  Sun 15:41
  Report: ~/shoots/con-day1/people-report/index.html
Preview only. Check the report, then re-run with --move or --copy.
```

`people-report/index.html` shows every group as a contact sheet, with a dashed
line between the times of day a group was shot and a note on photos with two
or more people. `people-report/groups.json` has the same grouping for scripts.
Rename the `person-NN` folders to the cosplayers' names after the move.

## How it decides

It never looks at who a face belongs to. It reads what each person *wears*:

1. **Costume colours.** The MediaPipe multiclass selfie segmenter (the same
   Apache-2.0 model the hair masks use, downloaded once) splits a small
   preview into hair, clothes, accessories and skin. The colours of the
   costume, wig and props become a colour signature, with the part around the
   face (wig, headpiece) and the rest of the costume compared separately as
   well as together. Skin is left out, so skin tone plays no part.
2. **Sessions.** In capture order, shots stay together while the pause between
   them is at most `--gap` (default 10 min) and the costume does not change. A
   change needs two shots in a row that differ from the shots before them, so
   one prop close-up or wide shot does not split a session.
3. **Linking.** Sessions from different times of day join when their average
   signatures match closely (`--link`, default 0.20). The bar is strict on
   purpose: one cosplayer left in two folders costs a drag-and-drop, while two
   people in one folder could send someone the wrong photos.

Shots with two or more people, and shots with nobody in them (props, venue,
details), stay in the session they were taken in but never decide a split or
a link. Shots without a capture time are grouped by colour alone, and ones
with nobody in them go to `unsorted/`.

Every file sharing a stem moves together (`DSCF1234.RAF`, `.JPG`, `.xmp`), as
with `./run split`. The preview comes from a same-name JPEG when there is one,
else the JPEG embedded in a Fuji RAF. Other RAW files without a JPEG are placed
by capture time only.

## Options

| Option | Default | What it does |
|---|---|---|
| `--gap` | `10m` | A pause longer than this always ends a session |
| `--link` | `0.20` | How alike two sessions must be to join (0–1, lower is stricter) |
| `--split` | `0.40` | How big a costume change splits a session (0–1) |
| `--no-link` | off | Never join sessions across the day |
| `--move` / `--copy` | preview | Put each group in `by-person/person-NN/` |
| `-o` | `FOLDER/by-person` | Where the person folders go |
| `-r` | off | Include subfolders (earlier `by-person/` output is skipped) |
| `--mtime-fallback` | off | Use file modified time when EXIF has no capture time |
| `--no-report` / `--json` | | Skip the HTML report / print the groups as JSON |

## What it gets wrong

- **Matching colours.** Two people in black suits, or a group cosplay in one
  uniform, look alike to it. They can share a group, most often when shot back
  to back. Lower `--split` or check the report.
- **Very different framing.** A cosplayer shot full length by a window in the
  morning and in tight close-up in the afternoon can stay as two groups.
  Raise `--link` a little (0.30) or merge the folders by hand.
- **Costume changes.** Someone who changes costume becomes a new group; that is
  by design, since the colours are all it sees.

## Calibration (2026-09-26)

Measured on a mock convention day, run in 10 random orders: two real
cosplayers each shot in two sessions hours apart (9 photos of one costume in 7
outdoor locations; Alex's 2 Fujifilm X-T4 frames of a white-wig, black-vinyl
cosplay plus 4 crops of them), 37 full-length fashion and street photos of
other people (several with darker skin; public examples from the IDM-VTON and
OOTDiffusion repositories), and 12 video clips (LivePortrait examples) as
4-shot sessions of other people, with 30 s to 4 min between subjects.

| Setting | Pairs put together correctly | Same-person pairs found | Cosplayer seen twice rejoined |
|---|---|---|---|
| defaults (`--link 0.20 --split 0.40`) | 97.8% | 92.5% | 7-location costume: 10/10 runs; window vs close-up: 0/10 |
| `--no-link` | 97.5% | 79.8% | never (by design) |
| `--link 0.30` | 93.4% | 92.5% | same as defaults |

Stress case: adding 70 celebrity headshots, mostly dark suits, drops the
first column to 86.7% at the defaults. That is the matching-colours limit
above.

End to end on a 26-photo folder (the two X-T4 frames at full 26 MP plus 24
smaller photos with EXIF times): 11 groups, all correct, in 4.7 s including
start-up. Previews decode at reduced size, so a 26 MP JPEG costs tens of
milliseconds.

Not yet run on a real convention shoot; one full day's folder would show
whether the defaults need moving.
