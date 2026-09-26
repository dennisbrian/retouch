# Find duplicates across a shoot

At a convention the same pose often gets shot again minutes later, after a
chat, a lens change or a second cosplayer's turn. Burst grouping only joins
frames shot within about two seconds of each other, so those repeats end up
scattered through the shoot. `./run dupes` finds them wherever they fall.

```bash
./run dupes ~/shoots/2026-09-20                 # report only
./run dupes ~/shoots/2026-09-20 -r              # include subfolders
./run dupes ~/shoots/2026-09-20 --faces         # keeper = sharpest face with open eyes
./run dupes ~/shoots/2026-09-20 --strict        # only near-identical frames
./run dupes ~/shoots/2026-09-20 --move-extras   # move non-keepers aside
```

It writes `duplicates-report/index.html` (a contact sheet of every group, the
suggested keeper outlined in green) and `duplicates-report/duplicates.json`
into the shoot folder. Nothing is moved or deleted unless you pass
`--move-extras`, which moves every non-keeper into `<shoot>/duplicates/`,
keeping subfolders. It never overwrites a file there; move the files back to
undo. Run it on the original photos, before a batch.

## What counts as a duplicate

Same pose and framing: the subject stands the same way, arms and head in the
same place, and the framing moved by no more than about 10% (you stepped
sideways or zoomed a little). A blink, a small head tilt, a different exposure
or white balance still count as duplicates; the keeper choice handles those.
A raised arm, a turn of the body or a different crop make a different photo.

The suggested keeper is the sharpest, best-exposed frame. With `--faces` the
grouped frames are also checked for face focus and closed eyes (the same
ranking as burst picks in the app's Shoot tab), which takes a second or two
per grouped frame.

## How it works

Each photo is read once at low resolution (about 0.2 s for a 26 MP JPEG).
Pairs are aligned over small shifts and zooms, then compared cell by cell so a
moved arm on a small full-length figure still counts as a change. A frame
joins a group only if it matches every frame already in it, so a slow drift
through a pose sequence doesn't chain into one big group. No model weights
are used.

## What was checked

- 1,944 frames from 17 public talking-head and full-length video clips: no
  pair from two different clips was grouped; frames 0.1 s apart were grouped
  every time on full-length clips and 76% of the time on tight face close-ups
  (a moving mouth fills much of a close-up, so it reads as a change).
- 94 public portraits, several people photographed many times on different
  days: no false groups.
- Two real 26 MP con frames of the same cosplayer 7 minutes apart in different
  poses: kept apart. Copies of each with 4-8% reframing, a 1.5° tilt or
  +0.2 EV were grouped with their original, on lighter and darker skin.

## Limits

- In tight face close-ups a change of expression can split frames you'd call
  duplicates; `--strict` makes that more likely, not less.
- Portrait and landscape frames of the same pose are never grouped.
- Mirror images, and crops tighter than about 10%, count as different photos.
- It hasn't yet been run on a full real con shoot; one would show whether the
  thresholds need moving.
