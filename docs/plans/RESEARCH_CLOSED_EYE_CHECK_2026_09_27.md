# Closed-eye flag and burst picks: glasses, darker skin, bursts (2026-09-27)

The closed-eye flag (`face_quality.classify_eyes_open`, PR #9) and the burst
blink check (`shoot_intelligence.rank_burst_candidates`) were calibrated on
talking-head clips without glasses and mostly lighter skin. This note measures
them on glasses, darker skin (simulated), sunglasses and simulated bursts, and
records the two burst misfires that were fixed. Renders are unaffected; only
flags, burst ranks and the review page can change.

Public material was fetched for measurement only and is not in the repo:
Yale Face Database (via github.com/zonagit/HadoopSparkEigenfaces), deepface
`tests/unit/dataset`, dlib `examples/faces`, face_recognition examples,
LivePortrait `assets/examples/driving` (12 clips, 4 with glasses).

## Stills

| Set | Faces | Result |
|---|---|---|
| Yale "glasses" (thin frames) | 16 | 16 open read `yes` |
| Yale "sleepy" (eyes closed) | 15 | 14 `no`, 1 `uncertain` (0.154) |
| Yale "wink" | 15 | 13 `no`, 2 `uncertain` |
| Yale other open conditions (incl. harsh side light) | 104 | 0 false `no`; the one `no` (rightlight, 0.080) has closed eyes |
| deepface / dlib / face_recognition (few darker-skinned subjects, 2 with thick glasses) | 127 | 0 `no`; 33 `uncertain`, 26 of them faces under 64 px |
| Open-eyed stills with painted-on opaque sunglasses | 70 | 69 `yes`, 1 `uncertain` |
| Alex's 4 cosplay photos (white face paint, red contacts, wig over one eye) | 4 | 4 `yes`, eyes open |

Sunglasses never read as closed, but they read as `yes`: the landmark model
draws a neutral open eye behind an opaque lens. That is harmless for burst
ranking (every frame of that person scores the same) but means `yes` is not
proof the eyes are visible.

## Darker skin (simulated)

Every frame of the 12 clips (4,412 frames) was re-measured after darkening
the whole frame to about 0.35x linear light with a brown shift. The eyes-open
state agreed with the original on 93.5-100% of frames per clip (median
aperture shift -0.004 to +0.013); the lowest agreement is a clip whose eyes
hover on the 0.15 closed line. No systematic bias: this checks landmark
stability under low contrast, not real darker-skinned blinks, which no public
clip here had.

## Glasses in video

In the 4 glasses clips every `no` frame checked by eye was a closed or nearly
closed eye (d9: 32 sampled of 113, d13: the 12 lowest of 13). Open eyes behind strong lens
glare (d19) read 0.17-0.19, so `uncertain`, never `no`.

## Bursts: two misfires, both fixed

Simulated 5-frame bursts (every 6th frame, about 5 fps) over the 12 clips:
2,095 frames, 186 flagged by the absolute band and 12 more by the relative
check. Checked by eye, 11 of those 12 were half-closed eyes or downward looks
with lowered lids; one was plainly open.

1. **Open eyes next to a wide-eyed frame.** A subject raising their brows
   read 0.49; a normal open frame at 0.27 fell under 0.55x of that and was
   flagged. Every eye the relative check caught correctly read 0.15-0.25, so
   the relative check now only applies below `BLINK_RELATIVE_CEILING` = 0.25.
   On the simulated bursts this removes the false flag and one borderline
   downward look (0.251).
2. **Two people swapping the bigger face.** The relative check compared the
   main face of each frame with the widest main face in the burst, so when two
   cosplayers of similar size swapped which face was bigger, one person's
   eyes were compared with the other's. A real composite (a wide-eyed and a
   narrow-eyed subject side by side, 4 frames) flagged 2 of 4 open-eyed frames
   before the fix and 0 after. Faces are now tracked across the burst by bbox
   centre (`SUBJECT_TRACK_MAX_SHIFT` = 0.5 face widths) and each face is
   compared with its own track, which also lets a second subject's half-blink
   be caught relative to their own eyes (previously only the main face had a
   relative check).

## Not covered

- No real con bursts (Alex has not uploaded one); the clip "bursts" are
  talking heads, not a camera's burst mode.
- No real darker-skinned blinks; darker skin is simulated or stills only.
- Thick cosplay goggles and visors were not tested beyond painted lenses.
- Faces under 64 px stay `uncertain` by design.
