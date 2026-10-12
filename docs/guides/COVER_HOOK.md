# Pick the cover and slides 2-3

`./run cover` ranks one shoot's retouched photos as Instagram covers, suggests
a cover plus slides 2 and 3, and exports the cover's 4:5 crop. It only reads
the photos. Its files go to `FOLDER/cover-hook/`.

```bash
./run cover ~/photos_out                   # offline: 4 of the 6 checks
./run cover ~/photos_out --vision          # plus one Claude look per top candidate
./run cover ~/photos_out --vision --vision-top 12
./run cover ~/photos_out -r                # include subfolders
./run cover a.jpg b.jpg c.jpg --no-crop    # a hand-picked list, scores only
```

## The six checks

| # | Check | Weight | Measured by |
|---|---|---|---|
| 1 | The cover is a finished photo, not a poster, collage or screenshot | 30 | Claude look (`--vision`) |
| 2 | The subject survives the 4:5 feed crop and the 3:4 grid crop | 25 | `social_crops` crop plan, share of each head and of the person mask kept |
| 3 | One vivid colour dominates the subject | 15 | `image_analyzer.colour_pop_score` |
| 4 | The whole costume reads: full-body or pair over a tight half-body | 20 | Claude look; offline, an estimate from face count and face height |
| 5 | Clean light, not a colour cast, dim or crushed | 10 | `image_analyzer`; outdoor / studio over indoor only with `--vision` |
| 6 | Face sharp, eyes open | gate | `face_quality`: both eyes closed x0.6 (a wink is noted, not demoted), a face much softer than the rest of the shoot x0.8 |

A check that could not be measured is left out, and the score is taken over
the checks that were. Without `--vision`, check 1 is skipped and check 4 is an
estimate; the tool says so when it prints the ranking.

## How far to trust it

The weights are set by hand from a small study (124 covers from 12 cosplay
accounts, top third against bottom third by likes inside each account):

- One difference was large: poster and promo covers were 0 of 27 in the top
  third and 8 of 27 in the bottom third.
- Among real photos the leans were weak: saturated colour 14 of 21 against 10
  of 21, full-body or pair 15 against 10, dim indoor light and hard contrast
  leaning the other way.
- Eye contact, face size, centring and background clutter showed no
  difference and are not scored.

Treat the ranking as a shortlist to look at, not a prediction of likes.

## The Claude look

`--vision` asks one question per photo, for the best `--vision-top`
(default 8) photos of the offline ranking, on a 1024 px copy: is it a photo,
how is it framed, how many people, is the costume readable, what is the
setting. By default it runs the local Claude Code CLI (`claude -p`) with your
signed-in subscription, so no API key is needed. The Anthropic API is used only
when `ANTHROPIC_API_KEY` is set, the `anthropic` package is installed and
`RETOUCH_COVER_LOOK_MODEL` names a model. Answers are cached by file content in
`cover-hook/.cache/look.json`; `--refresh` asks again.

## Suggestion

- **Cover:** the best-scoring photo (never a poster or screenshot).
- **Slide 2:** the best photo with the other kind of framing: a tight shot
  after a full-body cover, a wider shot after a tight cover.
- **Slide 3:** the best photo left.

A slide is never a near-duplicate of one already chosen (same pose and
framing, checked with the [duplicate finder](DUPLICATES.md)).

## Output

- `cover-hook/cover_scores.json` and `cover_scores.csv`: every photo with its
  rank, score, per-check values and notes.
- `cover-hook/4x5/<name>_4x5.jpg`: the cover's feed crop (skip with `--no-crop`).
- `--manifest PATH`: also records the score on assets a shoot-review manifest
  already lists, when path and content match. The next shoot scan rebuilds
  that evidence and drops the entry.

## Limits

- A face on a poster or banner behind the subject can be counted as a second
  person offline; `--vision` corrects the framing.
- The crop check needs a face or a person mask. With neither it is skipped.
- The light check cannot tell a location without `--vision`.
- Scores compare photos inside one shoot. They are not comparable across shoots.
