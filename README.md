# 040COMBAT — Deurne showcase

A 26-second 9:16 showcase for 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s; the grid is fitted on its kick pattern. The intro builds
under the opening and pauses on a freeze. The drop lands on the first competition photo,
the breakdown carries the people, and the beat returns for the last burst and the end card.
Every splice is on a bar line.

| time | music | picture |
|---|---|---|
| 0–8.5 | intro (song bars 8–12) | Milosz's high kick lands on the first downbeat · a body-lock lift and drop · a women's round, her body kick on the beat · pads with the coach into a double-leg (the lift in slow motion) · the packed class shadowboxing (120 fps, half speed) · Milosz and Bilal trading under the wall logo |
| 8.5–9.5 | held breath: the intro's last instant rings out in a hall | his punch freezes in black & white |
| 9.5–15.5 | the drop (bars 32–35) | Milosz's hand raised · Imre's arms up · the two together, above and below, a gold hairline between · back into motion with a kid on the pads with his coach and a round in the full room |
| 15.5–21.6 | the breakdown (bars 52–54, 59) | his laugh after the round · a coach on the mat with a pair · the team photo from the faces out, then the whole room under the logo on the 808 |
| 21.6–26.7 | the beat returns (bars 60–62) | four hits: her combination, a scramble, Milosz's jab with the room watching, a pad strike under the wall logo that freezes, goes black & white and becomes **040COMBAT / DEURNE** |

Real sound from the floor cuts through the music in a few places: the kick, the
takedown, her kick, the pads, the punch the music stops on, the laugh, and the last
strike.

**Grade.** Each shot is matched first. The white balance is neutralised on near-neutral
midtones (walls and mats), then the black and white points and the exposure are set to a
common level (`measure`). One gentle look goes on top: filmic contrast, natural skin,
deep neutral blacks, and a touch of the logo's gold in the highlights. The film pass is
light: a little halation and bloom, and fine grain. The photos keep their own grade.

**Identity.** The colours are black, gold (#C09B53, measured from the logo) and off-white.
The end card is set in Barlow Condensed (`brand/fonts`, OFL). The logo itself is never
pasted on: it appears as the real sign on the gym wall, in black and white behind the
type.

```bash
python edits/showcase.py                                  # renders/040COMBAT_showcase_9x16.mp4 (+ .wav)
python edits/showcase.py --stills 0.6 9.6 --scale 0.5     # review frames
python edits/showcase.py --edl                            # cut list
```

`AFTERFILM_FFMPEG` points at a full ffmpeg build. The engine (`afterfilm/`) comes from the
District98 after-movie project: timeline, decoding, transitions and film finish.

## Footage handover

Footage, photos, music and renders never go in git, because the repo is public. They sit
on a **draft** release tagged `footage`, which only collaborators can see. The file names
in `edits/showcase.py` are the names on that release; download them into `footage/`.
