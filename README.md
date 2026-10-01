# 040COMBAT — Deurne showcase

A 24-second 9:16 showcase for 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s; the grid is fitted on its kick pattern. There is one bar
of build, then the intro under the action, a held breath on a freeze, the drop for the
achievements and a burst, the breakdown for the people, and the beat back for the last
hits and the end card. Every splice is on a bar line.

| time | music | picture |
|---|---|---|
| 0–2 | build (bar 7) | the hook: nine impacts cut on the eighth notes, each on its own sound (a takedown, ground and pound in gold gloves, her kick, a cross, a jab, a face, an inversion, gold gloves, the wall sign), then three staccato punch-ins on Milosz |
| 2–8 | intro (bars 8–11) | the kick lands on the downbeat in slow motion, trailing light · a whip into a body-lock throw · her combination · pads with the coach into a double-leg · a jump-cut exchange under the wall logo |
| 8–9 | held breath: the intro's last instant rings out in a hall | his punch freezes on a negative frame and the picture closes to a slit |
| 9–15 | the drop (bars 32–35) | the slit bursts open, gilded · Milosz's hand raised, a whip up to Imre, the two slam together above and below · five hits: gold gloves, her knee, a kid on the pads in light trails, the full room trading, a roll to the back |
| 15–19 | the breakdown (bars 52, 59) | his laugh after the round · the packed class in slow motion · the whole room lit out of silhouette on the 808 |
| 19–24 | the beat returns (bars 60–62) | three pad combinations under the wall sign, each punched in closer · the uppercut freezes, 040COMBAT slams in on the next beat, DEURNE on the one after |

Real sound from the floor cuts through the music: the kick, the throw, her punches, the
pads, the double-leg, the punch the music stops on, the photos slamming together, the
knee, the laugh, the last three hits.

**One look: black & white with gold accents.** A hard panchromatic black & white, with
neutral graphite shadows and off-white paper highlights. Only what is really gold or yellow
in the frame keeps colour, printed in the logo's gold (#C09B53): the gold gloves, the yellow
shin guards, the stars on the wall sign. At the drop and on the photos the highlights gild
and the light glows warm. Each shot is first matched for black point, white point and
exposure, so phone footage from different rooms prints the same. On top: fine grain, gate
weave, gold flashes on the hits and single negative frames.

**Identity.** The logo is never pasted on: it appears as the real sign on the gym wall.
It flashes in the build, sits over the exchange, and stands in black and white behind the
type at the end. The type is Barlow Condensed ExtraBold (`brand/fonts`, OFL).

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
