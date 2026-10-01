# 040COMBAT — Deurne showcase

A 26-second 9:16 showcase for 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s; the grid is fitted on its kick pattern. There is one bar
of build, then the intro under the action, a held breath on a freeze, the drop for the
achievements and a burst, the breakdown for the people, and the beat back for the last
hits and the end card. Every splice is on a bar line.

| time | music | picture |
|---|---|---|
| 0–2 | build (bar 7) and the kick's own impact, reversed through a hall | out of black the gym's lights strike on · feet, gold gloves, a face, the wall logo strobe in · three staccato punch-ins on Milosz |
| 2–8 | intro (bars 8–11) | the kick lands on the downbeat in slow motion, trailing light · a whip into a body-lock throw · her combination · pads with the coach into a double-leg · a jump-cut exchange under the wall logo |
| 8–9 | held breath: the intro's last instant rings out in a hall | his punch freezes on a negative frame and the picture closes to a slit |
| 9–15 | the drop (bars 32–35) | the drop bursts the slit open in gold · Milosz's hand raised, a whip up to Imre, the two slam together above and below · five hits: gold gloves, her knee, a kid on the pads in light trails, the full room trading, a roll to the back |
| 15–21 | the breakdown (bars 52–53, 59) | his laugh after the round · the coach on the mat · the team photo face by face · the whole room lit out of silhouette on the 808 |
| 21–26 | the beat returns (bars 60–62) | a switch kick, a scramble, a takedown under the logo · the uppercut freezes, 040COMBAT slams in on the next beat, DEURNE on the one after |

Real sound from the floor cuts through the music: the kick, the throw, her punches, the
pads, the double-leg, the punch the music stops on, the photos slamming together, the
knee, the laugh, the last three hits.

**Two looks**, like the District98 film's Iron and Ember:
- **IRON** is a hard panchromatic black and white printed on off-white paper. It covers the build, the intro, the people and the end card.
- **GOLD** prints the same tones through the logo's gold (#C09B53): near-black, through bronze, to warm paper. It carries the drop, the photos and the hits.

Each shot is first matched for black point, white point and exposure, so phone footage
from different rooms prints the same. On top sit fine grain, halation, gate weave, gold
exposure flashes on the hits and single negative frames.

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
