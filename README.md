# 040COMBAT — Deurne showcase

A 33-second 9:16 showcase for 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s; the grid is fitted on its kick pattern. Every splice is on
a bar line.

| time | music | picture |
|---|---|---|
| 0–3 | out of silence (bars 6½–7) | out of black, the light strikes behind Milosz and his partner: flickering silhouettes, rimmed where the light wraps them · on the bar line the light lands · three punch-ins on the eighth notes |
| 3–9 | the vocals (bars 8–10) | the kick lands on the first vocal downbeat in slow motion, trailing light · a whip into a throw, slowed at the top and landing on the beat · her combination · pads into a double-leg · an exchange under the wall logo |
| 9–10 | silence; the intro rings out in a hall | Bilal's hook freezes · the gym drains to black and earlier instants of the punch appear one by one, a stroboscopic multiple exposure · the last frames print them white |
| 10–20 | the drop (bars 32–36) | Milosz's hand raised · a whip up to Imre · the two slam together · the d'arce roll, slowed at the top · the arm-triangle squeeze · Milosz trading · 50-50 · the gi · the sit-back into the armbar |
| 20–24 | the breakdown (bars 52–53) | a coach on the mat with a pair · a kid on the pads with his coach |
| 24–26 | bar 59, an 808 and air | the team photo: a band of light sweeps across the room and finds every face · the wall sign surfaces above them |
| 26–28 | the beat returns (bar 60) | a combination under the wall sign, punched in closer · the uppercut |
| 28–33 | the song's drumless outro (bar 106) | the uppercut freezes; the room drains, the two of them fade to black · over the empty gym, barely there, the badge builds itself: the gold ring drawn by a line of light, the banner snapping open on the beat with one low impact, COMBAT rising, 040, the stars on the eighth notes, a glint, DEURNE |

Real sound from the floor cuts through the music: the kick, the throw landing, her
punches, the pads and the double-leg, Bilal's hook, the photos slamming together, the
d'arce roll, the armbar sit-back, the last combination, and the uppercut ringing out in a hall.

**Look: a relit black & white.** Every shot carries a people matte (Robust Video Matting,
`afterfilm/matte.py`). The grade uses it like a lighting pass:
- the room drops by up to about 1.8 stops behind the athletes, the bright walls and windows most;
- the athletes lift a touch and take local contrast (muscle, sweat, fabric);
- the ceiling darkens.

The image then prints through a filmic curve with graphite shadows and off-white paper
highlights. Each shot is first matched for black point, white point and exposure, so phone
footage from different rooms prints the same. Gold appears only as light: the gilded
competition photos and the exposure flashes on the big hits. Its one solid use is in the badge.

**The badge** (`edits/logo040.py`) is rebuilt as vector paths from the supplied logo, which
is not in git:
- the octagons are fitted as straight-edged polygons;
- the banner is measured;
- the letterforms are traced, and the stars rebuilt as true five-point stars.

Every part animates on its own and stays crisp at any size. The ring and banner frame use a
metallic gold gradient, the rim a dark silver.

**Mattes.** The RVM weights (GPL-3.0) are downloaded to `work/models/` and are not in git;
the mattes are cached in `work/mattes/`. For the 30-person team photo, which the video model
can't separate, the matte is the team block, feathered.

**Ads.** The song is not cleared for paid use. Every render also writes `*_no-music.mp4`:
the same picture with only the floor sounds, the hall tails and the logo impact, ready for a
licensed track at around 120 BPM.

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
