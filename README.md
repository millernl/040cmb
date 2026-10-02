# 040COMBAT — Deurne showcase

A 34-second 9:16 invitation to 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Not a fight trailer: an invitation. The skill and the range come first, then the people:
coaching, laughing, promotions, the hug. It ends on the two team photos under the wall sign.
The mission is "I want to be a part of this": competence, confidence, community.

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s. Every splice is on a bar line.

| time | music | picture |
|---|---|---|
| 0–3 | out of silence (bars 6½–7) | out of black, the light comes on behind Milosz and his partner · it lands on the bar line · three punch-ins on the eighth notes |
| 3–9 | the vocals (bars 8–10) | the kick lands on the first vocal downbeat in slow motion · the throw · her combination · pads into a double-leg · an exchange under the wall sign |
| 9–10 | silence | Bilal's hook freezes, the room falls away around them |
| 10–20 | the drop (bars 32–36) | burn to white → Milosz's hand raised · Imre · the two together · the d'arce roll · the arm-triangle · Milosz trading · 50-50 · the gi · the armbar |
| 20–26 | the breakdown (bars 52–54) | burn to white → a coach on the mat with a pair · a kid on the pads with his coach · Milosz laughing after a round · a new belt and a hug under the wall sign · a congratulation |
| 26–28 | bar 59, an 808 and air | the purple-belt hug in slow motion, the room cheering |
| 28–30 | the beat returns (bar 60) | the gi roll · a combination under the wall sign · the uppercut |
| 30–34 | the song's drumless outro (bar 106) | the uppercut holds · burn to white → the two teams, no-gi above and gi below, both under the wall sign · fade to black |

Real sound from the floor sits under the music: the kick, the throw landing, her punches,
the pads and the double-leg, Bilal's hook, the d'arce roll, the armbar, the laugh, the room
cheering at the promotion, the last combination, and the uppercut ringing out in a hall.

**Look: one warm, low-key colour grade.** It is calibrated on the Arma BJJ reference
(`footage/new/arma_bjj.mp4`), measured tone by tone:
- shadows are brown-black;
- an amber cast runs through the midtones;
- highlights are cream;
- saturation falls off toward the highlights.

Each shot is first white-balanced and matched for black point, white point and exposure, so
footage from different phones and rooms prints the same. Skin, gold and amber keep their
colour; cool colours are muted, not erased.

A people matte (Robust Video Matting, `afterfilm/matte.py`) lets the room sit lower than the
athletes. A warm halation, bloom, fine grain and a soft vignette finish the image. The weights
are GPL-3.0 and live in `work/models/`, not in git.

**Transitions.** As in the reference, chapters change through a short burn to warm white;
everything else is a cut, plus two whips. There are no strobes, negative frames or graphics.

**Ads.** The song is not cleared for paid use. Every render also writes `*_no-music.mp4`:
the same picture with only the floor sounds and hall tails, ready for a licensed track at
around 120 BPM.

`edits/logo040.py` rebuilds the badge as vector paths. The v5 outro used it; this cut ends on
the real sign on the wall instead.

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
