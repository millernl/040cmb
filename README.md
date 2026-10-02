# 040COMBAT — Deurne showcase

A 30-second 9:16 showcase for 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only.

## The edit (`edits/showcase.py`)

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s; the grid is fitted on its kick pattern. Every splice is on
a bar line.

| time | music | picture |
|---|---|---|
| 0–2 | build (bar 7) | the hook: nine impacts on the eighth notes, each on its own sound, every other one opening on a white-silhouette frame · three punch-ins on Milosz |
| 2–8 | intro (bars 8–10) | the kick lands on the first vocal downbeat in slow motion · a whip into a body-lock throw · her combination · pads into a double-leg · an exchange under the wall logo |
| 8–9 | silence; the intro rings out in a hall | Bilal's hook freezes, the gym drains to black around the two of them, the last two frames print them white |
| 9–19 | the drop (bars 32–36) | Milosz's hand raised · a whip up to Imre · the two slam together · then strikes and grappling in turn: the packed class, the d'arce roll slowed at the top, her knee, the arm-triangle squeeze, Milosz trading, a sweep from the bottom, the armbar given room |
| 19–23 | the breakdown (bars 52–53) | a hard exchange that ends in his laugh · a kid on the pads with his coach |
| 23–25 | bar 59, an 808 and air | the whole room lit out of silhouette |
| 25–30 | the beat returns (bars 60–61) | three pad combinations under the wall sign, each punched in closer · the uppercut freezes, the room goes dark, 040COMBAT slams in behind the striker's legs, DEURNE on the next beat |

Real sound from the floor cuts through the music: the hook's impacts, the kick, the throw,
her punches, the pads, the double-leg, Bilal's hook, the photos slamming together, the
knee, the exchange and the laugh, the last combinations and the uppercut.

**Look.** Black and white throughout: a hard panchromatic print with graphite shadows and
off-white paper highlights. Each shot is first matched for black point, white point and
exposure, so phone footage from different rooms prints the same. Gold appears only as
light: the gilded highlights of the competition photos and exposure flashes on the big hits.
It is also the colour of DEURNE in the type.

**Mattes.** `afterfilm/matte.py` runs Robust Video Matting (Lin et al.; GPL-3.0 weights,
downloaded to `work/models/`, not in git) on the shots that need people separated from the
gym. These drive the white-silhouette frames, the gym draining away in the silence, and the
type standing behind the striker at the end. The mattes are cached in `work/mattes/`.

**Ads.** The song is not cleared for paid use. Every render also writes `*_no-music.mp4`:
the same picture with only the floor sounds and hall tails, ready for a licensed track at
around 120 BPM.

**Identity.** The logo is never pasted on: it appears as the real sign on the gym wall. The
type is Barlow Condensed ExtraBold (`brand/fonts`, OFL).

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
