# 040COMBAT — Deurne showcase

A 38-second 9:16 invitation to 040COMBAT, the MMA, kickboxing and grappling gym in
Deurne. It is cut from the gym's own training footage and competition photos only, and ends on
the gym's real logo.

## The edit (`edits/showcase.py`)

Not a fight trailer: an invitation. The skill and the range come first, then the people:
laughing, coaching, promotions, the hug. It ends on the two teams and the badge. The mission
is "I want to be a part of this": competence, confidence, community.

It is built like the District98 Homecoming film:
- the brand mark opens as a door into the footage;
- the first chapter plays in a 4:5 window that closes to a slit in the silence and bursts
  open on the drop;
- one disciplined grade runs through it all;
- it ends on the logo.

Cut to Future & Metro Boomin's *Everyday Hustle* (`music/`, not in git). The track runs at
119.67 BPM, so a bar is 2.006 s. Every splice is on a bar line.

| time | music | picture |
|---|---|---|
| 0–3 | out of silence (bars 6½–7) | out of black the gold octagon draws itself and Milosz and his partner come up inside it · on the bar line the camera flies through it into the 4:5 window · three punch-ins |
| 3–9 | the vocals (bars 8–10) | the kick lands on the first vocal downbeat in slow motion · the throw · her combination · pads into a double-leg · an exchange under the wall sign |
| 9–10 | silence | Bilal's hook freezes, the room falls away, the window closes to a slit |
| 10–20 | the drop (bars 32–36) | the slit bursts open · Milosz's hand raised · Imre · the two together · the d'arce roll · the arm-triangle · 50-50 · the gi · the armbar |
| 20–26 | the breakdown (bars 52–54) | burn to white → Milosz laughing after the round · a coach explaining on the mat · a new belt under the wall sign |
| 26–28 | bar 59, an 808 and air | the purple-belt hug in slow motion, the room cheering |
| 28–30 | the beat returns (bar 60) | the kids sparring in gold gloves · a combination under the wall sign · the uppercut |
| 30–38 | the song's drumless outro (bar 106) | the uppercut holds · burn to white → the no-gi team · the camera pulls back and the octagon closes around them · the gi team inside it · the octagon settles and the badge builds: banner, COMBAT, 040, the stars · DEURNE |

Real sound from the floor sits under the music: the kick, the throw landing, her punches,
the pads and the double-leg, Bilal's hook, the d'arce roll, the armbar, the laugh, the room
cheering at the promotion, the kids, the last combination, and the uppercut ringing out in a
hall. Two clean sub impacts sit under the kick and the banner.

**Look (`edits/look040.py`).** One grade, built like a colourist's node tree:
1. **Match.** White balance, black and white points and exposure are matched per shot, so
   footage from different phones and rooms starts equal.
2. **Separation.** A people matte (Robust Video Matting, `afterfilm/matte.py`) sets a broad
   pool of light where the people are. The room sits about a stop lower and less saturated.
   It is never an outline: a matte-shaped lift reads as a cut-out with a glow.
3. **Colour.** Skin and gold keep their saturation; blues, greens and LED casts are pulled
   down. Then a print-film curve and a gentle S.
4. **Balance.** Split toning: cool graphite in the shadows, warm ivory in the highlights.
   Highlights stop just short of pure white.
5. **Diffusion.** A Pro-Mist-style bloom on the highlights. Halation, fine grain and a soft
   vignette come from afterfilm's film pass.

The matte weights are GPL-3.0 and live in `work/models/`, not in git.

**The logo (`edits/logo_art.py`).** The gym's own PDF (`work/logo2/`, not in git) is split
into its vector layers by cutting its content stream: rim, body, gold ring and frame, banner,
COMBAT, 040, stars. Each is rendered on its own, crisp at any size. In the logo the ring's
sides are hidden behind the banner. `full_ring()` completes them from the ring's own edges
for the door.

**Transitions.** Chapters change through a short burn to warm white. Everything else is a
cut, plus two whips.

**Ads.** The song is not cleared for paid use. Every render also writes `*_no-music.mp4`:
the same picture with only the floor sounds, hall tails and impacts, ready for a licensed
track at around 120 BPM.

`edits/logo040.py` (a traced rebuild of the badge) is kept for reference; v7 uses the real
logo art instead.

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
