"""040COMBAT, Deurne — a 24-second 9:16 showcase.

One look: a hard panchromatic black & white in which only what is really gold or yellow — the
gold gloves, the yellow shin guards, the stars on the wall sign — prints in the logo's gold.
At the drop the highlights gild and glow. Cut to Future & Metro Boomin's "Everyday Hustle"
(music/, not in git; 119.67 BPM, bar = 2.006 s).

  0.0–2.0    the hook · nine impacts cut on the eighth notes, each on its own sound: a
             takedown, ground and pound in gold gloves, her kick, a scramble, a punch, a face,
             an inversion, Milosz, the wall logo · three staccato punch-ins on Milosz
  2.0–8.0    intro · the kick lands on the downbeat in slow motion, trailing light · a whip
             into a body-lock throw · her combination · pads with the coach into a double-leg
             · a jump-cut exchange under the wall logo
  8.0–9.0    the music holds its breath; his punch freezes and the frame closes to a slit
  9.0–15.0   the drop bursts it open, gilded · Milosz's hand raised, a whip to Imre, the two
             slam together above and below · five hits: gold gloves, her knee, a kid on the pads
             in light trails, the full room trading, a roll to the back
  15.0–19.1  the breakdown · his laugh after the round, the packed class in slow motion, the
             whole room lit out of silhouette on the 808
  19.1–24.2  the beat returns · three pad combinations under the wall sign, each punched in
             closer, and the uppercut that freezes into 040COMBAT, then DEURNE, on the beats

Footage and photos are read from footage/ (the draft release 'footage'; never in git).

    python edits/showcase.py                       # renders/040COMBAT_showcase_9x16.mp4
    python edits/showcase.py --stills 0.5 9.6 --scale 0.5
    python edits/showcase.py --edl
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from afterfilm import brand, fx, gfx, media  # noqa: E402
from afterfilm.timeline import Clip, FrameBuffer, Overlay, Shot, Timeline, VideoSource  # noqa: E402

FOOT = ROOT / "footage"
MUSIC = ROOT / "music" / "Everyday_Hustle.m4a"
FPS = 30

# ── identity ─────────────────────────────────────────────────────────────────
GOLD = (192, 155, 83)                       # measured from the supplied logo
OFFWHITE = (242, 237, 228)
FONT = "BarlowCondensed-ExtraBold"

# ── the music's grid ─────────────────────────────────────────────────────────
BEAT, OFF = 0.50143, 0.047                  # fitted on the kick pattern across the track
BAR = 4 * BEAT


def SB(j):
    """Song time of bar j's downbeat."""
    return OFF + BAR * j


V_KICK = BAR                                 # 2.006: one bar of build, then the first downbeat
V_PAUSE = V_KICK + 3 * BAR                   # 8.023: the intro stops on a bar line
V_DROP = V_PAUSE + 2 * BEAT                  # 9.026: two beats of held breath
V_BREAK = V_DROP + 3 * BAR                   # 15.043: into the breakdown
V_PRE = V_BREAK + BAR                        # 17.049: the breakdown's last bar — an 808, then air
V_RETURN = V_PRE + BAR                       # 19.055: the beat returns
V_CARD = V_RETURN + BAR                      # 21.060: the end card
V_LAST = V_CARD + BAR                        # 23.066: the last downbeat
DURATION = V_LAST + 1.15

# music: (video start, song start, song end, gain dB)
MUSIC_PLAN = [(0.0, SB(7), SB(11), 3.0),
              (V_DROP, SB(32), SB(35), 0.0),
              (V_BREAK, SB(52), SB(53), 2.0),
              (V_PRE, SB(59), SB(62) + 0.9, 1.0)]


def vb(t0, k):
    return t0 + k * BEAT


def F(t):
    return int(round(t * FPS))


# ── sources ──────────────────────────────────────────────────────────────────
_SRC = {}
DENOISE = {"action.mp4": "hqdn3d=2:1.5:4:3", "wall.drill.near.the.end.mp4": "hqdn3d=1.5:1.5:3:3",
           "uitleg.mma.mp4": "hqdn3d=1.5:1.5:3:3"}


def src(name):
    if name not in _SRC:
        _SRC[name] = VideoSource(str(FOOT / name), rescue=False, extra_pre=DENOISE.get(name))
    return _SRC[name]


class StillSource:
    """A photograph as a one-frame source; framing happens in the clip (reframe)."""

    def __init__(self, name):
        self.path = str(FOOT / name)
        self.img = np.asarray(ImageOps.exif_transpose(Image.open(self.path)).convert("RGB"))

    def read(self, t0, t1, fps, size):
        w, h = size
        ih, iw = self.img.shape[:2]
        # cover the request, but never upscale past the photo's own pixels: reframe does that once
        s = min(max(w / iw, h / ih), max(1.0, 1080 / iw, 1920 / ih))
        out = cv2.resize(self.img, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA)
        return FrameBuffer(out[None], t0, fps)


def measure(frames):
    """Per-shot match before the look: black and white points and a common exposure."""
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = np.stack([cv2.resize(frames[i], (96, 170), interpolation=cv2.INTER_AREA) for i in idx]).astype(np.float32) / 255
    y = small.reshape(-1, 3) @ np.array(BW_MIX, np.float32)
    lo, mid, hi = np.percentile(y, [1.0, 50, 99.6])
    black = max(0.0, lo - 0.01) * 0.85
    scale = float(np.clip(0.97 / max(hi - black, 1e-3), 1.0, 1.35))
    mid2 = (mid - black) * scale
    expo = float(np.clip(np.log2(0.34 / max(mid2, 1e-3)) * 0.5, -0.4, 0.5))
    return dict(black=float(black), scale=scale, expo=expo)


NOMATCH = dict(black=0.0, scale=1.0, expo=0.0)


class MClip(Clip):
    """A clip with per-frame framing and grade: staccato zoom steps, a keyed track, light
    trails and exposure that change inside the shot. It measures itself when decoded."""

    def setup(self, zsteps=None, track=None, zfn=None, expo_fn=None, gold_fn=None, echo_fn=None):
        self.zsteps, self.trackt, self.zfn = zsteps, track, zfn
        self.expo_fn, self.gold_fn, self.echo_fn = expo_fn, gold_fn, echo_fn
        return self

    def prepare(self, dur, fps, size):
        fresh = self.buf is None
        super().prepare(dur, fps, size)
        if fresh and "match" not in self.grade:
            self.grade = {**self.grade, "match": measure(self.buf.frames)}

    def framing(self, lt, dur):
        p = float(np.clip(lt / max(dur, 1e-6), 0, 1))
        if getattr(self, "zfn", None):
            return self.zfn(lt)
        if getattr(self, "zsteps", None):
            st = [s for s in self.zsteps if s[0] <= lt + 1e-6] or self.zsteps[:1]
            t0, z, cx, cy, drift = (list(st[-1]) + [0.0])[:5]
            return z * (1 + drift * max(lt - t0, 0)), cx, cy
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * p
        if getattr(self, "trackt", None):
            ks = self.trackt
            ts = [k[0] for k in ks]
            return z, float(np.interp(lt, ts, [k[1] for k in ks])), float(np.interp(lt, ts, [k[2] for k in ks]))
        cx, cy = self.center_at(p)
        return z, cx, cy

    def frame(self, lt, dur, look, freeze_at=None, out_size=None, **over):
        st = self.src_time(min(lt, freeze_at) if freeze_at is not None else lt)
        img = self.buf.frame(st)
        echo = self.echo_fn(lt) if getattr(self, "echo_fn", None) else self.echo
        if echo > 0.01:
            # chronophotography: earlier instants of the move linger as light
            for k in range(1, 6):
                img = np.maximum(img, self.buf.frame(st - k * 0.05) * (echo * 0.75 ** k))
        z, cx, cy = self.framing(lt, dur)
        img = fx.reframe(img, z, cx, cy, out_size=out_size or (img.shape[1], img.shape[0]))
        g = {**self.grade, **over}
        if getattr(self, "expo_fn", None):
            g["exposure"] = g.get("exposure", 0.0) + self.expo_fn(lt)
        if getattr(self, "gold_fn", None):
            g["gold"] = self.gold_fn(lt)
        return look.grade(img, **g)


def clip(name, t_in, speed=1.0, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", echo=0.0, decode_zoom=None,
         zsteps=None, track=None, zfn=None, expo_fn=None, gold_fn=None, echo_fn=None, **grade):
    s = src(name)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    sw, sh = sorted((s.info["width"], s.info["height"]))      # portrait after rotation
    dw = int(min(sw, max(1080, round(1080 * zmax))) // 2 * 2)
    dh = int(min(sh, max(1920, round(1920 * zmax))) // 2 * 2)
    c = MClip(s, t_in, speed=speed, zoom=zoom, center=center, note=note, echo=echo, grade=grade, decode=(dw, dh))
    return c.setup(zsteps, track, zfn, expo_fn, gold_fn, echo_fn)


def photo(name, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", decode_zoom=None, zfn=None, zsteps=None, expo_fn=None, **grade):
    grade.setdefault("match", NOMATCH)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    c = MClip(StillSource(name), 0.0, speed=0.0, zoom=zoom, center=center, note=note, grade=grade,
              decode=(int(1080 * zmax * 1.05) // 2 * 2, int(1920 * zmax * 1.05) // 2 * 2))
    return c.setup(zsteps=zsteps, zfn=zfn, expo_fn=expo_fn)


# ── the look ─────────────────────────────────────────────────────────────────
BW_MIX = (0.50, 0.40, 0.10)                  # orange-filter panchromatic: skin glows, mats sink
PAPER = np.array([1.0, 0.982, 0.948], np.float32)          # whites: off-white paper, never blue
GRAPHITE = np.array([0.975, 0.985, 1.0], np.float32)       # shadows: neutral graphite, never brown
METAL = np.array([1.16, 0.90, 0.50], np.float32)
METAL = METAL / float(METAL @ fx.LUMA)                     # the logo's gold, at unit luminance
GLOW = np.array([1.0, 0.76, 0.40], np.float32)


class CombatLook(fx.Look):
    """Black & white with gold accents. Each shot is matched first (black point, white point,
    exposure), so phone footage from different rooms prints the same. Only real gold and
    yellow (hue 36–64°, saturated) keeps colour, printed in the logo's gold; skin never does.
    `gold` (0–1) gilds the highlights and lets the light glow — the drop and the photos."""

    def __init__(self, w, h):
        super().__init__(w, h, accent=GOLD)

    def grade(self, img, match=None, gold=0.0, exposure=0.0, contrast=0.46, accents=1.0, **_):
        x = np.clip(img, 0, 1).astype(np.float32)
        e = exposure
        if match:
            x = np.clip((x - match["black"]) * match["scale"], 0, 1)
            e += match["expo"]
        v = x @ np.array(BW_MIX, np.float32)
        if e:
            v = np.clip(v * (2.0 ** e), 0, 1)
        v = fx.filmic(v)
        v = np.clip(v + contrast * (v * v * (3 - 2 * v) - v), 0, 1)
        vv = v[..., None]
        out = vv * (GRAPHITE + (PAPER - GRAPHITE) * vv)
        warm = fx.smoothstep(0.55, 1.0, vv) * (0.10 + 0.30 * gold)
        out = out * (1 - warm) + np.clip(vv * METAL, 0, 1) * warm
        if accents:
            hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            m = (fx.smoothstep(34, 40, hsv[..., 0]) * (1 - fx.smoothstep(62, 70, hsv[..., 0]))
                 * fx.smoothstep(0.36, 0.55, hsv[..., 1]) * fx.smoothstep(0.2, 0.36, hsv[..., 2]))
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None] * accents
            out = out * (1 - m) + np.clip((0.22 + 0.95 * vv) * METAL * 0.95, 0, 1) * m
        amt = 0.12 + 0.3 * gold
        hot = np.clip((v - 0.62) / 0.38, 0, 1)
        h, w = v.shape
        small = cv2.resize(hot, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
        k = min(w, h) / 1080 / 4
        g = cv2.GaussianBlur(small, (0, 0), 20 * k + 1) * 0.7 + cv2.GaussianBlur(small, (0, 0), 64 * k + 1) * 0.6
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        out = 1 - (1 - out) * (1 - np.clip(g * amt, 0, 1) * GLOW)
        return (0.006 + out * 0.994).astype(np.float32)


# ── transitions of our own ───────────────────────────────────────────────────
def t_smear_v(a, b, p, ctx, direction=1):
    """fx.t_smear, vertically: a whip up (or down) caught in four frames."""
    h, w = b.shape[:2]
    if p < 0.5:
        img, k, sgn = a, (p / 0.5) ** 1.2, 1
    else:
        img, k, sgn = b, ((1 - p) / 0.5) ** 1.5, -1
    if k < 0.02:
        return img
    off = -direction * sgn * k * 0.08 * h
    out = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, off]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    kk = int(k * 0.22 * h)
    if kk > 1:
        out = cv2.blur(out, (1, kk))
    return np.clip(out * (1 + 0.12 * k), 0, 1)


fx.TRANSITIONS["smear_v"] = t_smear_v
fx.TRANSITIONS["smear_up"] = lambda a, b, p, c: t_smear_v(a, b, p, c, direction=-1)
WHIP = 0.14                                  # a whip's length; the shot starts half of it early


# ── per-frame events ─────────────────────────────────────────────────────────
GOLD_TINT = (1.0, 0.80, 0.46)


def strike(levels):
    """Fluorescent tubes catching: exposure (stops) per local frame, then steady."""
    def fn(lt):
        n = int(lt * FPS)
        return levels[n] if n < len(levels) else 0.0
    return fn


def ease_snap(lt, amt, t=0.35, drift=0.03):
    """A snap zoom that settles, then keeps pushing slowly."""
    return 1.0 + amt * (1 - fx.expo_out(lt / t)) + drift * lt


# ── the edit ─────────────────────────────────────────────────────────────────
E8 = BEAT / 2                                 # an eighth note
# (start, length, clip, source time of the impact, zoom, centre, note) — every cut lands on a hit
HOOK = [(0 * E8, E8, "wall.drill.near.the.end.mp4", 61.36, 1.10, (0.5, 0.50), "a takedown under the logo"),
        (1 * E8, E8, "action.mp4", 6.07, 1.25, (0.45, 0.45), "ground and pound, gold gloves"),
        (2 * E8, 0.75 * E8, "alternative.shot.mp4", 8.30, 1.15, (0.5, 0.5), "her body kick"),
        (2.75 * E8, 0.75 * E8, "Milosz.Bilal.mp4", 37.00, 1.0, (0.5, 0.5), "a cross on the pads, the room watching"),
        (3.5 * E8, 0.75 * E8, "kickboxing.drills.mp4", 9.55, 1.12, (0.5, 0.45), "a jab into the pads"),
        (4.25 * E8, 0.5 * E8, "Dilbrien.x.Milosz.scramble.mp4", 40.24, 2.8, (0.56, 0.16), "a face"),
        (4.75 * E8, 0.5 * E8, "Michael.x.Dilbrien.scramble.mp4", 56.91, 1.3, (0.5, 0.5), "an inversion"),
        (5.25 * E8, 0.5 * E8, "KB3.mp4", 8.62, 1.3, (0.45, 0.45), "gold gloves"),
        (5.75 * E8, 0.5 * E8, "Milosz.Bilal.mp4", 0.46, 3.2, (0.45, 0.075), "the wall sign")]


def build(w=1080, h=1920):
    look = CombatLook(w, h)
    tl = Timeline(w, h, FPS, DURATION, look)
    S = tl.add
    neg, flashes = set(), []                 # single negative frames, gold exposure flashes

    # 0 · the hook: nine impacts on the eighth notes, accelerating, each on its own sound
    E = BEAT / 2
    for k, (t0, d, name, s_imp, z, c, note) in enumerate(HOOK):
        S(Shot(t0, d, clip(name, s_imp - 0.06, zoom=(z, z * 1.04), center=c, note=note), punch=0.05 if k < 5 else 0.0))
    neg.update({F(HOOK[4][0]) + 1})
    # Milosz: three staccato punch-ins, then the kick lands wide on the downbeat in slow motion
    t = 1.60
    imp = V_KICK - t
    S(Shot(t, vb(V_KICK, 2) - t, clip(
        "kickboxing.drills.mp4", 21.39 - imp,
        speed=[(0, 1.0), (imp + 0.01, 1.0), (imp + 0.06, 0.28), (imp + 0.62, 0.28), (imp + 0.82, 1.0), (9, 1.0)],
        zsteps=[(0, 1.0, 0.5, 0.5), (0.13, 1.2, 0.70, 0.42), (0.27, 1.42, 0.72, 0.40), (imp, 1.04, 0.5, 0.5, 0.05)],
        echo_fn=lambda lt: 0.75 * fx.smoothstep(imp, imp + 0.08, lt) * (1 - fx.smoothstep(imp + 0.6, imp + 0.8, lt)),
        note="Milosz's high kick lands on the downbeat"), shake=0.0))
    neg.add(F(V_KICK) + 1)
    flashes.append((V_KICK, 0.75, 0.2))

    # 2 · intro: kick → throw → her hands → pads into a double-leg → the exchange
    t = vb(V_KICK, 2)                                                    # 3.008
    S(Shot(t - WHIP / 2, BEAT * 2 + WHIP / 2, clip(
        "Dilbrien.x.Milosz.scramble.mp4", 19.95 - WHIP / 2, zoom=(1.36, 1.36),
        track=[(0, 0.50, 0.31), (0.32, 0.56, 0.31), (0.70, 0.48, 0.31), (1.05, 0.60, 0.34)],
        note="a body-lock lift and throw"), trans=("smear", WHIP)))
    t = vb(V_KICK, 4)                                                    # 4.011
    S(Shot(t, 2 * BEAT, clip("alternative.shot.mp4", 13.85, zoom=(1.06, 1.12), center=(0.5, 0.48),
                              note="her combination in the full room"), trans=("strobe", 0.13)))
    t = vb(V_KICK, 6)                                                    # 5.014
    S(Shot(t, 4 * BEAT, clip("action.mp4", 3.45, speed=[(0, 1.0), (1.35, 1.0), (1.40, 0.6), (1.85, 0.6), (1.90, 1.0), (9, 1.0)],
                              zoom=(1.0, 1.08), center=(0.5, 0.52), note="pads with the coach, into a double-leg")))
    neg.add(F(vb(V_KICK, 6) + 1.96))
    # a jump-cut exchange under the wall logo, the second half punched in; his punch freezes
    t = vb(V_KICK, 10)                                                   # 7.020
    S(Shot(t, BEAT, clip("Milosz.Bilal.KB2.mp4", 8.55, zoom=(1.0, 1.03), note="Milosz and Bilal trade under the wall logo")))
    t = vb(V_KICK, 11)                                                   # 7.522
    S(Shot(t, V_DROP - t, clip("Milosz.Bilal.KB2.mp4", 9.50, zoom=(1.16, 1.18), center=(0.5, 0.5),
                                note="…punched in → his punch freezes"), freeze_at=V_PAUSE - t))
    neg.add(F(V_PAUSE) + 2)

    # 3 · the drop, in gold: the achievements
    t = V_DROP
    S(Shot(t, 2 * BEAT, photo("Milosz.Celebration.jpg", center=(0.40, 0.42), gold=1.0, exposure=-0.15, decode_zoom=1.2,
                              zfn=lambda lt: (ease_snap(lt, 0.16), 0.40, 0.42), note="Milosz, hand raised"), shake=0.7))
    flashes.append((V_DROP, 0.9, 0.3))
    t = vb(V_DROP, 2)
    S(Shot(t - WHIP / 2, 2 * BEAT + WHIP / 2, photo("Imre.Celebration.jpg", gold=1.0, exposure=-0.1, decode_zoom=1.2,
                                                    zfn=lambda lt: (ease_snap(lt, 0.12), 0.5, 0.36), note="Imre, arms up"),
           trans=("smear_up", WHIP)))
    t = vb(V_DROP, 4)                                                    # 11.032
    dip = Diptych(w, h)
    S(Shot(t, 2 * BEAT, render=dip.render, clips=dip.clips))
    neg.add(F(t + Diptych.SLAM) + 1)
    flashes.append((t + Diptych.SLAM, 0.6, 0.18))
    # five hits, gold and iron in turn
    t = vb(V_DROP, 6)                                                    # 12.035
    S(Shot(t - WHIP / 2, BEAT + WHIP / 2, clip("KB3.mp4", 8.45 - WHIP / 2, zoom=(1.05, 1.1), gold=1.0,
                                               note="a round in the full room, gold gloves"), trans=("smear", WHIP)))
    t = vb(V_DROP, 7)
    S(Shot(t, BEAT, clip("good.kickboxing.clip.mp4", 4.06, zoom=(1.36, 1.42), center=(0.60, 0.38), note="her knee"),
           trans=("strobe", 0.1)))
    t = vb(V_DROP, 8)
    S(Shot(t, 2 * BEAT, clip("extra.mp4", 9.15, zoom=(1.14, 1.2), gold=1.0, echo_fn=lambda lt: 0.5 * fx.smoothstep(0.05, 0.25, lt),
                              note="a kid on the pads with his coach, the room full, in light trails")))
    t = vb(V_DROP, 10)
    S(Shot(t, BEAT, clip("Milosz.Bilal.mp4", 36.67, zoom=(1.08, 1.12), note="trading with the room watching"), punch=0.04))
    t = vb(V_DROP, 11)
    S(Shot(t, V_BREAK - t, clip("RNC.mp4", 2.55, zoom=(1.08, 1.14), center=(0.5, 0.45), gold=1.0,
                                 note="a roll to the back"), punch=0.04))

    # 4 · the breakdown: the people
    t = V_BREAK
    S(Shot(t, 2 * BEAT, clip("Milosz.Bilal.KB2.mp4", 19.95, speed=0.8, zoom=(1.45, 1.5), center=(0.40, 0.36), contrast=0.36,
                              note="his laugh after the round")))
    t = vb(V_BREAK, 2)
    S(Shot(t, 2 * BEAT, clip("boxing.instructies.mp4", 28.45, speed=0.5, zoom=(1.08, 1.0), center=(0.5, 0.5), contrast=0.36,
                              note="the packed class, 120 fps slowed to half")))
    # the whole room, lit out of silhouette on the 808
    S(Shot(V_PRE, V_RETURN - V_PRE, photo("Of.voor.communityy.jpg", zsteps=[(0, 1.05, 0.5, 0.46, 0.035)], decode_zoom=1.12,
                                          contrast=0.36, expo_fn=strike([-7, -7, -2.0, -7, -0.6, -3.5, -0.3, -1.2, 0, 0, -0.5]),
                                          note="the whole room under the logo")))

    # 5 · the beat returns: three pad combinations under the wall sign, each punched in closer,
    #     their hits on the eighth notes; then the uppercut
    for k, (s_hit, z, note) in enumerate([(12.16, 1.0, "a combination on the pads"), (15.04, 1.16, "…closer"),
                                          (18.07, 1.34, "…closer")]):
        t = vb(V_RETURN, k)
        S(Shot(t, BEAT, clip("wall.drill.near.the.end.mp4", s_hit - 0.03, zoom=(z, z * 1.02), center=(0.5, 0.46), gold=0.4,
                              note=note), punch=0.03))
    t = vb(V_RETURN, 3)
    card = EndCard(w, h, clip("wall.drill.near.the.end.mp4", 8.98 - (V_CARD - t), note="the uppercut → the end card"))
    S(Shot(t, DURATION - t, render=card.picture, clips=[card.clip]))
    tl.overlays.append(Overlay(V_CARD, DURATION, card.draw, "post"))
    neg.update({F(V_CARD) + 1, F(card.T1)})
    flashes.append((V_CARD, 0.8, 0.25))

    # global per-frame events: gold flashes, single negative frames
    def events(t, img, tl):
        for t0, k, d in flashes:
            if t0 <= t < t0 + d * 2.5:
                img = fx.flash(img, k * max(0.0, 1 - (t - t0) / d), GOLD_TINT)
        if F(t) in neg:
            img = (1.0 - img @ fx.LUMA)[..., None] * PAPER       # a single frame printed in negative
        return img
    tl.overlays.insert(0, Overlay(0.0, DURATION, events, "pre"))

    # the pause: the frozen punch closes to a slit; the drop bursts it open
    def bars(t):
        a0 = V_PAUSE + 0.14
        if a0 <= t < V_DROP:
            return 1.29 * fx.ease_in((t - a0) / (V_DROP - a0)) ** 0.8
        if V_DROP <= t < V_DROP + 0.16:
            return 1.29 * (1 - fx.expo_out((t - V_DROP) / 0.16))
        return 0.0
    tl.bars = bars
    tl.bar_ratio = 2.39
    tl.cinema_at = lambda t: {"mono": 1.0, "streaks": 0.0, "halation": 0.3, "bloom": 0.22, "weave": 0.35}
    tl.grain_at = lambda t: 0.75
    tl.vignette_at = lambda t: 0.6
    tl.audio_plan = sounds()
    return tl


class Diptych:
    """The two achievements slam together, Milosz from above, Imre from below."""
    SLAM = 0.16

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.top = MClip(StillSource("Milosz.Celebration.jpg"), 0.0, speed=0.0, center=(0.42, 0.33),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.15, "gold": 1.0}).setup()
        self.bot = MClip(StillSource("Imre.Celebration.jpg"), 0.0, speed=0.0, center=(0.5, 0.28),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.1, "gold": 1.0}).setup()
        self.top.zoom = self.bot.zoom = (1.0, 1.04)
        self.clips = [self.top, self.bot]

    def render(self, shot, lt, tl):
        w, h = self.w, self.h
        hh = h // 2
        a = self.top.frame(lt, shot.dur, tl.look, out_size=(w, hh))
        b = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, h - hh))
        e = fx.ease_in(min(lt / self.SLAM, 1.0))                 # accelerate into the slam
        dy = int(round((1 - e) * hh))
        img = np.zeros((h, w, 3), np.float32)
        if dy < hh:
            img[0:hh - dy] = a[dy:hh]                            # Milosz descends from above
            img[hh + dy:h] = b[0:h - hh - dy]                    # Imre rises from below
        k = lt - self.SLAM
        if k >= 0:
            img = fx.reframe(img, 1 + 0.025 * np.exp(-k / 0.08))                  # the impact
        return img


class EndCard:
    """The uppercut freezes in iron and drifts up toward the wall logo while the room goes
    dark; 040COMBAT slams in on the next beat, DEURNE on the one after."""

    def __init__(self, w, h, clip):
        self.w, self.h, self.clip = w, h, clip
        self.T0 = V_CARD
        self.T1, self.T2 = V_CARD + BEAT, V_CARD + 2 * BEAT
        f = brand.font(FONT, 100)
        tw, _ = gfx.measure("040COMBAT", f, 0.0)
        self.size = 100 * (0.86 * w) / tw
        self.f_big = brand.font(FONT, self.size)
        self.f_small = brand.font(FONT, self.size * 0.36)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        self.shade = 0.5 + 0.5 * fx.smoothstep(0.38, 0.66, yy)

    def picture(self, shot, lt, tl):
        w, h = self.w, self.h
        hit = self.T0 - shot.start
        if lt < hit:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h), gold=0.6)
        k = lt - hit
        img = self.clip.frame(hit, shot.dur, tl.look, out_size=(w, h), contrast=0.5)
        img = fx.reframe(img, 1.0 + 0.05 * fx.ease_out(k / 3.0), 0.5, 0.36)
        dark = fx.ease_out(fx.window(k, 0.05, 0.5))
        return img * (1 - dark * self.shade * 0.86)

    def _slam(self, c, text, font, x, y, lt, color, tracking):
        if lt < 0:
            return
        s = 1.0 + 0.14 * (1 - fx.expo_out(lt / 0.14))
        c.save()
        c.translate(x, y)
        c.scale(s, s)
        gfx.text(c, text, font, 0, 0, color, 1.0, tracking=tracking, align="center")
        c.restore()

    def draw(self, t, img, tl):
        w, h = self.w, self.h
        layer = tl.layer
        layer.clear()
        c = layer.c
        y1 = h * 0.71
        pulse = 1.0 + 0.012 * np.exp(-max(t - V_LAST, 0) / 0.1) * (t >= V_LAST)
        c.save()
        c.translate(w / 2, y1)
        c.scale(pulse, pulse)
        c.translate(-w / 2, -y1)
        self._slam(c, "040COMBAT", self.f_big, w / 2, y1, t - self.T1, OFFWHITE, 0.0)
        self._slam(c, "DEURNE", self.f_small, w / 2, y1 + self.size * 0.48, t - self.T2, GOLD, 0.16)
        c.restore()
        return layer.over(img)


# ── sound ────────────────────────────────────────────────────────────────────
def sounds():
    """Real sound from the floor: (video time, file, src t0, src t1, dB)."""
    hook = [(HOOK[0][0], "wall.drill.near.the.end.mp4", 61.36, 61.66, -5),
            (HOOK[1][0], "action.mp4", 6.03, 6.25, -4),
            (HOOK[2][0], "alternative.shot.mp4", 8.26, 8.45, -5),
            (HOOK[3][0], "Milosz.Bilal.mp4", 36.90, 37.15, -5)]
    combos = [(vb(V_RETURN, k), "wall.drill.near.the.end.mp4", s_hit - 0.03, s_hit + 0.45, -4)
              for k, s_hit in enumerate([12.16, 15.04, 18.07])]
    return hook + [
        (V_KICK - 0.03, "kickboxing.drills.mp4", 21.36, 21.75, -2),         # the kick lands
        (vb(V_KICK, 2) + (20.80 - 19.95), "Dilbrien.x.Milosz.scramble.mp4", 20.80, 21.15, -8),
        (vb(V_KICK, 4) + (14.40 - 13.85), "alternative.shot.mp4", 14.40, 14.98, -5),
        (vb(V_KICK, 6) + (3.50 - 3.45), "action.mp4", 3.50, 4.10, -5),       # the pads
        (vb(V_KICK, 6) + 1.92, "action.mp4", 5.18, 5.45, -6),               # the double-leg lands
        (V_PAUSE - 0.03, "Milosz.Bilal.KB2.mp4", 9.97, 10.45, -1),          # the punch the music stops on
        (vb(V_DROP, 4) + Diptych.SLAM - 0.02, "wall.drill.near.the.end.mp4", 8.95, 9.30, -6),
        (vb(V_DROP, 7) + (4.28 - 4.06), "good.kickboxing.clip.mp4", 4.28, 4.55, -6),
        (vb(V_DROP, 10) + (36.90 - 36.67), "Milosz.Bilal.mp4", 36.90, 37.30, -6),
        (V_BREAK + 0.06, "Milosz.Bilal.KB2.mp4", 20.0, 20.75, -5),           # the laugh
    ] + combos + [
        (V_CARD - 0.03, "wall.drill.near.the.end.mp4", 8.95, 9.40, -1),      # the uppercut
    ]


def _hall(sr, rt60=1.8, seed=4):
    from scipy import signal
    rng = np.random.default_rng(seed)
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * (10 ** (-3 * t / rt60))[:, None]
    ir = signal.sosfilt(signal.butter(2, 3500 / (sr / 2), output="sos"), ir, axis=0)
    ir = np.concatenate([np.zeros((int(0.02 * sr), 2)), ir])
    return (ir / np.sqrt((ir ** 2).sum(0, keepdims=True))).astype(np.float32)


def _verb(x, ir):
    from scipy import signal
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c]) for c in range(2)], 1).astype(np.float32)


def mix(plan, sr=48000):
    from scipy import signal
    n = int(round(DURATION * sr))
    out = np.zeros((n, 2), np.float32)
    song = media.read_audio(str(MUSIC), 0.0, None, sr=sr, channels=2)      # read once, slice in memory
    xf = int(0.012 * sr)
    for k, (v0, s0, s1, gdb) in enumerate(MUSIC_PLAN):
        lead = 0.0 if k == 0 else 0.015                                   # splice just ahead of the downbeat
        a, b = int(round((s0 - lead) * sr)), int(round(s1 * sr))
        seg = song[a:b].copy() * 10 ** (gdb / 20)
        if k == 0:
            fin = int(0.35 * sr)
            seg[:fin] *= np.linspace(0, 1, fin)[:, None] ** 2              # out of black
        else:
            seg[:xf] *= np.sin(np.linspace(0, np.pi / 2, xf))[:, None]
        seg[-xf:] *= np.cos(np.linspace(0, np.pi / 2, xf))[:, None]
        i = int(round((v0 - lead) * sr))
        seg = seg[:n - i]
        out[i:i + len(seg)] += seg
        if k == 0:
            # the pause: the intro's last instant rings out in a hall instead of stopping dead
            tail = song[int((s1 - 0.18) * sr):int(s1 * sr)]
            wet = _verb(tail, _hall(sr))
            wet *= np.linspace(1, 0, len(wet))[:, None] ** 1.5
            j = int(round(V_PAUSE * sr))
            m = min(len(wet), int(round((V_DROP - V_PAUSE - 0.02) * sr)))
            out[j:j + m] += wet[:m] * 10 ** (-9 / 20)
    # the build: the kick's own impact, reversed through the hall, swells into the downbeat
    hit = media.read_audio(str(FOOT / "kickboxing.drills.mp4"), 21.36, 0.4, sr=sr, channels=2)
    hit = signal.sosfilt(signal.butter(2, 120 / (sr / 2), "high", output="sos"), hit, axis=0)
    rev = _verb(hit[::-1].copy(), _hall(sr, 1.6))[::-1]
    rev = rev / (np.abs(rev).max() + 1e-6) * 0.35
    j = int(round(V_KICK * sr)) - len(rev)
    if j < 0:
        rev, j = rev[-j:], 0
    out[j:j + len(rev)] += rev
    # the end: the last downbeat rings, then the room
    j = int(round(V_LAST * sr))
    fade = int(0.8 * sr)
    out[j:j + fade] *= np.linspace(1, 0, min(fade, n - j))[:, None] ** 2
    out[j + fade:] = 0
    ring = song[int(SB(62) * sr):int((SB(62) + 0.4) * sr)]
    wet = _verb(ring, _hall(sr, 2.2))
    m = min(len(wet), n - j)
    out[j:j + m] += wet[:m] * 10 ** (-12 / 20)
    # real sound from the floor
    hp = signal.butter(2, 140 / (sr / 2), "high", output="sos")
    for v, name, t0, t1, db in plan:
        a = media.read_audio(str(FOOT / name), t0, t1 - t0, sr=sr, channels=2)
        a = signal.sosfilt(hp, a, axis=0).astype(np.float32)
        f = int(0.004 * sr)
        a[:f] *= np.linspace(0, 1, f)[:, None]
        a[-int(0.06 * sr):] *= np.linspace(1, 0, int(0.06 * sr))[:, None]
        a = a / (np.abs(a).max() + 1e-6) * 10 ** (db / 20) * 0.5
        i = max(0, int(round(v * sr)))
        a = a[:n - i]
        out[i:i + len(a)] += a
    k = int(0.03 * sr)
    out[-k:] *= np.linspace(1, 0, k)[:, None]
    peak = np.abs(out).max()
    if peak > 0.97:
        out *= 0.97 / peak
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="renders/040COMBAT_showcase_9x16.mp4")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--edl", action="store_true")
    a = ap.parse_args()
    W, H = int(1080 * a.scale) // 2 * 2, int(1920 * a.scale) // 2 * 2
    tl = build(W, H)
    if a.edl:
        print(tl.edl())
        sys.exit()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.stills:
        print(tl.stills(a.stills, str(out.with_suffix("")) + "_{t:05.2f}.png"))
    else:
        wav = str(out.with_suffix(".wav"))
        media.write_wav(wav, mix(tl.audio_plan))
        tl.render(str(out), wav=wav)
