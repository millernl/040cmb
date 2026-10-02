"""040COMBAT, Deurne — a 34-second 9:16 invitation.

Not a fight trailer: an invitation. The skill and the range first, then the people — coaching,
laughing, promotions, the hug — and it ends on the two team photos under the wall sign.

The look is one warm, low-key colour grade, calibrated on the Arma BJJ reference (measured tone
by tone: brown-black shadows, an amber cast through the mids, cream highlights, saturation that
falls off toward the top). Each shot is white-balanced and exposure-matched first; skin, gold and
amber keep their colour, cool colours are muted, not erased. A people matte (Robust Video
Matting, afterfilm/matte.py) lets the room sit lower than the athletes. Chapters change through
a short burn to white, as in the reference; everything else is a cut. Cut to Future & Metro
Boomin's "Everyday Hustle" (music/, not in git; 119.67 BPM, bar = 2.006 s).

  0.0–3.0    out of black, the light comes on behind Milosz and his partner · the light lands ·
             three punch-ins on the eighth notes
  3.0–9.0    the kick lands on the first vocal downbeat in slow motion · the throw · her
             combination · pads into a double-leg · an exchange under the wall sign
  9.0–10.0   silence · Bilal's hook freezes, the room falls away around them
  10.0–20.1  burn to white → the drop · Milosz's hand raised, Imre, the two together · the d'arce
             roll · the arm-triangle · Milosz trading · 50-50 · the gi · the armbar
  20.1–26.1  burn to white → the people · a coach on the mat · a kid on the pads with his coach ·
             Milosz laughing after a round · a new belt and a hug · a congratulation
  26.1–28.1  the 808 · the purple-belt hug, in slow motion, the room cheering
  28.1–30.1  the beat returns · the gi roll · a combination under the wall sign · the uppercut
  30.1–34.5  the uppercut holds · burn to white → the two teams, no-gi above, gi below, both under
             the wall sign · the song's drumless outro

Footage and photos are read from footage/ (the draft release 'footage'; never in git). The
newer clips live in footage/new/.

    python edits/showcase.py                       # renders/040COMBAT_showcase_9x16.mp4 (+ _no-music)
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
from afterfilm import fx, matte, media  # noqa: E402
from afterfilm.timeline import Clip, FrameBuffer, Overlay, Shot, Timeline, VideoSource  # noqa: E402

FOOT = ROOT / "footage"
MUSIC = ROOT / "music" / "Everyday_Hustle.m4a"
FPS = 30

# ── the music's grid ─────────────────────────────────────────────────────────
BEAT, OFF = 0.50143, 0.047                  # fitted on the kick pattern across the track
BAR = 4 * BEAT


def SB(j):
    """Song time of bar j's downbeat."""
    return OFF + BAR * j


def vb(t0, k):
    return t0 + k * BEAT


V_LAND = BAR                                 # 2.006: the light lands
V_KICK = 1.5 * BAR                           # 3.009: the kick, on the first vocal downbeat (song bar 8)
V_PAUSE = V_KICK + 3 * BAR                   # 9.026: the intro stops on a bar line
V_DROP = V_PAUSE + 2 * BEAT                  # 10.029
V_BREAK = V_DROP + 5 * BAR                   # 20.057: the people
V_808 = V_BREAK + 3 * BAR                    # 26.074
V_RETURN = V_808 + BAR                       # 28.080
V_CARD = V_RETURN + BAR                      # 30.086: the uppercut
V_TEAM = V_CARD + 2 * BEAT                   # 31.089: the teams
DURATION = V_TEAM + 3.4

# music: (video start, song start, song end, gain dB, fade in s, fade out s)
MUSIC_PLAN = [(0.0, SB(6.5), SB(11), 3.0, 0.6, 0.012),
              (V_DROP, SB(32), SB(37), 0.0, 0.012, 0.012),
              (V_BREAK, SB(52), SB(55), 2.0, 0.012, 0.012),
              (V_808, SB(59), SB(61) + 0.35, 1.0, 0.012, 0.35),
              (V_CARD, SB(106), SB(106) + DURATION - V_CARD, 0.0, 0.04, 1.6)]

# ── sources ──────────────────────────────────────────────────────────────────
_SRC = {}
DENOISE = {"action.mp4": "hqdn3d=2:1.5:4:3", "wall.drill.near.the.end.mp4": "hqdn3d=1.5:1.5:3:3",
           "uitleg.mma.mp4": "hqdn3d=2:1.5:4:3", "armbar.mp4": "hqdn3d=1.5:1.5:3:3"}
SOFT = "hqdn3d=2.5:2:5:4"                     # the WhatsApp-quality clips


def src(name):
    if name not in _SRC:
        _SRC[name] = VideoSource(str(FOOT / name), rescue=False,
                                 extra_pre=DENOISE.get(name, SOFT if name.startswith("new/") else None))
    return _SRC[name]


class StillSource:
    """A photograph as a one-frame source; framing happens in the clip (reframe)."""

    def __init__(self, name):
        self.path = str(FOOT / name)
        self.img = np.asarray(ImageOps.exif_transpose(Image.open(self.path)).convert("RGB"))

    def read(self, t0, t1, fps, size):
        w, h = size
        ih, iw = self.img.shape[:2]
        s = min(max(w / iw, h / ih), max(1.0, 1080 / iw, 1920 / ih))
        out = cv2.resize(self.img, (int(round(iw * s)), int(round(ih * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
        return FrameBuffer(out[None], t0, fps)


LUMA = fx.LUMA


def measure(frames):
    """Per-shot match before the look: white balance on near-neutral midtones (walls, mats,
    gis), black and white points, and a common exposure."""
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = np.stack([cv2.resize(frames[i], (96, 170), interpolation=cv2.INTER_AREA) for i in idx]).astype(np.float32) / 255
    px = small.reshape(-1, 3)
    y = px @ LUMA
    ch = px.max(1) - px.min(1)
    m = (y > 0.15) & (y < 0.85) & (ch < 0.12)
    sel = px[m] if m.mean() > 0.03 else px[(y > 0.1) & (y < 0.9)]
    mean = sel.mean(0)
    gain = np.clip((mean @ LUMA / mean) ** 0.8, 0.8, 1.25).astype(np.float32)
    yg = (px * gain) @ LUMA
    lo, mid, hi = np.percentile(yg, [1.0, 50, 99.6])
    black = max(0.0, lo - 0.01) * 0.85
    scale = float(np.clip(0.97 / max(hi - black, 1e-3), 1.0, 1.35))
    expo = float(np.clip(np.log2(0.32 / max((mid - black) * scale, 1e-3)) * 0.5, -0.4, 0.5))
    return dict(gain=gain, black=float(black), scale=scale, expo=expo)


NOMATCH = dict(gain=np.ones(3, np.float32), black=0.0, scale=1.0, expo=0.0)


class MClip(Clip):
    """A clip with per-frame framing (staccato zoom steps, keyed tracks), light trails and the
    people matte, which the grade uses to sit the room lower. It measures itself when decoded."""

    def setup(self, zsteps=None, track=None, zfn=None, echo_fn=None, use_matte=True):
        self.zsteps, self.trackt, self.zfn, self.echo_fn = zsteps, track, zfn, echo_fn
        self.use_matte = use_matte
        self.abuf, self.a0 = None, 0.0
        return self

    def prepare(self, dur, fps, size):
        fresh = self.buf is None
        super().prepare(dur, fps, size)
        if fresh and "match" not in self.grade:
            self.grade = {**self.grade, "match": measure(self.buf.frames)}
        if fresh and self.abuf is None and self.use_matte:
            if isinstance(self.source, StillSource):
                self.abuf = matte.still(self.source.img)
            else:
                self.a0 = self.src_time(0)
                self.abuf = matte.mattes(self.source.path, self.a0, self.src_time(dur) + 0.1, FPS)

    def local_speed(self, lt):
        if isinstance(self.speed, (list, tuple)):
            return float(np.interp(lt, [k[0] for k in self.speed], [k[1] for k in self.speed]))
        return float(self.speed)

    def sample(self, st, lt):
        """Real time and faster: the nearest whole source frame (blending two ghosts fast
        action). Slow motion: blend neighbours."""
        b = self.buf
        if self.local_speed(lt) >= 0.75 or len(b.frames) == 1:
            i = int(np.clip(round((st - b.t0) * b.fps), 0, len(b.frames) - 1))
            return b.frames[i].astype(np.float32) / 255.0
        return b.frame(st)

    def framing(self, lt, dur):
        p = float(np.clip(lt / max(dur, 1e-6), 0, 1))
        if self.zfn:
            return self.zfn(lt)
        if self.zsteps:
            st = [s for s in self.zsteps if s[0] <= lt + 1e-6] or self.zsteps[:1]
            t0, z, cx, cy, drift = (list(st[-1]) + [0.0])[:5]
            return z * (1 + drift * max(lt - t0, 0)), cx, cy
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * p
        if self.trackt:
            ts = [k[0] for k in self.trackt]
            return z, float(np.interp(lt, ts, [k[1] for k in self.trackt])), float(np.interp(lt, ts, [k[2] for k in self.trackt]))
        cx, cy = self.center_at(p)
        return z, cx, cy

    def alpha(self, lt, dur, out_size, freeze_at=None):
        """The people matte at local time lt, framed exactly like the picture."""
        st = self.src_time(min(lt, freeze_at) if freeze_at is not None else lt)
        ab = self.abuf
        a = ab[int(np.clip(round((st - self.a0) * FPS), 0, len(ab) - 1))] if ab.ndim == 3 else ab
        z, cx, cy = self.framing(lt, dur)
        return np.clip(fx.reframe(np.ascontiguousarray(a), z, cx, cy, out_size=out_size), 0, 1)

    def frame(self, lt, dur, look, freeze_at=None, out_size=None, **over):
        size = out_size or (self.buf.frames.shape[2], self.buf.frames.shape[1])
        st = self.src_time(min(lt, freeze_at) if freeze_at is not None else lt)
        img = self.sample(st, lt if freeze_at is None or lt < freeze_at else freeze_at)
        echo = self.echo_fn(lt) if self.echo_fn else self.echo
        if echo > 0.01:
            # chronophotography: earlier instants of the move linger as light
            for k in range(1, 6):
                img = np.maximum(img, self.buf.frame(st - k * 0.05) * (echo * 0.75 ** k))
        z, cx, cy = self.framing(lt, dur)
        img = fx.reframe(img, z, cx, cy, out_size=size)
        g = {**self.grade, **over}
        if "alpha" not in g and self.abuf is not None:
            g["alpha"] = self.alpha(lt, dur, size, freeze_at)
        return look.grade(img, **g)


def clip(name, t_in, speed=1.0, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", echo=0.0, decode_zoom=None,
         zsteps=None, track=None, zfn=None, echo_fn=None, **grade):
    s = src(name)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    sw, sh = sorted((s.info["width"], s.info["height"]))      # portrait after rotation
    dw = int(min(sw, max(1080, round(1080 * zmax))) // 2 * 2)
    dh = int(min(sh, max(1920, round(1920 * zmax))) // 2 * 2)
    if sw < 1080:                                              # a small source: decode at its own size
        dw, dh = sw // 2 * 2, sh // 2 * 2
    c = MClip(s, t_in, speed=speed, zoom=zoom, center=center, note=note, echo=echo, grade=grade, decode=(dw, dh))
    return c.setup(zsteps, track, zfn, echo_fn)


def photo(name, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", decode_zoom=None, zfn=None, zsteps=None,
          use_matte=True, **grade):
    grade.setdefault("match", NOMATCH)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    c = MClip(StillSource(name), 0.0, speed=0.0, zoom=zoom, center=center, note=note, grade=grade,
              decode=(int(1080 * zmax * 1.05) // 2 * 2, int(1920 * zmax * 1.05) // 2 * 2))
    return c.setup(zsteps=zsteps, zfn=zfn, use_matte=use_matte)


# ── the look ─────────────────────────────────────────────────────────────────
# Arma's cast, measured tone by tone on its reference film (chroma offsets per luma decile)
TINT_L = np.linspace(0.05, 0.95, 10).astype(np.float32)
TINT = np.array([[5.0, -1.2, -2.7], [13.5, -2.4, -15.6], [16.5, -2.7, -22.3], [18.6, -3.1, -24.1],
                 [18.1, -2.8, -25.6], [17.3, -2.8, -23.5], [13.5, -1.9, -20.7], [10.7, -1.2, -19.4],
                 [6.4, -0.2, -16.9], [7.4, -0.7, -14.9]], np.float32) / 255
BLACK = np.array([0.020, 0.015, 0.012], np.float32)       # warm charcoal, never pure black


class WarmLook(fx.Look):
    """One warm, low-key grade for every shot.

    match (white balance, black/white points, exposure) → hue-selective saturation (skin, gold
    and amber keep it, cool colours are muted) → the room sits lower than the athletes (alpha,
    the people matte) → a filmic print curve → Arma's cast, tone by tone → warm charcoal blacks.
    """

    def __init__(self, w, h):
        super().__init__(w, h)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        self.ceiling = (1 - fx.smoothstep(0.0, 0.3, yy)).astype(np.float32)

    def grade(self, img, match=None, alpha=None, exposure=0.0, contrast=0.32, drop=1.5, lift=0.2, clar=0.32,
              top=0.25, sat_w=1.05, sat_c=0.5, tint=0.6, **_):
        x = np.clip(img, 0, 1).astype(np.float32)
        e = exposure - 0.35
        if match:
            x = np.clip((x * match["gain"] - match["black"]) * match["scale"], 0, 1)
            e += match["expo"]
        y = x @ LUMA
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        warmh = np.clip(1 - np.minimum(np.abs(((hsv[..., 0] - 28 + 180) % 360) - 180) / 38, 1), 0, 1)
        chroma = (x - y[..., None]) * (sat_c + (sat_w - sat_c) * warmh)[..., None]
        h, w = y.shape
        k = w / 1080
        if alpha is not None and drop > 0:
            a = alpha.astype(np.float32)
            am = cv2.GaussianBlur(a, (0, 0), 12 * k)
            small = cv2.resize(a, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
            halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 110 * k / 8), (w, h), interpolation=cv2.INTER_LINEAR)
            light = np.clip(0.6 * am + 0.7 * halo, 0, 1)
            room = fx.smoothstep(0.35, 0.9, y)
            f = 2.0 ** (-(drop * (0.55 + 0.45 * room)) * (1 - light) + lift * am)
            if self.ceiling.shape[0] == h:
                f = f * (1 - top * self.ceiling * (1 - am))
            y2 = (y ** 2.2 * f) ** (1 / 2.2)
            chroma = chroma * (y2 / np.maximum(y, 1e-4))[..., None] ** 0.8
            base = cv2.GaussianBlur(y2, (0, 0), 9 * k)
            y = y2 + (y2 - base) * clar * am
        y = np.clip(y * 2.0 ** e, 0, 1)
        yt = fx.filmic(y)
        yt = np.clip(yt + contrast * (yt * yt * (3 - 2 * yt) - yt), 0, 1)
        chroma = chroma * (yt / np.maximum(y, 1e-4))[..., None] ** 0.6
        cast = np.stack([np.interp(yt, TINT_L, TINT[:, c]) for c in range(3)], -1) * tint
        out = np.clip(yt[..., None] + chroma + cast, 0, 1)
        return (BLACK + out * (1 - BLACK)).astype(np.float32)


# ── transitions ──────────────────────────────────────────────────────────────
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


CREAM = np.array([1.0, 0.965, 0.90], np.float32)


def t_burn(a, b, p, ctx):
    """The reference's chapter change: the outgoing shot overexposes into warm white, a few frames
    of pure light, and the incoming shot comes back out of it, washed, then settles."""
    if p < 0.5:
        img, k = a, (p / 0.5) ** 1.6
    else:
        img, k = b, ((1 - p) / 0.5) ** 1.3
    if k < 0.01:
        return img
    lit = np.clip(img * (1 + 3.2 * k), 0, 1)
    out = lit * (1 - k ** 1.5) + CREAM * k ** 1.5
    return np.clip(out, 0, 1)


fx.TRANSITIONS["smear_up"] = lambda a, b, p, c: t_smear_v(a, b, p, c, direction=-1)
fx.TRANSITIONS["burn"] = t_burn
WHIP = 0.14                                  # a whip's length; the shot starts half of it early
BURN = 0.44                                  # a burn's length; its white peak lands on the beat


def ramp(points):
    """Speed keys [(local t, speed), ...] with 0.03 s ramps between the steps."""
    out = []
    for i, (t, s) in enumerate(points):
        if i and abs(s - points[i - 1][1]) > 1e-6:
            out.append((t, points[i - 1][1]))
            out.append((t + 0.03, s))
        elif not i:
            out.append((t, s))
    out.append((99.0, points[-1][1]))
    return out


def t_in_for(speed, lt_hit, src_hit):
    """The clip's t_in so that source time src_hit plays at local time lt_hit."""
    probe = Clip(None, 0.0, speed=speed)
    return src_hit - probe.src_time(lt_hit)


def ease_snap(lt, amt, t=0.45, drift=0.025):
    """A gentle push that settles, then keeps drifting."""
    return 1.0 + amt * (1 - fx.expo_out(lt / t)) + drift * lt


# ── the opening ──────────────────────────────────────────────────────────────
STRIKES = {14: 0.85, 15: 0.4, 27: 1.0, 28: 0.5, 38: 0.7, 47: 0.95, 48: 0.6, 54: 1.0}


def backlight(n):
    lv = 0.0
    for f, s in STRIKES.items():
        if f <= n:
            lv = max(lv, s * np.exp(-(n - f) / 2.2))
    return float(lv)


class Opening:
    """Out of black the light comes on behind Milosz and his partner: silhouettes, rimmed where
    the light wraps them. On the bar line it lands; three punch-ins on the eighth notes; the kick
    lands on the downbeat in slow motion, trailing light."""

    def __init__(self):
        sp = ramp([(0.0, 0.45), (V_LAND, 1.0), (V_KICK, 0.28), (V_KICK + 0.56, 1.0)])
        self.clip = clip("kickboxing.drills.mp4", t_in_for(sp, V_KICK, 21.39), speed=sp,
                         zsteps=[(0, 1.12, 0.55, 0.47, -0.03), (V_LAND, 1.0, 0.5, 0.5),
                                 (vb(V_LAND, 0.5), 1.2, 0.68, 0.42), (vb(V_LAND, 1.0), 1.42, 0.70, 0.40),
                                 (vb(V_LAND, 1.5), 1.66, 0.71, 0.38), (V_KICK, 1.04, 0.5, 0.5, 0.05)],
                         echo_fn=lambda lt: 0.6 * fx.smoothstep(V_KICK, V_KICK + 0.08, lt) * (1 - fx.smoothstep(V_KICK + 0.6, V_KICK + 0.8, lt)),
                         decode_zoom=1.7, note="the light comes on behind Milosz; punch-ins; the kick lands")

    def picture(self, shot, lt, tl):
        w, h = tl.w, tl.h
        if lt < 0.25:
            return np.broadcast_to(BLACK, (h, w, 3)).copy()
        img = self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        if lt >= V_LAND:
            return img
        a = self.clip.alpha(lt, shot.dur, (w, h))
        L = backlight(int(lt * FPS))
        k = w / 1080
        a_s = cv2.GaussianBlur(a, (0, 0), 1.5 * k)[..., None]
        rim = np.clip(a - cv2.GaussianBlur(a, (0, 0), 7 * k), 0, 1)[..., None] * 2.0
        bg = img * (0.02 + 0.98 * L)
        glow = cv2.GaussianBlur(bg, (0, 0), 24 * k) * 0.3 * L
        subj = img * 0.05 + rim * CREAM * 0.8 * L
        return np.clip(BLACK + bg * (1 - a_s) + subj * a_s + glow, 0, 1)


# ── the hook freezes ─────────────────────────────────────────────────────────
class Hold:
    """A frame that freezes on a beat and holds while the room falls away around the people
    and the camera keeps drifting in. Used for Bilal's hook and for the last uppercut."""

    def __init__(self, clip, start, t_freeze, push=0.06, centre=(0.52, 0.42), dim=0.8):
        self.clip, self.start, self.tf = clip, start, t_freeze
        self.push, self.centre, self.dim = push, centre, dim

    def picture(self, shot, lt, tl):
        w, h = tl.w, tl.h
        fz = self.tf - self.start
        if lt < fz:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        k = lt - fz
        img = self.clip.frame(fz, shot.dur, tl.look, out_size=(w, h))
        a = cv2.GaussianBlur(self.clip.alpha(fz, shot.dur, (w, h)), (0, 0), 4)[..., None]
        fall = fx.ease_out(fx.window(k, 0.05, 0.6)) * self.dim
        out = BLACK + (img - BLACK) * (1 - fall * (1 - a))
        return fx.reframe(out, 1.0 + self.push * fx.ease_out(k / 1.2), *self.centre)


class Diptych:
    """The two achievements meet, Milosz from above, Imre from below."""
    SLIDE = 0.32

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.top = MClip(StillSource("Milosz.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.42, 0.33),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": 0.1, "drop": 0.6}).setup()
        self.bot = MClip(StillSource("Imre.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.5, 0.28),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": 0.1, "drop": 0.6}).setup()
        self.clips = [self.top, self.bot]

    def render(self, shot, lt, tl):
        w, h = self.w, self.h
        hh = h // 2
        a = self.top.frame(lt, shot.dur, tl.look, out_size=(w, hh))
        b = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, h - hh))
        e = fx.expo_out(min(lt / self.SLIDE, 1.0))
        dy = int(round((1 - e) * hh))
        img = np.broadcast_to(BLACK, (h, w, 3)).copy()
        if dy < hh:
            img[0:hh - dy] = a[dy:hh]
            img[hh + dy:h] = b[0:h - hh - dy]
        img[hh - 3:hh + 3] = BLACK
        return img


# ── the end: the two teams ───────────────────────────────────────────────────
class Teams:
    """No-gi above, gi below, both standing under the wall sign. They come out of the white
    together and drift gently against each other; the film fades to black on them."""
    GAP = 8

    def __init__(self, w, h):
        self.w, self.h = w, h
        ph = (h - self.GAP) // 2
        self.ph = ph
        self.top = photo("Of.voor.communityy.jpg", zfn=lambda lt: (1.0 + 0.035 * fx.ease_out(lt / 3.4), 0.5, 0.44),
                         use_matte=False, exposure=0.15, note="the no-gi team under the wall sign")
        self.bot = photo("new/Gi.team.jpg", zfn=lambda lt: (1.04 - 0.035 * fx.ease_out(lt / 3.4), 0.5, 0.50),
                         decode_zoom=1.1, use_matte=False, exposure=0.15, note="the gi team under the wall sign")
        self.clips = [self.top, self.bot]

    def render(self, shot, lt, tl):
        w, h, ph = self.w, self.h, self.ph
        img = np.broadcast_to(BLACK, (h, w, 3)).copy()
        img[:ph] = self.top.frame(lt, shot.dur, tl.look, out_size=(w, ph))
        img[h - ph:] = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, ph))
        fade = fx.ease_in(fx.window(lt, shot.dur - 0.9, shot.dur - 0.05))
        return BLACK + (img - BLACK) * (1 - fade)


# ── the edit ─────────────────────────────────────────────────────────────────
def build(w=1080, h=1920):
    look = WarmLook(w, h)
    tl = Timeline(w, h, FPS, DURATION, look)
    S = tl.add
    flashes = []                              # soft exposure flashes: (time, strength, length)

    # 0 · the opening, into the kick
    op = Opening()
    S(Shot(0.0, vb(V_KICK, 2), render=op.picture, clips=[op.clip]))
    flashes += [(V_LAND, 0.25, 0.12), (V_KICK, 0.45, 0.18)]

    # 1 · the intro: kick → throw → her combination → pads into a double-leg → the exchange
    t = vb(V_KICK, 2) - WHIP / 2                                         # the throw lands on the beat
    sp = ramp([(0.0, 1.0), (0.38, 0.5), (0.80, 1.0)])
    S(Shot(t, vb(V_KICK, 4.5) - t, clip("Dilbrien.x.Milosz.scramble.mp4", t_in_for(sp, vb(V_KICK, 4) - t, 1.09), speed=sp,
                                        zoom=(1.5, 1.5), track=[(0, 0.40, 0.29), (0.8, 0.43, 0.32), (1.35, 0.46, 0.40)],
                                        note="the throw, slowed at the top"), trans=("smear", WHIP)))
    t = vb(V_KICK, 4.5)
    S(Shot(t, vb(V_KICK, 7) - t, clip("alternative.shot.mp4", 14.02, zoom=(1.25, 1.32), center=(0.58, 0.42),
                                      note="her combination")))
    t = vb(V_KICK, 7)
    sp = ramp([(0.0, 1.0), (0.85, 0.6), (1.40, 1.0)])
    S(Shot(t, vb(V_KICK, 10) - t, clip("action.mp4", 3.95, speed=sp, zoom=(1.0, 1.08), center=(0.5, 0.52),
                                       note="pads with the coach, into a double-leg")))
    t = vb(V_KICK, 10)
    S(Shot(t, BEAT, clip("Milosz.Bilal.KB2.mp4", 8.55, zoom=(1.0, 1.03), note="Milosz and Bilal trade under the wall sign")))
    t = vb(V_KICK, 11)
    hook = Hold(clip("Milosz.Bilal.KB2.mp4", 9.50, zoom=(1.16, 1.16), note="Bilal's hook freezes"), t, V_PAUSE)
    S(Shot(t, V_DROP + BURN / 2 - t, render=hook.picture, clips=[hook.clip]))

    # 2 · the drop: the achievements, then the range
    t = V_DROP - BURN / 2
    S(Shot(t, 2 * BEAT + BURN / 2, photo("Milosz.Celebration.jpg", center=(0.40, 0.42), exposure=0.1, drop=0.6, decode_zoom=1.2,
                                         zfn=lambda lt: (ease_snap(lt, 0.10), 0.40, 0.42), note="Milosz, hand raised"),
           trans=("burn", BURN)))
    t = vb(V_DROP, 2)
    S(Shot(t - WHIP / 2, 2 * BEAT + WHIP / 2, photo("Imre.Celebration.jpg", exposure=0.1, drop=0.6, decode_zoom=1.2,
                                                    zfn=lambda lt: (ease_snap(lt, 0.08), 0.5, 0.36), note="Imre, arms up"),
           trans=("smear_up", WHIP)))
    t = vb(V_DROP, 4)
    dip = Diptych(w, h)
    S(Shot(t, 2 * BEAT, render=dip.render, clips=dip.clips))
    t = vb(V_DROP, 6)                                                    # the d'arce roll, slowed at the top
    sp = ramp([(0.0, 1.0), (0.70, 0.45), (1.80, 1.0)])
    S(Shot(t - WHIP / 2, vb(V_DROP, 10) - t + WHIP / 2, clip(
        "Makhachev.D.arce.mp4", 19.30, speed=sp, zoom=(1.16, 1.08), center=(0.5, 0.5),
        echo_fn=lambda lt: 0.3 * fx.smoothstep(0.7, 0.85, lt) * (1 - fx.smoothstep(1.7, 1.9, lt)), note="the d'arce roll"),
        trans=("smear", WHIP)))
    t = vb(V_DROP, 10)
    S(Shot(t, 3 * BEAT, clip("arm.triangle.mp4", 4.30, zoom=(1.45, 1.62), center=(0.42, 0.56), note="the arm-triangle squeeze")))
    t = vb(V_DROP, 13)
    S(Shot(t, BEAT, clip("Milosz.Bilal.mp4", 36.67, zoom=(1.08, 1.12), note="Milosz trading, the room watching")))
    t = vb(V_DROP, 14)
    S(Shot(t, BEAT, clip("50-50.mp4", 1.45, zoom=(1.12, 1.16), center=(0.55, 0.45), note="50-50")))
    t = vb(V_DROP, 15)
    S(Shot(t, 2 * BEAT, clip("limited.gi.footage.mp4", 2.20, zoom=(1.2, 1.28), center=(0.45, 0.42), note="the gi: side control")))
    t = vb(V_DROP, 17)
    S(Shot(t, V_BREAK + BURN / 2 - t, clip("armbar.mp4", 6.90, zoom=(1.12, 1.22), center=(0.5, 0.55),
                                           note="the sit-back into the armbar")))

    # 3 · the people: coaching, laughing, promotions
    t = V_BREAK - BURN / 2
    S(Shot(t, 3 * BEAT + BURN / 2, clip("uitleg.mma.mp4", 5.0 - BURN / 2 * 0.8, speed=0.8, zoom=(1.6, 1.72), center=(0.52, 0.33),
                                        note="a coach on the mat with a pair"), trans=("burn", BURN)))
    t = vb(V_BREAK, 3)
    S(Shot(t, 2 * BEAT, clip("extra.mp4", 8.95, speed=0.7, zoom=(1.12, 1.2), center=(0.5, 0.45),
                              note="a kid on the pads with his coach")))
    t = vb(V_BREAK, 5)
    S(Shot(t, 2 * BEAT, clip("Milosz.Bilal.KB2.mp4", 19.95, speed=0.8, zoom=(1.42, 1.48), center=(0.40, 0.36),
                              note="Milosz laughing after a round")))
    t = vb(V_BREAK, 7)
    S(Shot(t, 3 * BEAT, clip("new/Graduatie.mp4", 0.30, speed=0.8, zoom=(1.62, 1.7), center=(0.55, 0.59),
                              note="a new belt and a hug under the wall sign")))
    t = vb(V_BREAK, 10)
    S(Shot(t, 2 * BEAT, clip("new/Graduatie_2.mp4", 0.30, speed=0.85, zoom=(1.5, 1.56), center=(0.5, 0.57),
                              note="a congratulation")))
    # the 808: the purple-belt hug, slowed
    S(Shot(V_808, BAR, clip("new/WhatsApp_Video_2026-08-24_at_20.41.55.mp4", 6.95, speed=0.6, zoom=(1.55, 1.62),
                            center=(0.45, 0.48), note="the purple-belt hug, in slow motion")))

    # 4 · the beat returns: the gi roll, a combination under the wall sign, the uppercut
    S(Shot(V_RETURN, BEAT, clip("new/shark_tank.mp4", 27.25, zoom=(1.12, 1.16), center=(0.5, 0.52), note="the gi roll")))
    for k, (s_hit, z, note) in enumerate([(12.16, 1.08, "a combination on the pads"), (15.04, 1.24, "…closer")]):
        S(Shot(vb(V_RETURN, 1 + k), BEAT, clip("wall.drill.near.the.end.mp4", s_hit - 0.03, zoom=(z, z * 1.02),
                                               center=(0.5, 0.46), note=note)))
    t = vb(V_RETURN, 3)
    up = Hold(clip("wall.drill.near.the.end.mp4", 8.98 - (V_CARD - t), note="the uppercut holds"), t, V_CARD,
              push=0.05, centre=(0.5, 0.42), dim=0.7)
    S(Shot(t, V_TEAM + BURN / 2 - t, render=up.picture, clips=[up.clip]))
    flashes.append((V_CARD, 0.35, 0.18))

    # 5 · the teams
    teams = Teams(w, h)
    S(Shot(V_TEAM - BURN / 2, DURATION - V_TEAM + BURN / 2, render=teams.render, clips=teams.clips, trans=("burn", BURN)))

    def events(t, img, tl):
        for t0, k, d in flashes:
            if t0 <= t < t0 + d * 2.5:
                img = fx.flash(img, k * max(0.0, 1 - (t - t0) / d), tuple(CREAM))
        return img
    tl.overlays.insert(0, Overlay(0.0, DURATION, events, "pre"))
    tl.cinema_at = lambda t: {"mono": 0.0, "streaks": 0.0, "halation": 0.4, "bloom": 0.3, "weave": 0.25}
    tl.grain_at = lambda t: 0.6
    tl.vignette_at = lambda t: 0.55
    tl.audio_plan = sounds()
    return tl


# ── sound ────────────────────────────────────────────────────────────────────
def sounds():
    """Real sound from the floor: (video time, file, src t0, src t1, dB)."""
    combos = [(vb(V_RETURN, 1 + k), "wall.drill.near.the.end.mp4", s_hit - 0.03, s_hit + 0.45, -5)
              for k, s_hit in enumerate([12.16, 15.04])]
    return [
        (V_KICK - 0.03, "kickboxing.drills.mp4", 21.36, 21.75, -3),         # the kick lands
        (vb(V_KICK, 4) - 0.03, "Dilbrien.x.Milosz.scramble.mp4", 1.06, 1.45, -6),   # the throw lands
        (vb(V_KICK, 4.5) + (14.40 - 14.02), "alternative.shot.mp4", 14.40, 14.98, -6),
        (vb(V_KICK, 7) + 0.03, "action.mp4", 3.98, 4.60, -6),               # the pads
        (vb(V_KICK, 7) + 1.42, "action.mp4", 5.15, 5.45, -7),               # the double-leg lands
        (V_PAUSE - 0.03, "Milosz.Bilal.KB2.mp4", 9.97, 10.45, -2),          # the hook the music stops on
        (vb(V_DROP, 6) + 0.62, "Makhachev.D.arce.mp4", 19.92, 20.30, -8),   # the roll
        (vb(V_DROP, 13) + (36.90 - 36.67), "Milosz.Bilal.mp4", 36.90, 37.30, -7),
        (vb(V_DROP, 17) + (8.05 - 6.90), "armbar.mp4", 8.04, 8.40, -7),     # the sit-back
        (vb(V_BREAK, 5), "Milosz.Bilal.KB2.mp4", 19.95, 20.75, -8),          # the laugh
        (V_808 + 0.1, "new/WhatsApp_Video_2026-08-24_at_20.41.55.mp4", 0.4, 2.4, -10),   # the room cheering
    ] + combos + [
        (V_CARD - 0.03, "wall.drill.near.the.end.mp4", 8.95, 9.40, -2),      # the uppercut
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


def mix(plan, sr=48000, music=True):
    """The mix. music=False: the floor sounds alone (with the hall tails), for laying a licensed
    track under it in an ad."""
    from scipy import signal
    n = int(round(DURATION * sr))
    out = np.zeros((n, 2), np.float32)
    song = media.read_audio(str(MUSIC), 0.0, None, sr=sr, channels=2)      # read once, slice in memory
    if not music:
        song = np.zeros_like(song)
    for k, (v0, s0, s1, gdb, fin, fout) in enumerate(MUSIC_PLAN):
        lead = 0.0 if k == 0 else 0.015                                   # splice just ahead of the downbeat
        a, b = int(round((s0 - lead) * sr)), int(round(s1 * sr))
        seg = song[a:b].copy() * 10 ** (gdb / 20)
        nfi, nfo = max(int(fin * sr), 1), max(int(fout * sr), 1)
        seg[:nfi] *= np.sin(np.linspace(0, np.pi / 2, nfi))[:, None] ** (2 if k == 0 else 1)
        seg[-nfo:] *= np.cos(np.linspace(0, np.pi / 2, nfo))[:, None]
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
    rev = rev / (np.abs(rev).max() + 1e-6) * 0.3
    j = int(round(V_KICK * sr)) - len(rev)
    if j < 0:
        rev, j = rev[-j:], 0
    out[j:j + len(rev)] += rev
    # real sound from the floor
    hp = signal.butter(2, 140 / (sr / 2), "high", output="sos")
    for v, name, t0, t1, db in plan:
        a = media.read_audio(str(FOOT / name), t0, t1 - t0, sr=sr, channels=2)
        a = signal.sosfilt(hp, a, axis=0).astype(np.float32)
        f = int(0.004 * sr)
        a[:f] *= np.linspace(0, 1, f)[:, None]
        fo = int(min(0.25, (t1 - t0) / 3) * sr)
        a[-fo:] *= np.linspace(1, 0, fo)[:, None]
        a = a / (np.abs(a).max() + 1e-6) * 10 ** (db / 20) * 0.5
        i = max(0, int(round(v * sr)))
        a = a[:n - i]
        out[i:i + len(a)] += a
        if name == "wall.drill.near.the.end.mp4" and abs(v - (V_CARD - 0.03)) < 1e-6:
            wet = _verb(a, _hall(sr, 2.4))[:n - i] * 0.45                  # the uppercut rings out
            out[i:i + len(wet)] += wet
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
        # the same picture with the floor sounds only, for ads (the song is not cleared for paid use)
        import subprocess
        me = str(out.with_name(out.stem + "_no-music.wav"))
        media.write_wav(me, mix(tl.audio_plan, music=False))
        subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(out), "-i", me, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
                        str(out.with_name(out.stem + "_no-music.mp4"))], check=True)
