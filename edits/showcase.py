"""040COMBAT, Deurne — a 33-second 9:16 showcase.

The look: a relit black & white. Every shot carries a people matte (Robust Video Matting,
afterfilm/matte.py); the grade drops the room up to ~1.8 stops behind the athletes, lifts them a
touch and adds local contrast on them only, then prints through a filmic curve with graphite
shadows and paper highlights. Gold appears only as light (the gilded competition photos, flashes
on the big hits, the glint on the wall sign) and in the badge. Cut to Future & Metro Boomin's
"Everyday Hustle" (music/, not in git; 119.67 BPM, bar = 2.006 s).

  0.0–3.0    out of black, the light strikes behind Milosz and his partner: flickering
             silhouettes · the light lands · three punch-ins on the eighth notes
  3.0–9.0    the kick lands on the first vocal downbeat in slow motion, trailing light · a whip
             into a throw, slowed at the top, landing on the beat · her combination · pads into
             a double-leg · an exchange under the wall logo
  9.0–10.0   silence · Bilal's hook freezes; the gym drains to black and the hook's path
             appears as a stroboscopic multiple exposure; the last frames print them white
  10.0–20.1  the drop · Milosz's hand raised, Imre, the two slam together · the d'arce roll ·
             the arm-triangle squeeze · Milosz trading · 50-50 · the gi · the armbar
  20.1–24.1  the breakdown · a coach on the mat with a pair · a kid on the pads with his coach
  24.1–26.1  the 808 · the team photo: a light sweeps across the room and finds every face; the
             wall sign glints gold as it passes
  26.1–28.1  the beat returns · a combination under the wall sign, punched in closer, the uppercut
  28.1–33.3  the uppercut freezes; the room drains, the two of them fade to black; over the empty
             gym, barely there, the 040COMBAT badge builds itself — the gold ring drawn by a
             line of light, the banner snapping open on the beat, the letters rising, the stars —
             a glint, DEURNE · the song's drumless outro

Footage and photos are read from footage/ (the draft release 'footage'; never in git).

    python edits/showcase.py                       # renders/040COMBAT_showcase_9x16.mp4 (+ _no-music)
    python edits/showcase.py --stills 0.5 9.6 --scale 0.5
    python edits/showcase.py --edl
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import skia
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "edits"))
from afterfilm import brand, fx, gfx, matte, media  # noqa: E402
from afterfilm.timeline import Clip, FrameBuffer, Overlay, Shot, Timeline, VideoSource  # noqa: E402
import logo040  # noqa: E402

FOOT = ROOT / "footage"
MUSIC = ROOT / "music" / "Everyday_Hustle.m4a"
FPS = 30

# ── identity ─────────────────────────────────────────────────────────────────
GOLD = (192, 155, 83)                       # measured from the supplied logo
OFFWHITE = (242, 237, 228)
GOLD_TINT = (1.0, 0.80, 0.46)

# ── the music's grid ─────────────────────────────────────────────────────────
BEAT, OFF = 0.50143, 0.047                  # fitted on the kick pattern across the track
BAR = 4 * BEAT


def SB(j):
    """Song time of bar j's downbeat."""
    return OFF + BAR * j


def vb(t0, k):
    return t0 + k * BEAT


def F(t):
    return int(round(t * FPS))


V_LAND = BAR                                 # 2.006: the light lands
V_KICK = 1.5 * BAR                           # 3.009: the kick, on the first vocal downbeat (song bar 8)
V_PAUSE = V_KICK + 3 * BAR                   # 9.026: the intro stops on a bar line
V_DROP = V_PAUSE + 2 * BEAT                  # 10.029
V_BREAK = V_DROP + 5 * BAR                   # 20.057
V_PRE = V_BREAK + 2 * BAR                    # 24.069: the 808 bar
V_RETURN = V_PRE + BAR                       # 26.074
V_CARD = V_RETURN + BAR                      # 28.080: the uppercut
U0 = V_CARD + 2 * BEAT                       # the badge starts building
DURATION = V_CARD + 5.2

# music: (video start, song start, song end, gain dB, fade in s, fade out s)
MUSIC_PLAN = [(0.0, SB(6.5), SB(11), 3.0, 0.6, 0.012),
              (V_DROP, SB(32), SB(37), 0.0, 0.012, 0.012),
              (V_BREAK, SB(52), SB(54), 2.0, 0.012, 0.012),
              (V_PRE, SB(59), SB(61) + 0.35, 1.0, 0.012, 0.35),
              (V_CARD, SB(106), SB(106) + DURATION - V_CARD, 0.0, 0.04, 1.4)]

# ── sources ──────────────────────────────────────────────────────────────────
_SRC = {}
DENOISE = {"action.mp4": "hqdn3d=2:1.5:4:3", "wall.drill.near.the.end.mp4": "hqdn3d=1.5:1.5:3:3",
           "uitleg.mma.mp4": "hqdn3d=2:1.5:4:3", "armbar.mp4": "hqdn3d=1.5:1.5:3:3"}


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
        s = min(max(w / iw, h / ih), max(1.0, 1080 / iw, 1920 / ih))
        out = cv2.resize(self.img, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA)
        return FrameBuffer(out[None], t0, fps)


BW_MIX = np.array([0.50, 0.40, 0.10], np.float32)          # orange-filter panchromatic: skin glows


def measure(frames):
    """Per-shot match before the look: black and white points and a common exposure."""
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = np.stack([cv2.resize(frames[i], (96, 170), interpolation=cv2.INTER_AREA) for i in idx]).astype(np.float32) / 255
    y = small.reshape(-1, 3) @ BW_MIX
    lo, mid, hi = np.percentile(y, [1.0, 50, 99.6])
    black = max(0.0, lo - 0.01) * 0.85
    scale = float(np.clip(0.97 / max(hi - black, 1e-3), 1.0, 1.35))
    mid2 = (mid - black) * scale
    expo = float(np.clip(np.log2(0.34 / max(mid2, 1e-3)) * 0.5, -0.4, 0.5))
    return dict(black=float(black), scale=scale, expo=expo)


NOMATCH = dict(black=0.0, scale=1.0, expo=0.0)


class MClip(Clip):
    """A clip with per-frame framing (staccato zoom steps, keyed tracks), light trails and the
    people matte, which the grade uses to relight the shot. It measures itself when decoded."""

    def setup(self, zsteps=None, track=None, zfn=None, echo_fn=None):
        self.zsteps, self.trackt, self.zfn, self.echo_fn = zsteps, track, zfn, echo_fn
        self.abuf, self.a0 = None, 0.0
        return self

    def prepare(self, dur, fps, size):
        fresh = self.buf is None
        super().prepare(dur, fps, size)
        if fresh and "match" not in self.grade:
            self.grade = {**self.grade, "match": measure(self.buf.frames)}
        if fresh and self.abuf is None:
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
    c = MClip(s, t_in, speed=speed, zoom=zoom, center=center, note=note, echo=echo, grade=grade, decode=(dw, dh))
    return c.setup(zsteps, track, zfn, echo_fn)


def photo(name, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", decode_zoom=None, zfn=None, zsteps=None, **grade):
    grade.setdefault("match", NOMATCH)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    c = MClip(StillSource(name), 0.0, speed=0.0, zoom=zoom, center=center, note=note, grade=grade,
              decode=(int(1080 * zmax * 1.05) // 2 * 2, int(1920 * zmax * 1.05) // 2 * 2))
    return c.setup(zsteps=zsteps, zfn=zfn)


# ── the look ─────────────────────────────────────────────────────────────────
PAPER = np.array([1.0, 0.982, 0.948], np.float32)          # highlights: off-white paper
GRAPHITE = np.array([0.965, 0.98, 1.0], np.float32)        # shadows: neutral graphite
METAL = np.array([1.16, 0.90, 0.50], np.float32)
METAL = METAL / float(METAL @ fx.LUMA)                     # the logo's gold at unit luminance


class CombatLook(fx.Look):
    """A relit black & white. alpha (the people matte, framed like the picture) drives the
    relight: the room drops up to `drop` stops (bright walls and windows most), the athletes
    lift `lift` stops and take local contrast (`clar`), the ceiling darkens (`top`). Then a
    filmic print curve with graphite shadows and paper highlights. `gold` gilds the highlights."""

    def __init__(self, w, h):
        super().__init__(w, h, accent=GOLD)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        self.ceiling = (1 - fx.smoothstep(0.0, 0.32, yy)).astype(np.float32)

    def grade(self, img, match=None, alpha=None, gold=0.0, exposure=0.0, contrast=0.5, drop=1.6, lift=0.22,
              clar=0.8, top=0.3, **_):
        x = np.clip(img, 0, 1).astype(np.float32)
        e = exposure
        if match:
            x = np.clip((x - match["black"]) * match["scale"], 0, 1)
            e += match["expo"]
        v = x @ BW_MIX
        if e:
            v = np.clip(v * (2.0 ** e), 0, 1)
        h, w = v.shape
        k = w / 1080
        if alpha is not None and drop > 0:
            a = alpha.astype(np.float32)
            am = cv2.GaussianBlur(a, (0, 0), 12 * k)
            small = cv2.resize(a, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
            halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 110 * k / 8), (w, h), interpolation=cv2.INTER_LINEAR)
            light = np.clip(0.6 * am + 0.7 * halo, 0, 1)
            room = fx.smoothstep(0.35, 0.9, v)
            lin = v ** 2.2
            lin = lin * 2.0 ** (-(drop * (0.55 + 0.45 * room)) * (1 - light) + lift * am)
            if self.ceiling.shape[0] == h:
                lin = lin * (1 - top * self.ceiling * (1 - am))
            v = lin ** (1 / 2.2)
            base = cv2.GaussianBlur(v, (0, 0), 9 * k)
            v = v + (v - base) * (clar * am + 0.15)
        v = v + (v - cv2.GaussianBlur(v, (0, 0), 1.5 * k)) * 0.3
        v = fx.filmic(np.clip(v, 0, 1))
        v = np.clip(v + contrast * (v * v * (3 - 2 * v) - v), 0, 1)
        vv = v[..., None]
        out = vv * (GRAPHITE + (PAPER - GRAPHITE) * vv)
        if gold > 0:
            warm = fx.smoothstep(0.5, 1.0, vv) * 0.4 * gold
            out = out * (1 - warm) + np.clip(vv * METAL, 0, 1) * warm
        return (0.005 + out * 0.995).astype(np.float32)


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


def ease_snap(lt, amt, t=0.35, drift=0.03):
    """A snap zoom that settles, then keeps pushing slowly."""
    return 1.0 + amt * (1 - fx.expo_out(lt / t)) + drift * lt


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


# ── the opening ──────────────────────────────────────────────────────────────
# backlight strikes (local frame → brightness), like tubes catching; they decay like phosphor
STRIKES = {11: 0.9, 12: 0.45, 19: 0.6, 26: 1.0, 27: 0.3, 28: 0.75, 35: 0.45, 39: 0.95, 40: 0.6,
           45: 0.7, 46: 0.25, 47: 0.85, 48: 0.5, 52: 1.0, 54: 0.65, 55: 0.95, 57: 0.4}


def backlight(n):
    lv = 0.0
    for f, s in STRIKES.items():
        if f <= n:
            lv = max(lv, s * np.exp(-(n - f) / 1.4))
    return float(lv)


class Opening:
    """Out of black the light strikes behind Milosz and his partner: silhouettes, rimmed where the
    light wraps their edges. On the bar line the light lands; three punch-ins on the eighth notes;
    the kick lands on the downbeat in slow motion, trailing light."""

    def __init__(self):
        sp = ramp([(0.0, 0.45), (V_LAND, 1.0), (V_KICK, 0.28), (V_KICK + 0.56, 1.0)])
        self.clip = clip("kickboxing.drills.mp4", t_in_for(sp, V_KICK, 21.39), speed=sp,
                         zsteps=[(0, 1.12, 0.55, 0.47, -0.03), (V_LAND, 1.0, 0.5, 0.5),
                                 (vb(V_LAND, 0.5), 1.2, 0.68, 0.42), (vb(V_LAND, 1.0), 1.42, 0.70, 0.40),
                                 (vb(V_LAND, 1.5), 1.66, 0.71, 0.38), (V_KICK, 1.04, 0.5, 0.5, 0.05)],
                         echo_fn=lambda lt: 0.75 * fx.smoothstep(V_KICK, V_KICK + 0.08, lt) * (1 - fx.smoothstep(V_KICK + 0.6, V_KICK + 0.8, lt)),
                         decode_zoom=1.7, note="the light strikes behind Milosz; punch-ins; the kick lands")

    def picture(self, shot, lt, tl):
        w, h = tl.w, tl.h
        if lt < 0.2:
            return np.full((h, w, 3), 0.005, np.float32)
        img = self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        if lt >= V_LAND:
            return img
        a = self.clip.alpha(lt, shot.dur, (w, h))
        L = backlight(int(lt * FPS))
        k = w / 1080
        a_s = cv2.GaussianBlur(a, (0, 0), 1.5 * k)[..., None]
        rim = np.clip(a - cv2.GaussianBlur(a, (0, 0), 7 * k), 0, 1)[..., None] * 2.2
        bg = img * (0.015 + 0.985 * L)
        glow = cv2.GaussianBlur(bg, (0, 0), 24 * k) * 0.35 * L          # the haze lit from behind
        subj = img * 0.035 + rim * PAPER * 0.85 * L
        return np.clip(bg * (1 - a_s) + subj * a_s + glow, 0, 1)


# ── the hook freezes: the void ──────────────────────────────────────────────
class Void:
    """Bilal's hook freezes on the intro's last beat. In the silence the gym drains to black
    around them and earlier instants of the punch appear one by one as a stroboscopic multiple
    exposure; the last two frames before the drop print them as white silhouettes."""
    TRAIL = (9.70, 9.76, 9.82, 9.88, 9.93)

    def __init__(self, clip, start):
        self.clip, self.start = clip, start

    def picture(self, shot, lt, tl):
        w, h = tl.w, tl.h
        fz = V_PAUSE - self.start
        if lt < fz:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        k = lt - fz
        img = self.clip.frame(fz, shot.dur, tl.look, out_size=(w, h), contrast=0.6)
        a = self.clip.alpha(fz, shot.dur, (w, h))
        if V_DROP - (self.start + lt) <= 2 / FPS + 1e-6:
            return (0.005 + a[..., None] * PAPER * 0.97).astype(np.float32)
        drain = fx.ease_out(fx.window(k, 0.04, 0.45))
        out = img * (1 - 0.96 * drain * (1 - a[..., None]))
        for i, s in enumerate(self.TRAIL):
            appear = fx.ease_out(fx.window(k, 0.10 + 0.07 * i, 0.22 + 0.07 * i))
            if appear <= 0:
                continue
            lt_s = s - self.clip.t_in
            g = self.clip.frame(lt_s, shot.dur, tl.look, out_size=(w, h), contrast=0.6)
            ga = self.clip.alpha(lt_s, shot.dur, (w, h))[..., None]
            out = np.maximum(out, g * ga * appear * (0.35 + 0.1 * i) * drain)
        a3 = a[..., None]
        out = out * (1 - a3) + img * a3
        z = 1.0 + 0.07 * fx.ease_in_out(k / (V_DROP - V_PAUSE))
        return fx.reframe(out, z, 0.52, 0.42)


class Diptych:
    """The two achievements slam together, Milosz from above, Imre from below."""
    SLAM = 0.16

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.top = MClip(StillSource("Milosz.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.42, 0.33),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.15, "gold": 1.0, "drop": 0.8}).setup()
        self.bot = MClip(StillSource("Imre.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.5, 0.28),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.1, "gold": 1.0, "drop": 0.8}).setup()
        self.clips = [self.top, self.bot]

    def render(self, shot, lt, tl):
        w, h = self.w, self.h
        hh = h // 2
        a = self.top.frame(lt, shot.dur, tl.look, out_size=(w, hh))
        b = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, h - hh))
        e = fx.ease_in(min(lt / self.SLAM, 1.0))                 # accelerate into the slam
        dy = int(round((1 - e) * hh))
        img = np.full((h, w, 3), 0.005, np.float32)
        if dy < hh:
            img[0:hh - dy] = a[dy:hh]
            img[hh + dy:h] = b[0:h - hh - dy]
        k = lt - self.SLAM
        if k >= 0:
            img = fx.reframe(img, 1 + 0.025 * np.exp(-k / 0.08))
        return img


# ── the team: a light sweeps across the room ─────────────────────────────────
class GroupReveal:
    """On the 808: black. A soft band of light sweeps across the team photo and finds every
    face; once found they stay lit, the room stays low. The wall sign glints gold as it passes."""
    LOGO = (0.33, 0.62, 0.215, 0.365)          # the wall sign in the photo, normalised

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.clip = photo("Of.voor.communityy.jpg", zsteps=[(0, 1.18, 0.5, 0.47, 0.03)], decode_zoom=1.28,
                          drop=1.6, clar=0.6, lift=0.35, note="the team: a light sweeps across the room")
        X = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        self.X = np.repeat(X, h, 0)
        # RVM is a video model for a few people; on a 30-person group photo it finds only some
        # of them. The team stands as one block, so the matte is that block, feathered, merged
        # with whatever the model did find.
        yy, xx = np.mgrid[0:960, 0:540].astype(np.float32)
        xx, yy = xx / 539, yy / 959
        box = (fx.smoothstep(-0.02, 0.0, xx) * (1 - fx.smoothstep(0.905, 0.93, xx))
               * fx.smoothstep(0.385, 0.405, yy) * (1 - fx.smoothstep(0.62, 0.64, yy)))
        img = self.clip.source.img
        found = cv2.resize(matte.still(img, size=(1080, 1920)), (540, 960))
        self.clip.abuf = np.maximum(box, found).astype(np.float32)

    def picture(self, shot, lt, tl):
        w, h = self.w, self.h
        img = self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        a = cv2.GaussianBlur(self.clip.alpha(lt, shot.dur, (w, h)), (0, 0), 3)
        xs = -0.25 + 1.5 * fx.ease_in_out(fx.window(lt, 0.0, 1.15))
        band = np.exp(-((self.X - xs) / 0.085) ** 2)
        lit = np.clip((xs - self.X) / 0.18 + 0.5, 0, 1)
        settle = fx.smoothstep(1.0, 1.6, lt)
        people = np.maximum(lit, settle) * 0.97 + 0.03 + band * 0.4
        room = 0.06 + 0.5 * np.maximum(lit, settle) + band * 0.3
        f = (a * people + (1 - a) * room)[..., None]
        out = img * f
        # the wall sign glints gold as the light passes, framed with the photo
        z, cx, cy = self.clip.framing(lt, shot.dur)
        x0, x1, y0, y1 = self.LOGO
        ax = (x0 + x1) / 2
        g = np.exp(-((xs - ax) / 0.10) ** 2)
        if g > 0.02:
            m = np.zeros((h, w), np.float32)
            s = z
            px0 = int((0.5 + (x0 - cx) * s) * w); px1 = int((0.5 + (x1 - cx) * s) * w)
            py0 = int((0.5 + (y0 - cy) * s * 1.0) * h); py1 = int((0.5 + (y1 - cy) * s) * h)
            m[max(py0, 0):max(py1, 0), max(px0, 0):max(px1, 0)] = 1.0
            hot = fx.smoothstep(0.35, 0.8, out @ fx.LUMA) * cv2.GaussianBlur(m, (0, 0), 6)
            out = out + (hot * g * 0.55)[..., None] * (METAL * np.array([1.0, 0.85, 0.6], np.float32))
        return np.clip(out, 0, 1)


# ── the outro: freeze, drain to black, the badge builds over the empty gym ────
def _gym_backdrop(w, h):
    """Gym.jpg, the empty gym, processed into an out-of-focus, low black & white with its
    ceiling tubes glowing. Covers the frame with room to push."""
    im = np.asarray(ImageOps.exif_transpose(Image.open(FOOT / "Gym.jpg")).convert("RGB")).astype(np.float32) / 255
    W, H = int(w * 1.12), int(h * 1.12)
    ih, iw = im.shape[:2]
    s = H / ih
    big = cv2.resize(im, (int(iw * s), H), interpolation=cv2.INTER_AREA)
    x0 = int(0.40 * big.shape[1] - W / 2)
    crop = big[:, max(0, x0):max(0, x0) + W]
    v = crop @ BW_MIX
    k = w / 1080
    hot = cv2.GaussianBlur(np.clip((v - 0.8) / 0.2, 0, 1), (0, 0), 22 * k)
    v = cv2.GaussianBlur(v, (0, 0), 5 * k)
    v = fx.filmic(np.clip(v * 2 ** -2.0, 0, 1)) + hot * 0.22
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    v = v * (0.55 + 0.45 * np.exp(-((yy - 0.42) / 0.45) ** 2))
    return np.clip(v[..., None] * PAPER, 0, 1).astype(np.float32)


class Outro:
    def __init__(self, w, h, clip):
        self.w, self.h, self.clip = w, h, clip
        self.backdrop = _gym_backdrop(w, h)
        self.P = logo040.paths()
        b = self.P["outer"].computeTightBounds()
        self.bx, self.by = (b.left() + b.right()) / 2, (b.top() + b.bottom()) / 2
        self.s = 0.64 * w / b.width()
        self.cx, self.cy = w / 2, h * 0.445
        self.bottom = self.cy + b.height() / 2 * self.s
        ro, ri = np.array(self.P["ring_pts"]), self._ring_inner_pts()
        mid = (ro + ri) / 2
        self.ring_w = float(np.linalg.norm(ro - ri, axis=1).mean() * np.cos(np.pi / 8))   # corner offset → edge thickness
        self.ring_path = self._from_top(mid)
        self.font = brand.font("BarlowCondensed-SemiBold", 0.034 * h)
        g0, g1 = skia.Point(b.left(), b.top()), skia.Point(b.right(), b.bottom())
        self.metal = skia.GradientShader.MakeLinear(
            [g0, g1], [skia.Color(236, 205, 140), skia.Color(192, 155, 83), skia.Color(132, 98, 44), skia.Color(205, 170, 100)],
            [0.0, 0.38, 0.72, 1.0])
        self.silver = skia.GradientShader.MakeLinear(
            [g0, g1], [skia.Color(150, 150, 146), skia.Color(96, 96, 94), skia.Color(62, 62, 60), skia.Color(110, 110, 106)],
            [0.0, 0.4, 0.75, 1.0])

    def _ring_inner_pts(self):
        data = __import__("json").loads(logo040.CACHE.read_text())
        return np.array(data["ring_inner"])

    def _from_top(self, poly):
        """The ring's centre line as a closed path that starts at the middle of its top edge."""
        n = len(poly)
        i = min(range(n), key=lambda i: (poly[i][1] + poly[(i + 1) % n][1]))
        a, b = poly[i], poly[(i + 1) % n]
        start = (a + b) / 2
        p = skia.Path()
        p.moveTo(*start)
        for j in range(1, n + 1):
            p.lineTo(*poly[(i + j) % n])
        p.lineTo(*start)
        return p

    # the picture under the badge
    def picture(self, shot, lt, tl):
        w, h = self.w, self.h
        hit = V_CARD - shot.start
        if lt < hit:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        k = lt - hit
        img = self.clip.frame(hit, shot.dur, tl.look, out_size=(w, h), contrast=0.55)
        a = self.clip.alpha(hit, shot.dur, (w, h))[..., None]
        drain = fx.ease_out(fx.window(k, 0.06, 0.45))
        fig = 1 - fx.ease_in_out(fx.window(k, 0.5, 1.05))
        out = img * (1 - a) * (1 - 0.97 * drain) + img * a * fig
        out = fx.reframe(out, 1.0 + 0.06 * fx.ease_out(k / 1.2), 0.5, 0.40)
        bgk = fx.ease_in_out(fx.window(k, 0.8, 1.9)) * (1 - fx.ease_in(fx.window(k, DURATION - V_CARD - 0.9, DURATION - V_CARD - 0.1)))
        if bgk > 0:
            z = 1.0 + 0.06 * fx.ease_out(fx.window(k, 0.8, DURATION - V_CARD))
            bg = fx.reframe(self.backdrop, z, 0.5, 0.5, out_size=(w, h))
            out = out + bg * bgk * 0.75
        return np.clip(out, 0, 1)

    # the badge, drawn crisp after the film pass
    def draw(self, t, img, tl):
        u = t - U0
        if u < 0:
            return img
        P, layer = self.P, tl.layer
        layer.clear()
        c = layer.c
        fade = 1 - fx.ease_in(fx.window(t, DURATION - 0.75, DURATION - 0.05))
        push = 1.0 + 0.035 * fx.ease_out(u / 4.0)
        c.save()
        c.translate(self.cx, self.cy)
        c.scale(self.s * push, self.s * push)
        c.translate(-self.bx, -self.by)
        A = fade
        col = lambda r, g, b, a=1.0: skia.Paint(AntiAlias=True, Color4f=skia.Color4f(r, g, b, a * A))  # noqa: E731

        def metal(a=1.0, stroke=0.0):
            p = skia.Paint(AntiAlias=True, Shader=self.metal, Alphaf=a * A)
            if stroke:
                p.setStyle(skia.Paint.kStroke_Style)
                p.setStrokeWidth(stroke)
                p.setStrokeJoin(skia.Paint.kMiter_Join)
            return p
        # the body and its silver rim settle in behind
        e_body = fx.ease_out(fx.window(u, 0.18, 0.7))
        if e_body > 0:
            c.save()
            sb = 0.94 + 0.06 * e_body
            c.translate(self.bx, self.by)
            c.scale(sb, sb)
            c.translate(-self.bx, -self.by)
            c.drawPath(P["outer"], skia.Paint(AntiAlias=True, Shader=self.silver, Alphaf=e_body * A))
            c.drawPath(P["body"], col(0.015, 0.015, 0.015, 1.0 if e_body > 0.5 else e_body * 2))
            c.restore()
        # the gold ring, drawn by a line of light from the top centre both ways
        p = fx.ease_in_out(fx.window(u, 0.0, 0.55))
        if p > 0:
            if p < 0.999:
                for tr in (skia.TrimPathEffect.Make(0.0, p / 2), skia.TrimPathEffect.Make(1 - p / 2, 1.0)):
                    pm = metal(1.0, self.ring_w)
                    pm.setPathEffect(tr)
                    c.drawPath(self.ring_path, pm)
                # the hot heads of the line
                meas = skia.PathMeasure(self.ring_path, True)
                L = meas.getLength()
                for d in (L * p / 2, L * (1 - p / 2)):
                    pos, _ = meas.getPosTan(d)
                    for r, al in ((self.ring_w * 2.2, 0.18), (self.ring_w * 1.0, 0.45), (self.ring_w * 0.45, 0.9)):
                        c.drawCircle(pos.x(), pos.y(), r, col(1.0, 0.93, 0.78, al))
            else:
                c.drawPath(self.ring_path, metal(1.0, self.ring_w))
        # the banner snaps open from the centre on the beat
        e_ban = fx.expo_out(fx.window(u, 0.42, 0.78))
        bnr, rb, fr = P["banner"], P["rim_banner"], P["frame"]
        if e_ban > 0:
            half = (bnr.width() / 2 + rb) * e_ban
            mx = (bnr.left() + bnr.right()) / 2
            c.save()
            c.clipRect(skia.Rect.MakeLTRB(mx - half, bnr.top() - rb - 4, mx + half, bnr.bottom() + rb + 4))
            c.drawRRect(skia.RRect.MakeRectXY(bnr.makeOutset(rb, rb), rb * 0.9, rb * 0.9),
                        skia.Paint(AntiAlias=True, Shader=self.silver, Alphaf=A))
            c.drawRRect(skia.RRect.MakeRectXY(bnr.makeOutset(rb * 0.4, rb * 0.4), rb * 0.5, rb * 0.5), col(0.015, 0.015, 0.015))
            c.drawRRect(skia.RRect.MakeRectXY(bnr, 24, 24), metal())
            c.drawRRect(skia.RRect.MakeRectXY(bnr.makeInset(fr, fr), 14, 14), col(0.02, 0.02, 0.02))
            c.restore()
        # COMBAT rises into the banner, letter by letter
        inner = bnr.makeInset(fr + 2, fr + 2)
        for i, (gp, box) in enumerate(P["combat"]):
            e = fx.expo_out(fx.window(u, 0.58 + 0.055 * i, 0.9 + 0.055 * i))
            if e <= 0:
                continue
            c.save()
            c.clipRect(inner)
            c.translate(0, (1 - e) * box[3] * 1.15)
            c.drawPath(gp, col(0.96, 0.95, 0.93, min(1.0, e * 1.4)))
            c.restore()
        # 040 drops in above
        ring_inner = P["ring_inner"]
        for i, (gp, box) in enumerate(P["o40"]):
            e = fx.expo_out(fx.window(u, 0.9 + 0.06 * i, 1.25 + 0.06 * i))
            if e <= 0:
                continue
            c.save()
            c.clipPath(ring_inner, doAntiAlias=True)
            c.translate(0, -(1 - e) * box[3] * 0.9)
            c.drawPath(gp, col(0.96, 0.95, 0.93, e))
            c.restore()
        # the stars, one per eighth note
        for i, (sx, sy, R) in enumerate(P["stars"]):
            x = fx.window(u, 1.24 + 0.125 * i, 1.24 + 0.125 * i + 0.3)
            if x <= 0:
                continue
            sc = 1 + 0.22 * np.sin(np.pi * min(x * 1.35, 1.0)) * (1 - x) * 2 if x < 1 else 1.0
            sc = fx.ease_out(min(x * 2.2, 1.0)) * sc
            c.save()
            c.translate(sx, sy)
            c.scale(sc, sc)
            c.drawPath(logo040.star_path(0, 0, R * 1.08, 0.42), metal())
            c.restore()
        # a glint crosses the badge
        gx = fx.window(u, 1.78, 2.38)
        if 0 < gx < 1:
            b = P["outer"].computeTightBounds()
            pos = b.left() - 0.3 * b.width() + 1.6 * b.width() * fx.ease_in_out(gx)
            sh = skia.GradientShader.MakeLinear(
                [skia.Point(pos - 95, self.by - 45), skia.Point(pos + 95, self.by + 45)],
                [skia.Color4f(1, 1, 1, 0).toColor(), skia.Color4f(1, 0.95, 0.82, 0.55 * A).toColor(), skia.Color4f(1, 1, 1, 0).toColor()],
                [0.0, 0.5, 1.0])
            gp = skia.Paint(AntiAlias=True, Shader=sh, BlendMode=skia.BlendMode.kSrcATop)
            c.drawRect(b.makeOutset(40, 40), gp)
        c.restore()
        # DEURNE, quietly, under the badge
        e_d = fx.ease_out(fx.window(u, 1.95, 2.6))
        if e_d > 0:
            gfx.text(c, "DEURNE", self.font, self.cx, self.bottom + 0.055 * self.h, OFFWHITE, 0.72 * e_d * A,
                     tracking=0.62 + 0.25 * (1 - e_d), align="center")
        return layer.over(img)


# ── the edit ─────────────────────────────────────────────────────────────────
def build(w=1080, h=1920):
    look = CombatLook(w, h)
    tl = Timeline(w, h, FPS, DURATION, look)
    S = tl.add
    flashes = []                              # gold exposure flashes: (time, strength, length)
    neg = set()                               # single frames printed in negative

    # 0 · the opening, into the kick
    op = Opening()
    S(Shot(0.0, vb(V_KICK, 2), render=op.picture, clips=[op.clip]))
    flashes += [(V_LAND, 0.35, 0.12), (V_KICK, 0.75, 0.2)]
    neg.add(F(V_KICK) + 1)

    # 1 · the intro: kick → throw → her combination → pads into a double-leg → the exchange
    t = vb(V_KICK, 2) - WHIP / 2                                         # the throw lands on the beat
    sp = ramp([(0.0, 1.0), (0.38, 0.5), (0.80, 1.0)])
    S(Shot(t, vb(V_KICK, 4.5) - t, clip("Dilbrien.x.Milosz.scramble.mp4", t_in_for(sp, vb(V_KICK, 4) - t, 1.09), speed=sp,
                                        zoom=(1.5, 1.5), track=[(0, 0.40, 0.29), (0.8, 0.43, 0.32), (1.35, 0.46, 0.40)],
                                        note="the throw, slowed at the top"), trans=("smear", WHIP)))
    t = vb(V_KICK, 4.5)
    S(Shot(t, vb(V_KICK, 7) - t, clip("alternative.shot.mp4", 14.02, zoom=(1.25, 1.32), center=(0.58, 0.42),
                                      note="her combination"), trans=("strobe", 0.13)))
    t = vb(V_KICK, 7)
    sp = ramp([(0.0, 1.0), (0.85, 0.6), (1.40, 1.0)])
    S(Shot(t, vb(V_KICK, 10) - t, clip("action.mp4", 3.95, speed=sp, zoom=(1.0, 1.08), center=(0.5, 0.52),
                                       note="pads with the coach, into a double-leg")))
    t = vb(V_KICK, 10)
    S(Shot(t, BEAT, clip("Milosz.Bilal.KB2.mp4", 8.55, zoom=(1.0, 1.03), note="Milosz and Bilal trade under the wall logo")))
    t = vb(V_KICK, 11)
    void = Void(clip("Milosz.Bilal.KB2.mp4", 9.50, zoom=(1.16, 1.16), note="Bilal's hook freezes → the void"), t)
    S(Shot(t, V_DROP - t, render=void.picture, clips=[void.clip]))

    # 2 · the drop: the achievements, then the range
    t = V_DROP
    S(Shot(t, 2 * BEAT, photo("Milosz.Celebration.jpg", center=(0.40, 0.42), gold=1.0, exposure=-0.15, drop=0.8, decode_zoom=1.2,
                              zfn=lambda lt: (ease_snap(lt, 0.16), 0.40, 0.42), note="Milosz, hand raised"), shake=0.7))
    flashes.append((V_DROP, 0.9, 0.3))
    t = vb(V_DROP, 2)
    S(Shot(t - WHIP / 2, 2 * BEAT + WHIP / 2, photo("Imre.Celebration.jpg", gold=1.0, exposure=-0.1, drop=0.8, decode_zoom=1.2,
                                                    zfn=lambda lt: (ease_snap(lt, 0.12), 0.5, 0.36), note="Imre, arms up"),
           trans=("smear_up", WHIP)))
    t = vb(V_DROP, 4)
    dip = Diptych(w, h)
    S(Shot(t, 2 * BEAT, render=dip.render, clips=dip.clips))
    flashes.append((t + Diptych.SLAM, 0.6, 0.18))
    neg.add(F(t + Diptych.SLAM) + 1)
    t = vb(V_DROP, 6)                                                    # the d'arce roll, slowed at the top
    sp = ramp([(0.0, 1.0), (0.70, 0.45), (1.80, 1.0)])
    S(Shot(t - WHIP / 2, vb(V_DROP, 10) - t + WHIP / 2, clip(
        "Makhachev.D.arce.mp4", 19.30, speed=sp, zoom=(1.16, 1.08), center=(0.5, 0.5),
        echo_fn=lambda lt: 0.4 * fx.smoothstep(0.7, 0.85, lt) * (1 - fx.smoothstep(1.7, 1.9, lt)), note="the d'arce roll"),
        trans=("smear", WHIP)))
    t = vb(V_DROP, 10)
    S(Shot(t, 3 * BEAT, clip("arm.triangle.mp4", 4.30, zoom=(1.45, 1.62), center=(0.42, 0.56), note="the arm-triangle squeeze")))
    t = vb(V_DROP, 13)
    S(Shot(t, BEAT, clip("Milosz.Bilal.mp4", 36.67, zoom=(1.08, 1.12), note="Milosz trading, the room watching"), punch=0.04))
    t = vb(V_DROP, 14)
    S(Shot(t, BEAT, clip("50-50.mp4", 1.45, zoom=(1.12, 1.16), center=(0.55, 0.45), note="50-50"), punch=0.03))
    t = vb(V_DROP, 15)
    S(Shot(t, 2 * BEAT, clip("limited.gi.footage.mp4", 2.20, zoom=(1.2, 1.28), center=(0.45, 0.42), note="the gi: side control")))
    t = vb(V_DROP, 17)
    S(Shot(t, V_BREAK - t, clip("armbar.mp4", 6.90, zoom=(1.12, 1.22), center=(0.5, 0.55), note="the sit-back into the armbar")))

    # 3 · the breakdown: coaching
    t = V_BREAK
    S(Shot(t, 4 * BEAT, clip("uitleg.mma.mp4", 5.0, speed=0.8, zoom=(1.6, 1.72), center=(0.52, 0.33), contrast=0.45,
                              note="a coach on the mat with a pair")))
    t = vb(V_BREAK, 4)
    S(Shot(t, 4 * BEAT, clip("extra.mp4", 8.95, speed=0.7, zoom=(1.12, 1.22), center=(0.5, 0.45), contrast=0.45,
                              note="a kid on the pads with his coach")))
    # the 808: the team
    grp = GroupReveal(w, h)
    S(Shot(V_PRE, V_RETURN - V_PRE, render=grp.picture, clips=[grp.clip]))

    # 4 · the beat returns: a combination under the wall sign, punched in closer; the uppercut
    for k, (s_hit, z, note) in enumerate([(12.16, 1.0, "a combination on the pads"), (15.04, 1.16, "…closer"),
                                          (18.07, 1.34, "…closer")]):
        S(Shot(vb(V_RETURN, k), BEAT, clip("wall.drill.near.the.end.mp4", s_hit - 0.03, zoom=(z, z * 1.02),
                                           center=(0.5, 0.46), note=note), punch=0.03))
    t = vb(V_RETURN, 3)
    out = Outro(w, h, clip("wall.drill.near.the.end.mp4", 8.98 - (V_CARD - t), note="the uppercut → the badge"))
    S(Shot(t, DURATION - t, render=out.picture, clips=[out.clip]))
    tl.overlays.append(Overlay(U0, DURATION, out.draw, "post"))
    flashes.append((V_CARD, 0.8, 0.25))

    def events(t, img, tl):
        for t0, k, d in flashes:
            if t0 <= t < t0 + d * 2.5:
                img = fx.flash(img, k * max(0.0, 1 - (t - t0) / d), GOLD_TINT)
        if F(t) in neg:
            img = (1.0 - img @ fx.LUMA)[..., None] * PAPER
        return img
    tl.overlays.insert(0, Overlay(0.0, DURATION, events, "pre"))
    tl.cinema_at = lambda t: {"mono": 1.0, "streaks": 0.0, "halation": 0.3, "bloom": 0.22, "weave": 0.35}
    tl.grain_at = lambda t: 0.75
    tl.vignette_at = lambda t: 0.55
    tl.audio_plan = sounds()
    return tl


# ── sound ────────────────────────────────────────────────────────────────────
def sounds():
    """Real sound from the floor: (video time, file, src t0, src t1, dB)."""
    combos = [(vb(V_RETURN, k), "wall.drill.near.the.end.mp4", s_hit - 0.03, s_hit + 0.45, -4)
              for k, s_hit in enumerate([12.16, 15.04, 18.07])]
    return [
        (V_KICK - 0.03, "kickboxing.drills.mp4", 21.36, 21.75, -2),         # the kick lands
        (vb(V_KICK, 4) - 0.03, "Dilbrien.x.Milosz.scramble.mp4", 1.06, 1.45, -5),   # the throw lands
        (vb(V_KICK, 4.5) + (14.40 - 14.02), "alternative.shot.mp4", 14.40, 14.98, -5),
        (vb(V_KICK, 7) + 0.03, "action.mp4", 3.98, 4.60, -5),               # the pads
        (vb(V_KICK, 7) + 1.42, "action.mp4", 5.15, 5.45, -6),               # the double-leg lands
        (V_PAUSE - 0.03, "Milosz.Bilal.KB2.mp4", 9.97, 10.45, -1),          # the hook the music stops on
        (vb(V_DROP, 4) + Diptych.SLAM - 0.02, "wall.drill.near.the.end.mp4", 8.95, 9.30, -6),
        (vb(V_DROP, 6) + 0.62, "Makhachev.D.arce.mp4", 19.92, 20.30, -7),   # the roll
        (vb(V_DROP, 13) + (36.90 - 36.67), "Milosz.Bilal.mp4", 36.90, 37.30, -6),
        (vb(V_DROP, 17) + (8.05 - 6.90), "armbar.mp4", 8.04, 8.40, -6),     # the sit-back
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


def _impact(sr, dur=2.2):
    """One clean, deep landing: a sub that drops 58 → 34 Hz under a soft felt transient."""
    from scipy import signal
    t = np.arange(int(dur * sr)) / sr
    f = 34 + 24 * np.exp(-t / 0.18)
    sub = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.55)
    rng = np.random.default_rng(3)
    click = signal.lfilter(*signal.butter(2, 900 / (sr / 2)), rng.standard_normal(len(t))) * np.exp(-t / 0.012) * 0.3
    return ((sub * 0.9 + click)[:, None].repeat(2, 1)).astype(np.float32)


def mix(plan, sr=48000, music=True):
    """The mix. music=False: the floor sounds alone (with the hall tails and the impact), for
    laying a licensed track under it in an ad."""
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
    rev = rev / (np.abs(rev).max() + 1e-6) * 0.35
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
        a[-int(0.06 * sr):] *= np.linspace(1, 0, int(0.06 * sr))[:, None]
        a = a / (np.abs(a).max() + 1e-6) * 10 ** (db / 20) * 0.5
        i = max(0, int(round(v * sr)))
        a = a[:n - i]
        out[i:i + len(a)] += a
        if name == "wall.drill.near.the.end.mp4" and abs(v - (V_CARD - 0.03)) < 1e-6:
            wet = _verb(a, _hall(sr, 2.4))[:n - i] * 0.45                  # the uppercut rings out
            out[i:i + len(wet)] += wet
    # one clean low impact as the banner snaps open
    imp = _impact(sr) * 0.42
    j = int(round((U0 + 0.5) * sr))
    m = min(len(imp), n - j)
    out[j:j + m] += imp[:m]
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
