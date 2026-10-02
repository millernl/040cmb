"""040COMBAT, Deurne — a 38.5-second 9:16 invitation.

Built the way the District98 film is: the brand mark opens as a door into the footage, the first
chapter plays in a 4:5 cinema window that closes in the silence and bursts open on the drop, one
disciplined grade runs through everything, and the film ends on the logo. Here the door is
040COMBAT's own gold octagon and the end is the real badge (work/logo2, the gym's PDF, split into
its vector layers by edits/logo_art.py) building itself after the teams. A few devices come from
the Lemon Haus film: lights powering on over the group, ghosted step-printed memories, flash pops,
and smears between shots.

  0.0–2.0    out of black the gold octagon draws itself and Milosz and his partner come up inside
             it; on the bar line the camera flies through it into the 4:5 window
  2.0–3.0    two steps in on the exchange, on the beat and the eighth note
  3.0–9.0    the kick lands on the first vocal downbeat in slow motion · the throw · her
             combination · pads into a double-leg · an exchange under the wall sign
  9.0–10.0   silence · Bilal's hook freezes, the room falls away, and the window closes to a line
             of warm light that charges up
  10.0–20.1  the line bursts open on the drop: Milosz's hand raised, Imre, then the two of them
             together in black and white, each landing with a camera flash · the d'arce roll · the
             arm-triangle · 50-50 · the gi · the armbar
  20.1–26.1  burn to white → the people · Milosz laughing after the round, warm like film · a coach
             explaining on the mat · then the promotions become memories (black and white,
             step-printed, back in the 4:5 window): a new belt under the wall sign
  26.1–28.1  the 808 · the purple-belt hug, in slow motion, the room cheering
  28.1–30.1  the beat returns, full frame and in colour · the kids sparring in gold gloves · the
             fighter steps in for the uppercut
  30.1–38.5  the uppercut holds and the lights go out · they flicker back on over the no-gi team,
             warm like a real photograph · the camera pulls back and the octagon closes around them ·
             the gi team inside it · the octagon settles and the badge builds itself: banner, COMBAT,
             040, the stars · DEURNE · the outro's four-bar phrase ends, and so does the film. After
             its first line the outro plays without its voice (edits/separate.py)

The grade is edits/look040.py. Footage and photos are read from footage/ (the draft release
'footage'; never in git); the newer clips live in footage/new/.

    python edits/showcase.py                       # renders/040COMBAT_showcase_9x16.mp4
    python edits/showcase.py --no-music-version    # + _no-music.mp4, the floor sounds only (for ads)
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
sys.path.insert(0, str(ROOT / "edits"))
from afterfilm import brand, fx, gfx, matte, media  # noqa: E402
from afterfilm.timeline import Clip, FrameBuffer, Overlay, Shot, Timeline, VideoSource  # noqa: E402
import logo_art  # noqa: E402
import look040  # noqa: E402

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


V_LAND = BAR                                 # 2.006: through the door, on the bar line
V_KICK = 1.5 * BAR                           # 3.009: the kick, on the first vocal downbeat (song bar 8)
V_PAUSE = V_KICK + 3 * BAR                   # 9.026: the intro stops on a bar line
V_DROP = V_PAUSE + 2 * BEAT                  # 10.029
V_BREAK = V_DROP + 5 * BAR                   # 20.057: the people
V_808 = V_BREAK + 3 * BAR                    # 26.074
V_RETURN = V_808 + BAR                       # 28.080
V_CARD = V_RETURN + BAR                      # 30.086: the uppercut
V_TEAM = V_CARD + 2 * BEAT                   # 31.089: the team
PB0, PB1 = vb(V_TEAM, 2.0), vb(V_TEAM, 4.0)  # the pull-back into the octagon
V_GI = vb(V_TEAM, 4.5)                       # the gi team, inside it
U0 = vb(V_GI, 2.0)                           # the badge builds
DURATION = V_CARD + 4 * BAR + 0.45           # the outro's four-bar phrase and its decay; the next one never starts
MEM0 = V_BREAK + 8 * BEAT                    # the promotions, in the 4:5 window and in black and white

# music: (video start, song start, song end, gain dB, fade in s, fade out s)
MUSIC_PLAN = [(0.0, SB(6.5), SB(11), 3.0, 0.6, 0.012),
              (V_DROP, SB(32), SB(37), 0.0, 0.012, 0.012),
              (V_BREAK, SB(52), SB(55), 2.0, 0.012, 0.012),
              (V_808, SB(59), SB(61) + 0.35, 1.0, 0.012, 0.35),
              (V_CARD, SB(106), SB(110) + 0.42, 0.0, 0.04, 0.3)]
INST_FROM = SB(106.9)                        # after "this shit don't pop in the hood": the outro without its voice

# ── sources ──────────────────────────────────────────────────────────────────
_SRC = {}
DENOISE = {"action.mp4": "hqdn3d=2:1.5:4:3", "wall.drill.near.the.end.mp4": "hqdn3d=1.5:1.5:3:3",
           "uitleg.mma.mp4": "hqdn3d=2:1.5:4:3", "armbar.mp4": "hqdn3d=1.5:1.5:3:3"}
SOFT = "hqdn3d=2.5:2:5:4"                     # the WhatsApp-quality clips


def src(name):
    if name not in _SRC:
        soft = SOFT if name.startswith("new/") and "fill-in" not in name else None
        _SRC[name] = VideoSource(str(FOOT / name), rescue=False, extra_pre=DENOISE.get(name, soft))
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


NOMATCH = look040.NOMATCH


class MClip(Clip):
    """A clip with per-frame framing (staccato zoom steps, keyed tracks, or a framing function),
    light trails and the people matte, which the grade uses to sit the room lower. It measures
    itself (look040.measure) when decoded."""

    def setup(self, zsteps=None, track=None, zfn=None, echo_fn=None, use_matte=True, post=None, step=1):
        self.zsteps, self.trackt, self.zfn, self.echo_fn = zsteps, track, zfn, echo_fn
        self.use_matte = use_matte
        self.abuf, self.a0 = None, 0.0
        self.post, self.step, self._held = post, step, {}
        return self

    def prepare(self, dur, fps, size):
        fresh = self.buf is None
        super().prepare(dur, fps, size)
        if fresh and "match" not in self.grade:
            self.grade = {**self.grade, "match": look040.measure(self.buf.frames)}
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
        if self.step > 1 and freeze_at is None:
            # step printing: each instant held for `step` frames, the two before it lingering as
            # ghosts — the long-shutter, under-cranked look of a memory
            q = self.step / FPS
            k = int(np.floor(lt / q + 1e-6))
            for old in [key for key in self._held if key[0] < k - 3]:
                del self._held[old]
            held = []
            for j in (k, k - 1, k - 2):
                key = (j, out_size)
                if key not in self._held:
                    self._held[key] = self._frame(max(j, 0) * q, dur, look, None, out_size, over)
                held.append(self._held[key])
            return 0.6 * held[0] + 0.27 * held[1] + 0.13 * held[2]
        return self._frame(lt, dur, look, freeze_at, out_size, over)

    def _frame(self, lt, dur, look, freeze_at, out_size, over):
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
        out = look.grade(img, **g)
        return self.post(out) if self.post else out


def clip(name, t_in, speed=1.0, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", echo=0.0, decode_zoom=None,
         zsteps=None, track=None, zfn=None, echo_fn=None, post=None, step=1, **grade):
    s = src(name)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    sw, sh = sorted((s.info["width"], s.info["height"]))      # portrait after rotation
    dw = int(min(sw, max(1080, round(1080 * zmax))) // 2 * 2)
    dh = int(min(sh, max(1920, round(1920 * zmax))) // 2 * 2)
    if sw < 1080:                                              # a small source: decode at its own size
        dw, dh = sw // 2 * 2, sh // 2 * 2
    c = MClip(s, t_in, speed=speed, zoom=zoom, center=center, note=note, echo=echo, grade=grade, decode=(dw, dh))
    return c.setup(zsteps, track, zfn, echo_fn, post=post, step=step)


def photo(name, zoom=(1.0, 1.0), center=(0.5, 0.5), note="", decode_zoom=None, zfn=None, zsteps=None,
          use_matte=True, post=None, **grade):
    grade.setdefault("match", NOMATCH)
    zmax = decode_zoom or max([max(zoom)] + [z[1] for z in (zsteps or [])])
    c = MClip(StillSource(name), 0.0, speed=0.0, zoom=zoom, center=center, note=note, grade=grade,
              decode=(int(1080 * zmax * 1.05) // 2 * 2, int(1920 * zmax * 1.05) // 2 * 2))
    return c.setup(zsteps=zsteps, zfn=zfn, use_matte=use_matte, post=post)


# ── the look ─────────────────────────────────────────────────────────────────
GRADE = dict(sat=0.75, skin_keep=1.1, contrast=0.32, high_tint=(0.034, 0.014, -0.031),
             shadow_tint=(-0.008, 0.0035, 0.0115), expo=-0.22, mist=0.18, sep_drop=1.0, sep_desat=0.45,
             toe=0.014, cool_cut=0.4, green_cut=0.3, white=0.96)
INK = np.array([0.014, 0.013, 0.013], np.float32)              # the film's black
CREAM = np.array([1.0, 0.965, 0.90], np.float32)


class Look(fx.Look):
    """look040's grade (match → separation → hue-weighted colour → print curve → split tone →
    diffusion) with the film's settings."""

    def grade(self, img, **kw):
        return look040.grade(img, look=GRADE, **kw)


MONO_W = np.array([0.32, 0.58, 0.10], np.float32)     # skin a little brighter than plain luma


def mono(img, contrast=0.4):
    """The film's black and white, for the moments: silver, deep, a breath of warmth in the
    highlights — after the grade, so the separation and the diffusion carry over."""
    y = np.clip(img @ MONO_W, 0, 1)
    y = y + contrast * (y * y * (3 - 2 * y) - y)
    y = 0.012 + y * (0.955 - 0.012)
    hi = fx.smoothstep(0.45, 1.0, y)[..., None]
    return np.clip(y[..., None] * (1 + hi * np.array([0.0, -0.012, -0.035], np.float32)), 0, 1).astype(np.float32)


def analog(img, k=1.0, halo=0.2):
    """A real photograph: warm, the blacks lifted to a brown paper black, the highlights a creamy
    white, the colour a touch softer, and the lights haloed warm the way film does it."""
    x = np.clip(img, 0, 1).astype(np.float32)
    y = x @ look040.LUMA
    x = y[..., None] + (x - y[..., None]) * 0.9
    x = x * np.array([1.03, 1.0, 0.935], np.float32)
    h, w = y.shape
    hot = cv2.resize(np.clip((y - 0.72) / 0.28, 0, 1), (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(hot, (0, 0), 6 * w / 1080), (w, h), interpolation=cv2.INTER_LINEAR)
    x = x + glow[..., None] * np.array([0.9, 0.38, 0.14], np.float32) * halo
    black = np.array([0.055, 0.043, 0.035], np.float32)
    white = np.array([0.965, 0.945, 0.895], np.float32)
    out = black + np.clip(x, 0, 1) * (white - black)
    return (img + (out - img) * k).astype(np.float32) if k < 1 else out.astype(np.float32)


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


def t_burn(a, b, p, ctx):
    """A chapter change: the outgoing shot overexposes into warm white, a few frames of pure
    light, and the incoming shot comes back out of it."""
    if p < 0.5:
        img, k = a, (p / 0.5) ** 1.6
    else:
        img, k = b, ((1 - p) / 0.5) ** 1.3
    if k < 0.01:
        return img
    lit = np.clip(img * (1 + 3.2 * k), 0, 1)
    return np.clip(lit * (1 - k ** 1.5) + CREAM * k ** 1.5, 0, 1)


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
    return 1.0 + amt * (1 - fx.expo_out(lt / t)) + drift * lt


# ── the badge, on screen ─────────────────────────────────────────────────────
RIM_PT = 230.444                             # the logo's width, page points
BADGE_W = 0.66                               # its width on screen at rest, of the frame width


def _place(rgba, P, S, cx, cy, w, h):
    """A page-space RGBA layer (P × P px, straight alpha) to screen: scale S about the page
    centre, centred at (cx, cy). Returns premultiplied rgb and alpha."""
    m = np.float32([[S, 0, cx - S * P / 2], [0, S, cy - S * P / 2]])
    pre = np.concatenate([rgba[..., :3] * rgba[..., 3:4], rgba[..., 3:4]], -1)
    out = cv2.warpAffine(pre, m, (w, h), flags=cv2.INTER_AREA if S < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    return out[..., :3], out[..., 3:4]


def _place_mask(mask, P, S, cx, cy, w, h):
    m = np.float32([[S, 0, cx - S * P / 2], [0, S, cy - S * P / 2]])
    return cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _over(img, rgb_pre, a, k=1.0):
    return img * (1 - a * k) + rgb_pre * k


# ── the opening: the octagon door ────────────────────────────────────────────
DRAW = (0.22, 0.86)                          # the gold ring draws itself, from the top round to the bottom
FILL = (0.50, 1.05)                          # the footage comes up inside it
FLY0 = 1.42                                  # the fly-through starts
K_FLY = 4.4                                  # how far the octagon grows: past the 4:5 window
F_REST = 0.55                                # the footage's scale inside the octagon at rest
RING_R = 95.0                                # the ring's outer corner radius, page points


def _zoom_blur(layer, cx, cy, ratio, n):
    """A zoom blur: the layer averaged over scales 1 → 1/ratio about (cx, cy), i.e. over the part
    of the frame's exposure when it was still smaller."""
    if n <= 1:
        return layer
    h, w = layer.shape[:2]
    acc = np.zeros_like(layer)
    for sc in ratio ** (-np.linspace(0.0, 1.0, n)):
        m = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        out = cv2.warpAffine(layer, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acc += out.reshape(acc.shape)
    return acc / n


class Door:
    """Out of black the gold octagon draws itself and the footage comes up inside it (the District
    film's glyph door, made 040's); on the bar line the camera flies through it. Inside the door the
    picture is held small, so the two of them read in the octagon; as the camera moves in, the
    picture grows to fill the frame and the ring flies past. The kickboxing shot plays on through
    the landing, the punch-ins and the kick, in one continuous clip."""
    P = 3200

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy = w / 2, h / 2
        self.ring, self.win = logo_art.full_ring(self.P)
        self.S0 = BADGE_W * w / (RIM_PT / 300 * self.P)        # page px → screen px at rest
        self.win_half = self.win.sum(1).max() / 2 * self.S0     # the window's half-width on screen at rest
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.theta = np.abs(np.arctan2(xx - self.cx, -(yy - self.cy))) / np.pi    # 0 at the top, 1 at the bottom
        sp = ramp([(0.0, 0.45), (V_LAND, 1.0), (V_KICK, 0.28), (V_KICK + 0.62, 1.0)])

        def framing(lt):
            # landed wide; then two steps in on the exchange (both faces, the gloves on the
            # pads), and back out wide for the kick
            if lt < V_LAND:
                return 1.0, 0.5, 0.5
            if lt < vb(V_LAND, 1.0):
                return 1.0 + 0.04 * (lt - V_LAND), 0.5, 0.5
            if lt < vb(V_LAND, 1.5):
                return 1.28, 0.5, 0.48
            if lt < V_KICK:
                return 1.5, 0.6, 0.45                  # with him as he steps back to load the kick
            return 1.04 * (1 + 0.05 * (lt - V_KICK)), 0.5, 0.5
        self.clip = clip("kickboxing.drills.mp4", t_in_for(sp, V_KICK, 21.39), speed=sp, zfn=framing, decode_zoom=1.7,
                         echo_fn=lambda lt: 0.75 * fx.smoothstep(V_KICK, V_KICK + 0.08, lt) * (1 - fx.smoothstep(V_KICK + 0.65, V_KICK + 0.85, lt)),
                         note="the octagon door → punch-ins → the kick lands")

    def k_at(self, lt):
        """The octagon's scale (1 = at rest)."""
        if lt < FLY0:
            return 0.94 + 0.06 * fx.expo_out(fx.window(lt, DRAW[0], FILL[1])) + 0.02 * max(lt - FILL[1], 0)
        k0 = 1.0 + 0.02 * (FLY0 - FILL[1])
        return k0 * (K_FLY / k0) ** (fx.window(lt, FLY0, V_LAND) ** 2.2)

    def f_at(self, k):
        """The footage's scale for an octagon at scale k: it travels with the octagon, eases into
        full frame, and always covers the window."""
        x = F_REST * k
        soft = x / (1 + x ** 6) ** (1 / 6)
        cover = (self.win_half * k + 3) / (self.w / 2)
        return float(min(1.0, max(soft, cover)))

    def picture(self, shot, lt, tl):
        w, h, cx, cy = self.w, self.h, self.cx, self.cy
        if lt < DRAW[0]:
            return np.broadcast_to(INK, (h, w, 3)).copy()
        img = self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        if lt >= V_LAND:
            return img
        k, kp = self.k_at(lt), self.k_at(lt - 0.5 / FPS)
        f, fp = self.f_at(k), self.f_at(kp)
        # the picture, held inside the octagon (anchored a little high, on their faces)
        if f < 0.999:
            ay = 0.5 - 0.035 * (1 - f) / (1 - F_REST)
            m = np.float32([[f, 0, cx - f * 0.5 * w], [0, f, cy - f * ay * h]])
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img = _zoom_blur(img, cx, cy, f / fp, int(np.clip(0.5 * w * (f / fp - 1) / 1.5, 1, 16)))
        # the ring and its window at this frame's scale, zoom-blurred over the shutter
        S = self.S0 * k
        win = _place_mask(self.win, self.P, S, cx, cy, w, h)
        rgb, a = _place(self.ring, self.P, S, cx, cy, w, h)
        ratio = k / kp
        n = int(np.clip(RING_R * self.P / 300 * S * (ratio - 1) / 1.5, 1, 24))
        if n > 1:
            ring = _zoom_blur(np.concatenate([rgb, a], -1), cx, cy, ratio, n)
            rgb, a = ring[..., :3], ring[..., 3:4]
            win = _zoom_blur(win, cx, cy, ratio, n)
        # the ring draws itself: both ways from the top, meeting at the bottom
        d = fx.ease_in_out(fx.window(lt, *DRAW))
        draw = np.clip((d * 1.04 - self.theta) / 0.04, 0, 1)[..., None]
        fill = fx.ease_in_out(fx.window(lt, *FILL))
        out = INK + (img - INK) * (win[..., None] * fill)
        return np.clip(out * (1 - a * draw) + rgb * draw, 0, 1)


# ── a frame that holds ───────────────────────────────────────────────────────
class Hold:
    """A frame that freezes on a beat and holds while the room falls away around the people and
    the camera keeps drifting in. Bilal's hook (into the slit) and the last uppercut."""

    def __init__(self, clip, start, t_freeze, push=0.06, centre=(0.5, 0.5), dim=0.8, out=None, edge=4):
        self.clip, self.start, self.tf = clip, start, t_freeze
        self.push, self.centre, self.dim, self.out, self.edge = push, centre, dim, out, edge

    def picture(self, shot, lt, tl):
        w, h = tl.w, tl.h
        fz = self.tf - self.start
        if lt < fz:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        k = lt - fz
        img = self.clip.frame(fz, shot.dur, tl.look, out_size=(w, h))
        a = cv2.GaussianBlur(self.clip.alpha(fz, shot.dur, (w, h)), (0, 0), self.edge * w / 1080)[..., None]
        fall = fx.ease_out(fx.window(k, 0.05, 0.6)) * self.dim
        out = INK + (img - INK) * (1 - fall * (1 - a))
        if self.out:                                    # the people go too: the lights are off
            out = INK + (out - INK) * (1 - fx.ease_in(fx.window(shot.start + lt, *self.out)))
        return fx.reframe(out, 1.0 + self.push * fx.ease_out(k / 1.2), *self.centre)


class Burst:
    """The drop: the slit bursts open on Milosz's photograph, the camera snapping back from close
    with a zoom blur on the first frames."""
    SNAP, T = 0.22, 0.35

    def __init__(self, w, h):
        self.w, self.h = w, h
        z = lambda lt: (1.0 + self.SNAP * (1 - fx.expo_out(lt / self.T)) + 0.02 * lt, 0.40, 0.42)  # noqa: E731
        self.z = z
        self.clip = photo("Milosz.Celebration.jpg", exposure=0.12, sep_drop=0.5, decode_zoom=1.3, zfn=z,
                          note="Milosz, hand raised")

    def picture(self, shot, lt, tl):
        img = self.clip.frame(lt, shot.dur, tl.look, out_size=(self.w, self.h))
        ratio = self.z(lt)[0] / self.z(max(lt - 0.5 / FPS, 0))[0]
        n = int(np.clip(abs(np.log(ratio)) * self.h * 0.6 / 1.5, 1, 16))
        return _zoom_blur(img, self.w * 0.40, self.h * 0.42, ratio, n) if n > 1 else img


class Champions:
    """The two achievements again, now together and in black and white: Milosz drops in from
    above on the beat, Imre rises from below on the eighth note, each landing with a camera flash
    and a smear of motion; on the next beat the pair jumps a step closer."""
    SLIDE = 0.12
    LEAD = 0.10                               # the shot starts this early so Milosz lands on the beat

    def __init__(self, w, h):
        self.w, self.h = w, h
        g = {"match": NOMATCH, "exposure": 0.18, "sep_drop": 0.5}
        self.top = MClip(StillSource("Milosz.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.42, 0.33),
                         decode=(w * 2, h), grade=dict(g)).setup(post=mono)
        self.bot = MClip(StillSource("Imre.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.04), center=(0.5, 0.28),
                         decode=(w * 2, h), grade=dict(g)).setup(post=mono)
        self.clips = [self.top, self.bot]

    def _half(self, img, lt, land, sign):
        """One half sliding into its slot (sign -1: from above, +1: from below), its offset in px."""
        hh = img.shape[0]
        x = fx.window(lt, land - self.SLIDE, land)
        off = sign * hh * (1 - fx.expo_out(x)) if x < 1 else 0.0
        if 0 < x < 1:
            x2 = fx.window(lt - 1 / FPS, land - self.SLIDE, land)
            v = abs(hh * (fx.expo_out(x) - fx.expo_out(x2)))          # px moved this frame
            k = int(v * 0.6)
            if k > 1:
                img = cv2.blur(img, (1, k))
        if lt >= land:                                                  # the flash
            f = np.exp(-(lt - land) / 0.07)
            img = img + (1 - img) * (0.85 * f)
        return img, int(round(off)) if x > 0 else None

    def render(self, shot, lt, tl):
        w, h = self.w, self.h
        hh = h // 2
        a = self.top.frame(lt, shot.dur, tl.look, out_size=(w, hh))
        b = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, h - hh))
        img = np.broadcast_to(INK, (h, w, 3)).copy()
        a, oa = self._half(a, lt, self.LEAD, -1)
        b, ob = self._half(b, lt, self.LEAD + BEAT / 2, 1)
        if oa is not None and -oa < hh:
            img[0:hh + oa] = a[-oa:hh] if oa < 0 else a[:hh]
        if ob is not None and ob < h - hh:
            img[hh + ob:h] = b[0:h - hh - ob]
        line = max(2, int(round(3 * w / 1080)))
        img[hh - line:hh + line] = INK
        # on the next beat: a step closer, and a softer flash across both
        u = lt - (self.LEAD + BEAT)
        if u >= 0:
            img = fx.reframe(img, 1.0 + 0.035 * fx.expo_out(min(u / 0.12, 1.0)), 0.5, 0.5)
            f = np.exp(-u / 0.08) * 0.35
            img = img + (1 - img) * f
        return np.clip(img, 0, 1)


# ── the end: the team, the octagon closes around them, the badge builds ─────
def _gym_backdrop(w, h):
    """The empty gym (Gym.jpg), out of focus and low, its ceiling tubes glowing: the ground the
    badge stands on, barely there."""
    im = np.asarray(ImageOps.exif_transpose(Image.open(FOOT / "Gym.jpg")).convert("RGB")).astype(np.float32) / 255
    ih, iw = im.shape[:2]
    s = h * 1.1 / ih
    big = cv2.resize(im, (int(iw * s), int(h * 1.1)), interpolation=cv2.INTER_AREA)
    x0 = int(0.42 * big.shape[1] - w * 0.55)
    crop = big[int(h * 0.05):int(h * 0.05) + h, max(0, x0):max(0, x0) + w]
    k = w / 1080
    y = crop @ look040.LUMA
    hot = cv2.GaussianBlur(np.clip((y - 0.82) / 0.18, 0, 1), (0, 0), 26 * k)
    soft = cv2.GaussianBlur(crop, (0, 0), 14 * k)
    g = look040.grade(soft, look={**GRADE, "mist": 0.0, "expo": -2.9, "toe": 0.0, "sat": 0.5})
    g = g + hot[..., None] * np.array([1.0, 0.86, 0.66], np.float32) * 0.07
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g = g * (0.5 + 0.5 * np.exp(-((yy - 0.42) / 0.5) ** 2)) * 0.36        # barely there: ~5% grey at most
    return np.clip(g, 0, 1).astype(np.float32)


class Elem:
    """A screen-space piece of the badge (premultiplied rgb + alpha, cropped to its box)."""

    def __init__(self, rgb_pre, a):
        ys, xs = np.nonzero(a[..., 0] > 1e-3)
        if len(xs) == 0:
            self.empty = True
            return
        self.empty = False
        pad = 2
        self.y0, self.y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
        self.x0, self.x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
        self.rgb = rgb_pre[self.y0:self.y1, self.x0:self.x1].copy()
        self.a = a[self.y0:self.y1, self.x0:self.x1].copy()
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def draw(self, img, k=1.0, dy=0.0, scale=1.0, clip=None, xclip=None):
        """Composite over img. clip: (y0, y1) screen rows it may show in; xclip: (x0, x1)."""
        if self.empty or k <= 0.002 or scale <= 0.01:
            return img
        H, W = img.shape[:2]
        rgb, a = self.rgb, self.a
        if abs(scale - 1) > 1e-3 or abs(dy) > 0.25:
            hh, ww = a.shape[:2]
            m = np.float32([[scale, 0, ww / 2 * (1 - scale)], [0, scale, hh / 2 * (1 - scale) + dy]])
            rgb = cv2.warpAffine(rgb, m, (ww, hh), flags=cv2.INTER_LINEAR)
            a = cv2.warpAffine(a, m, (ww, hh), flags=cv2.INTER_LINEAR)[..., None]
        y0, y1, x0, x1 = self.y0, self.y1, self.x0, self.x1
        sub_a = a * k
        if clip is not None:
            rows = np.arange(y0, y1)[:, None, None]
            sub_a = sub_a * ((rows >= clip[0]) & (rows < clip[1]))
        if xclip is not None:
            cols = np.arange(x0, x1)[None, :, None]
            sub_a = sub_a * np.clip(np.minimum(cols - xclip[0], xclip[1] - cols) / 1.5 + 0.5, 0, 1)
        region = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = region * (1 - sub_a) + rgb * (sub_a / np.maximum(a, 1e-6)) * (a > 1e-6)
        return img


LIGHTS = [0.0, 0.0, 0.2, 0.22, 0.04, 0.0, 0.58, 0.6, 0.3, 0.92, 1.0, 1.05, 1.03, 1.01]   # per frame from V_TEAM
CLUNKS = [(2, 0.16), (6, 0.2), (9, 0.3)]                        # (frame, gain): the banks of lights going on


def lights(u):
    """The gym's lights coming on, fluorescent: a flicker, a false start, then on."""
    n = int(np.floor(u * FPS + 1e-6))
    return LIGHTS[n] if 0 <= n < len(LIGHTS) else (1.0 if n >= 0 else 0.0)


class Finale:
    """The uppercut has held and the lights have gone; then they come back on, flickering, on the
    no-gi team (the Lemon Haus group reveal). The camera
    pulls back and the gold octagon closes around them; inside it, on the beat, the gi team. The
    octagon settles to the size of the badge, the window goes black, and the badge builds itself
    from the logo's own layers: the banner opens, COMBAT rises letter by letter, 040 drops in, the
    stars land on the eighth notes, and DEURNE settles underneath. The empty gym is there behind
    it, barely."""
    P = 1800
    K_MID = 1.75                              # the octagon's scale while the teams are in it

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx, self.cy = w / 2, h * 0.47
        self.S0 = BADGE_W * w / (RIM_PT / 300 * self.P)
        self.L = logo_art.layers(self.P)
        self.ring, self.win = logo_art.full_ring(self.P)
        self.rim_oct, self.rim_wings = logo_art.rim_parts(self.P)
        self.backdrop = _gym_backdrop(w, h)
        self.nogi = photo("Of.voor.communityy.jpg", zfn=lambda lt: (1.0 + 0.04 * min(lt / 2.0, 1.0), 0.5, 0.5),
                          use_matte=False, exposure=0.12, post=lambda x: analog(x, 1.0, halo=0.1),
                          note="the no-gi team under the wall sign")
        self.win_half = self.win.sum(1).max() / 2 * self.S0     # the window's half-width on screen at rest
        self.gi = photo("new/Gi.team.jpg", zoom=(1.0, 1.03), center=(0.5, 0.5), decode_zoom=1.1, use_matte=False,
                        exposure=0.12, post=lambda x: analog(x, 1.0, halo=0.1), note="the gi team, inside the octagon")
        self.clips = [self.nogi, self.gi]
        self.font = brand.font("BarlowCondensed-SemiBold", 0.030 * h)
        self.type_layer = gfx.Layer(w, h)
        # the badge at rest, screen space
        R = lambda rgba: _place(rgba, self.P, self.S0, self.cx, self.cy, w, h)  # noqa: E731
        self.rest = {n: R(self.L[n]) for n in ("body", "banner", "bits", "o40", "combat", "stars")}
        self.rest["ring"] = R(self.ring)
        self.rest["rim_oct"], self.rest["rim_wings"] = R(self.rim_oct), R(self.rim_wings)
        _, frame = logo_art.ring_and_frame(self.P)
        self.rest["frame"] = R(frame)
        self.win_rest = _place_mask(self.win, self.P, self.S0, self.cx, self.cy, w, h)[..., None]
        self.letters = self._split("combat")
        self.digits = self._split("o40")
        self.starl = self._split("stars")
        s = self.S0 * self.P / 300
        bx0, by0, bx1, by1 = logo_art.BANNER_PT
        self.banner_rows = (self.cy + (by0 + 1.5 - 150) * s, self.cy + (by1 - 1.5 - 150) * s)
        self.banner_half = (bx1 - bx0) / 2 * s + 10
        self.badge_bottom = self.cy + (242.44 - 150) * s

    def _split(self, name):
        from scipy import ndimage
        rgb, a = self.rest[name]
        lab, n = ndimage.label(a[..., 0] > 0.02, structure=np.ones((3, 3)))
        parts = []
        for i in range(1, n + 1):
            m = ndimage.binary_dilation(lab == i, iterations=2)[..., None].astype(np.float32)
            e = Elem(rgb * m, a * m)
            if not e.empty and e.a.size > 40:
                parts.append(e)
        return sorted(parts, key=lambda e: e.cx)

    def _layer(self, img, name, k=1.0, xclip=None):
        rgb, a = self.rest[name]
        if xclip is not None:
            cols = np.arange(self.w)[None, :, None]
            a2 = a * np.clip(np.minimum(cols - xclip[0], xclip[1] - cols) / 1.5 + 0.5, 0, 1)
            return img * (1 - a2 * k) + rgb * (a2 / np.maximum(a, 1e-6)) * (a > 1e-6) * k
        return _over(img, rgb, a, k)

    def k_at(self, t):
        """The octagon's scale: it closes in from outside the frame, holds round the teams while
        drifting back, and settles to the badge's size just before it builds."""
        if t < PB1:
            return 5.4 * (self.K_MID / 5.4) ** fx.ease_in_out(fx.window(t, PB0, PB1))
        s0 = U0 - 0.65
        k1 = self.K_MID * (1 - 0.03 * fx.window(t, PB1, s0))
        return k1 * (1 / k1) ** fx.ease_in_out(fx.window(t, s0, U0 + 0.05))

    def f_at(self, k):
        """The no-gi photo's scale: full frame until the window is about to cut into it, then just
        wider than the window."""
        return float(min(1.0, (self.win_half * k + 2) / (self.w / 2)))

    def picture(self, shot, lt, tl):
        w, h, cx, cy = self.w, self.h, self.cx, self.cy
        t = shot.start + lt
        if t < PB0:
            img = self.nogi.frame(lt, shot.dur, tl.look, out_size=(w, h))
            L = lights(t - V_TEAM)
            if L >= 1.0:
                return np.clip(img * L, 0, 1)
            # the highlights (the tubes, the white mats) catch first
            return np.maximum(np.clip(img, 0, 1) ** (1 + 2.5 * (1 - L)) * L, INK)
        k = self.k_at(t)
        bg_k = fx.ease_in_out(fx.window(t, PB0 + 0.1, PB1 + 0.25))
        out = INK + (self.backdrop - INK) * bg_k
        if abs(k - 1) > 1e-3:
            S, P = self.S0 * k, self.P
            for lay in (self.rim_oct, self.L["body"]):
                out = _over(out, *_place(lay, P, S, cx, cy, w, h))
            win = _place_mask(self.win, P, S, cx, cy, w, h)[..., None]
            ring = _place(self.ring, P, S, cx, cy, w, h)
        else:
            out = _over(out, *self.rest["rim_oct"])
            out = _over(out, *self.rest["body"])
            win, ring = self.win_rest, self.rest["ring"]
        # the photo inside the window
        u = t - U0
        if t < V_GI:
            f = self.f_at(k)
            q = (1 - f) / (1 - self.f_at(self.K_MID))
            ph = self.nogi.frame(lt, shot.dur, tl.look, out_size=(w, h))
            ay, ty = 0.5 + 0.015 * q, h / 2 + (cy - h / 2) * q         # the team's rows in the window's full-width band
            m = np.float32([[f, 0, cx - f * 0.5 * w], [0, f, ty - f * ay * h]])
            ph = cv2.warpAffine(ph, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            n = int(self.win_half * 2 * k) + 10
            n += n % 2
            g = self.gi.frame(lt, shot.dur, tl.look, out_size=(n, n))
            # lift the team into the window's full-width band; the mats continue below
            g = cv2.warpAffine(g, np.float32([[1, 0, 0], [0, 1, -0.08 * n]]), (n, n), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            ph = np.broadcast_to(INK, (h, w, 3)).copy()
            y0, x0 = int(round(cy - n / 2)), int(round(cx - n / 2))
            ph[max(y0, 0):y0 + n, max(x0, 0):x0 + n] = g[max(-y0, 0):, max(-x0, 0):][:h - max(y0, 0), :w - max(x0, 0)]
        wk = win * (1 - fx.ease_in_out(fx.window(u, 0.0, 0.45)))
        out = out * (1 - wk) + ph * wk
        out = _over(out, *ring)
        if u < 0:
            return np.clip(out, 0, 1)
        # the badge builds
        eb = fx.expo_out(fx.window(u, 0.30, 0.68))
        half = self.banner_half * eb
        xc = (cx - half, cx + half)
        for name in ("rim_wings", "banner", "frame", "bits"):
            out = self._layer(out, name, xclip=xc)
        clip_b = (int(self.banner_rows[0]), int(self.banner_rows[1]))
        for i, L in enumerate(self.letters):
            x = fx.expo_out(fx.window(u, 0.46 + 0.05 * i, 0.80 + 0.05 * i))
            if x > 0:
                out = L.draw(out, k=min(1.0, x * 1.6), dy=(1 - x) * (L.y1 - L.y0) * 1.1, clip=clip_b)
        clip_o = (0, int(self.banner_rows[0]) - 4)
        for i, L in enumerate(self.digits):
            x = fx.expo_out(fx.window(u, 0.85 + 0.06 * i, 1.20 + 0.06 * i))
            if x > 0:
                out = L.draw(out, k=x, dy=-(1 - x) * (L.y1 - L.y0) * 0.9, clip=clip_o)
        for i, L in enumerate(self.starl):
            x = fx.window(u, 1.25 + 0.125 * i, 1.6 + 0.125 * i)
            if x > 0:
                sc = fx.ease_out(min(x * 2.4, 1.0)) * (1 + 0.06 * np.sin(np.pi * min(x * 1.2, 1.0)) * (1 - x))
                out = L.draw(out, k=min(1.0, x * 3), scale=sc)
        z = self.push(u)
        if z > 1.0005:                                  # the camera keeps breathing in on the badge
            m = np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]])
            out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return np.clip(out, 0, 1)

    @staticmethod
    def push(u):
        return 1.0 + 0.03 * fx.smoothstep(0.3, DURATION - U0, u)

    def draw_type(self, t, img, tl):
        """DEURNE, quietly, under the badge — drawn after the film pass so it stays crisp."""
        u = t - U0
        e = fx.ease_out(fx.window(u, 2.0, 2.7))
        fade = 1 - fx.ease_in(fx.window(t, DURATION - 0.7, DURATION - 0.05))
        if e <= 0:
            return img if fade >= 1 else INK + (img - INK) * fade
        layer = self.type_layer                 # its own layer: the timeline composites tl.layer again after this
        layer.clear()
        y = self.cy + (self.badge_bottom + 0.06 * self.h - self.cy) * self.push(u)
        gfx.text(layer.c, "DEURNE", self.font, self.cx, y, (236, 230, 220),
                 0.7 * e, tracking=0.58 + 0.22 * (1 - e), align="center")
        out = layer.over(img)
        return INK + (out - INK) * fade


# ── the edit ─────────────────────────────────────────────────────────────────
def build(w=1080, h=1920):
    look = Look(w, h)
    tl = Timeline(w, h, FPS, DURATION, look)
    S = tl.add
    flashes = []                              # soft exposure flashes: (time, strength, length)

    # 0 · the door, the punch-ins, the kick
    door = Door(w, h)
    S(Shot(0.0, vb(V_KICK, 2), render=door.picture, clips=[door.clip]))
    flashes += [(V_KICK, 0.7, 0.2)]

    # 1 · the intro: kick → throw → her combination → pads into a double-leg → the exchange
    t = vb(V_KICK, 2) - WHIP / 2                                         # the throw lands on the beat
    sp = ramp([(0.0, 1.0), (0.38, 0.5), (0.80, 1.0)])
    S(Shot(t, vb(V_KICK, 4.5) - t, clip("Dilbrien.x.Milosz.scramble.mp4", t_in_for(sp, vb(V_KICK, 4) - t, 1.09), speed=sp,
                                        zoom=(1.5, 1.5), track=[(0, 0.40, 0.29), (0.8, 0.43, 0.32), (1.35, 0.46, 0.38)],
                                        mist=0.08, lift=0.03, note="the throw, slowed at the top"), trans=("smear", WHIP)))
    t = vb(V_KICK, 4.5)
    S(Shot(t, vb(V_KICK, 7) - t, clip("alternative.shot.mp4", 14.02, zoom=(1.25, 1.32), center=(0.58, 0.45),
                                      mist=0.06, sep_drop=0.6, lift=0.03, note="her combination")))
    t = vb(V_KICK, 7)
    sp = ramp([(0.0, 1.0), (0.85, 0.6), (1.40, 1.0)])
    S(Shot(t, vb(V_KICK, 10) - t, clip("action.mp4", 3.95, speed=sp, zoom=(1.08, 1.14), center=(0.5, 0.56),
                                       lift=0.055, sep_drop=0.75, exposure=0.1, note="pads with the coach, into a double-leg")))
    t = vb(V_KICK, 10)
    S(Shot(t, BEAT, clip("Milosz.Bilal.KB2.mp4", 8.55, zoom=(1.12, 1.15), center=(0.5, 0.42),
                         lift=0.055, sep_drop=0.75, exposure=0.1, note="Milosz and Bilal trade under the wall sign")))
    t = vb(V_KICK, 11)
    hook = Hold(clip("Milosz.Bilal.KB2.mp4", 9.50, zoom=(1.2, 1.2), center=(0.5, 0.46), lift=0.055, sep_drop=0.75, exposure=0.1,
                     note="Bilal's hook freezes"), t, V_PAUSE, push=0.08, centre=(0.5, 0.5))
    S(Shot(t, V_DROP - t, render=hook.picture, clips=[hook.clip]))

    # 2 · the drop: the achievements, then the range on the mats
    # the slit has closed to a line of light; on the drop it bursts open on the photographs
    burst = Burst(w, h)
    S(Shot(V_DROP, 2 * BEAT, render=burst.picture, clips=[burst.clip]))
    flashes.append((V_DROP, 0.9, 0.2))
    t = vb(V_DROP, 2)
    S(Shot(t - WHIP / 2, 2 * BEAT + WHIP / 2 - Champions.LEAD, photo("Imre.Celebration.jpg", exposure=0.12, sep_drop=0.5, decode_zoom=1.2,
                                                    zfn=lambda lt: (ease_snap(lt, 0.08), 0.5, 0.36), note="Imre, arms up"),
           trans=("smear_up", WHIP)))
    t = vb(V_DROP, 4) - Champions.LEAD
    champs = Champions(w, h)
    S(Shot(t, vb(V_DROP, 6) - t, render=champs.render, clips=champs.clips))
    t = vb(V_DROP, 6)                                                    # the d'arce roll, slowed at the top
    sp = ramp([(0.0, 1.0), (0.70, 0.45)])
    S(Shot(t - WHIP / 2, 3 * BEAT + WHIP / 2, clip(
        "Makhachev.D.arce.mp4", 19.30, speed=sp, zoom=(1.16, 1.1), center=(0.5, 0.5),
        echo_fn=lambda lt: 0.3 * fx.smoothstep(0.7, 0.85, lt), note="the d'arce roll"), trans=("smear", WHIP)))
    t = vb(V_DROP, 9)
    S(Shot(t, 3 * BEAT, clip("arm.triangle.mp4", 4.30, zoom=(1.45, 1.6), center=(0.42, 0.56), note="the arm-triangle squeeze")))
    t = vb(V_DROP, 12)
    S(Shot(t - WHIP / 2, 2 * BEAT + WHIP / 2, clip("50-50.mp4", 3.55 - WHIP / 2, zoom=(1.1, 1.15), center=(0.5, 0.5),
                                                   note="50-50: the coach takes the legs"), trans=("smear", WHIP)))
    t = vb(V_DROP, 14)
    S(Shot(t, 3 * BEAT, clip("limited.gi.footage.mp4", 1.9, zoom=(1.18, 1.26), center=(0.45, 0.45), note="the gi: side control")))
    t = vb(V_DROP, 17)
    S(Shot(t - WHIP / 2, V_BREAK + BURN / 2 - t + WHIP / 2, clip("armbar.mp4", 6.90 - WHIP / 2, zoom=(1.12, 1.22),
                                                                  center=(0.5, 0.55), note="the sit-back into the armbar"),
           trans=("smear_up", WHIP)))

    # 3 · the people
    t = V_BREAK - BURN / 2
    sp = ramp([(0.0, 1.0), (0.45 + BURN / 2, 0.6)])
    S(Shot(t, 4 * BEAT + BURN / 2, clip("Milosz.Bilal.KB2.mp4", 19.55 - BURN / 2, speed=sp, zoom=(1.66, 1.74),
                                        track=[(0, 0.29, 0.34), (2.4, 0.37, 0.33)],
                                        post=lambda x: analog(x, 0.85, halo=0.04), mist=0.08,
                                        note="Milosz laughing after the round, warm like film"), trans=("burn", BURN)))
    t = vb(V_BREAK, 4)
    S(Shot(t, 4 * BEAT, clip("uitleg.mma.mp4", 5.0, speed=0.8, zoom=(1.6, 1.72), center=(0.52, 0.33),
                              note="a coach on the mat with a pair")))
    # the promotions: phone clips, so they become memories — black and white, step-printed, in
    # the 4:5 window
    S(Shot(MEM0, 4 * BEAT, clip("new/Graduatie.mp4", 0.45, speed=0.75, zoom=(1.46, 1.52), center=(0.615, 0.67),
                                post=mono, step=3, mist=0.4, note="a new belt and a hug under the wall sign")))
    S(Shot(V_808, BAR, clip("new/WhatsApp_Video_2026-08-24_at_20.41.55.mp4", 6.95, speed=0.6, zoom=(1.35, 1.42),
                            center=(0.47, 0.40), post=mono, step=3, mist=0.4, note="the purple-belt hug, in slow motion")))

    # 4 · the beat returns: the kids in gold gloves, then the fighter steps in for the uppercut
    S(Shot(V_RETURN, 2 * BEAT, clip("new/fill-in.mp4", 0.60, zoom=(1.04, 1.08), center=(0.55, 0.5), echo=0.3,
                                    note="the kids sparring, gold gloves")))
    t = vb(V_RETURN, 2)
    up = Hold(clip("wall.drill.near.the.end.mp4", 8.98 - (V_CARD - t), mist=0.0, exposure=-0.15, sep_drop=0.6, white=0.93,
                   note="he steps in: the uppercut holds"), t, V_CARD,
              push=0.05, centre=(0.5, 0.42), dim=0.45, out=(V_TEAM - 0.32, V_TEAM - 0.03), edge=1.5)
    S(Shot(t, V_TEAM - t, render=up.picture, clips=[up.clip]))
    flashes.append((V_CARD, 0.12, 0.15))
    up_t = (t, V_TEAM)

    # 5 · the finale: the lights come on over the team
    fin = Finale(w, h)
    S(Shot(V_TEAM, DURATION - V_TEAM, render=fin.picture, clips=fin.clips))
    tl.overlays.append(Overlay(U0, DURATION, fin.draw_type, "top"))

    def events(t, img, tl):
        for t0, k, d in flashes:
            if t0 <= t < t0 + d * 2.5:
                img = fx.flash(img, k * max(0.0, 1 - (t - t0) / d), tuple(CREAM))
        return img
    tl.overlays.insert(0, Overlay(0.0, DURATION, events, "pre"))

    # the 4:5 window: on through the first chapter, closing to a slit in the silence; the drop
    # bursts it open (the burn's white peak hides the snap)
    full_bar = (h - w / 0.8) / 2
    a_line = (h / 2 - 1.5 * h / 1920) / full_bar                # the window closed to a 3-px line

    def bars(t):
        if MEM0 <= t < V_RETURN:                       # the promotions, back in the window
            return fx.ease_out(fx.window(t, MEM0, MEM0 + 0.1))
        if t >= V_DROP:                                # the line bursts open
            return a_line * (1 - fx.expo_out(fx.window(t, V_DROP, V_DROP + 0.12)))
        return 1.0 + (a_line - 1.0) * fx.ease_in(fx.window(t, V_PAUSE + 0.10, V_DROP - 0.12)) ** 0.8
    tl.bars = bars
    tl.bar_ratio = 0.8

    def line(t, img, tl):
        """As the window closes the last of the picture becomes a line of warm light, which
        charges up and bursts with the drop."""
        win = h - 2 * min(int(round(full_bar * bars(t))), h // 2 + 1)
        k = (1 - np.clip(win / (0.07 * h), 0, 1)) ** 2
        if t >= V_DROP:
            k = np.exp(-(t - V_DROP) / 0.05)
        if k < 0.01:
            return img
        k = k * (1 + 0.6 * fx.smoothstep(V_DROP - 0.12, V_DROP, t))
        y = np.arange(h, dtype=np.float32)[:, None] - h / 2
        x = np.arange(w, dtype=np.float32)[None, :] - w / 2
        sx = 0.75 + 0.25 * np.exp(-(x / (0.45 * w)) ** 2)
        core = np.exp(-(y / (1.6 * h / 1920)) ** 2)
        halo = np.exp(-(y / (14 * h / 1920 * (1 + k))) ** 2) * 0.35
        glow = np.clip((core + halo) * sx * k, 0, 1.5)[..., None]
        return np.clip(img + glow * np.array([1.0, 0.93, 0.80], np.float32), 0, 1)
    tl.overlays.append(Overlay(V_PAUSE, V_DROP + 0.25, line, "top"))
    tl.cinema_at = lambda t: {"mono": 0.0, "streaks": 0.0,
                              "halation": 0.08 if up_t[0] <= t < up_t[1] else (0.25 if t < PB1 else 0.12), "bloom": 0.0,
                              "weave": 0.25 if t < PB1 else 0.0}
    mono_t = [(vb(V_DROP, 4) - Champions.LEAD, vb(V_DROP, 6)), (MEM0, V_RETURN)]
    tl.grain_at = lambda t: 0.85 if any(a <= t < b for a, b in mono_t) else (0.55 if t < PB1 else 0.35)
    tl.vignette_at = lambda t: 0.35 if t < PB1 else 0.2
    tl.audio_plan = sounds()
    return tl


# ── sound ────────────────────────────────────────────────────────────────────
def sounds():
    """Real sound from the floor: (video time, file, src t0, src t1, dB)."""
    return [
        (V_KICK - 0.03, "kickboxing.drills.mp4", 21.36, 21.75, -1),         # the kick lands
        (vb(V_KICK, 4) - 0.03, "Dilbrien.x.Milosz.scramble.mp4", 1.06, 1.45, -6),   # the throw lands
        (vb(V_KICK, 4.5) + (14.40 - 14.02), "alternative.shot.mp4", 14.40, 14.98, -6),
        (vb(V_KICK, 7) + 0.03, "action.mp4", 3.98, 4.60, -6),               # the pads
        (vb(V_KICK, 7) + 1.42, "action.mp4", 5.15, 5.45, -7),               # the double-leg lands
        (V_PAUSE - 0.03, "Milosz.Bilal.KB2.mp4", 9.97, 10.45, -2),          # the hook the music stops on
        (vb(V_DROP, 6) + 0.62, "Makhachev.D.arce.mp4", 19.92, 20.30, -8),   # the roll
        (vb(V_DROP, 17) + (8.05 - 6.90), "armbar.mp4", 8.04, 8.40, -7),     # the sit-back
        (V_BREAK + 0.45, "Milosz.Bilal.KB2.mp4", 20.0, 20.9, -8),            # the laugh
        (V_808 + 0.1, "new/WhatsApp_Video_2026-08-24_at_20.41.55.mp4", 0.4, 2.4, -10),   # the room cheering
        (V_RETURN, "new/fill-in.mp4", 0.60, 1.60, -9),
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


def _impact(sr, dur=2.2):
    """One clean, deep landing: a sub that drops 58 → 34 Hz under a soft felt transient."""
    from scipy import signal
    t = np.arange(int(dur * sr)) / sr
    f = 34 + 24 * np.exp(-t / 0.18)
    sub = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.55)
    rng = np.random.default_rng(3)
    click = signal.lfilter(*signal.butter(2, 900 / (sr / 2)), rng.standard_normal(len(t))) * np.exp(-t / 0.012) * 0.3
    return ((sub * 0.9 + click)[:, None].repeat(2, 1)).astype(np.float32)


def _clunk(sr, dur=0.5):
    """A bank of gym lights switching on: a dull breaker thud (a short 70 → 42 Hz drop) under a
    soft, dark click — low and clean."""
    from scipy import signal
    t = np.arange(int(dur * sr)) / sr
    f = 42 + 28 * np.exp(-t / 0.03)
    thud = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.09)
    rng = np.random.default_rng(7)
    click = signal.lfilter(*signal.butter(2, 1800 / (sr / 2)), rng.standard_normal(len(t))) * np.exp(-t / 0.006) * 0.35
    return ((thud * 0.8 + click)[:, None].repeat(2, 1)).astype(np.float32)


def _instrumental_from(seg, s0, sr):
    """The song segment (song time s0 onward) with its voice taken out from INST_FROM: the original up
    to the gap after the line, then the separated instrumental (edits/separate.py), crossfaded."""
    from scipy import signal
    import separate
    s1 = s0 + len(seg) / sr
    inst = separate.instrumental(MUSIC, s0, s1 - s0)                        # 44.1 kHz
    inst = signal.resample_poly(inst, sr // 300, separate.SR // 300, axis=0).astype(np.float32)[:len(seg)]
    inst *= 10 ** (1.5 / 20)                                               # most of what the voice carried
    if len(inst) < len(seg):
        inst = np.concatenate([inst, np.zeros((len(seg) - len(inst), 2), np.float32)])
    j = int(round((INST_FROM - s0) * sr))
    x = int(0.06 * sr)
    ramp = np.clip((np.arange(len(seg)) - (j - x // 2)) / x, 0, 1)[:, None]
    ramp = np.sin(ramp * np.pi / 2) ** 2
    return (seg * (1 - ramp) + inst * ramp).astype(np.float32)


def mix(plan, sr=48000, music=True):
    """The mix. music=False: the floor sounds alone (with the hall tails and the impacts), for
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
        seg = song[a:b].copy()
        if music and s0 < INST_FROM < s1:
            seg = _instrumental_from(seg, s0 - lead, sr)
        seg *= 10 ** (gdb / 20)
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
    # two clean low impacts: under the kick, and as the banner snaps open
    # into the drop: Bilal's hook, reversed through the hall, draws in to the downbeat
    hk = media.read_audio(str(FOOT / "Milosz.Bilal.KB2.mp4"), 9.97, 0.4, sr=sr, channels=2)
    hk = signal.sosfilt(signal.butter(2, 120 / (sr / 2), "high", output="sos"), hk, axis=0)
    rv = _verb(hk[::-1].copy(), _hall(sr, 1.3, seed=9))[::-1]
    rv = rv / (np.abs(rv).max() + 1e-6) * 0.22
    rv = rv[-int((V_DROP - V_PAUSE - 0.1) * sr):]
    j = int(round(V_DROP * sr)) - len(rv)
    out[j:j + len(rv)] += rv
    for at, g in ((V_KICK, 0.28), (V_DROP, 0.3), (U0 + 0.3, 0.36)):
        imp = _impact(sr) * g
        j = int(round(at * sr))
        m = min(len(imp), n - j)
        out[j:j + m] += imp[:m]
    # the lights coming on over the team
    for nf, g in CLUNKS:
        c = _clunk(sr) * g
        j = int(round((V_TEAM + nf / FPS) * sr))
        out[j:j + len(c)] += c[:n - j]
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
    ap.add_argument("--no-music-version", action="store_true", help="also write *_no-music.mp4 (floor sounds only)")
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
        if not a.no_music_version:
            sys.exit()
        # the same picture with the floor sounds only, for ads (the song is not cleared for paid use)
        import subprocess
        me = str(out.with_name(out.stem + "_no-music.wav"))
        media.write_wav(me, mix(tl.audio_plan, music=False))
        subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(out), "-i", me, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
                        str(out.with_name(out.stem + "_no-music.mp4"))], check=True)
