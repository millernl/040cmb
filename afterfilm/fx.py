"""Pixel work: the District98 grade, film finish, and transitions.

Frames are float32 HxWx3 in [0, 1], display-referred.
"""
import numpy as np
import cv2

from . import brand

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ── easing ────────────────────────────────────────────────────────────────────
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    return clamp01(x) ** 3


def ease_in_out(x):
    x = clamp01(x)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2) if isinstance(x, np.ndarray) else \
        (4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2)


def expo_out(x):
    x = float(clamp01(x))
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def expo_in_out(x):
    x = float(clamp01(x))
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def window(t, t0, t1):
    """Progress 0→1 of t across [t0, t1]."""
    return float(clamp01((t - t0) / max(t1 - t0, 1e-6)))


def luma(img):
    return img @ LUMA


def _chroma(color):
    c = brand.f32(color)
    return c - float(c @ LUMA)


# ── the grade ────────────────────────────────────────────────────────────────
# Looks share one pipeline; a preset is a set of knobs. Colour arc of the film:
# 'iron' (high-contrast black & white) → 'color' (stage light remapped into the
# brand's green / gold) → 'silver' (clean, faded colour for the heart).
PRESETS = {
    "iron":  dict(mono=1.0, sat=0.0, remap=0.0, contrast=0.42, expo=0.05, split=0.45, warmth=0.0),
    "color": dict(mono=0.0, sat=0.92, remap=1.0, contrast=0.30, expo=0.0, split=1.0, warmth=0.1),
    # the heart: clean, faded colour — whites stay white, no sepia wash
    "silver": dict(mono=0.0, sat=0.55, remap=1.0, contrast=0.26, expo=0.06, split=0.55, warmth=0.0),
    # red/orange: the shot's black & white tones rendered through the red solo's own palette
    "ember": dict(mono=1.0, sat=0.0, remap=0.0, contrast=0.42, expo=0.0, split=0.0, warmth=0.0, ember=1.0),
    # another act's own stage light, as it was: filmic, no hue remap, no brand tint
    "stage": dict(mono=0.0, sat=1.0, remap=0.0, contrast=0.3, expo=0.0, split=0.0, warmth=0.0),
}

# Luma → colour of the red-solo shots as the 'color' look renders them (measured from
# 2:09:46–2:10:35 and 1:09:27), extended past their brightest pixels to a warm white.
_EMBER_L = np.array([0.0, 0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
                     0.475, 0.525, 0.575, 0.625, 0.675, 0.8, 1.0], np.float32)
_EMBER_RGB = np.array([[4, 6, 5], [11, 11, 10], [24, 17, 10], [57, 27, 7], [86, 34, 6], [124, 42, 7],
                       [155, 51, 9], [179, 61, 12], [199, 73, 15], [213, 86, 17], [224, 103, 20],
                       [226, 117, 22], [228, 134, 20], [230, 149, 17], [231, 162, 40], [240, 190, 110],
                       [248, 225, 185]], np.float32) / 255.0
EMBER_SCALE = 0.56     # a white in the black & white grade lands on the solo's bright orange
EMBER_GAMMA = 1.2      # keeps the mids in the deep reds, as on the red stage


def ember(v):
    """Black & white tone (0..1) → the red/orange palette."""
    L = np.clip(v, 0, 1) ** EMBER_GAMMA * EMBER_SCALE
    return np.stack([np.interp(L, _EMBER_L, _EMBER_RGB[:, c]) for c in range(3)], -1).astype(np.float32)
BW_MIX = np.array([0.50, 0.40, 0.10], np.float32)        # orange-filter panchromatic: skin glows, violet light sinks

# hue remap: stage purples/blues → desaturated teal, reds/oranges/yellows → gold, greens → '98 Green'
_H_IN = np.array([0, 25, 45, 65, 100, 140, 180, 210, 240, 270, 300, 330, 360], np.float32)
_H_OUT = np.array([18, 32, 40, 48, 105, 125, 168, 184, 192, 200, 240, 350, 378], np.float32)
_S_MUL = np.array([0.9, 1.0, 0.92, 0.8, 0.6, 0.7, 0.65, 0.55, 0.38, 0.28, 0.3, 0.55, 0.9], np.float32)


def filmic(x):
    """ACES-style filmic tone curve on display-referred input (mid-grey held)."""
    lin = np.power(np.clip(x, 0, 1), 2.2) * 0.8
    y = (lin * (2.51 * lin + 0.03)) / (lin * (2.43 * lin + 0.59) + 0.14)
    return np.power(np.clip(y, 0, 1), 1 / 2.2)


class Look:
    """The District98 film look — one place to tune the picture."""

    def __init__(self, w, h, accent=brand.GOLD, seed=98):
        self.w, self.h = w, h
        self.shadow_tint = _chroma(brand.GREEN)
        self.high_tint = _chroma(accent)
        self.accent = brand.f32(accent)
        self.black = np.array([0.016, 0.022, 0.019], np.float32)   # off-black leaning green (#1C2120 family)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        self.vignette = (1 - 0.38 * smoothstep(0.3, 1.0, r) ** 1.3)[..., None].astype(np.float32)
        rng = np.random.default_rng(seed)
        self.grain = []
        for _ in range(12):
            g = rng.standard_normal((int(h / 1.25), int(w / 1.25))).astype(np.float32)   # finer, film-like
            g = cv2.GaussianBlur(g, (0, 0), 0.55)
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
            self.grain.append(g / (g.std() + 1e-6))
        self.phase = rng.uniform(0, 6.28, 6)

    def remap(self, x, amt):
        hls = cv2.cvtColor(x, cv2.COLOR_RGB2HLS)
        h, s = hls[..., 0].copy(), hls[..., 2].copy()
        h2 = np.interp(h, _H_IN, _H_OUT)
        hls[..., 0] = np.mod(h + amt * (h2 - h), 360.0)
        hls[..., 2] = np.clip(s * (1 + amt * (np.interp(h, _H_IN, _S_MUL) - 1)), 0, 1)
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def grade(self, img, look="color", exposure=0.0, **over):
        p = {**PRESETS[look], **{k: v for k, v in over.items() if v is not None}}
        x = np.clip(img, 0, 1).astype(np.float32)
        e = p["expo"] + exposure
        if e:
            x = np.clip(x * (2.0 ** e), 0, 1)
        mono = float(p["mono"])
        if p["remap"] and mono < 1:
            x = self.remap(x, p["remap"])
        y = luma(x)[..., None]
        x = y + (x - y) * p["sat"] * (1 - mono)
        if mono:
            bw = (np.clip(img, 0, 1) * (2.0 ** e)) @ BW_MIX
            x = x * (1 - mono) + np.clip(bw, 0, 1)[..., None] * mono
        x = filmic(x)
        c = p["contrast"]
        x = x + c * (x * x * (3 - 2 * x) - x)
        if p.get("ember"):
            return ember(luma(x))
        y = luma(x)[..., None]
        if p["split"]:
            sh = (1 - smoothstep(0.05, 0.55, y)) * smoothstep(0.0, 0.18, y)
            hi = smoothstep(0.5, 1.0, y)
            x = x + p["split"] * (sh * self.shadow_tint * 0.38 + hi * self.high_tint * 0.2)
        if p["warmth"]:
            w_ = p["warmth"]
            x = x * np.array([1 + 0.05 * w_, 1 + 0.01 * w_, 1 - 0.07 * w_], np.float32)
        x = self.black + x * (1 - self.black)
        return np.clip(x, 0, 1).astype(np.float32)

    def cinema(self, img, n, mono=0.0, halation=1.0, streaks=1.0, bloom=1.0, weave=1.0):
        """What a lens and a film stock do to light: halation, anamorphic streaks,
        bloom, a whisper of fringing, gate weave."""
        w, h = self.w, self.h
        k = min(w, h) / 1080                     # blur radii follow the frame: 4K looks like 1080p
        q = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        ql = luma(q)
        out = img
        if halation:
            hot = np.clip((ql - 0.62) / 0.38, 0, 1) ** 1.5
            halo = cv2.GaussianBlur(hot, (0, 0), 2.2 * k) - hot * 0.4
            halo = cv2.resize(np.clip(halo, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
            col = np.array([1.0, 0.34, 0.14], np.float32) * (1 - mono) + np.array([0.9, 0.86, 0.8], np.float32) * mono
            out = out + halo[..., None] * col * 0.26 * halation
        if streaks:
            pts = np.clip((ql - 0.8) / 0.2, 0, 1) ** 2
            st = cv2.GaussianBlur(pts, (0, 0), sigmaX=w / 4 * 0.09, sigmaY=0.6)
            st = st * 9.0 + cv2.GaussianBlur(pts, (0, 0), sigmaX=w / 4 * 0.025, sigmaY=0.5) * 3.0
            st = cv2.resize(np.clip(st, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
            col = np.array([1.0, 0.86, 0.62], np.float32) * (1 - mono) + np.array([0.92, 0.94, 1.0], np.float32) * mono
            out = out + st[..., None] * col * 0.32 * streaks
        if bloom:
            hot = np.clip(q - 0.66, 0, None) / 0.34
            glow = cv2.GaussianBlur(hot, (0, 0), 5 * k) * 0.55 + cv2.GaussianBlur(hot, (0, 0), 20 * k) * 0.9
            glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
            out = 1 - (1 - np.clip(out, 0, 1)) * (1 - np.clip(glow * 0.3 * bloom, 0, 1))
        # lateral fringing, stronger toward the corners
        fr = 0.0011
        rch = cv2.warpAffine(out[..., 0], cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + fr), (w, h),
                             borderMode=cv2.BORDER_REFLECT)
        bch = cv2.warpAffine(out[..., 2], cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 - fr), (w, h),
                             borderMode=cv2.BORDER_REFLECT)
        out = np.stack([rch, out[..., 1], bch], -1)
        if weave:
            ph = self.phase
            s = h / 1080
            dx = (0.55 * np.sin(n * 0.31 + ph[0]) + 0.3 * np.sin(n * 0.87 + ph[1])) * s * weave
            dy = (0.45 * np.sin(n * 0.23 + ph[2]) + 0.25 * np.sin(n * 1.13 + ph[3])) * s * weave
            out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            out = out * (1 + 0.006 * np.sin(n * 1.7 + ph[4]) + 0.004 * np.sin(n * 2.9 + ph[5]))
        return np.clip(out, 0, 1).astype(np.float32)

    def bloom(self, img, strength=0.32, threshold=0.68):
        return self.cinema(img, 0, halation=0, streaks=0, weave=0, bloom=strength / 0.3)

    def finish(self, img, frame_no, grain=1.0, vignette=1.0):
        if vignette:
            img = img * (1 - vignette + vignette * self.vignette)
        if grain:
            g = self.grain[frame_no % len(self.grain)]
            y = luma(img)
            amt = 0.024 * grain * (0.15 + 0.85 * (1 - np.abs(y - 0.42) * 1.5).clip(0, 1))
            img = img + (g * amt)[..., None]
        return np.clip(img, 0, 1)

    def write_cube(self, path, look="color", n=33):
        """Export a look as a .cube LUT for reuse in Premiere / Resolve / CapCut."""
        r = np.linspace(0, 1, n, dtype=np.float32)
        b, g, rr = np.meshgrid(r, r, r, indexing="ij")
        grid = np.stack([rr, g, b], -1).reshape(1, -1, 3)
        out = self.grade(grid, look=look).reshape(-1, 3)
        with open(path, "w") as f:
            f.write(f'TITLE "District98 {look}"\n')
            f.write(f"LUT_3D_SIZE {n}\n")
            for v in out:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


# ── geometry helpers ─────────────────────────────────────────────────────────
def reframe(img, zoom=1.0, cx=0.5, cy=0.5, out_size=None, rot=0.0):
    """Crop/scale around a normalised centre — Ken Burns, punch-ins, pans."""
    h, w = img.shape[:2]
    ow, oh = out_size or (w, h)
    s = zoom * max(ow / w, oh / h)
    # keep the crop inside the frame
    half_w, half_h = ow / (2 * s), oh / (2 * s)
    px = float(np.clip(cx * w, half_w, w - half_w)) if half_w * 2 <= w else w / 2
    py = float(np.clip(cy * h, half_h, h - half_h)) if half_h * 2 <= h else h / 2
    m = cv2.getRotationMatrix2D((px, py), rot, s)
    m[0, 2] += ow / 2 - px
    m[1, 2] += oh / 2 - py
    interp = cv2.INTER_CUBIC if s > 1.4 else cv2.INTER_LINEAR      # vertical crops upscale ~2.7x
    return cv2.warpAffine(img, m, (ow, oh), flags=interp, borderMode=cv2.BORDER_REFLECT)


def shift(img, dx=0, dy=0):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def hblur(img, px):
    k = int(px)
    if k < 2:
        return img
    return cv2.blur(img, (k, 1))


def mix(a, b, m):
    if m.ndim == 2:
        m = m[..., None]
    return a + (b - a) * m


# ── transitions: f(A, B, p, ctx) → frame ─────────────────────────────────────
def t_slit(a, b, p, ctx):
    """A vertical slit of stage light opens from the centre (from black or from A)."""
    h, w = b.shape[:2]
    if a is None:
        a = np.zeros_like(b)
    e = ease_in_out(p) ** 1.15
    half = e * (w / 2 + 40)
    x = np.abs(np.arange(w, dtype=np.float32) - w / 2)
    m = clamp01((half - x) / 2.5)[None, :, None]
    rim = np.exp(-((x - half) / (6 + 40 * e)) ** 2)[None, :, None] * (1 - e) ** 0.6
    bz = reframe(b, 1.0 + 0.06 * (1 - e))
    out = a * (1 - m) + bz * m
    return np.clip(out + rim * np.array([1.0, 0.93, 0.8], np.float32) * 0.9, 0, 1)


def t_stripes(a, b, p, ctx, n=9, stagger=0.55):
    """Light blinds: vertical bands open from their centres, left to right, with a
    thin hot edge — the hard light-and-shadow stripes from the moodboard."""
    h, w = b.shape[:2]
    sw = w / n
    x = np.arange(w, dtype=np.float32)
    i = np.floor(x / sw)
    local = clamp01((p * (1 + stagger) - (i / n) * stagger))
    local = local * local * (3 - 2 * local)
    d = np.abs(x - (i + 0.5) * sw)
    half = local * sw / 2 + 0.5
    m = clamp01((half - d) / 1.5)
    edge = np.exp(-((d - half) / 2.2) ** 2) * (local > 0.01) * (local < 0.99)
    bz = shift(b, dx=(1 - ease_out(p)) * 50)
    az = shift(a, dx=-ease_in(p) * 30)
    out = mix(az, bz, np.broadcast_to(m[None, :], (h, w)))
    return np.clip(out + edge[None, :, None] * 0.55, 0, 1)


def t_swoosh(a, b, p, ctx, flip=False):
    """A calligraphic ribbon — the sweep of the glyph's arm — wipes A away."""
    h, w = b.shape[:2]
    e = expo_in_out(p)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    base = -0.35 * w + e * 1.7 * w
    curve = base + 0.16 * w * np.sin(np.pi * (y * 1.15 + 0.15)) - 0.22 * w * (y - 0.5)
    if flip:
        x = w - x
    d = x - curve                                   # >0: still A, <0: B revealed
    m = clamp01(-d / 2.0 + 0.5)
    thick = 26 + 70 * np.sin(np.pi * y) ** 2        # brush pressure along the stroke
    ribbon = np.exp(-((d - thick * 0.6) / thick) ** 2) * np.sin(np.pi * p) ** 0.7
    green = brand.f32(brand.GREEN)
    az = shift(a, dx=(1 if flip else -1) * -e * 80)
    bz = reframe(b, 1.05 - 0.05 * e)
    out = mix(az, bz, m)
    col = green * 0.6 + 0.4 * np.clip(ribbon[..., None] * 1.4, 0, 1)
    return np.clip(out * (1 - ribbon[..., None] * 0.85) + col * ribbon[..., None] * 0.95, 0, 1)


def t_whip(a, b, p, ctx, direction=1):
    """Whip pan with motion blur."""
    h, w = b.shape[:2]
    e = ease_in_out(p)
    off = e * w * direction
    blur = np.sin(np.pi * p) ** 1.5 * 140
    out = shift(a, dx=-off) + shift(b, dx=direction * w - off)
    return np.clip(hblur(out, blur), 0, 1)


def t_smear(a, b, p, ctx, direction=1):
    """A camera whip caught in four frames: the outgoing shot starts to smear, the cut
    lands on the incoming shot at full smear, and it settles. Nothing slides across the
    frame; the motion lives in the blur. Centre it on the beat (start half a length early)."""
    h, w = b.shape[:2]
    if p < 0.5:
        img, k, sgn = a, (p / 0.5) ** 1.2, 1
    else:
        img, k, sgn = b, ((1 - p) / 0.5) ** 1.5, -1
    if k < 0.02:
        return img
    off = -direction * sgn * k * 0.08 * w           # content drifts against the whip
    m = np.float32([[1, 0, off], [0, 1, 0]])
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = hblur(out, k * 0.28 * w)
    return np.clip(out * (1 + 0.12 * k), 0, 1)


def t_strobe(a, b, p, ctx, n=4):
    """The incoming shot strobes in on single frames — B, A, B, A — then holds."""
    i = int(p * n + 1e-6)
    return flash(b, 0.2) if i % 2 == 0 else a


def t_cut(a, b, p, ctx):
    return b


def t_dissolve(a, b, p, ctx):
    e = smoothstep(0, 1, p)
    return a * (1 - e) + b * e


def t_burn(a, b, p, ctx):
    """Dissolve through warm light — for the slow, emotional section."""
    e = smoothstep(0.15, 0.85, p)
    glow = np.sin(np.pi * p) ** 2 * 0.45
    out = a * (1 - e) + b * e
    return np.clip(1 - (1 - out) * (1 - glow * np.array([1.0, 0.82, 0.55], np.float32)), 0, 1)


def t_dip(a, b, p, ctx):
    """Fade through black — a breath between sections."""
    if p < 0.5:
        return a * (1 - smoothstep(0, 0.5, p))
    return b * smoothstep(0.5, 1.0, p)


def flash(img, k, tint=(1.0, 0.95, 0.86)):
    """Exposure flash on a hit; k in [0, 1]."""
    if k <= 0:
        return img
    return np.clip(img * (1 + 1.6 * k) + k * 0.55 * np.array(tint, np.float32), 0, 1)


TRANSITIONS = {
    "slit": t_slit,
    "stripes": t_stripes,
    "swoosh": t_swoosh,
    "swoosh_r": lambda a, b, p, c: t_swoosh(a, b, p, c, flip=True),
    "whip": t_whip,
    "whip_l": lambda a, b, p, c: t_whip(a, b, p, c, direction=-1),
    "cut": t_cut,
    "smear": t_smear,
    "smear_l": lambda a, b, p, c: t_smear(a, b, p, c, direction=-1),
    "strobe": t_strobe,
    "dissolve": t_dissolve,
    "burn": t_burn,
    "dip": t_dip,
}
