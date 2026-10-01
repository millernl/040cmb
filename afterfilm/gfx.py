"""Vector graphics on top of the picture: type, the glyph, the letterbox.

Everything is drawn with Skia straight into numpy buffers, so shapes stay crisp at
any scale (the glyph zoom-through magnifies the mark ~60x).
"""
import functools

import numpy as np
import skia
from scipy import ndimage

from . import brand, fx


class Layer:
    """Premultiplied RGBA overlay the size of the frame."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.buf = np.zeros((h, w, 4), np.uint8)
        self.surface = skia.Surface(self.buf, colorType=skia.kRGBA_8888_ColorType,
                                    alphaType=skia.kPremul_AlphaType)
        self.c = self.surface.getCanvas()

    def clear(self):
        self.c.clear(skia.ColorTRANSPARENT)

    def over(self, img):
        self.surface.flushAndSubmit()
        a = self.buf[..., 3:4].astype(np.float32) / 255.0
        if not a.any():
            return img
        return img * (1 - a) + self.buf[..., :3].astype(np.float32) / 255.0


class Mask:
    """8-bit coverage mask the size of the frame, for shapes used as windows."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.buf = np.zeros((h, w, 4), np.uint8)
        self.surface = skia.Surface(self.buf, colorType=skia.kRGBA_8888_ColorType,
                                    alphaType=skia.kPremul_AlphaType)
        self.c = self.surface.getCanvas()

    def draw(self, fn):
        self.c.clear(skia.ColorTRANSPARENT)
        self.c.save()
        fn(self.c, skia.Paint(AntiAlias=True, Color=skia.ColorWHITE))
        self.c.restore()
        self.surface.flushAndSubmit()
        return self.buf[..., 3].astype(np.float32) / 255.0


# ── type ─────────────────────────────────────────────────────────────────────
def measure(text, font, tracking=0.0):
    widths = font.getWidths(font.textToGlyphs(text))
    return float(sum(widths)) + tracking * font.getSize() * max(len(text) - 1, 0), widths


def text(c, s, font, x, y, color=brand.WHITE, alpha=1.0, tracking=0.0, align="left",
         letter_alpha=None, letter_dy=None):
    """Tracked text. letter_alpha / letter_dy: optional per-letter callables i → value."""
    if alpha <= 0 or not s:
        return
    total, widths = measure(s, font, tracking)
    x0 = x - total * {"left": 0.0, "center": 0.5, "right": 1.0}[align]
    pos = x0
    for i, (ch, w) in enumerate(zip(s, widths)):
        a = alpha * (letter_alpha(i) if letter_alpha else 1.0)
        if a > 0.003 and ch != " ":
            dy = letter_dy(i) if letter_dy else 0.0
            c.drawString(ch, pos, y + dy, font, skia.Paint(AntiAlias=True, Color4f=brand.skcolor(color, a)))
        pos += w + tracking * font.getSize()


def typewriter(c, s, font, x, y, p, color=brand.WHITE, alpha=1.0, tracking=0.08, align="left", cursor=True):
    """Mono label that types itself on (p: 0→1), block cursor riding the last char."""
    n = int(round(fx.clamp01(p) * len(s)))
    total, widths = measure(s, font, tracking)
    x0 = x - total * {"left": 0.0, "center": 0.5, "right": 1.0}[align]
    text(c, s[:n], font, x0, y, color, alpha, tracking)
    if cursor and 0 < p < 1:
        cx = x0 + sum(widths[:n]) + tracking * font.getSize() * n
        size = font.getSize()
        c.drawRect(skia.Rect.MakeXYWH(cx + 1, y - size * 0.78, size * 0.55, size * 0.95),
                   skia.Paint(Color4f=brand.skcolor(color, alpha * 0.9)))


def text_path(s, font, tracking=0.0):
    """Outline of a string as one path (origin at the baseline start)."""
    path = skia.Path()
    glyphs = font.textToGlyphs(s)
    widths = font.getWidths(glyphs)
    x = 0.0
    for g, w in zip(glyphs, widths):
        gp = font.getPath(g)
        if gp is not None:
            gp.offset(x, 0)
            path.addPath(gp)
        x += w + tracking * font.getSize()
    return path


# ── the glyph ────────────────────────────────────────────────────────────────
@functools.cache
def glyph_pivot():
    """The deepest point inside the glyph (in glyph units) — zooming into it fills
    the frame soonest, so it's the door the camera flies through."""
    g, (gw, gh) = brand.glyph()
    scale = 800 / gh
    a = brand.raster_path(g, scale, pad=0) > 0.5
    dist = ndimage.distance_transform_edt(a)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    b = g.computeTightBounds()
    return b.left() + x / scale, b.top() + y / scale, dist.max() / scale


def glyph_matrix(w, h, p, base_height, center=(0.5, 0.5), rot0=-3.0):
    """Transform for the zoom-through: p=0 → glyph sits at base_height, centred;
    p=1 → deep inside the pivot, frame fully covered."""
    g, (gw, gh) = brand.glyph()
    px, py, depth = glyph_pivot()
    s0 = base_height / gh
    s1 = (np.hypot(w, h) / 2) / depth * 1.25            # pivot's inscribed disc covers the frame
    s = s0 * (s1 / s0) ** p
    # slide from 'glyph centred' to 'pivot centred' as we zoom
    gx, gy = gw / 2, gh / 2
    k = fx.smoothstep(0.0, 0.6, p)
    ax, ay = gx + (px - gx) * k, gy + (py - gy) * k
    m = skia.Matrix()
    m.preTranslate(w * center[0], h * center[1])
    m.preRotate(rot0 * (1 - p))
    m.preScale(s, s)
    m.preTranslate(-ax, -ay)
    return m


def glyph_window(mask, w, h, p, base_height, center=(0.5, 0.5), rot0=-3.0):
    g, _ = brand.glyph()
    m = glyph_matrix(w, h, p, base_height, center, rot0)

    def draw(c, paint):
        c.concat(m)
        c.drawPath(g, paint)

    return mask.draw(draw)


def rim(img, m, strength, color=(1.0, 0.93, 0.82)):
    """Hairline of light along a mask's edge, so a window's shape reads over dark footage."""
    if strength <= 0:
        return img
    m2 = m[..., 0] if m.ndim == 3 else m
    gy, gx = np.gradient(m2)
    edge = np.clip(np.hypot(gx, gy) * 2.2, 0, 1)[..., None]
    return np.clip(img + edge * strength * np.asarray(color, np.float32), 0, 1)


def paste_alpha(img, alpha, color, cx, cy, opacity=1.0):
    """Composite a small alpha raster centred at (cx, cy)."""
    h, w = alpha.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = img.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    a = alpha[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0, None] * opacity
    col = np.asarray(color, np.float32)
    region = img[dy0:dy1, dx0:dx1]
    img[dy0:dy1, dx0:dx1] = region * (1 - a) + col * a if col.ndim == 1 else region * (1 - a) + col[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0] * a
    return img


def glyph_drawon(img, p, cx, cy, height, color=brand.WHITE, glow=True):
    """Write the glyph stroke by stroke (p: 0→1)."""
    alpha, t = brand.glyph_drawon(int(height))
    soft = 0.07
    reveal = fx.clamp01((p * (1 + soft) - t) / soft) * alpha
    col = brand.f32(color)
    img = paste_alpha(img, reveal, col, cx, cy)
    if glow and 0 < p < 1:
        # the wet 'ink head' glows as it travels
        head = np.exp(-((t - p) / 0.035) ** 2) * alpha * (t <= p + 0.02)
        halo = ndimage.gaussian_filter(head, 6) * 2.2
        img = paste_alpha(img, np.clip(halo, 0, 1), np.array([1.0, 0.9, 0.72], np.float32), cx, cy, 0.55)
    return img


# ── lockup (end card) ────────────────────────────────────────────────────────
def lockup_geometry(w, h, height):
    g, letters, (lw, lh) = brand.lockup()
    s = height / lh
    return g, letters, s, (w - lw * s) / 2, (h - lh * s) / 2


def draw_wordmark(c, w, h, height, p, color=brand.WHITE, cy_offset=0.0):
    """Wordmark letters arrive one by one, tracking in from wide."""
    g, letters, s, x0, y0 = lockup_geometry(w, h, height)
    n = len(letters)
    b_all = [lp.computeTightBounds() for lp in letters]
    mid = (b_all[0].left() + b_all[-1].right()) / 2
    for i, (lp, b) in enumerate(zip(letters, b_all)):
        li = fx.clamp01((p * (1 + 0.6) - i / n * 0.6))
        e = fx.expo_out(li)
        if e <= 0.002:
            continue
        spread = (1 - e) * 0.9          # letters start 90% wider apart
        dx = ((b.left() + b.right()) / 2 - mid) * spread
        c.save()
        c.translate(x0 + dx * s, y0 + cy_offset + (1 - e) * 10)
        c.scale(s, s)
        c.drawPath(lp, skia.Paint(AntiAlias=True, Color4f=brand.skcolor(color, float(fx.smoothstep(0, 0.7, li)))))
        c.restore()


def lockup_glyph_box(w, h, height, cy_offset=0.0):
    """Where the glyph sits inside the lockup: (centre x, centre y, glyph height) in px."""
    g, letters, s, x0, y0 = lockup_geometry(w, h, height)
    b = g.computeTightBounds()
    return x0 + (b.left() + b.right()) / 2 * s, y0 + cy_offset + (b.top() + b.bottom()) / 2 * s, b.height() * s


# ── letterbox ────────────────────────────────────────────────────────────────
def letterbox(img, amount, ratio=2.39):
    """Cinema bars; amount 0 (16:9 full frame) → 1 (full scope)."""
    if amount <= 0:
        return img, 0
    h, w = img.shape[:2]
    bar = min(int(round((h - w / ratio) / 2 * amount)), h // 2 + 1)   # amount > 1 closes toward black
    if bar > 0:
        img[:bar] = 0
        img[h - bar:] = 0
    return img, bar


def bar_labels(c, w, h, bar, alpha, labels):
    """Mono labels sitting in the bottom bar, placed like the identity guide's page
    footers (left 8% · right-centre 72% · right 93%)."""
    if alpha <= 0 or bar < 24:
        return
    f = brand.font(brand.MONO, round(h * 0.0145))
    y = h - bar / 2 + f.getSize() * 0.35
    for (s, xpos, align) in zip(labels, (0.081, 0.724, 0.933), ("left", "left", "right")):
        if s:
            text(c, s.upper(), f, w * xpos, y, (200, 204, 200), alpha, tracking=0.06, align=align)
