"""District98 identity as code: palette, type, and the logo as vector paths.

Source: 'District 98 — Logo & Identity Design', Phase 5, Bart de Graaff, Oct 2021.
The guide sets type in Apercu Mono / Arial Nova / Dunbar Text; the free stand-ins
here are DM Mono / Inter Tight / Outfit (SIL OFL, see brand/fonts/LICENSE-*).
"""
import functools
import re
from pathlib import Path

import numpy as np
import skia

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "brand"
FONTS = BRAND / "fonts"


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


GREEN = rgb("#3D5E3C")    # '98 Green' — the core colour
BLACK = rgb("#000000")
INK = rgb("#1C2120")      # off-black
WHITE = rgb("#FFFFFF")
ORANGE = rgb("#F56732")   # accent option 1, 'Burnt orange'
GOLD = rgb("#F7C27C")     # accent option 2, 'Gold'
SKIN = rgb("#EBCCC1")     # accent option 3, 'Skin'

MONO = "dm-mono-400-normal"
MONO_LIGHT = "dm-mono-300-normal"
MONO_MEDIUM = "dm-mono-500-normal"
DISPLAY_THIN = "inter-tight-200-normal"
DISPLAY_LIGHT = "inter-tight-300-normal"
DISPLAY = "inter-tight-400-normal"
DISPLAY_ITALIC = "inter-tight-200-italic"
GEOMETRIC = "outfit-300-normal"


def f32(color, alpha=1.0):
    return np.array(color, np.float32) / 255.0


def skcolor(color, alpha=1.0):
    return skia.Color4f(color[0] / 255, color[1] / 255, color[2] / 255, alpha)


@functools.cache
def typeface(name):
    tf = skia.Typeface.MakeFromFile(str(FONTS / f"{name}.ttf"))
    if tf is None:
        raise FileNotFoundError(name)
    return tf


def font(name, size):
    f = skia.Font(typeface(name), size)
    f.setSubpixel(True)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setHinting(skia.FontHinting.kNone)
    return f


def parse_svg_path(d):
    """Minimal SVG path parser — the extracted logo only uses absolute M L C H V Z."""
    path = skia.Path()
    toks = re.findall(r"[MLCHVZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    i, cmd, x, y = 0, None, 0.0, 0.0
    num = lambda k: float(toks[i + k])
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd == "Z":
                path.close()
                continue
        if cmd == "M":
            x, y = num(0), num(1); i += 2
            path.moveTo(x, y); cmd = "L"
        elif cmd == "L":
            x, y = num(0), num(1); i += 2
            path.lineTo(x, y)
        elif cmd == "C":
            path.cubicTo(num(0), num(1), num(2), num(3), num(4), num(5))
            x, y = num(4), num(5); i += 6
        elif cmd == "H":
            x = num(0); i += 1
            path.lineTo(x, y)
        elif cmd == "V":
            y = num(0); i += 1
            path.lineTo(x, y)
        else:
            raise ValueError(f"unsupported path token {toks[i]!r}")
    return path


def _svg(name):
    svg = (BRAND / f"logo_{name}.svg").read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    return vb, [parse_svg_path(d) for d in re.findall(r' d="([^"]+)"', svg)]


@functools.cache
def lockup():
    """Glyph path, per-letter wordmark paths, and (w, h) — all in lockup coordinates."""
    vb, _ = _svg("lockup")
    _, glyph = _svg("glyph")
    _, letters = _svg("wordmark")
    g = skia.Path()
    for p in glyph:
        g.addPath(p)
    g.offset(-vb[0], -vb[1])
    for p in letters:
        p.offset(-vb[0], -vb[1])
    return g, letters, (vb[2], vb[3])


@functools.cache
def glyph():
    """The dancer glyph alone, origin at its top-left, plus (w, h)."""
    vb, paths = _svg("glyph")
    g = skia.Path()
    for p in paths:
        g.addPath(p)
    g.offset(-vb[0], -vb[1])
    return g, (vb[2], vb[3])


def raster_path(path, scale, pad=4):
    """Rasterise a path to a float32 alpha mask (antialiased)."""
    b = path.computeTightBounds()
    w = int(np.ceil(b.width() * scale)) + 2 * pad
    h = int(np.ceil(b.height() * scale)) + 2 * pad
    surf = skia.Surface.MakeRaster(skia.ImageInfo.Make(w, h, skia.kAlpha_8_ColorType, skia.kPremul_AlphaType))
    c = surf.getCanvas()
    c.translate(pad, pad)
    c.scale(scale, scale)
    c.translate(-b.left(), -b.top())
    c.drawPath(path, skia.Paint(AntiAlias=True, Color=skia.ColorWHITE))
    a = np.frombuffer(surf.makeImageSnapshot().tobytes(), np.uint8).reshape(h, w)
    return a.astype(np.float32) / 255.0


@functools.cache
def glyph_drawon(height_px):
    """Alpha mask of the glyph and a per-pixel 'ink time' map in [0, 1].

    The ink time follows the strokes: skeletonise the glyph, walk the skeleton
    geodesically from the top-left tip (the dancer's raised hand), then give every
    glyph pixel the time of its nearest skeleton point. Thresholding that map with a
    rising value writes the mark stroke by stroke, like a brush.
    """
    from collections import deque
    from scipy import ndimage
    from skimage.morphology import skeletonize

    g, (gw, gh) = glyph()
    alpha = raster_path(g, height_px / gh, pad=8)
    mask = alpha > 0.5
    skel = skeletonize(mask)
    # hairline tips break off the skeleton; drop the crumbs so those pixels inherit
    # the time of the stroke they belong to instead of flashing in early
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    sizes = ndimage.sum(skel, lab, range(1, n + 1))
    skel = np.isin(lab, 1 + np.nonzero(sizes >= 0.05 * sizes.max())[0])
    ys, xs = np.nonzero(skel)
    geo = np.full(skel.shape, -1.0, np.float32)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    offset = 0.0
    while remaining:
        # start each connected piece at its top-left-most point
        sy, sx = min(remaining, key=lambda p: p[0] + p[1])
        q = deque([(sy, sx)])
        geo[sy, sx] = offset
        remaining.discard((sy, sx))
        top = offset
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in remaining:
                        geo[ny, nx] = geo[y, x] + (1.4142 if dy and dx else 1.0)
                        top = max(top, geo[ny, nx])
                        remaining.discard((ny, nx))
                        q.append((ny, nx))
        offset = top * 0.35       # later pieces start while the first is still inking
    _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    t = geo[iy, ix]
    t = t / max(t.max(), 1e-6)
    t = ndimage.gaussian_filter(t, 1.5)      # soften skeleton-branch seams
    return alpha, t.astype(np.float32)
