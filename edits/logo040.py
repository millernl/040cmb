"""The 040COMBAT badge as vector paths, rebuilt from the supplied logo (work/logo_supplied.jpg,
not in git — the mark is the gym's).

The 320 px image is upscaled 8x and split by colour. The octagons and the banner are straight-
edged, so they are fitted as polygons; the letterforms and stars are traced as smoothed
contours. Everything shares one coordinate space (the 8x canvas) and comes back as skia paths,
so the badge renders crisply at any size and each part can animate on its own.

    python edits/logo040.py        # rebuild work/logo/paths.json and a preview
"""
import functools
import json
from pathlib import Path

import cv2
import numpy as np
import skia
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "work" / "logo_supplied.jpg"
CACHE = ROOT / "work" / "logo" / "paths.json"
S = 8


def _sm(m, s):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), s) > 0.5


def _masks():
    from PIL import Image
    im = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32) / 255
    h, w = im.shape[:2]
    big = np.clip(cv2.resize(im, (w * S, h * S), interpolation=cv2.INTER_CUBIC), 0, 1)
    hsv = cv2.cvtColor(big, cv2.COLOR_RGB2HSV)
    lum = big @ np.array([0.299, 0.587, 0.114], np.float32)
    badge = ndimage.binary_fill_holes(_sm(lum < 0.93, 6))
    gold = _sm((hsv[..., 1] > 0.35) & (hsv[..., 0] > 25) & (hsv[..., 0] < 60) & (hsv[..., 2] > 0.35), 4)
    white = _sm((lum > 0.72) & (hsv[..., 1] < 0.25), 3.5)
    return badge, gold, white


def _hull_poly(mask, n=8):
    """The convex hull of a mask, simplified until it has n corners."""
    ys, xs = np.nonzero(mask)
    hull = cv2.convexHull(np.stack([xs, ys], -1).astype(np.int32))
    eps = 2.0
    p = hull
    while len(p) > n and eps < 200:
        p = cv2.approxPolyDP(hull, eps, True)
        eps *= 1.15
    return p[:, 0, :].astype(float)


def _inset(poly, d):
    """Move every edge of a convex polygon inward by d and intersect neighbours."""
    c = poly.mean(0)
    n = len(poly)
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        t = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-t[1], t[0]])
        if np.dot(c - a, nrm) < 0:
            nrm = -nrm
        lines.append((a + nrm * d, t))
    out = []
    for i in range(n):
        (p1, t1), (p2, t2) = lines[i - 1], lines[i]
        A = np.array([t1, -t2]).T
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t1 * s[0])
    return np.array(out)


def _contours(mask, eps=2.0, min_area=1500):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if abs(cv2.contourArea(c)) < min_area:
            continue
        p = c[:, 0, :].astype(np.float32)
        k = 6
        pad = np.concatenate([p[-k:], p, p[:k]])
        p = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in range(2)], -1)
        p = cv2.approxPolyDP(np.ascontiguousarray(p.reshape(-1, 1, 2), np.float32), eps, True)[:, 0, :]
        out.append(p.astype(float).tolist())
    return out


def _glyphs(mask, min_area=1500):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = []
    for i in sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_LEFT]):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        out.append(dict(box=[int(v) for v in stats[i, :4]], contours=_contours(lab == i)))
    return out


def build_paths():
    badge, gold, white = _masks()
    # the banner: the rows where the badge is widest (it overhangs the octagon)
    widths = badge.sum(1)
    rows = np.nonzero(widths > widths.max() * 0.97)[0]
    b0, b1 = int(rows.min()), int(rows.max())
    mid = (b0 + b1) // 2
    gx = np.nonzero(gold[mid - 50:mid + 50].any(0))[0]
    bl, br = int(gx.min()), int(gx.max())
    gcol = np.nonzero(gold[:, (bl + br) // 2 - 300:(bl + br) // 2 + 300][b0 - 60:b1 + 60].any(1))[0] + b0 - 60
    # the banner's gold frame: its outer edge rows near b0/b1
    top_rows = gcol[gcol < mid]
    bot_rows = gcol[gcol > mid]
    bt = int(top_rows.min()) if len(top_rows) else b0
    bb = int(bot_rows.max()) if len(bot_rows) else b1
    frame = int(np.argmin(gold[mid, bl:bl + 300])) or 16
    # octagons fitted with the banner rows masked out
    keep = np.ones_like(badge)
    keep[bt - 20:bb + 21] = False
    ring_outer = _hull_poly(gold & keep & ~_near_banner(badge.shape, bl, br, bt, bb))
    # the light rim and the black body are parallel to the ring: measured down the centre line
    x = int(ring_outer[:, 0].mean())
    ring_top = int(np.nonzero(gold[:bt - 30, x])[0].min())
    rim_top = int(np.nonzero(badge[:, x])[0].min())
    lum_col = _lum()[:, x]
    body_top = rim_top + int(np.argmax(lum_col[rim_top:ring_top] < 0.3))
    outer = _inset(ring_outer, -(ring_top - rim_top))
    body = _inset(ring_outer, -(ring_top - body_top))
    rim_banner = ring_top - body_top
    col = gold[:bt - 30, (bl + br) // 2]
    edges = np.nonzero(np.diff(col.astype(np.int8)))[0]
    t_ring = float(edges[1] - edges[0]) if len(edges) >= 2 else 40.0
    ring_inner = _inset(ring_outer, t_ring)
    inner_mask = np.zeros(badge.shape, np.uint8)
    cv2.fillPoly(inner_mask, [ring_inner.astype(np.int32)], 1)
    inner_mask = inner_mask > 0
    combat = white.copy()
    combat[:bt + frame + 6] = False
    combat[bb - frame - 6:] = False
    combat[:, :bl + frame + 6] = False
    combat[:, br - frame - 6:] = False
    o40 = white & inner_mask
    o40[bt - 10:] = False
    stars = gold & cv2.erode(inner_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    stars[:bb + 10] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(stars.astype(np.uint8))
    star_specs = []                                   # (cx, cy, R): regular five-point stars, point up
    for i in sorted(range(1, n), key=lambda i: st[i, 0]):
        x0, y0, w0, h0, a0 = st[i]
        if a0 < 2500:
            continue
        R = w0 / 1.902
        star_specs.append([x0 + w0 / 2, y0 + h0 / 2 + 0.0955 * R, R])
    data = dict(canvas=badge.shape[0], outer=outer.tolist(), body=body.tolist(), ring_outer=ring_outer.tolist(),
                ring_inner=ring_inner.tolist(), banner=[bl, bt, br, bb], frame=frame,
                rim_banner=rim_banner, combat=_glyphs(combat), o40=_glyphs(o40), stars=star_specs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data))
    return data


def _near_banner(shape, bl, br, bt, bb):
    m = np.zeros(shape, bool)
    m[bt - 40:bb + 41, bl - 40:br + 41] = True
    return m


@functools.cache
def _lum():
    from PIL import Image
    im = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32) / 255
    h, w = im.shape[:2]
    big = np.clip(cv2.resize(im, (w * S, h * S), interpolation=cv2.INTER_CUBIC), 0, 1)
    return big @ np.array([0.299, 0.587, 0.114], np.float32)


def star_path(cx, cy, R, r_ratio=0.382, rot=0.0):
    p = skia.Path()
    pts = []
    for k in range(10):
        a = -np.pi / 2 + rot + k * np.pi / 5
        r = R if k % 2 == 0 else R * r_ratio
        pts.append(skia.Point(cx + r * np.cos(a), cy + r * np.sin(a)))
    p.addPoly(pts, True)
    return p


@functools.cache
def paths():
    data = json.loads(CACHE.read_text()) if CACHE.exists() else build_paths()

    def poly(pts):
        p = skia.Path()
        p.addPoly([skia.Point(*q) for q in pts], True)
        return p

    def glyph(g):
        p = skia.Path()
        p.setFillType(skia.PathFillType.kEvenOdd)
        for c in g["contours"]:
            p.addPoly([skia.Point(*q) for q in c], True)
        return p

    bl, bt, br, bb = data["banner"]
    return dict(canvas=data["canvas"], outer=poly(data["outer"]), body=poly(data["body"]),
                ring_outer=poly(data["ring_outer"]), ring_inner=poly(data["ring_inner"]),
                ring_pts=np.array(data["ring_outer"]), banner=skia.Rect.MakeLTRB(bl, bt, br, bb),
                frame=data["frame"], rim_banner=data["rim_banner"],
                combat=[(glyph(g), g["box"]) for g in data["combat"]],
                o40=[(glyph(g), g["box"]) for g in data["o40"]],
                stars=[tuple(sp) for sp in data["stars"]])


def draw_badge(c, P, alpha=1.0):
    """The finished badge, in canvas units (the caller sets the transform)."""
    col = lambda r, g, b, a=1.0: skia.Paint(AntiAlias=True, Color4f=skia.Color4f(r, g, b, a * alpha))  # noqa: E731
    gold = (192 / 255, 155 / 255, 83 / 255)
    b = P["banner"]
    rb = P["rim_banner"]
    c.drawPath(P["outer"], col(0.80, 0.80, 0.80))
    c.drawRRect(skia.RRect.MakeRectXY(b.makeOutset(rb, rb), rb * 0.9, rb * 0.9), col(0.80, 0.80, 0.80))
    c.drawPath(P["body"], col(0.02, 0.02, 0.02))
    c.drawPath(P["ring_outer"], col(*gold))
    c.drawPath(P["ring_inner"], col(0.02, 0.02, 0.02))
    c.drawRRect(skia.RRect.MakeRectXY(b, 24, 24), col(*gold))
    f = P["frame"]
    c.drawRRect(skia.RRect.MakeRectXY(b.makeInset(f, f), 14, 14), col(0.02, 0.02, 0.02))
    for g, _ in P["combat"] + P["o40"]:
        c.drawPath(g, col(0.96, 0.96, 0.96))
    for cx, cy, R in P["stars"]:
        c.drawPath(star_path(cx, cy, R), col(*gold))


def preview(path):
    from PIL import Image
    P = paths()
    n = 900
    buf = np.zeros((n, n, 4), np.uint8)
    surf = skia.Surface(buf, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    c = surf.getCanvas()
    c.clear(skia.Color4f(0.03, 0.03, 0.03, 1))
    c.scale(n / P["canvas"], n / P["canvas"])
    draw_badge(c, P)
    surf.flushAndSubmit()
    Image.fromarray(buf[..., :3].copy()).save(path)


if __name__ == "__main__":
    d = build_paths()
    print({k: (len(v) if isinstance(v, list) else v) for k, v in d.items()})
    paths.cache_clear()
    preview(str(CACHE.parent / "rebuilt.png"))
