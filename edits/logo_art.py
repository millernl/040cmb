"""The real 040COMBAT logo (work/logo2/040_Combat_logo.pdf, the gym's own file — not in git),
split into its vector layers so each part can animate on its own and stay crisp at any size.

The PDF draws the badge in a fixed order; the content stream is cut into those groups and each
is rendered alone (pymupdf), with alpha:

  rim      the light-grey outer silhouette (octagon + banner wings)
  body     the black octagon
  stars    three gold stars (axial gold gradient)
  o40      the white "040"
  banner   the black banner
  bits     four small gold caps at the banner ends
  gold     the gold octagon ring and banner frame (one compound path, axial gold gradient)
  combat   the white "COMBAT", black-outlined

`layers(px)` returns float32 RGBA (straight alpha) arrays of the page at px × px, cached.
`ring_and_frame(px)` separates the gold into the ring and the banner frame; `full_ring(px)` is the
ring completed behind the banner (the door), and `rim_parts(px)` splits the rim into the octagon
and the banner wings.
"""
import functools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "work" / "logo2" / "040_Combat_logo.pdf"
CACHE = ROOT / "work" / "logo2" / "layers"
PAGE = 300.0                                   # points

# content-stream line ranges (inclusive) of each group, read off the file
GROUPS = {
    "body": [(4, 25)],
    "rim": [(26, 100)],
    "stars": [(102, 155)],
    "o40": [(156, 225), "Q"],
    "banner": [(156, 158), (226, 238), "Q"],
    "bits": [(240, 271)],
    "gold": [(272, 368)],
    "combat": [(369, -4)],
}
ORDER = ["rim", "body", "gold", "bits", "banner", "stars", "o40", "combat"]   # draw order (as in the PDF)
BANNER_PT = (40.454, 122.889, 259.545, 177.111)      # the black banner, page points (top-down y)
OCT_PT = (62.160, 62.161, 237.839, 237.840)           # the black octagon


def _stream(lines, spec):
    out = ["q", "0 300 300 -300 re", "W n", "/GS0 gs"]
    for part in spec:
        if isinstance(part, str):
            out.append(part)
        else:
            a, b = part
            b = len(lines) + b if b < 0 else b
            out += lines[a:b + 1]
    out += ["Q", ""]
    return "\n".join(out).encode("latin1")


def render(name, px):
    import pymupdf
    doc = pymupdf.open(str(PDF))
    page = doc[0]
    lines = page.read_contents().decode("latin1").split("\n")
    xref = page.get_contents()[0]
    for x in page.get_contents()[1:]:
        doc.update_stream(x, b"")
    doc.update_stream(xref, _stream(lines, GROUPS[name]))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(px / PAGE, px / PAGE), alpha=True)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).astype(np.float32) / 255
    return a[:px, :px]


@functools.cache
def layers(px=2400):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in GROUPS:
        f = CACHE / f"{name}_{px}.npy"
        if f.exists():
            out[name] = np.load(f).astype(np.float32) / 255
        else:
            out[name] = render(name, px)
            np.save(f, (out[name] * 255).astype(np.uint8))
    return out


@functools.cache
def ring_and_frame(px=2400):
    """The gold, split: the octagon ring (outside the banner) and the banner frame (inside it)."""
    g = layers(px)["gold"]
    s = px / PAGE
    x0, y0, x1, y1 = BANNER_PT
    pad = 3.0
    m = np.zeros(g.shape[:2], np.float32)
    m[int((y0 - pad) * s):int((y1 + pad) * s), int((x0 - pad) * s):int((x1 + pad) * s)] = 1.0
    ring, frame = g.copy(), g.copy()
    ring[..., 3] *= 1 - m
    frame[..., 3] *= m
    return ring, frame


def composite(names, px=2400, bg=None):
    L = layers(px)
    out = np.zeros((px, px, 3), np.float32) if bg is None else bg.copy()
    for n in ORDER:
        if n in names:
            a = L[n][..., 3:4]
            out = out * (1 - a) + L[n][..., :3] * a
    return out


# ── the door: a complete octagon ring and its interior ───────────────────────
OCT_SIDE_X = (78.079, 221.921)                  # the ring's vertical sides (outer edge), points
OCT_SIDE_Y = (121.469, 178.530)                 # their extent, top-down points
SH1 = dict(x0=-136.1016, width=452.8886, bounds=(0.390143, 0.618083, 0.890418),
           dark=(0.631373, 0.431373, 0.160784), light=(0.878431, 0.784314, 0.388235))


def gold_at(x_pt):
    """The PDF's axial gold (Sh1) at page x, as RGB."""
    t = (np.asarray(x_pt, np.float32) - SH1["x0"]) / SH1["width"]
    b0, b1, b2 = SH1["bounds"]
    d, l = np.array(SH1["dark"], np.float32), np.array(SH1["light"], np.float32)
    up = np.clip((t - b0) / (b1 - b0), 0, 1)
    down = np.clip((t - b1) / (b2 - b1), 0, 1)
    k = np.where(t < b1, up, 1 - down)[..., None]
    return d + (l - d) * k


@functools.cache
def full_ring(px=3200):
    """The gold octagon ring, complete. In the logo its two vertical sides are hidden behind the
    banner; here the ring is cut just outside the banner and the cut ends are extended straight
    down to meet each other. The gold is an axial gradient along x, so a copied row is exact.
    Returns (rgba, window): window is everything inside the ring's inner edge, tucked under it."""
    from scipy import ndimage
    g = layers(px)["gold"]
    s = px / PAGE
    bx0, by0, bx1, by1 = BANNER_PT
    pad = 1.4
    r0, r1 = int((by0 - pad) * s), int((by1 + pad) * s) + 1         # the rows the banner covers
    c0, c1 = int((bx0 - pad) * s), int((bx1 + pad) * s) + 1
    ring = g.copy()
    ring[r0:r1, c0:c1, 3] = 0
    above, below = ring[r0 - 1].copy(), ring[r1].copy()
    runs = np.diff(np.r_[0, (above[:, 3] > 0.5).astype(np.int8), 0])
    if (runs == 1).sum() != 2:
        raise RuntimeError(f"expected two ring stubs above the banner, found {(runs == 1).sum()}")
    row = np.where((above[:, 3] >= below[:, 3])[:, None], above, below)
    ring[r0:r1, c0:c1] = row[None, c0:c1]
    lab, _ = ndimage.label(ring[..., 3] < 0.5)
    inside = lab == lab[px // 2, px // 2]
    inside = ndimage.binary_dilation(inside, iterations=max(2, int(0.6 * s)))
    return ring.astype(np.float32), inside.astype(np.float32)


def rim_parts(px=2400):
    """The grey rim split into the octagon outline and the banner wings."""
    r = layers(px)["rim"]
    s = px / PAGE
    x0, y0, x1, y1 = BANNER_PT
    pad = 8.0
    wing = np.zeros(r.shape[:2], np.float32)
    rows = slice(int((y0 - pad) * s), int((y1 + pad) * s))
    wing[rows, : int((OCT_SIDE_X[0] - 6) * s)] = 1
    wing[rows, int((OCT_SIDE_X[1] + 6) * s):] = 1
    octa, wings = r.copy(), r.copy()
    octa[..., 3] *= 1 - wing
    wings[..., 3] *= wing
    return octa, wings


if __name__ == "__main__":
    from PIL import Image
    L = layers(1200)
    tiles = []
    for n in ORDER:
        rgb, a = L[n][..., :3], L[n][..., 3:4]
        tiles.append(rgb * a + 0.16 * (1 - a))
    ring, frame = ring_and_frame(1200)
    for x in (ring, frame):
        tiles.append(x[..., :3] * x[..., 3:4] + 0.16 * (1 - x[..., 3:4]))
    tiles.append(composite(set(ORDER), 1200, bg=np.full((1200, 1200, 3), 0.16, np.float32)))
    row1 = np.concatenate(tiles[:6], 1)
    row2 = np.concatenate(tiles[6:] + [np.full_like(tiles[0], 0.16)] * (12 - len(tiles)), 1)
    im = np.concatenate([row1, row2], 0)
    Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).resize((im.shape[1] // 3, im.shape[0] // 3), Image.LANCZOS).save(
        str(CACHE.parent / "layers.jpg"), quality=88)
    print("ok")
