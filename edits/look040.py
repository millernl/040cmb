"""The 040COMBAT look: one grade for every shot, built like a colorist's node tree.

  1  match       white balance on near-neutral midtones (walls, mats, gis), black/white points,
                 a common exposure — so footage from different phones and rooms starts equal
  2  separation  the people matte (RVM) sits the room lower and a little less saturated than
                 the athletes, through a broad pool of light, never an outline: the eye goes to
                 people, the gym stays readable
  3  colour      hue-selective saturation (skin and gold keep it, blues/greens/magentas are
                 pulled down), then a print-film tone curve with a soft toe and a long shoulder
  4  balance     split toning: shadows toward a cool graphite, highlights toward warm ivory —
                 the separation premium grades live on, kept subtle
  5  diffusion   a Pro-Mist-style highlight bloom: bright areas glow softly into their
                 surroundings, which takes the digital edge off phone footage
  6  (film)      halation, grain and weave come from afterfilm's cinema/finish passes

Every parameter lives in LOOK so the whole film can be retuned in one place.
"""
import cv2
import numpy as np

from afterfilm import fx

LUMA = fx.LUMA

LOOK = dict(
    sep_drop=0.6,          # stops the room sits below the athletes
    sep_desat=0.35,        # saturation the room loses
    sep_pool=110,          # radius of the pool of light around the people (px at 1080 wide)
    sep_gain=2.2,          # how quickly the pool reaches full light
    sep_soft=8,            # softness of the lift on the people themselves
    sat=0.78,              # global saturation after the hue weights
    skin_keep=1.0,         # weight for skin/gold hues (15–50°)
    cool_cut=0.45,         # weight for blues/cyans
    green_cut=0.35,        # weight for greens (gym mats and walls go green under LEDs)
    contrast=0.20,         # print S-curve on top of the filmic curve
    toe=0.018,             # black level after the print (lifted, never crushed)
    shadow_tint=(-0.006, 0.002, 0.010),   # cool graphite in the shadows
    high_tint=(0.022, 0.008, -0.022),     # warm ivory in the highlights
    mist=0.22,             # highlight diffusion
    expo=-0.12,            # overall exposure trim (stops)
    white=1.0,             # the brightest the print gets (below 1: highlights stay off pure white)
)


def measure(frames):
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = np.stack([cv2.resize(frames[i], (96, 170), interpolation=cv2.INTER_AREA) for i in idx]).astype(np.float32) / 255
    px = small.reshape(-1, 3)
    y = px @ LUMA
    ch = px.max(1) - px.min(1)
    m = (y > 0.15) & (y < 0.85) & (ch < 0.12)
    sel = px[m] if m.mean() > 0.03 else px[(y > 0.1) & (y < 0.9)]
    mean = sel.mean(0)
    gain = np.clip((mean @ LUMA / mean) ** 0.85, 0.8, 1.25).astype(np.float32)
    yg = (px * gain) @ LUMA
    lo, mid, hi = np.percentile(yg, [0.8, 50, 99.6])
    black = max(0.0, lo - 0.008) * 0.9
    scale = float(np.clip(0.98 / max(hi - black, 1e-3), 1.0, 1.35))
    expo = float(np.clip(np.log2(0.30 / max((mid - black) * scale, 1e-3)) * 0.55, -0.45, 0.55))
    return dict(gain=gain, black=float(black), scale=scale, expo=expo)


NOMATCH = dict(gain=np.ones(3, np.float32), black=0.0, scale=1.0, expo=0.0)


def _hue_weights(x, L):
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]

    def bump(c, width):
        d = np.abs(((h - c + 180) % 360) - 180)
        return np.clip(1 - d / width, 0, 1)
    skin = np.maximum(bump(30, 26), bump(12, 14) * 0.8)
    cool = np.maximum(bump(205, 50), bump(250, 40))
    green = bump(120, 45)
    w = 1.0 + (L["skin_keep"] - 1.0) * skin
    w = w * (1 - (1 - L["cool_cut"]) * cool) * (1 - (1 - L["green_cut"]) * green)
    return w


def grade(img, match=None, alpha=None, exposure=0.0, look=None, **over):
    L = {**LOOK, **(look or {}), **{k: v for k, v in over.items() if k in LOOK}}
    x = np.clip(img, 0, 1).astype(np.float32)
    e = exposure + L["expo"]
    if match:
        x = np.clip((x * match["gain"] - match["black"]) * match["scale"], 0, 1)
        e += match["expo"]
    lin = x ** 2.2 * (2.0 ** e)
    h, w = lin.shape[:2]
    k = w / 1080
    sat_w = _hue_weights(np.clip(x, 0, 1), L) * L["sat"]
    if alpha is not None and L["sep_drop"] > 0:
        # a broad pool of light where the people are, not an outline around them: a tight
        # matte-shaped lift reads as a cut-out with a glow (the cheap look); this does not
        a = alpha.astype(np.float32)
        small = cv2.resize(a, (max(w // 16, 1), max(h // 16, 1)), interpolation=cv2.INTER_AREA)
        pool = cv2.GaussianBlur(small, (0, 0), L["sep_pool"] * k / 16)
        pool = cv2.resize(np.clip(pool * L["sep_gain"], 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        near = cv2.GaussianBlur(a, (0, 0), L["sep_soft"] * k)
        light = np.clip(np.maximum(pool, near * 0.85), 0, 1)
        lin = lin * (2.0 ** (-L["sep_drop"] * (1 - light)))[..., None]
        sat_w = sat_w * (1 - L["sep_desat"] * (1 - light))
    y = lin @ LUMA
    lin = y[..., None] + (lin - y[..., None]) * sat_w[..., None]
    lin = np.clip(lin, 0, None)
    # print: filmic toe/shoulder in display space, then a gentle S
    disp = fx.filmic(np.clip(lin ** (1 / 2.2), 0, 1))
    c = L["contrast"]
    disp = disp + c * (disp * disp * (3 - 2 * disp) - disp)
    Y = (disp @ LUMA)[..., None]
    sh = (1 - fx.smoothstep(0.05, 0.5, Y)) * fx.smoothstep(0.0, 0.12, Y)
    hi = fx.smoothstep(0.45, 1.0, Y)
    disp = disp + sh * np.array(L["shadow_tint"], np.float32) + hi * np.array(L["high_tint"], np.float32)
    # diffusion: highlights bloom softly into their surroundings
    if L["mist"] > 0:
        q = cv2.resize(disp, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hot = np.clip((q - 0.55) / 0.45, 0, 1) ** 1.3
        glow = cv2.GaussianBlur(hot, (0, 0), 9 * k) * 0.6 + cv2.GaussianBlur(hot, (0, 0), 28 * k) * 0.4
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        disp = 1 - (1 - np.clip(disp, 0, 1)) * (1 - np.clip(glow * L["mist"], 0, 1))
    t = L["toe"]
    return np.clip(t + np.clip(disp, 0, 1) * (L["white"] - t), 0, 1).astype(np.float32)
