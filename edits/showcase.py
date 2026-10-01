"""040COMBAT, Deurne — a 26-second 9:16 showcase.

Cut to Future & Metro Boomin's "Everyday Hustle" (music/, not in git; 119.67 BPM, a bar is
2.006 s). The intro builds under the opening, pauses on a freeze, the drop lands on the
first competition photo, the breakdown carries the people, and the beat returns for the
last burst and the end card.

  0.0–8.5    intro · a high kick lands on the first downbeat, a takedown, a women's
             round, pads into a double-leg, the full class, Milosz trading in front of
             the wall logo
  8.5–9.5    the music holds its breath on a black & white freeze of his punch
  9.5–15.5   the drop · Milosz's hand raised, Imre's arms up, the two together; back
             into motion with a kid on the pads and a round
  15.5–21.6  the breakdown · his laugh after the round, a coach on the mat, the team
             photo from the faces out to the whole room
  21.6–26.7  the beat returns · four hits, the last a pad strike under the wall logo
             that freezes into 040COMBAT / DEURNE

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
from afterfilm.timeline import Clip, FrameBuffer, Shot, Timeline, VideoSource  # noqa: E402

FOOT = ROOT / "footage"
MUSIC = ROOT / "music" / "Everyday_Hustle.m4a"
FPS = 30

# ── identity ─────────────────────────────────────────────────────────────────
# measured from the supplied logo: black, a warm gold (#C09B53), white → off-white
INK = (8, 8, 8)
GOLD = (192, 155, 83)
OFFWHITE = (242, 237, 228)
FONT_HEAVY = "BarlowCondensed-ExtraBold"
FONT_TEXT = "BarlowCondensed-SemiBold"

# ── the music's grid ─────────────────────────────────────────────────────────
BEAT, OFF = 0.50143, 0.047                  # fitted on the kick pattern across the track


def SB(j):
    """Song time of bar j's downbeat."""
    return OFF + 4 * BEAT * j


BAR = 4 * BEAT
V_A0 = SB(8) - BEAT                          # one beat of pickup before the first downbeat
V_PAUSE = SB(12) - V_A0                      # 8.524: the intro stops on its bar line
V_DROP = V_PAUSE + 2 * BEAT                  # 9.527: two beats of held breath
V_BREAK = V_DROP + 3 * BAR                   # 15.544: into the breakdown
V_PRE = V_BREAK + 2 * BAR                    # 19.555: the breakdown's last bar (an 808, then air)
V_RETURN = V_PRE + BAR                       # 21.561: the beat returns
V_CARD = V_RETURN + BAR                      # 23.567: the end card
V_LAST = V_CARD + BAR                        # 25.572: the last downbeat
DURATION = V_LAST + 1.15

# music segments: (video start, song start, song end, gain dB) — the intro and the breakdown
# lifted a little so the quiet parts still carry on a phone
MUSIC_PLAN = [(0.0, V_A0, SB(12), 3.0),
              (V_DROP, SB(32), SB(35), 0.0),
              (V_BREAK, SB(52), SB(54), 2.0),
              (V_PRE, SB(59), SB(62) + 0.9, 1.0)]


def vb(t0, k):
    """Video time of beat k counted from t0."""
    return t0 + k * BEAT


# ── sources ──────────────────────────────────────────────────────────────────
_SRC = {}
# the dim 1080p clips get a light temporal denoise on decode; the rest are clean 4K
DENOISE = {"action.mp4": "hqdn3d=2:1.5:4:3", "wall.drill.near.the.end.mp4": "hqdn3d=1.5:1.5:3:3",
           "uitleg.mma.mp4": "hqdn3d=1.5:1.5:3:3"}


def src(name):
    if name not in _SRC:
        _SRC[name] = VideoSource(str(FOOT / name), rescue=False, extra_pre=DENOISE.get(name))
    return _SRC[name]


class StillSource:
    """A photograph as a one-frame source; framing happens in Clip.frame (reframe)."""

    def __init__(self, name):
        self.path = str(FOOT / name)
        self.img = np.asarray(ImageOps.exif_transpose(Image.open(self.path)).convert("RGB"))

    def read(self, t0, t1, fps, size):
        w, h = size
        ih, iw = self.img.shape[:2]
        s = max(w / iw, h / ih)
        out = cv2.resize(self.img, (max(w, int(round(iw * s))), max(h, int(round(ih * s)))), interpolation=cv2.INTER_AREA)
        return FrameBuffer(out[None], t0, fps)


def measure(frames):
    """Per-shot match: neutralise the white balance on near-neutral midtones (walls, mats),
    set black and white points, and bring the exposure to a common level."""
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = np.stack([cv2.resize(frames[i], (96, 170), interpolation=cv2.INTER_AREA) for i in idx]).astype(np.float32) / 255
    px = small.reshape(-1, 3)
    y = px @ fx.LUMA
    chroma = px.max(1) - px.min(1)
    neutral = (y > 0.12) & (y < 0.85) & (chroma < 0.10)
    sel = px[neutral] if neutral.mean() > 0.03 else px[(y > 0.1) & (y < 0.9)]
    mean = sel.mean(0)
    gain = np.clip((mean @ fx.LUMA / mean) ** 0.7, 0.85, 1.2)          # 70% of the way to neutral
    py = (px * gain) @ fx.LUMA
    lo, mid, hi = np.percentile(py, [1.0, 50, 99.6])
    black = max(0.0, lo - 0.01) * 0.85
    scale = float(np.clip(0.97 / max(hi - black, 1e-3), 1.0, 1.3))
    mid2 = (mid - black) * scale
    expo = float(np.clip(np.log2(0.36 / max(mid2, 1e-3)) * 0.45, -0.35, 0.45))
    return dict(gain=gain.astype(np.float32), black=float(black), scale=scale, expo=expo)


class MClip(Clip):
    """A clip that measures itself the first time it is decoded."""

    def prepare(self, dur, fps, size):
        fresh = self.buf is None
        super().prepare(dur, fps, size)
        if fresh and "match" not in self.grade:
            self.grade = {**self.grade, "match": measure(self.buf.frames)}


def clip(name, t_in, speed=1.0, zoom=(1.0, 1.0), center=(0.5, 0.5), center_end=None, note="", **grade):
    s = src(name)
    zmax = max(zoom)
    info = s.info
    sw, sh = sorted((info["width"], info["height"]))      # portrait after rotation
    dw = int(min(sw, max(1080, round(1080 * zmax))) // 2 * 2)
    dh = int(min(sh, max(1920, round(1920 * zmax))) // 2 * 2)
    return MClip(s, t_in, speed=speed, zoom=zoom, center=center, center_end=center_end, note=note,
                 grade=grade, decode=(dw, dh))


NOMATCH = dict(gain=np.ones(3, np.float32), black=0.0, scale=1.0, expo=0.0)


def photo(name, zoom=(1.0, 1.06), center=(0.5, 0.5), center_end=None, note="", **grade):
    grade.setdefault("match", NOMATCH)
    return MClip(StillSource(name), 0.0, speed=0.0, zoom=zoom, center=center, center_end=center_end, note=note,
                 grade=grade, decode=(int(1080 * max(zoom) * 1.05) // 2 * 2, int(1920 * max(zoom) * 1.05) // 2 * 2))


# ── the look ─────────────────────────────────────────────────────────────────
class CombatLook(fx.Look):
    """Matched shot by shot first (MClip.measure), then one gentle shared look: filmic
    contrast, natural skin, deep neutral blacks, a touch of warmth in the highlights."""

    SAT, CONTRAST = 0.88, 0.16

    def __init__(self, w, h):
        super().__init__(w, h, accent=GOLD)
        self.black = np.array([0.008, 0.008, 0.008], np.float32)
        self.warm = (brand.f32(GOLD) - float(brand.f32(GOLD) @ fx.LUMA)).astype(np.float32)

    def grade(self, img, match=None, mono=0.0, contrast=None, sat=None, exposure=0.0, warm=1.0, tint=None, **_):
        x = np.clip(img, 0, 1).astype(np.float32)
        e = exposure
        if match:
            x = np.clip((x * match["gain"] - match["black"]) * match["scale"], 0, 1)
            e += match["expo"]
        if tint is not None:
            x = np.clip(x * np.asarray(tint, np.float32), 0, 1)
        if e:
            x = np.clip(x * (2.0 ** e), 0, 1)
        y = luma_(x)
        s = (self.SAT if sat is None else sat) * (1 - mono)
        x = y + (x - y) * s
        if mono:
            bw = np.clip(x @ np.array([0.45, 0.45, 0.10], np.float32), 0, 1)[..., None]
            x = x * (1 - mono) + bw * mono
        x = fx.filmic(x)
        c = self.CONTRAST if contrast is None else contrast
        x = x + c * (x * x * (3 - 2 * x) - x)
        y = luma_(x)
        x = x + fx.smoothstep(0.55, 1.0, y) * self.warm * 0.07 * warm * (1 - mono)
        x = self.black + x * (1 - self.black)
        return np.clip(x, 0, 1).astype(np.float32)


def luma_(x):
    return (x @ fx.LUMA)[..., None]


# ── the edit ─────────────────────────────────────────────────────────────────
def kick_ramp(t_impact_local, slow=0.35, ease=0.08):
    return [(0, 1.0), (t_impact_local + 0.02, 1.0), (t_impact_local + 0.02 + ease, slow), (9, slow)]


def build(w=1080, h=1920):
    look = CombatLook(w, h)
    tl = Timeline(w, h, FPS, DURATION, look)
    S = tl.add

    # 0 · intro: the opening hit on the first downbeat
    t = 0.0
    S(Shot(t, vb(0.501, 1) - t, clip("kickboxing.drills.mp4", 21.39 - 0.501, speed=kick_ramp(0.501),
                                      zoom=(1.04, 1.12), center=(0.5, 0.5), note="Milosz's high kick lands on the downbeat")))
    t = vb(0.501, 1)                                                     # 1.003
    S(Shot(t, vb(0.501, 3) - t, clip("Dilbrien.x.Milosz.scramble.mp4", 19.95, zoom=(1.75, 1.85), center=(0.47, 0.40),
                                      note="a body-lock lift and drop"), punch=0.03))
    t = vb(0.501, 3)                                                     # 2.005
    S(Shot(t, vb(0.501, 5) - t, clip("alternative.shot.mp4", 8.30 - (vb(0.501, 4) - t), zoom=(1.0, 1.05),
                                      center=(0.5, 0.5), note="her body kick lands on the beat")))
    t = vb(0.501, 5)                                                     # 3.008
    # pads → double-leg: real time through the combination, the lift in slow motion
    S(Shot(t, vb(0.501, 10) - t, clip("action.mp4", 3.04, speed=[(0, 1.0), (1.74, 1.0), (1.80, 0.6), (2.28, 0.6), (2.34, 1.0), (9, 1.0)],
                                      zoom=(1.0, 1.06), center=(0.5, 0.52), note="pads with the coach, into a double-leg",
                                      tint=(1.025, 0.975, 1.0))))
    t = vb(0.501, 10)                                                    # 5.515
    S(Shot(t, vb(0.501, 12) - t, clip("boxing.instructies.mp4", 28.45, speed=0.5, zoom=(1.06, 1.0), center=(0.5, 0.5),
                                       note="the packed class shadowboxing, 120 fps slowed to half")))
    t = vb(0.501, 12)                                                    # 6.518
    S(Shot(t, V_DROP - t, clip("Milosz.Bilal.KB2.mp4", 10.00 - (V_PAUSE - t), zoom=(1.0, 1.04), center=(0.5, 0.5),
                                note="Milosz and Bilal trade under the wall logo → freeze"), freeze_at=V_PAUSE - t))

    # the drop: the achievements
    t = V_DROP
    S(Shot(t, 3 * BEAT, photo("Milosz.Celebration.jpg", zoom=(1.0, 1.07), center=(0.40, 0.42), note="Milosz, hand raised", exposure=-0.2),
           flash=0.7, shake=0.5))
    t += 3 * BEAT
    S(Shot(t, 3 * BEAT, photo("Imre.Celebration.jpg", zoom=(1.08, 1.0), center=(0.5, 0.38), note="Imre, arms up", exposure=-0.1), flash=0.15))
    t += 3 * BEAT
    diptych = Diptych(w, h)
    S(Shot(t, 2 * BEAT, render=diptych.render, clips=diptych.clips))
    t += 2 * BEAT                                                        # 13.538
    S(Shot(t, 2 * BEAT, clip("extra.mp4", 9.15, zoom=(1.12, 1.18), center=(0.5, 0.5), note="a kid on the pads with his coach, the room full"),
           trans=("smear", 0.13)))
    t += 2 * BEAT                                                        # 14.541
    S(Shot(t, V_BREAK - t, clip("KB3.mp4", 8.0, zoom=(1.0, 1.04), center=(0.5, 0.5), note="a round in the full room, gold gloves")))

    # the breakdown: the people
    t = V_BREAK
    S(Shot(t, 2 * BEAT, clip("Milosz.Bilal.KB2.mp4", 19.95, speed=0.8, zoom=(1.42, 1.48), center=(0.40, 0.36),
                              note="his laugh after the round")))
    t += 2 * BEAT                                                        # 16.547
    S(Shot(t, 2 * BEAT, clip("uitleg.mma.mp4", 4.45, zoom=(1.55, 1.62), center=(0.5, 0.40), note="the coach on the mat with them")))
    t += 2 * BEAT                                                        # 17.550
    S(Shot(t, V_PRE - t, photo("Of.voor.community.jpg", zoom=(2.1, 1.0), center=(0.42, 0.40), center_end=(0.52, 0.5),
                                note="the team: faces, out to the room")))
    S(Shot(V_PRE, V_RETURN - V_PRE, photo("Of.voor.communityy.jpg", zoom=(1.3, 1.04), center=(0.5, 0.45), center_end=(0.5, 0.47),
                                          note="the whole room under the logo"), flash=0.25))

    # the beat returns: four hits, the last one under the logo
    t = V_RETURN
    S(Shot(t, BEAT, clip("alternative.shot.mp4", 13.92, zoom=(1.05, 1.08), center=(0.5, 0.5), note="her combination")))
    t += BEAT
    S(Shot(t, BEAT, clip("Dilbrien.x.Milosz.scramble.mp4", 22.45, zoom=(1.5, 1.58), center=(0.42, 0.40), note="a scramble, legs flying")))
    t += BEAT
    S(Shot(t, BEAT, clip("Milosz.Bilal.mp4", 36.95, zoom=(1.08, 1.12), center=(0.5, 0.5), note="Milosz's jab, the room watching")))
    t += BEAT
    card = EndCard(w, h, clip("wall.drill.near.the.end.mp4", 8.98 - BEAT, note="a pad strike under the wall logo → the end card"))
    S(Shot(t, DURATION - t, render=card.picture, clips=[card.clip]))
    tl.overlays.append(_overlay(V_CARD, DURATION, card.draw))

    tl.cinema_at = lambda t: {"mono": 0.0, "streaks": 0.0, "halation": 0.1, "bloom": 0.1, "weave": 0.0}
    tl.grain_at = lambda t: 0.4
    tl.vignette_at = lambda t: 0.5
    tl.audio_plan = sounds()
    return tl


def _overlay(t0, t1, fn, stage="post"):
    from afterfilm.timeline import Overlay
    return Overlay(t0, t1, fn, stage)


class Diptych:
    """The two achievements together, above and below, a gold hairline between them."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        half = (w, h // 2)
        self.top = MClip(StillSource("Milosz.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.0, 1.05), center=(0.42, 0.33),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.2})
        self.bot = MClip(StillSource("Imre.Celebration.jpg"), 0.0, speed=0.0, zoom=(1.05, 1.0), center=(0.5, 0.28),
                         decode=(w * 2, h), grade={"match": NOMATCH, "exposure": -0.1})
        self.clips = [self.top, self.bot]
        self.half = half

    def render(self, shot, lt, tl):
        w, h = self.w, self.h
        hh = h // 2
        a = self.top.frame(lt, shot.dur, tl.look, out_size=(w, hh))
        b = self.bot.frame(lt, shot.dur, tl.look, out_size=(w, h - hh))
        img = np.concatenate([a, b], 0)
        # the halves slide into place from opposite sides
        e = fx.expo_out(lt / 0.28)
        dx = int(round((1 - e) * w * 0.18))
        img[:hh] = fx.shift(img[:hh], -dx, 0) if dx else img[:hh]
        img[hh:] = fx.shift(img[hh:], dx, 0) if dx else img[hh:]
        g = fx.ease_out(fx.window(lt, 0.05, 0.4))
        half = int(w * g / 2)
        if half > 0:
            col = brand.f32(GOLD)
            img[hh - 2:hh + 2, w // 2 - half:w // 2 + half] = col
        return img


class EndCard:
    """The strike under the wall logo freezes, goes black & white and drifts up toward the
    logo while the room darkens; 040COMBAT / DEURNE settle over the floor."""

    def __init__(self, w, h, clip):
        self.w, self.h, self.clip = w, h, clip
        self.f_big = brand.font(FONT_HEAVY, 100)
        tw, _ = gfx.measure("040COMBAT", self.f_big, 0.01)
        size = 100 * (0.80 * w) / tw
        self.f_big = brand.font(FONT_HEAVY, size)
        self.f_small = brand.font(FONT_TEXT, size * 0.34)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        self.shade = 0.5 + 0.5 * fx.smoothstep(0.4, 0.68, yy)            # the floor goes darkest, under the type

    def picture(self, shot, lt, tl):
        w, h = self.w, self.h
        if lt < BEAT:
            return self.clip.frame(lt, shot.dur, tl.look, out_size=(w, h))
        k = lt - BEAT
        mono = float(fx.smoothstep(0.0, 0.2, k))
        img = self.clip.frame(BEAT, shot.dur, tl.look, out_size=(w, h), mono=mono, contrast=0.32)
        img = fx.reframe(img, 1.0 + 0.05 * fx.ease_out(k / 3.0), 0.5, 0.36)
        dark = fx.ease_out(fx.window(k, 0.1, 0.7))
        img = img * (1 - dark * self.shade * 0.88)
        return fx.flash(img, max(0.0, 1 - k / 0.18) * 0.8)

    def draw(self, t, img, tl):
        lt = t - V_CARD
        w, h = self.w, self.h
        layer = tl.layer
        layer.clear()
        c = layer.c
        cy = h * 0.72
        size = self.f_big.getSize()
        # 040COMBAT rises out of a baseline mask
        e = fx.expo_out(fx.window(lt, 0.2, 0.8))
        if e > 0:
            c.save()
            c.clipRect(skia_rect(0, cy - size * 0.95, w, cy + size * 0.06))
            gfx.text(c, "040COMBAT", self.f_big, w / 2, cy + (1 - e) * size * 0.85, OFFWHITE, 1.0, tracking=0.01, align="center")
            c.restore()
        # a short gold rule, then DEURNE
        ry = cy + size * 0.30
        g = fx.ease_out(fx.window(lt, 0.5, 0.9))
        if g > 0:
            half = w * 0.06 * g
            c.drawRect(skia_rect(w / 2 - half, ry, w / 2 + half, ry + max(2.0, h / 640)), skia_paint(GOLD, 1.0))
        k = fx.ease_out(fx.window(lt, 0.65, 1.2))
        if k > 0:
            gfx.text(c, "DEURNE", self.f_small, w / 2, ry + self.f_small.getSize() * 1.25, GOLD, k,
                     tracking=0.6 + 0.3 * (1 - k), align="center")
        return layer.over(img)


def skia_rect(l, t, r, b):
    import skia
    return skia.Rect.MakeLTRB(l, t, r, b)


def skia_paint(color, a):
    import skia
    return skia.Paint(AntiAlias=True, Color4f=brand.skcolor(color, a))


# ── sound ────────────────────────────────────────────────────────────────────
def sounds():
    """Real sound from the footage, cut through the music: (video time, file, src t0, src t1, dB)."""
    action_v = lambda s: 3.008 + (s - 3.04)                  # noqa: E731  (real time through the pads)
    return [
        (0.501 - 0.03, "kickboxing.drills.mp4", 21.36, 21.70, -4),       # the kick lands
        (1.003 + (20.80 - 19.95), "Dilbrien.x.Milosz.scramble.mp4", 20.80, 21.10, -10),
        (2.507 - 0.03, "alternative.shot.mp4", 8.27, 8.55, -6),          # her kick
        (action_v(3.15), "action.mp4", 3.15, 4.15, -7),                  # the pads
        (V_PAUSE - 0.03, "Milosz.Bilal.KB2.mp4", 9.97, 10.45, -3),       # the punch the music stops on
        (V_BREAK + 0.06, "Milosz.Bilal.KB2.mp4", 20.0, 20.75, -6),       # the laugh
        (V_CARD - 0.03, "wall.drill.near.the.end.mp4", 8.95, 9.40, -3),  # the strike under the logo
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
        if k:
            seg[:xf] *= np.sin(np.linspace(0, np.pi / 2, xf))[:, None]
        seg[-xf:] *= np.cos(np.linspace(0, np.pi / 2, xf))[:, None]
        i = int(round((v0 - lead) * sr))
        seg = seg[:n - i]
        out[i:i + len(seg)] += seg
        if k == 0:
            # the pause: the intro's last instant rings out in a hall instead of stopping dead
            tail = song[int((s1 - 0.18) * sr):int(s1 * sr)]
            wet = np.stack([signal.fftconvolve(tail[:, c], _hall(sr)[:, c]) for c in range(2)], 1)
            wet *= np.linspace(1, 0, len(wet))[:, None] ** 1.5
            j = int(round(V_PAUSE * sr))
            m = min(len(wet), int(round((V_DROP - V_PAUSE - 0.02) * sr)))
            out[j:j + m] += wet[:m] * 10 ** (-9 / 20)
    # the end: the last downbeat rings, then the room
    j = int(round(V_LAST * sr))
    fade = int(0.8 * sr)
    out[j:j + fade] *= np.linspace(1, 0, min(fade, n - j))[:, None] ** 2
    out[j + fade:] = 0
    ring = song[int(SB(62) * sr):int((SB(62) + 0.4) * sr)]
    wet = np.stack([signal.fftconvolve(ring[:, c], _hall(sr, 2.2)[:, c]) for c in range(2)], 1)
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
        peak = np.abs(a).max() + 1e-6
        a = a / peak * 10 ** ((db + 4) / 20) * 0.5
        i = int(round(v * sr))
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
    ap.add_argument("--from", dest="t_from", type=float)
    ap.add_argument("--to", dest="t_to", type=float)
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
        frames = None
        if a.t_from is not None or a.t_to is not None:
            frames = range(int((a.t_from or 0) * FPS), int((a.t_to or DURATION) * FPS))
        tl.render(str(out), wav=wav, frames=frames)
