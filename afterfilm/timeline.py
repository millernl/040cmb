"""The edit: shots on one track, transitions between them, graphics on top.

Frame pipeline:  shot pictures (graded) → transition → 'pre' overlays → bloom →
'post' overlays → vignette + grain → letterbox → 'top' overlays (bar labels).
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import fx, gfx, media


# ── sources ──────────────────────────────────────────────────────────────────
class FrameBuffer:
    def __init__(self, frames, t0, fps):
        self.frames, self.t0, self.fps = frames, t0, fps

    def frame(self, t):
        n = len(self.frames)
        x = (t - self.t0) * self.fps
        i = int(np.clip(np.floor(x), 0, n - 1))
        j = min(i + 1, n - 1)
        f = float(np.clip(x - i, 0, 1))
        a = self.frames[i].astype(np.float32) / 255.0
        if j != i and 0.04 < f:
            # blend neighbours: smooth slow motion when stretching time
            a = a * (1 - f) + self.frames[j].astype(np.float32) * (f / 255.0)
        return a


class VideoSource:
    def __init__(self, path, rescue=True, extra_pre=None):
        self.path = path
        self.info = media.probe(path)
        self.rescue = rescue
        self.extra_pre = extra_pre           # e.g. "tmedian=radius=2" to strip strobe flashes

    def read(self, t0, t1, fps, size):
        native = self.info["fps"] * (2 if self.info["interlaced"] else 1)
        dfps = min(fps, native)
        frames = media.read_frames(self.path, t0, (t1 - t0) + 2 / dfps, dfps, size,
                                   pre=",".join(x for x in (self.extra_pre, media.RESCUE_PRE if self.rescue else None) if x) or None,
                                   post=media.RESCUE_POST if self.rescue else None)
        return FrameBuffer(frames, t0, dfps)


# ── clips & shots ────────────────────────────────────────────────────────────
@dataclass
class Clip:
    source: object
    t_in: float                          # source seconds at the clip's first frame
    speed: object = 1.0                  # float, or [(local_t, speed), ...] for ramps
    zoom: tuple = (1.0, 1.0)             # Ken Burns: zoom at start → end of shot
    center: tuple = (0.5, 0.5)           # framing centre, normalised
    center_end: Optional[tuple] = None
    track: Optional[list] = None         # [(p, cx), ...] horizontal framing keyed on shot progress p
    grade: dict = field(default_factory=dict)
    echo: float = 0.0                    # light trails: strength of trailing frames (0 = off)
    note: str = ""                       # what the shot is (for the EDL printout)
    hold: Optional[float] = None         # local seconds after which the clip is never read (a freeze)
    decode: Optional[tuple] = None       # (w, h) to decode at, overriding the timeline's
    buf: object = None

    def src_time(self, lt):
        if not isinstance(self.speed, (list, tuple)):
            return self.t_in + lt * self.speed
        ks = self.speed
        ts = np.linspace(0, max(lt, 0), max(int(lt * 200), 2))
        sp = np.interp(ts, [k[0] for k in ks], [k[1] for k in ks])
        return self.t_in + float(np.trapezoid(sp, ts))

    def min_speed(self):
        return min(k[1] for k in self.speed) if isinstance(self.speed, (list, tuple)) else self.speed

    def prepare(self, dur, fps, size):
        if self.buf is None:
            if self.hold is not None:
                dur = min(dur, self.hold + 0.1)
            dec_fps = fps / max(min(self.min_speed(), 1.0), 0.25)
            self.buf = self.source.read(self.src_time(0), self.src_time(dur), dec_fps, size)

    def release(self):
        self.buf = None

    def center_at(self, p):
        ce = self.center_end or self.center
        cx = self.center[0] + (ce[0] - self.center[0]) * p
        cy = self.center[1] + (ce[1] - self.center[1]) * p
        if self.track:
            ks = sorted(self.track)
            cx = float(np.interp(p, [k[0] for k in ks], [k[1] for k in ks]))
        return cx, cy

    def frame(self, lt, dur, look, freeze_at=None, out_size=None, **grade_over):
        st = self.src_time(min(lt, freeze_at) if freeze_at is not None else lt)
        img = self.buf.frame(st)
        if self.echo:
            # chronophotography: earlier instants of the move linger as light (lighten blend)
            for k in range(1, 6):
                img = np.maximum(img, self.buf.frame(st - k * 0.07) * (self.echo * 0.72 ** k))
        p = float(np.clip(lt / max(dur, 1e-6), 0, 1.2))
        z = self.zoom[0] + (self.zoom[1] - self.zoom[0]) * p
        cx, cy = self.center_at(min(p, 1.0))
        size = out_size or (img.shape[1], img.shape[0])
        moved = (cx, cy) != tuple(self.center)
        if abs(z - 1) > 1e-3 or moved or size != (img.shape[1], img.shape[0]):
            img = fx.reframe(img, z, cx, cy, out_size=size)
        return look.grade(img, **{**self.grade, **grade_over})


@dataclass
class Shot:
    start: float
    dur: float
    clip: Optional[Clip] = None
    trans: Optional[tuple] = None        # (name, seconds): transition from the previous shot
    punch: float = 0.0                   # scale kick on entry (0.06 = 6%)
    flash: float = 0.0                   # exposure flash on entry
    freeze_at: Optional[float] = None    # local seconds: freeze → duotone + slow push
    shake: float = 0.0                   # camera shake on impact, decays over ~0.4 s
    render: Optional[Callable] = None    # custom picture: render(shot, lt, tl) → frame
    clips: list = field(default_factory=list)

    @property
    def end(self):
        return self.start + self.dur

    def all_clips(self):
        return ([self.clip] if self.clip else []) + list(self.clips)

    def picture(self, t, tl):
        lt = t - self.start
        if self.render:
            img = self.render(self, lt, tl)
        elif self.freeze_at is not None and lt >= self.freeze_at:
            k = lt - self.freeze_at
            mono = float(fx.smoothstep(0.0, 0.14, k))
            img = self.clip.frame(lt, self.dur, tl.look, freeze_at=self.freeze_at, out_size=(tl.w, tl.h),
                                  mono=mono, contrast=0.5)
            img = fx.reframe(img, 1.0 + 0.045 * fx.ease_out(k / max(self.dur - self.freeze_at, 0.1)))
            img = fx.flash(img, max(0.0, 1 - k / 0.16) * 0.9)
        else:
            img = self.clip.frame(lt, self.dur, tl.look, out_size=(tl.w, tl.h))
        if self.punch:
            img = fx.reframe(img, 1 + self.punch * (1 - fx.ease_out(lt / 0.35)))
        if self.shake:
            k = np.exp(-lt / 0.16) * self.shake
            if k > 0.02:
                ph = self.start * 7.31
                dx = (np.sin(lt * 61 + ph) * 0.6 + np.sin(lt * 37 + 2 * ph) * 0.4) * 0.012 * k
                dy = (np.sin(lt * 53 + 3 * ph) * 0.6 + np.sin(lt * 29 + ph) * 0.4) * 0.012 * k
                img = fx.reframe(img, 1 + 0.03 * k, 0.5 + dx, 0.5 + dy, rot=np.sin(lt * 43 + ph) * 0.35 * k)
        if self.flash:
            img = fx.flash(img, self.flash * max(0.0, 1 - lt / 0.22))
        return img


@dataclass
class Overlay:
    t0: float
    t1: float
    fn: Callable                         # fn(t, img, tl) → img
    stage: str = "pre"                   # pre | post | top


# ── the timeline ─────────────────────────────────────────────────────────────
class Timeline:
    def __init__(self, w, h, fps, duration, look):
        self.w, self.h, self.fps, self.duration, self.look = w, h, fps, duration, look
        self.shots: list[Shot] = []
        self.overlays: list[Overlay] = []
        self.bars: Callable = lambda t: 0.0
        self.labels: Callable = lambda t: (None, 0.0)
        self.cinema_at: Callable = lambda t: {}
        self.grain_at: Callable = lambda t: 1.0
        self.vignette_at: Callable = lambda t: 1.0
        self.bar_ratio = 2.39            # the letterbox window's aspect (2.39 scope; 0.8 = 4:5 in vertical)
        self.decode_size = None          # None: decode at output size; (w, h): decode at that size
        self.layer = gfx.Layer(w, h)
        self.mask = gfx.Mask(w, h)
        self.audio = None                # stereo float32 @48k, or None

    def add(self, shot):
        self.shots.append(shot)
        return shot

    def overlay(self, t0, t1, stage="pre"):
        def deco(fn):
            self.overlays.append(Overlay(t0, t1, fn, stage))
            return fn
        return deco

    # coverage: each shot plays until the next shot's transition has finished
    def _need_until(self):
        shots = sorted(self.shots, key=lambda s: s.start)
        need = {}
        for i, s in enumerate(shots):
            until = s.end
            if i + 1 < len(shots) and shots[i + 1].trans:
                nxt = shots[i + 1]
                until = max(until, nxt.start + nxt.trans[1])
            need[id(s)] = until
        return shots, need

    def _active(self, t, shots):
        cur = None
        for s in shots:
            if s.start <= t < s.end:
                cur = s
        return cur

    def picture(self, t, shots, need):
        cur = self._active(t, shots)
        if cur is None:
            return np.zeros((self.h, self.w, 3), np.float32)
        b = cur.picture(t, self)
        if cur.trans and t < cur.start + cur.trans[1]:
            i = shots.index(cur)
            prev = shots[i - 1] if i > 0 and need[id(shots[i - 1])] > t else None
            a = prev.picture(t, self) if prev else None
            name, d = cur.trans
            p = (t - cur.start) / d
            if a is None and name != "slit":
                a = np.zeros_like(b)
            b = fx.TRANSITIONS[name](a, b, p, self)
        return b

    def _schedule(self, t, shots, need):
        for s in shots:
            live = s.start - 0.1 <= t <= need[id(s)] + 0.05
            for c in s.all_clips():
                if live and c.buf is None:
                    c.prepare(need[id(s)] - s.start + 0.1, self.fps, c.decode or self.decode_size or (self.w, self.h))
                elif not live and c.buf is not None:
                    c.release()

    def frame(self, n, shots=None, need=None):
        if shots is None:
            shots, need = self._need_until()
        t = n / self.fps
        self._schedule(t, shots, need)
        img = self.picture(t, shots, need)
        for stage in ("pre", "post"):
            for o in self.overlays:
                if o.stage == stage and o.t0 <= t < o.t1:
                    img = o.fn(t, img, self)
            if stage == "pre":
                img = self.look.cinema(img, n, **self.cinema_at(t))
        img = self.look.finish(img, n, grain=self.grain_at(t), vignette=self.vignette_at(t))
        img, bar = gfx.letterbox(img, self.bars(t), self.bar_ratio)
        self.layer.clear()
        labels, a = self.labels(t)
        if labels:
            gfx.bar_labels(self.layer.c, self.w, self.h, bar, a, labels)
        for o in self.overlays:
            if o.stage == "top" and o.t0 <= t < o.t1:
                img = o.fn(t, img, self)
        return self.layer.over(img)

    def render(self, path, wav=None, frames=None, crf=16, preset="slow", log_every=50):
        shots, need = self._need_until()
        n_total = int(round(self.duration * self.fps))
        frames = frames or range(n_total)
        enc = media.Encoder(path, self.w, self.h, self.fps, crf=crf, preset=preset)
        t0 = time.time()
        for k, n in enumerate(frames):
            enc.write(self.frame(n, shots, need))
            if log_every and k % log_every == 0:
                print(f"  frame {n}/{n_total}  {time.time() - t0:5.1f}s", flush=True)
        enc.close(wav)
        for s in shots:
            for c in s.all_clips():
                c.release()

    def stills(self, times, path_fmt):
        from PIL import Image
        shots, need = self._need_until()
        out = []
        for t in times:
            n = int(round(t * self.fps))
            img = self.frame(n, shots, need)
            p = path_fmt.format(t=t, n=n)
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(p)
            out.append(p)
        return out

    def edl(self):
        """Human-readable cut list."""
        rows = []
        for i, s in enumerate(sorted(self.shots, key=lambda s: s.start), 1):
            for c in s.all_clips():
                src = getattr(c.source, "path", type(c.source).__name__)
                tc = c.t_in
                rows.append(f"{i:02d}  rec {s.start:6.2f}–{s.end:6.2f}  src {int(tc // 3600):01d}:{int(tc % 3600 // 60):02d}:"
                            f"{tc % 60:05.2f}  {str(s.trans[0]) if s.trans else 'cut':8s}  {c.note}  [{src}]")
        return "\n".join(rows)
