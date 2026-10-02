"""People mattes: Robust Video Matting (Lin et al., 2021), run on the footage once and cached.

The model is recurrent, so it is warmed up on the half second before the requested range and
the alpha stays steady from frame to frame. Weights: rvm_mobilenetv3_fp32.onnx from the RVM
GitHub release (GPL-3.0), downloaded to work/models/ — not in git.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import media

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "work" / "models" / "rvm_mobilenetv3_fp32.onnx"
CACHE = ROOT / "work" / "mattes"
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return _SESSION


def mattes(path, t0, t1, fps, size=(540, 960), warm=0.5):
    """Alpha (float32, [N, h, w] in 0..1) for source seconds [t0, t1) at `fps`, at `size`."""
    w, h = size
    key = hashlib.sha1(f"{Path(path).name}|{t0:.3f}|{t1:.3f}|{fps}|{w}x{h}".encode()).hexdigest()[:16]
    cache = CACHE / f"{Path(path).stem}_{key}.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32) / 255.0
    ws = max(0.0, t0 - warm)
    frames = media.read_frames(str(path), ws, (t1 - ws) + 1 / fps, fps, size)
    sess = _session()
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    ratio = np.array([0.4], np.float32)
    out = []
    for f in frames:
        src = (f.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        fgr, pha, *rec = sess.run(None, {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                         "downsample_ratio": ratio})
        out.append(pha[0, 0])
    skip = int(round((t0 - ws) * fps))
    a = np.stack(out[skip:skip + int(round((t1 - t0) * fps)) + 1]).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, (np.clip(a, 0, 1) * 255).astype(np.uint8))
    return a


def still(img, size=(540, 960)):
    """Matte of a photograph: the recurrent model is fed the same frame until it settles."""
    import cv2
    ih, iw = img.shape[:2]
    s = min(size[0] / iw, size[1] / ih) if iw > ih else max(size[0] / iw, size[1] / ih)
    w, h = int(round(iw * s)) // 4 * 4, int(round(ih * s)) // 4 * 4
    key = hashlib.sha1(img[::17, ::17].tobytes()).hexdigest()[:16]
    cache = CACHE / f"still_{key}_{w}x{h}.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32) / 255.0
    src = (cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    sess = _session()
    rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
    for _ in range(8):
        fgr, pha, *rec = sess.run(None, {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                         "downsample_ratio": np.array([0.4], np.float32)})
    a = np.clip(pha[0, 0], 0, 1).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, (a * 255).astype(np.uint8))
    return a
