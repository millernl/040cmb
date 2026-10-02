"""The song without its vocal: an MDX-Net instrumental model (UVR-MDX-NET-Inst_HQ_3, ONNX, CPU) in
work/models/, not in git. The spectrogram layout and the chunking follow UVR's MDX inference:
44.1 kHz stereo, n_fft 6144, hop 1024, 3072 bins × 256 frames per chunk, chunks overlapping by
n_fft/2 on each side; each chunk is run on the mix and on its negative and the two are averaged.

    instrumental(path, t0, dur)  → float32 [n, 2] at 44.1 kHz, cached in work/stems/
"""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "work" / "models" / "UVR-MDX-NET-Inst_HQ_3.onnx"
CACHE = ROOT / "work" / "stems"
SR = 44100
N_FFT, HOP, DIM_F, DIM_T = 6144, 1024, 3072, 256
CHUNK = HOP * (DIM_T - 1)
TRIM = N_FFT // 2
WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann


def _stft(x):
    """[k, n] → [k, bins, frames] complex, centred (reflect-padded) like torch.stft."""
    xp = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode="reflect")
    frames = 1 + (xp.shape[1] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    return np.fft.rfft(xp[:, idx] * WIN, axis=-1).transpose(0, 2, 1)


def _istft(spec, length):
    frames = np.fft.irfft(spec.transpose(0, 2, 1), n=N_FFT, axis=-1) * WIN
    k, t, _ = frames.shape
    out = np.zeros((k, N_FFT + HOP * (t - 1)), np.float64)
    wsum = np.zeros(out.shape[1], np.float64)
    for i in range(t):
        out[:, i * HOP:i * HOP + N_FFT] += frames[:, i]
        wsum[i * HOP:i * HOP + N_FFT] += WIN ** 2
    out /= np.maximum(wsum, 1e-8)
    return out[:, N_FFT // 2:N_FFT // 2 + length]


def _to_model(waves):
    """[N, 2, CHUNK] → [N, 4, DIM_F, DIM_T]: (left re, left im, right re, right im)."""
    n = waves.shape[0]
    s = _stft(waves.reshape(-1, CHUNK))                          # [N*2, bins, T]
    s = np.stack([s.real, s.imag], 1).reshape(n, 4, -1, DIM_T)  # [N, 4, bins, T]
    return s[:, :, :DIM_F].astype(np.float32)


def _from_model(spec):
    n = spec.shape[0]
    bins = N_FFT // 2 + 1
    full = np.zeros((n, 4, bins, DIM_T), np.float32)
    full[:, :, :DIM_F] = spec
    full = full.reshape(n * 2, 2, bins, DIM_T)
    return _istft(full[:, 0] + 1j * full[:, 1], CHUNK).reshape(n, 2, CHUNK)


def demix(mix):
    """mix: [2, n] at 44.1 kHz → the instrumental, [2, n]."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    n = mix.shape[1]
    gen = CHUNK - 2 * TRIM
    pad = gen - n % gen
    mp = np.concatenate([np.zeros((2, TRIM)), mix, np.zeros((2, pad)), np.zeros((2, TRIM))], 1)
    waves = np.stack([mp[:, i:i + CHUNK] for i in range(0, n + pad, gen)])
    out = []
    for w in waves:                                             # one chunk at a time: memory stays small
        spec = _to_model(w[None])
        pred = (sess.run(None, {"input": spec})[0] - sess.run(None, {"input": -spec})[0]) / 2
        out.append(_from_model(pred)[0])
    tar = np.stack(out)[:, :, TRIM:-TRIM].transpose(1, 0, 2).reshape(2, -1)[:, :n]
    return tar


def instrumental(path, t0, dur):
    from afterfilm import media
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{Path(path).name}:{t0:.4f}:{dur:.4f}:{MODEL.name}".encode()).hexdigest()[:16]
    f = CACHE / f"inst_{key}.npy"
    if f.exists():
        return np.load(f)
    mix = media.read_audio(str(path), t0, dur, sr=SR, channels=2).T.astype(np.float64)
    inst = demix(mix).T.astype(np.float32)
    np.save(f, inst)
    return inst
