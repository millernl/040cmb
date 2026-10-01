"""ffmpeg plumbing: probe, decode frames/audio into numpy, encode the master."""
import json
import os
import shutil
import subprocess
from functools import cache

import numpy as np


@cache
def ffmpeg():
    exe = os.environ.get("AFTERFILM_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


@cache
def ffprobe():
    exe = os.environ.get("AFTERFILM_FFPROBE") or shutil.which("ffprobe")
    if not exe:
        cand = os.path.join(os.path.dirname(ffmpeg()), "ffprobe")
        exe = cand if os.path.exists(cand) else None
    return exe


@cache
def probe(path):
    out = subprocess.run([ffprobe(), "-v", "error", "-print_format", "json", "-show_format",
                          "-show_streams", str(path)], capture_output=True, check=True, text=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v.get("avg_frame_rate", "25/1").split("/"))
    return {
        "duration": float(info["format"].get("duration", v.get("duration", 0))),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": num / den if den else 25.0,
        "interlaced": v.get("field_order", "progressive") not in ("progressive", "unknown"),
        "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
        "codec": v.get("codec_name"),
    }


# Rescue chain for a low-bitrate export: undo block edges, calm the mosquito noise,
# upscale cleanly, then contrast-adaptive sharpening. Grain in the grade does the rest.
RESCUE_PRE = "deblock=filter=strong:block=8,hqdn3d=2.5:2:5:4"
RESCUE_POST = "cas=0.45"


def read_frames(path, t_in, dur, fps, size, deinterlace=None, pre=None, post=None):
    """Decode [t_in, t_in+dur) at `fps`, scaled+cropped to fill `size`. → uint8 [N,H,W,3]."""
    w, h = size
    info = probe(path)
    if deinterlace is None:
        deinterlace = info["interlaced"]
    vf = []
    if deinterlace:
        vf.append("bwdif=mode=send_field")          # 50i → 50p: every field becomes a frame
    if pre:
        vf.append(pre)
    vf += [f"fps={fps}", f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={w}:{h}"]
    if post:
        vf.append(post)
    warm = min(0.4, max(t_in, 0)) if pre else 0.0
    cmd = [ffmpeg(), "-v", "error", "-ss", f"{max(t_in, 0) - warm:.3f}", "-i", str(path), "-t", f"{dur + warm:.3f}",
           "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return frames[int(round(warm * fps)):]


def read_audio(path, t_in=0.0, dur=None, sr=48000, channels=2):
    cmd = [ffmpeg(), "-v", "error", "-ss", f"{max(t_in, 0):.3f}", "-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, channels).copy()


def write_wav(path, audio, sr=48000):
    import soundfile as sf
    sf.write(str(path), audio, sr, subtype="PCM_24")


class Encoder:
    """Pipe RGB frames into x264. Audio (a wav) is muxed in at close()."""

    def __init__(self, path, w, h, fps, crf=16, preset="slow"):
        self.path = str(path)
        self.video_tmp = self.path + ".video.mp4"
        self.w, self.h, self.fps = w, h, fps
        self.rng = np.random.default_rng(98)
        self.proc = subprocess.Popen(
            [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-movflags", "+faststart", self.video_tmp],
            stdin=subprocess.PIPE)

    def write(self, frame):
        # dither to 8-bit so the green gradients don't band
        f = frame * 255.0 + self.rng.uniform(-0.5, 0.5, frame.shape[:2])[..., None].astype(np.float32)
        self.proc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())

    def close(self, wav=None):
        self.proc.stdin.close()
        self.proc.wait()
        if wav:
            subprocess.run([ffmpeg(), "-v", "error", "-y", "-i", self.video_tmp, "-i", str(wav), "-map", "0:v",
                            "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                            "-movflags", "+faststart", self.path], check=True)
            os.remove(self.video_tmp)
        else:
            os.replace(self.video_tmp, self.path)
