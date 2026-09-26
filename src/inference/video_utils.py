"""Video frame sampling and visual-timestamp overlay used by all TEMPURA inference tasks.

A video is consumed as a sequence of frames sampled at a fixed rate (``fps``). Each frame
is optionally stamped with its timestamp (seconds) in the top-left corner, which is the
input format the released TEMPURA checkpoints were trained on.
"""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)  # Pillow >= 10.1 bundles a scalable default


def fps_frame_indices(num_video_frames: int, video_fps: float, sample_fps: float, max_num_frames: int = -1) -> List[int]:
    """Indices of frames sampled at ``sample_fps``, each taken from the middle of its interval.

    For ``sample_fps=1`` this yields frames at t = 0.5, 1.5, 2.5, ... seconds, i.e. one frame per
    second representing the clip [t-0.5, t+0.5).
    """
    duration = float(num_video_frames) / video_fps
    delta = 1.0 / sample_fps
    frame_seconds = np.arange(delta / 2, duration + delta / 2, delta)
    indices = np.around(frame_seconds * video_fps).astype(int)
    indices = [int(i) for i in indices if i < num_video_frames]
    if max_num_frames > 0 and len(indices) > max_num_frames:
        indices = indices[:max_num_frames]
    return indices


def _read_with_decord(video_path: str, sample_fps: float, max_num_frames: int) -> Tuple[List[Image.Image], List[float]]:
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    video_fps = float(vr.get_avg_fps())
    indices = fps_frame_indices(len(vr), video_fps, sample_fps, max_num_frames)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    return [Image.fromarray(f) for f in frames], [i / video_fps for i in indices]


def _read_with_pyav(video_path: str, sample_fps: float, max_num_frames: int) -> Tuple[List[Image.Image], List[float]]:
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        video_fps = float(stream.average_rate or stream.guessed_rate or 30)
        num_frames = stream.frames
        if not num_frames:  # some containers do not store the frame count; estimate from duration
            num_frames = int(float(stream.duration * stream.time_base) * video_fps) if stream.duration else 0
        if not num_frames:  # last resort: decode once to count
            num_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)
        wanted = set(fps_frame_indices(num_frames, video_fps, sample_fps, max_num_frames))
        frames, times = [], []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted:
                frames.append(frame.to_image())
                times.append(idx / video_fps)
            if idx >= max(wanted, default=-1):
                break
    return frames, times


def read_video_frames(video_path: str, sample_fps: float = 1.0, max_num_frames: int = -1) -> Tuple[List[Image.Image], List[float]]:
    """Return ``(frames, timestamps_in_seconds)`` for a video file or a directory of frames.

    A directory is treated as pre-extracted frames named ``<index>.jpg`` (index = frame order),
    already sampled at ``sample_fps``; timestamps are then ``index / sample_fps``.
    """
    if os.path.isdir(video_path):
        names = sorted((f for f in os.listdir(video_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))),
                       key=lambda x: int(os.path.splitext(x)[0]))
        if max_num_frames > 0:
            names = names[:max_num_frames]
        frames = [Image.open(os.path.join(video_path, n)).convert("RGB") for n in names]
        return frames, [i / sample_fps for i in range(len(frames))]
    try:
        return _read_with_decord(video_path, sample_fps, max_num_frames)
    except ImportError:
        return _read_with_pyav(video_path, sample_fps, max_num_frames)


def add_visual_timestamps(frames: List[Image.Image], timestamps: List[float]) -> List[Image.Image]:
    """Draw ``"{t:.2f}"`` on the top-left of every frame (in place) and return the list."""
    if not frames:
        return frames
    font = _load_font(max(8, frames[0].size[0] // 10))
    for frame, t in zip(frames, timestamps):
        draw = ImageDraw.Draw(frame)
        text = f"{t:.2f}"
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        draw.rectangle([-5, -5, x1 - x0 + 5, y1 - y0 + 5], fill=(0, 0, 0))
        draw.text((0, 0), text, fill="white", font=font)
    return frames


def load_video_for_model(video_path: str, sample_fps: float = 1.0, add_timestamp: bool = True,
                         max_num_frames: int = -1) -> Tuple[List[Image.Image], List[float]]:
    """Frames + timestamps ready to be fed to a TEMPURA model."""
    frames, times = read_video_frames(video_path, sample_fps, max_num_frames)
    if add_timestamp:
        add_visual_timestamps(frames, times)
    return frames, times
