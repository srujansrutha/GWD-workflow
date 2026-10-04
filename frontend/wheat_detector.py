"""Model loading, image decoding, detection and drawing for the wheat-head counter app.

Kept free of Streamlit code so it can be tested on its own:  python -c "import wheat_detector"
"""
from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:  # iPhone / modern camera photos (.heic, .heif)
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # optional dependency
    pass

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # an always-on app must never pip-install packages by itself

Image.MAX_IMAGE_PIXELS = 250_000_000  # refuse decompression bombs, allow big drone/camera photos

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_CANDIDATES = [
    os.environ.get("WHEAT_MODEL"),
    REPO_ROOT / "weights" / "wheat_yolo11s_gwhd21.pt",
    REPO_ROOT / "runs" / "detect" / "yolo11s_gwhd21" / "weights" / "best.pt",
]

TRAIN_IMGSZ = 1024      # the size the model was trained at
CONF_FLOOR = 0.10       # detect once at this floor; the UI slider filters the result instantly
DISPLAY_MAX_SIDE = 3000  # annotated images are shown and downloaded at most this wide


def find_weights() -> Path | None:
    for candidate in WEIGHT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


# ----------------------------------------------------------------------------- decoding
class ImageReadError(Exception):
    """The uploaded file is not an image we can decode."""


@dataclass
class DecodedImage:
    image: Image.Image      # RGB, EXIF-rotated
    fmt: str                # original file format, e.g. JPEG
    mode: str               # original colour mode, e.g. CMYK
    width: int
    height: int


def _to_rgb(im: Image.Image) -> Image.Image:
    """Convert any PIL mode to 8-bit RGB without the washed-out / clipped result of a naive convert."""
    if im.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):  # 16/32-bit grayscale (scientific TIFF)
        a = np.asarray(im, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
        return Image.fromarray(a.astype(np.uint8), "L").convert("RGB")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def decode_image(raw: bytes) -> DecodedImage:
    """Decode uploaded bytes (jpg, png, tif, webp, bmp, gif, heic, ...) into an upright RGB image."""
    try:
        im = Image.open(io.BytesIO(raw))
        fmt = im.format or "unknown"
        mode = im.mode
        im.seek(0)                      # first frame of GIF / multi-page TIFF
        im.load()
        im = ImageOps.exif_transpose(im)  # honour the camera's rotation flag
        rgb = _to_rgb(im)
    except Image.DecompressionBombError as e:
        raise ImageReadError("The image has too many pixels (limit is 250 megapixels).") from e
    except Exception as e:  # noqa: BLE001 - any decoder failure becomes a friendly message
        raise ImageReadError("This file is not a readable image.") from e
    return DecodedImage(rgb, fmt, mode, rgb.width, rgb.height)


# ----------------------------------------------------------------------------- detection
@dataclass
class Detection:
    boxes: np.ndarray    # (N, 4) float32 xyxy in original image pixels
    scores: np.ndarray   # (N,) float32
    width: int
    height: int
    ms: float            # inference time
    floor: float         # confidence floor actually used

    def at(self, conf: float) -> tuple[np.ndarray, np.ndarray]:
        keep = self.scores >= conf
        return self.boxes[keep], self.scores[keep]


class Detector:
    """Holds one YOLO model on the GPU for the lifetime of the process."""

    def __init__(self, weights: Path):
        import torch
        from ultralytics import YOLO

        self.weights = Path(weights)
        self._torch = torch
        self.on_gpu = torch.cuda.is_available()
        self.device = 0 if self.on_gpu else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.on_gpu else "CPU (no GPU found)"
        self._lock = threading.Lock()  # the model is not safe to call from two sessions at once

        t = time.perf_counter()
        self.model = YOLO(str(self.weights))
        # A first prediction moves the weights onto the GPU and warms up the CUDA kernels,
        # so the first real upload is as fast as every later one.
        self._predict(Image.new("RGB", (TRAIN_IMGSZ, TRAIN_IMGSZ), (110, 120, 60)), TRAIN_IMGSZ, 0.7, 0.25)
        self.load_seconds = time.perf_counter() - t

    @property
    def model_device(self) -> str:
        """Where the weights actually live, read from the loaded model (not assumed)."""
        try:
            return str(self.model.predictor.model.device)
        except Exception:  # noqa: BLE001
            return "unknown"

    def gpu_memory_gb(self) -> float | None:
        if not self.on_gpu:
            return None
        return self._torch.cuda.memory_reserved(0) / 1e9

    def _predict(self, image: Image.Image, imgsz: int, iou: float, floor: float):
        return self.model.predict(image, imgsz=imgsz, conf=floor, iou=iou, device=self.device,
                                  max_det=3000, verbose=False)[0]

    def detect(self, image: Image.Image, imgsz: int = TRAIN_IMGSZ, iou: float = 0.7) -> Detection:
        torch = self._torch
        with self._lock:
            result, used_floor = None, CONF_FLOOR
            t = time.perf_counter()
            # Very cluttered pictures can produce a huge number of candidate boxes. If the GPU runs
            # out of memory, retry with a stricter floor instead of crashing the always-on app.
            for floor in (CONF_FLOOR, 0.25, 0.5):
                try:
                    result, used_floor = self._predict(image, imgsz, iou, floor), floor
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if result is None:
                raise MemoryError("The GPU ran out of memory on this image. Try a smaller image or detection size.")
            ms = (time.perf_counter() - t) * 1000
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        return Detection(boxes, scores, image.width, image.height, ms, used_floor)


# ----------------------------------------------------------------------------- drawing
def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def annotate(image: Image.Image, boxes: np.ndarray, scores: np.ndarray, *, color: str = "#22c55e",
             style: str = "Outline", thickness: float = 1.0, show_conf: bool = False,
             badge: bool = True) -> Image.Image:
    """Return a copy of ``image`` (shrunk to DISPLAY_MAX_SIDE) with the boxes drawn on it."""
    scale = min(1.0, DISPLAY_MAX_SIDE / max(image.size))
    img = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS) if scale < 1 else image.copy()
    rgb = hex_to_rgb(color)
    short = min(img.size)
    line = max(1, round(short / 520 * thickness))

    if len(boxes):
        b = boxes * scale
        if style == "Filled":
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            for x1, y1, x2, y2 in b:
                d.rectangle([x1, y1, x2, y2], fill=(*rgb, 70), outline=(*rgb, 255), width=line)
            img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        else:
            d = ImageDraw.Draw(img)
            for x1, y1, x2, y2 in b:
                d.rectangle([x1, y1, x2, y2], outline=rgb, width=line)
        if show_conf:
            d = ImageDraw.Draw(img)
            f = _font(max(10, round(short / 75)))
            for (x1, y1, _, _), s in zip(b, scores):
                d.text((x1 + 2, max(0, y1 - f.size - 1)), f"{s:.2f}", fill=rgb, font=f,
                       stroke_width=max(1, line // 2), stroke_fill=(0, 0, 0))

    if badge:
        d = ImageDraw.Draw(img, "RGBA")
        f = _font(max(16, round(short / 24)))
        text = f"{len(boxes)} wheat heads" if len(boxes) != 1 else "1 wheat head"
        l, t, r, bt = d.textbbox((0, 0), text, font=f)
        pad = max(8, round(f.size * 0.5))
        x0 = y0 = max(8, round(short / 60))
        d.rounded_rectangle([x0, y0, x0 + (r - l) + 2 * pad, y0 + (bt - t) + 2 * pad], radius=pad,
                            fill=(15, 23, 42, 215), outline=(*rgb, 255), width=max(2, line))
        d.text((x0 + pad - l, y0 + pad - t), text, fill=(255, 255, 255), font=f)
    return img


def display_scale(width: int, height: int) -> float:
    return min(1.0, DISPLAY_MAX_SIDE / max(width, height))


def to_jpeg(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()
