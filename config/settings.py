"""
config/settings.py — Centralised configuration for the ANPR system.

All tuneable parameters live here so operators can adjust behaviour
without touching pipeline code.
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass
class Settings:
    # ── Input ─────────────────────────────────────────────────────────────
    source: str          = "0"           # camera index / video / RTSP / image path
    camera_id: str       = "CCTV-001"   # displayed in reports
    fps_target: int      = 30            # target processing FPS

    # ── Model ─────────────────────────────────────────────────────────────
    detector_model: str  = "yolov8n"    # yolov8n / yolov8s / yolov8m / yolov9c
    conf_thresh: float   = 0.40         # YOLO confidence threshold
    iou_thresh:  float   = 0.45         # NMS IoU threshold
    device: str          = "auto"       # auto / cpu / cuda / mps

    # ── GenAI enhancement ─────────────────────────────────────────────────
    use_genai: bool      = True
    esrgan_model: str    = "RealESRGAN_x4plus"   # or RealESRGAN_x4plus_anime_6B
    esrgan_scale: int    = 4
    esrgan_tile:  int    = 0            # 0 = no tiling; set 128-512 for low VRAM

    # ── OCR ───────────────────────────────────────────────────────────────
    ocr_languages: list  = field(default_factory=lambda: ["en"])
    ocr_gpu: bool        = False        # True if CUDA available

    # ── Safety compliance ─────────────────────────────────────────────────
    helmet_conf:   float = 0.50
    seatbelt_conf: float = 0.50
    # Temporal smoothing: require N consecutive frames to confirm violation
    violation_frames: int = 3

    # ── Plate detection ───────────────────────────────────────────────────
    plate_min_area:  int = 800          # px² — ignore smaller detections
    plate_max_skew: float = 30.0        # degrees — reject heavily angled plates

    # ── Output ────────────────────────────────────────────────────────────
    output_dir:      str = "outputs"
    save_violations: bool = True        # save annotated frames for each violation
    save_all_plates: bool = False       # save every recognised plate
    anonymise_faces: bool = True        # blur faces in saved images
    report_format:   str = "csv"        # csv / json / both

    # ── Display ───────────────────────────────────────────────────────────
    show_fps:          bool = True
    show_plate_crop:   bool = True      # show enhanced plate inset
    display_width:     int  = 1280
    display_height:    int  = 720

    # ── Indian plate regex (covers most state formats) ─────────────────
    plate_regex: str = (
        r"^[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{1,4}$"
    )

    def __post_init__(self):
        # Resolve device
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # Resolve source (int for webcam index)
        try:
            self.source = int(self.source)
        except (ValueError, TypeError):
            pass  # keep as string (path / URL)

        # Create output dirs
        base = ROOT / self.output_dir
        for sub in ("violations", "logs", "reports", "plates"):
            (base / sub).mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> Path:
        return ROOT / self.output_dir

    def save(self, path: str | None = None):
        path = path or str(self.output_path / "config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Settings":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
