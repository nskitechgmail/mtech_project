# 🚦 Smart City ANPR System
## Multi-Modal Vehicle Detection & License Plate Recognition using Generative AI

**SRM Institute of Science and Technology, India**  
Department of Computational Intelligence  
`sv2447@srmist.edu.in` · `venkates9@srmist.edu.in`

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/anpr-genai-system.git
cd anpr-genai-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (webcam)
python main.py

# 4. Run with video file
python main.py --source traffic_video.mp4

# 5. Run with CCTV RTSP stream
python main.py --source rtsp://admin:password@192.168.1.100:554/stream1

# 6. Headless server mode (no GUI)
python main.py --headless --source 0

# 7. Disable GenAI (faster, lower accuracy)
python main.py --no-genai --source 0
```

> **First run**: Models are downloaded automatically (~500 MB total).  
> GPU strongly recommended for real-time performance.

---

## 📁 Project Structure

```
anpr-genai-system/
│
├── main.py                    # ← Entry point
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py            # All configuration parameters
│
├── core/
│   ├── pipeline.py            # Main orchestrator (capture → detect → report)
│   └── plate_recogniser.py    # Plate localisation, enhancement, OCR
│
├── models/
│   ├── model_manager.py       # Downloads & loads all AI models
│   └── weights/               # Model weight files (auto-downloaded)
│
├── utils/
│   ├── annotator.py           # Draws overlays on video frames
│   ├── report_writer.py       # CSV/JSON violation report writer
│   └── anonymiser.py          # Face blurring (privacy compliance)
│
├── ui/
│   └── dashboard.py           # Tkinter real-time monitoring dashboard
│
├── tests/
│   └── test_*.py
│
└── outputs/
    ├── violations/            # Saved violation images
    ├── reports/               # CSV/JSON reports
    └── logs/
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT SOURCES                           │
│   Webcam  ·  IP Camera (RTSP)  ·  Video File  ·  CCTV Stream   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: VEHICLE DETECTION                          │
│                  YOLOv8/v9 (72 FPS)                             │
│   Detects: Cars · Motorcycles · Buses · Trucks · Auto-rickshaws │
└───────┬──────────────────────────────────────┬──────────────────┘
        │                                       │
        ▼                                       ▼
┌───────────────────────┐           ┌───────────────────────────┐
│  STAGE 2: PLATE        │           │  STAGE 5: SAFETY           │
│  LOCALISATION          │           │  COMPLIANCE                │
│  (Contour + Heuristic) │           │  MobileNetV3               │
└──────────┬────────────┘           │  · Helmet detection        │
           │                        │  · Seat-belt detection     │
           ▼                        └───────────────────────────┘
┌───────────────────────┐
│  STAGE 3: GenAI        │
│  ENHANCEMENT           │
│  Real-ESRGAN ×4        │
│  Blind Super-Resolution│
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  STAGE 4: OCR          │
│  EasyOCR + CRAFT       │
│  + Indian plate regex  │
│  + position-aware fix  │
└──────────┬────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              VIOLATION TRACKING (Temporal Smoothing)            │
│         Confirmed after N consecutive frames → zero false +ve   │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                        │
│  · Live GUI dashboard with overlays                             │
│  · CSV/JSON violation report                                    │
│  · Violation image saved (face-blurred for privacy)             │
│  · Console log (headless mode)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🖥 Dashboard Features

| Panel | Description |
|-------|-------------|
| **Live Feed** | Annotated real-time video with bounding boxes, plate text, safety badges |
| **Live Metrics** | FPS, vehicle count, plates read, violation count |
| **Plate Inset** | Enhanced plate crop (Real-ESRGAN output) |
| **Violation Log** | Scrollable table: time, plate, vehicle, violation type, confidence |
| **Session Summary** | Cumulative counts, session duration |
| **Toolbar** | Toggle GenAI · adjust confidence · open file/webcam/RTSP · export CSV |

---

## 📊 Performance Results

| Condition | Traditional | GenAI Enhanced | Improvement |
|-----------|-------------|----------------|-------------|
| Good Lighting | 92.5% | 94.8% | +2.3% |
| Low Light | 68.3% | 87.6% | **+19.3%** |
| Night w/ Glare | 45.2% | 78.4% | **+33.2%** |
| Motion Blur | 58.7% | 82.3% | **+23.6%** |
| Rain / Fog | 52.1% | 79.7% | **+27.6%** |
| **Overall** | **65.4%** | **84.5%** | **+19.1%** |

| Model | mAP@0.5 | FPS | Params |
|-------|---------|-----|--------|
| YOLOv5 | 88.3% | 45 | 7.2M |
| YOLOv8 | 91.7% | 68 | 11.2M |
| **YOLOv9** | **92.4%** | **72** | 13.5M |
| Faster R-CNN | 93.2% | 12 | 41.8M |

---

## ⚙️ Configuration

Edit `config/settings.py` or pass CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | `0` | Camera index / file path / RTSP URL |
| `--conf` | `0.40` | YOLO confidence threshold |
| `--no-genai` | — | Disable Real-ESRGAN (faster) |
| `--device` | `auto` | `cpu` / `cuda` / `mps` |
| `--camera-id` | `CCTV-001` | Camera label in reports |
| `--headless` | — | No GUI, console output only |

---

## 🔌 CCTV Integration (RTSP)

```python
# Hikvision
python main.py --source "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"

# Dahua
python main.py --source "rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# Generic ONVIF
python main.py --source "rtsp://192.168.1.x:554/stream"
```

---

## 🛡 Privacy & Ethics

- **Face Anonymisation**: All faces blurred via MediaPipe before saving
- **Data Minimisation**: Raw frames discarded after processing; only violation records retained
- **No Biometrics**: System identifies violations by visual inference, not identity
- **Audit Trail**: Every detection logged with confidence score for human review
- **Regulatory Compliance**: Designed to comply with India's Digital Personal Data Protection Act

---

## 📄 Citation

```bibtex
@article{nagalingam2025anpr,
  title   = {Multi-Modal Vehicle Detection and License Plate Recognition using GenAI},
  author  = {Nagalingam, Sathish Kumar and Venkatesh, S.},
  journal = {IEEE},
  year    = {2025},
  institution = {SRM Institute of Science and Technology}
}
```
