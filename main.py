#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SMART CITY ANPR SYSTEM — Multi-Modal Traffic Monitoring             ║
║   Vehicle Detection · License Plate Recognition · Safety Compliance         ║
║   with Generative AI Enhancement (Real-ESRGAN)                              ║
║                                                                              ║
║   SRM Institute of Science and Technology, India                            ║
║   Department of Computational Intelligence                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Entry point — run this file:
    python main.py                        # Launch GUI (webcam or file)
    python main.py --source 0             # Webcam (index 0)
    python main.py --source video.mp4     # Video file
    python main.py --source rtsp://...    # CCTV RTSP stream
    python main.py --source image.jpg     # Single image
    python main.py --headless --source 0  # Headless / server mode

Install:
    pip install -r requirements.txt
    # (first-run downloads models automatically ~500 MB)
"""

import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(__file__))

from core.pipeline   import ANPRPipeline
from ui.dashboard    import ANPRDashboard
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ANPR-Main")


def parse_args():
    p = argparse.ArgumentParser(description="Smart City ANPR System")
    p.add_argument("--source",   default="0",  help="Camera index, video path, or RTSP URL")
    p.add_argument("--headless", action="store_true", help="Run without GUI (log to file)")
    p.add_argument("--genai",    action="store_true", default=True, help="Enable GenAI enhancement")
    p.add_argument("--no-genai", dest="genai", action="store_false")
    p.add_argument("--conf",     type=float, default=0.40, help="Detection confidence threshold")
    p.add_argument("--device",   default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--output",   default="outputs", help="Output directory")
    p.add_argument("--camera-id", default="CCTV-001", help="Camera identifier for reports")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Settings(
        source    = args.source,
        use_genai = args.genai,
        conf_thresh= args.conf,
        device    = args.device,
        output_dir= args.output,
        camera_id = args.camera_id,
    )

    log.info("=" * 60)
    log.info("  Smart City ANPR System  —  SRM Institute")
    log.info("=" * 60)
    log.info(f"  Source   : {cfg.source}")
    log.info(f"  GenAI    : {'ENABLED (Real-ESRGAN)' if cfg.use_genai else 'DISABLED'}")
    log.info(f"  Device   : {cfg.device}")
    log.info(f"  Camera   : {cfg.camera_id}")
    log.info("=" * 60)

    if args.headless:
        # Server / headless mode
        pipeline = ANPRPipeline(cfg)
        pipeline.run_headless()
    else:
        # GUI dashboard
        app = ANPRDashboard(cfg)
        app.run()


if __name__ == "__main__":
    main()
