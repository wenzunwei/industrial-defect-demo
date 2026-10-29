"""Offline CLI: one image or a directory, with detection/segmentation/both."""

import argparse
import json
from pathlib import Path

from inference import DefectSystem
from inference.pipeline import read_image


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_images(source):
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() in EXTENSIONS:
        return source.parent, [source]
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if images:
            return source, images
    raise ValueError(f"No supported images found: {source}")


def main():
    parser = argparse.ArgumentParser(description="Industrial defect detection and segmentation (ONNX)")
    parser.add_argument("--source", type=Path, default=ROOT / "examples" / "inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--mode", choices=("both", "detect", "segment"), default="both")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    base, images = collect_images(args.source)
    output = args.output.resolve()
    if args.source.is_dir() and (output == base or output.is_relative_to(base) or base.is_relative_to(output)):
        raise ValueError("Keep input and output directories separate (neither may contain the other)")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    system = DefectSystem(args.models, args.mode, args.provider, args.threads,
                          args.conf, args.iou, args.threshold)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "INCOMPLETE"
    marker.write_text("Inference in progress; see summary.json after completion", encoding="utf-8")
    records = []
    for path in images:
        # Keep relative folders and the complete filename: a.jpg and a.png cannot collide.
        relative = path.relative_to(base)
        image = read_image(path)
        result = system.predict(image)
        metadata = system.save(image, result, output / relative, source=relative.as_posix())
        records.append({"source": relative.as_posix(), "result": (relative / "result.json").as_posix(),
                        "detection_count": len(result["detections"]) if result["detections"] is not None else None,
                        "foreground_pixels": metadata.get("segmentation", {}).get("foreground_pixels"),
                        "timing_ms": result["timing_ms"]})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    summary = {"status": "completed", "images": len(images), "mode": args.mode, "results": records,
               "timing_note": "Per-image preprocessing + ONNX inference + postprocessing; excludes load/save; not a benchmark"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    marker.unlink()


if __name__ == "__main__":
    main()
