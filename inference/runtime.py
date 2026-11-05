"""Standalone Mask R-CNN ONNX inference: numpy, OpenCV, onnxruntime only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode: {path}")
    data.tofile(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class MaskRCNNOnnx:
    def __init__(self, model_path, *, threads=2, allow_unverified=False):
        self.model_path = Path(model_path).resolve()
        self.manifest = json.loads(self.model_path.with_name("manifest.json").read_text(encoding="utf-8"))
        if not allow_unverified and (self.manifest.get("status") != "verified_cpu"
                                     or self.model_path.with_name("INCOMPLETE").exists()):
            raise ValueError("Use a completed, verified model directory")
        if file_hash(self.model_path) != self.manifest["onnx_sha256"]:
            raise ValueError("ONNX SHA256 mismatch")
        if threads < 1:
            raise ValueError("threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.session.get_outputs()]
        if self.session.get_inputs()[0].name != "images" or self.names != ["boxes", "labels", "scores", "masks"]:
            raise ValueError("Unexpected graph contract")

    def raw(self, image_bgr):
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR HxWx3 image")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)/255.
        arrays = self.session.run(self.names, {"images": images})
        result = dict(zip(self.names, arrays))
        n = len(result["scores"])
        h, w = image_bgr.shape[:2]
        if (result["boxes"].shape != (n, 4) or result["labels"].shape != (n,)
                or result["masks"].shape != (n, 1, h, w)):
            raise ValueError("Invalid original-resolution output geometry")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Non-finite ONNX output")
        return result

    def predict(self, image_bgr, score_threshold=None, mask_threshold=.5):
        score_threshold = self.manifest["default_score_threshold"] if score_threshold is None else score_threshold
        if not self.manifest["embedded_score_floor"] <= score_threshold <= 1 or not 0 < mask_threshold < 1:
            raise ValueError("score must be >= embedded floor and <=1; mask threshold must be in (0,1)")
        start = time.perf_counter()
        raw = self.raw(image_bgr)
        keep = raw["scores"] >= score_threshold
        boxes, labels, scores = (raw[k][keep] for k in ("boxes", "labels", "scores"))
        masks = raw["masks"][keep, 0]
        union = masks.max(axis=0) >= mask_threshold if len(masks) else np.zeros(image_bgr.shape[:2], bool)
        return {"boxes": boxes, "labels": labels, "scores": scores, "mask": union,
                "seconds": time.perf_counter()-start, "score_threshold": score_threshold, "mask_threshold": mask_threshold}


def save_prediction(image, prediction, prefix):
    prefix = Path(prefix)
    detected, overlay = image.copy(), image.copy()
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(detected, (x1, y1), (x2-1, y2-1), (0, 230, 0), 1, cv2.LINE_AA)
        cv2.putText(detected, f"{score:.3f}", (max(0, x1), max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 230, 0), 1)
    mask = prediction["mask"]
    overlay[mask] = (.55*image[mask]+.45*np.array([210, 180, 0])).astype(np.uint8)
    panels = np.hstack((image, detected, overlay))
    panels = cv2.copyMakeBorder(panels, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    for i, title in enumerate(("Input", "Detection", "Segmentation")):
        cv2.putText(panels, title, (i*image.shape[1]+5, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    save_image(str(prefix)+".mask.png", mask.astype(np.uint8)*255)
    save_image(str(prefix)+".preview.jpg", panels)
    row = {"shape_hw": list(image.shape[:2]), "boxes_xyxy": prediction["boxes"].tolist(),
           "labels": prediction["labels"].tolist(), "scores": prediction["scores"].tolist(),
           "foreground_pixels": int(mask.sum()), "score_threshold": prediction["score_threshold"],
           "mask_threshold": prediction["mask_threshold"], "elapsed_seconds_cpu": prediction["seconds"]}
    write_json(str(prefix)+".json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1]/"models/model.onnx")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", type=float)
    parser.add_argument("--mask-threshold", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (source == output or source in output.parents):
        raise ValueError("Output must be outside the input directory")
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if not paths:
        raise ValueError("No image files")
    if output.exists():
        raise FileExistsError(output)
    model = MaskRCNNOnnx(args.model, threads=args.threads)
    output.mkdir(parents=True)
    write_json(output/"summary.json", {"status": "running"})
    results = []
    for path in paths:
        image = read_image(path)
        prediction = model.predict(image, args.score, args.mask_threshold)
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        row = save_prediction(image, prediction, output/relative.parent/relative.name)
        results.append({"input": str(path), **row})
    summary = {"status": "completed", "images": len(results), "model_sha256": model.manifest["onnx_sha256"],
               "provider": "CPUExecutionProvider", "results": results}
    write_json(output/"summary.json", summary)
    print(json.dumps({"status": "completed", "images": len(results), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
