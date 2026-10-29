"""Two independent ONNX models behind a single original-image interface."""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("Expected a non-empty uint8 BGR image with three channels")


def detection_input(image, size=(640, 640), stride=32):
    """Match Ultralytics LetterBox(auto=True), BGR->RGB and /255."""
    h, w = image.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    dh, dw = (size[0] - rh) % stride, (size[1] - rw) % stride
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, {"scale": scale, "top": top, "left": left, "height": h, "width": w}


def segmentation_input(image, size=(640, 256)):
    """Match the trained segmenter: normalize RGB, then pad with normalized zero."""
    h, w = image.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size[0] - rh) // 2, (size[1] - rw) // 2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    resized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32)
    tensor = np.zeros((1, 3, *size), np.float32)
    tensor[0, :, top:top + rh, left:left + rw] = resized.transpose(2, 0, 1)
    return tensor, {"top": top, "left": left, "resized_h": rh, "resized_w": rw,
                    "height": h, "width": w}


def nms(boxes, scores, iou_threshold=0.7, max_det=100):
    """Single-class greedy NMS; boxes are xyxy in network-input coordinates."""
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    keep = []
    while len(order) and len(keep) < max_det:
        first, rest = order[0], order[1:]
        keep.append(int(first))
        low = np.maximum(boxes[first, :2], boxes[rest, :2])
        high = np.minimum(boxes[first, 2:], boxes[rest, 2:])
        intersection = np.maximum(high - low, 0).prod(axis=1)
        overlap = intersection / np.maximum(area[first] + area[rest] - intersection, 1e-9)
        order = rest[overlap <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def decode_detection(raw, geometry, conf=0.25, iou=0.7, max_det=100):
    # YOLO11 single-class export has [batch, 4 xywh + 1 class score, anchors].
    if raw.ndim != 3 or raw.shape[:2] != (1, 5):
        raise ValueError(f"Expected single-class raw detection [1,5,N], got {raw.shape}")
    rows = raw[0].T
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] > conf)]
    boxes = np.column_stack((rows[:, :2] - rows[:, 2:4] / 2,
                             rows[:, :2] + rows[:, 2:4] / 2))
    scores = rows[:, 4]
    keep = nms(boxes, scores, iou, max_det)
    boxes, scores = boxes[keep].copy(), scores[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry["left"]) / geometry["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry["top"]) / geometry["scale"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, geometry["width"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, geometry["height"])
    return [{"box_xyxy": box.tolist(), "confidence": float(score), "class_id": 0,
             "class_name": "defect"} for box, score in zip(boxes, scores)]


def restore_segmentation(logits, geometry, threshold=0.5):
    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or not np.isfinite(logits).all():
        raise ValueError(f"Expected finite segmentation logits [1,1,H,W], got {logits.shape}")
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -80, 80)))
    y, x = geometry["top"], geometry["left"]
    cropped = probability[y:y + geometry["resized_h"], x:x + geometry["resized_w"]]
    restored = cv2.resize(cropped, (geometry["width"], geometry["height"]),
                          interpolation=cv2.INTER_LINEAR)
    return (restored >= threshold).astype(np.uint8) * 255


def draw_detection(image, detections):
    output = image.copy()
    h, w = image.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["box_xyxy"]]
        cv2.rectangle(output, (min(x1, w-1), min(y1, h-1)),
                      (min(x2, w-1), min(y2, h-1)), (0, 230, 255), 2)
        text = f"defect {detection['confidence']:.2f}"
        label_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(output, text, (max(0, min(x1, w-label_width-2)), max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 255), 1, cv2.LINE_AA)
    return output


def draw_segmentation(image, mask):
    output = image.copy()
    foreground = mask > 0
    output[foreground] = (0.55*image[foreground] + 0.45*np.array([200, 170, 0])).astype(np.uint8)
    return output


class DefectSystem:
    """Load only the requested models; all returned coordinates refer to the original image."""

    def __init__(self, models_dir, mode="both", provider="cpu", threads=2,
                 conf=None, iou=None, threshold=None):
        if mode not in ("both", "detect", "segment"):
            raise ValueError("Mode must be both, detect or segment")
        if provider not in ("cpu", "cuda") or threads < 1:
            raise ValueError("Provider must be cpu/cuda and threads must be positive")
        self.models_dir, self.mode = Path(models_dir).resolve(), mode
        marker = self.models_dir / "INCOMPLETE"
        if marker.exists():
            raise RuntimeError("ONNX export is incomplete; re-export to a new model directory")
        self.manifest = json.loads((self.models_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported model manifest schema")
        det, seg = self.manifest["detection"], self.manifest["segmentation"]
        self.conf = det["confidence"] if conf is None else conf
        self.iou = det["iou"] if iou is None else iou
        self.threshold = seg["threshold"] if threshold is None else threshold
        if not all(np.isfinite(v) and 0 < v < 1 for v in (self.conf, self.iou, self.threshold)):
            raise ValueError("Confidence, NMS IoU and segmentation threshold must be between zero and one")
        import onnxruntime as ort
        requested = "CPUExecutionProvider" if provider == "cpu" else "CUDAExecutionProvider"
        if requested not in ort.get_available_providers():
            raise RuntimeError(f"{requested} unavailable; installed providers: {ort.get_available_providers()}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = [requested] if provider == "cpu" else [requested, "CPUExecutionProvider"]
        self.sessions = {}
        tasks = ("detection", "segmentation") if mode == "both" else (
            "detection" if mode == "detect" else "segmentation",)
        for task in tasks:
            config = self.manifest[task]
            path = (self.models_dir / config["file"]).resolve()
            if path.parent != self.models_dir or sha256(path) != config["onnx_sha256"]:
                raise ValueError(f"Model file/path/hash does not match manifest: {task}")
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            if requested not in session.get_providers():
                raise RuntimeError(f"Requested provider failed to initialize: {requested}")
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != config["input_name"] or
                    outputs[0].name != config["output_name"] or inputs[0].type != "tensor(float)"):
                raise ValueError(f"Unexpected ONNX model interface: {task}")
            self.sessions[task] = session

    def run_model(self, task, tensor):
        config = self.manifest[task]
        return self.sessions[task].run([config["output_name"]], {config["input_name"]: tensor})[0]

    def predict(self, image):
        check_image(image)
        result = {"image_size": {"height": image.shape[0], "width": image.shape[1]},
                  "mode": self.mode, "detections": None, "mask": None, "timing_ms": {}}
        for task in self.sessions:
            started = time.perf_counter()
            config = self.manifest[task]
            if task == "detection":
                tensor, geometry = detection_input(image, tuple(config["input_size"]), config["stride"])
                raw = self.run_model(task, tensor)
                result["detections"] = decode_detection(raw, geometry, self.conf, self.iou, config["max_det"])
            else:
                tensor, geometry = segmentation_input(image, tuple(config["input_size"]))
                raw = self.run_model(task, tensor)
                result["mask"] = restore_segmentation(raw, geometry, self.threshold)
            result["timing_ms"][task] = (time.perf_counter() - started) * 1000
        return result

    def save(self, image, result, output_dir, source=None):
        """Save PNG/JSON artifacts. Never modify the input image or source file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        artifacts = {}
        combined = image.copy()
        if result["mask"] is not None:
            mask = result["mask"]
            overlay = draw_segmentation(image, mask)
            write_image(output_dir / "mask.png", mask)
            write_image(output_dir / "segmentation.png", overlay)
            artifacts.update(mask="mask.png", segmentation="segmentation.png")
            combined = overlay
        if result["detections"] is not None:
            write_image(output_dir / "detection.png", draw_detection(image, result["detections"]))
            artifacts["detection"] = "detection.png"
            combined = draw_detection(combined, result["detections"])
        write_image(output_dir / "combined.png", combined)
        artifacts["combined"] = "combined.png"
        metadata = {k: v for k, v in result.items() if k != "mask"}
        metadata.update(source=str(source) if source is not None else None, artifacts=artifacts,
                        thresholds={"detection_confidence": self.conf, "nms_iou": self.iou,
                                    "segmentation": self.threshold},
                        providers={task: session.get_providers() for task, session in self.sessions.items()},
                        model_sha256={task: self.manifest[task]["onnx_sha256"] for task in self.sessions})
        if result["mask"] is not None:
            metadata["segmentation"] = {"foreground_pixels": int(np.count_nonzero(result["mask"])),
                                        "mask_size": list(result["mask"].shape)}
        (output_dir / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding="utf-8")
        return metadata
