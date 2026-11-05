"""Export the joint checkpoint; no retraining/downloads or training-code edits."""
import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from jointcv.commands import load_trained
from jointcv.common import WORKSPACE, read_image, sha256, write_json


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        prediction = self.model([images[0]])[0]
        return prediction["boxes"], prediction["labels"], prediction["scores"], prediction["masks"]


class NativeResizeTransform(GeneralizedRCNNTransform):
    """Match eager Python float64 size arithmetic instead of tracing float32.

    Stock torchvision's tracing resize can round a nominal 1024 to 1023 for
    some aspect ratios. Feature/image tensors remain FP32; only shape arithmetic
    uses doubles, matching the eager scale-factor calculation and floor.
    """
    def resize(self, image, target=None):
        if target is not None:
            raise ValueError("This transform is for inference export only")
        shape = torch._shape_as_tensor(image)[-2:].to(dtype=torch.float64)
        # Keep constants and reductions rank-1: the legacy exporter silently
        # narrows rank-0 floating constants to FP32 during scalar promotion.
        short = torch.tensor([float(self.min_size[-1])], dtype=torch.float64, device=image.device)
        long = torch.tensor([float(self.max_size)], dtype=torch.float64, device=image.device)
        scale = torch.minimum(torch.div(short, shape.min().reshape(1)),
                              torch.div(long, shape.max().reshape(1)))
        size = torch.floor(shape * scale).to(dtype=torch.int64)
        image = F.interpolate(image[None], size=[size[0], size[1]], mode="bilinear", align_corners=False)[0]
        return image, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WORKSPACE/"runs/maskrcnn_m1/weights/best_joint.pt")
    parser.add_argument("--example", type=Path, default=WORKSPACE/"examples/inputs/example_3.png")
    parser.add_argument("--output", type=Path, default=WORKSPACE/"exports/maskrcnn")
    parser.add_argument("--score-floor", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.score_floor < 1:
        raise ValueError("score-floor must be in (0,1)")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    marker = output/"INCOMPLETE"
    marker.write_text("Export/verification is not complete.\n", encoding="utf-8")
    started = time.monotonic()
    torch.set_num_threads(2)
    model, config, provenance = load_trained(args.weights.resolve(), torch.device("cpu"))
    # Low-score proposals cannot suppress higher-score boxes in descending-score
    # NMS. Encoding the deployment floor avoids computing discarded instance masks.
    model.roi_heads.score_thresh = args.score_floor
    original_transform = model.transform
    model.transform = NativeResizeTransform(original_transform.min_size, original_transform.max_size,
                                             original_transform.image_mean, original_transform.image_std,
                                             size_divisible=original_transform.size_divisible).eval()
    wrapper = ExportWrapper(model).eval()
    original = read_image(args.example)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    dummy = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0)/255.
    try:
        path = output/"model.onnx"
        with torch.no_grad():
            torch.onnx.export(wrapper, dummy, str(path), opset_version=17, dynamo=False,
                              input_names=["images"], output_names=["boxes", "labels", "scores", "masks"],
                              dynamic_axes={"images": {2: "height", 3: "width"},
                                            "boxes": {0: "detections"}, "labels": {0: "detections"},
                                            "scores": {0: "detections"},
                                            "masks": {0: "detections", 2: "height", 3: "width"}},
                              do_constant_folding=True)
        onnx.checker.check_model(str(path), full_check=True)
        graph = onnx.load(str(path), load_external_data=False)
        operators = sorted(set(node.op_type for node in graph.graph.node))
        manifest = {
            "schema_version": 1, "status": "exported_not_verified", "file": path.name,
            "onnx_sha256": sha256(path), "bytes": path.stat().st_size,
            "source_checkpoint": str(args.weights.resolve()), "source_sha256": provenance["weights_sha256"],
            "checkpoint_epoch": provenance["epoch"], "architecture": "Mask R-CNN ResNet50-FPN v2",
            "opset": 17, "precision": "float32", "batch": 1, "dynamic_height_width": True,
            "input": {"name": "images", "shape": [1, 3, "height", "width"], "dtype": "float32",
                      "preprocessing": "Original-size RGB /255 only; internal ImageNet normalization and aspect-preserving resize are embedded"},
            "internal_resize": {"min_size": config["model"]["min_size"], "max_size": config["model"]["max_size"]},
            "shape_arithmetic": "float64 scale/floor to match eager Python resize; learned feature tensors remain float32",
            "outputs": {"boxes": "[N,4] xyxy in original image coordinates; exclusive x2/y2",
                        "labels": "[N] int64, defect class=1", "scores": "[N] probabilities",
                        "masks": "[N,1,H,W] original-resolution mask probabilities, already sigmoid/pasted"},
            "embedded_score_floor": args.score_floor,
            "embedded_nms_iou": config["evaluation"]["nms_iou"], "max_detections": config["evaluation"]["max_detections"],
            "default_mask_threshold": .5, "default_score_threshold": args.score_floor,
            "operators": operators,
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "onnx", "onnxruntime")},
            "export_seconds": time.monotonic()-started,
            "note": "Deployment export with embedded score floor; not an AP evaluation graph for lower confidence thresholds. Runtime verification is required, especially empty outputs and new spatial sizes.",
        }
        write_json(output/"manifest.json", manifest)
        print({"status": "exported_not_verified", "path": str(path), "bytes": path.stat().st_size,
               "seconds": manifest["export_seconds"], "operators": operators}, flush=True)
    except BaseException as error:
        write_json(output/"export_error.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
