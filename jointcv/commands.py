import argparse
import copy
import importlib.metadata
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import (EXPECTED_WEIGHT_SHA256, WORKSPACE, checkpoint_load, code_hashes, get_device,
                     load_config, read_image, seed_everything, sha256, write_image, write_json)
from .data import DefectDataset, load_manifest, smoke_subset
from .metrics import evaluate
from .model import build_model
from .training import train


def reserve_output(category, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Invalid output name")
    path = WORKSPACE / category / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_trained(path, device):
    checkpoint = checkpoint_load(path)
    if checkpoint.get("kind") not in ("inference_weights", "training_state"):
        raise ValueError("Use this experiment's best_joint.pt, best_box.pt, best_dice.pt or last.pt")
    config = copy.deepcopy(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config, {"epoch": checkpoint["epoch"], "mode": checkpoint["mode"],
                           "weights_sha256": sha256(path), "training_code_hashes": checkpoint["code_hashes"]}


def check(config_path):
    config = load_config(config_path)
    prepared, records = load_manifest(config)
    weights = (WORKSPACE / config["model"]["weights"]).resolve()
    digest = sha256(weights)
    if digest != EXPECTED_WEIGHT_SHA256:
        raise ValueError("COCO weight checksum mismatch")
    # A normal and a positive record exercise the two target layouts.
    dataset = DefectDataset(prepared, smoke_subset(records, "train", 1, 1))
    shapes = [{"id": r["id"], "image": list(image.shape), "boxes": list(target["boxes"].shape),
               "masks": list(target["masks"].shape)} for image, target, r in dataset]
    return {"status": "ready", "weights_sha256": digest,
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "numpy", "pycocotools")},
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "target_shapes": shapes, "training_started": False}


def evaluate_command(args):
    device = get_device(args.device)
    seed_everything(42, 2)
    model, config, provenance = load_trained(args.weights, device)
    if args.calibrate and args.split != "val":
        raise ValueError("Threshold calibration is only allowed on val")
    if provenance["mode"] == "smoke" and args.split == "test":
        raise ValueError("Smoke checkpoints are not eligible for a final test report")
    if args.thresholds:
        calibration = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        if calibration["split"] != "val" or calibration["weights_sha256"] != provenance["weights_sha256"]:
            raise ValueError("Calibration must come from val and the exact same checkpoint")
        if calibration["manifest_sha256"] != config["data"]["manifest_sha256"]:
            raise ValueError("Calibration dataset changed")
        if args.split == "test" and calibration.get("subset", True):
            raise ValueError("Final test calibration must use the full validation split")
        selected = calibration["selected"]
        if selected is None:
            raise ValueError("No calibration point met the normal-image FP constraint")
        config["evaluation"]["score_threshold"] = selected["score_threshold"]
        config["evaluation"]["mask_threshold"] = selected["mask_threshold"]
    prepared, records = load_manifest(config)
    rows = smoke_subset(records, args.split, 3, 3) if args.smoke_subset else [r for r in records if r["split"] == args.split]
    output = reserve_output("evaluations", args.name)
    started = time.monotonic()
    result = evaluate(model, DefectDataset(prepared, rows), config, device, output,
                      calibrate=args.calibrate, save_masks=args.save_masks)
    result.update({"status": "completed", "split": args.split, "subset": args.smoke_subset,
                   "checkpoint": provenance, "config": config, "evaluation_code_hashes": code_hashes(),
                   "elapsed_seconds": time.monotonic()-started})
    write_json(output / "summary.json", result)
    if args.calibrate:
        write_json(output / "calibration.json", {"split": "val", "subset": args.smoke_subset,
                                                 "weights_sha256": provenance["weights_sha256"],
                                                 "manifest_sha256": config["data"]["manifest_sha256"],
                                                 **result["calibration"]})
    return {"output": str(output), **{k: result[k] for k in ("images", "box", "mask", "foreground", "operating_point")}}




@torch.inference_mode()
def predict(args):
    seed_everything(42, 2)
    device = get_device(args.device)
    model, config, provenance = load_trained(args.weights, device)
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("Place output outside the input directory to avoid rescanning generated images")
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if not paths:
        raise ValueError("No input images")
    score_threshold = config["evaluation"]["score_threshold"]
    pixel_threshold = config["evaluation"]["mask_threshold"]
    if args.thresholds:
        calibration = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        if calibration["split"] != "val" or calibration["weights_sha256"] != provenance["weights_sha256"] or not calibration["selected"]:
            raise ValueError("Threshold file must be valid val calibration for this checkpoint")
        score_threshold = calibration["selected"]["score_threshold"]
        pixel_threshold = calibration["selected"]["mask_threshold"]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in paths:
        original = read_image(path)
        tensor = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))).float()/255.
        raw = model([tensor.to(device)])[0]
        keep = raw["scores"] >= score_threshold
        boxes, scores = raw["boxes"][keep].cpu().numpy(), raw["scores"][keep].cpu().numpy()
        masks = raw["masks"][keep, 0].cpu().numpy()
        mask = masks.max(axis=0) >= pixel_threshold if len(masks) else np.zeros(original.shape[:2], bool)
        overlay, detection = original.copy(), original.copy()
        overlay[mask] = (.55*original[mask]+.45*np.array([0, 0, 255])).astype(np.uint8)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(detection, (x1, y1), (x2-1, y2-1), (0, 230, 0), 1)
            cv2.putText(detection, f"{score:.2f}", (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 230, 0), 1)
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        destination = output / relative.parent / relative.name  # include source extension to avoid collisions
        write_image(Path(str(destination)+".mask.png"), mask.astype(np.uint8)*255)
        write_image(Path(str(destination)+".preview.jpg"), np.hstack((original, detection, overlay)))
        record = {"input": str(path), "shape_hw": list(original.shape[:2]), "boxes_xyxy": boxes.tolist(),
                  "scores": scores.tolist(), "foreground_pixels": int(mask.sum())}
        write_json(Path(str(destination)+".json"), record)
        rows.append(record)
    summary = {"status": "completed", "images": len(rows), "checkpoint": provenance,
               "score_threshold": score_threshold, "mask_threshold": pixel_threshold, "results": rows}
    write_json(output / "summary.json", summary)
    return {"status": "completed", "images": len(rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description="KSDD2 shared-backbone Mask R-CNN experiments (local weights only)")
    sub = parser.add_subparsers(dest="command", required=True)
    checker = sub.add_parser("check", help="Check environment, checksum, split and target shapes")
    checker.add_argument("--config", default=str(WORKSPACE / "configs/m0.yaml"))
    trainer = sub.add_parser("train", help="Train M0/M1 or resume an epoch-boundary checkpoint")
    trainer.add_argument("--config", default=str(WORKSPACE / "configs/m0.yaml"))
    trainer.add_argument("--name")
    trainer.add_argument("--resume")
    trainer.add_argument("--device")
    trainer.add_argument("--smoke", action="store_true", help="2 epochs, 8 train and 6 val images; pipeline checks only")
    trainer.add_argument("--stop-after-epoch", type=int, help="Stop after a completed epoch without changing the configured LR horizon; useful for resume checks")
    evaluator = sub.add_parser("evaluate", help="COCO AP, original-resolution Dice/IoU and optional val calibration")
    evaluator.add_argument("--weights", required=True)
    evaluator.add_argument("--name", required=True)
    evaluator.add_argument("--split", choices=("val", "test"), default="val")
    evaluator.add_argument("--device", default="cuda:0")
    evaluator.add_argument("--calibrate", action="store_true")
    evaluator.add_argument("--thresholds")
    evaluator.add_argument("--save-masks", action="store_true")
    evaluator.add_argument("--smoke-subset", action="store_true")
    predictor = sub.add_parser("predict", help="Single-image or batch inference, saved masks/boxes/previews")
    predictor.add_argument("--weights", required=True)
    predictor.add_argument("--input", required=True)
    predictor.add_argument("--output", required=True)
    predictor.add_argument("--device", default="cuda:0")
    predictor.add_argument("--thresholds")
    args = parser.parse_args()
    if args.command == "check":
        result = check(args.config)
    elif args.command == "train":
        result = train(args.config, args.name, args.resume, args.smoke, args.device, args.stop_after_epoch)
    elif args.command == "evaluate":
        result = evaluate_command(args)
    else:
        result = predict(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
