"""COCO box/mask AP and original-resolution foreground metrics."""
import contextlib
import copy
import csv
import io
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from .common import read_image, portable_relative, write_image, write_json
from .data import collate


class ForegroundMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.normal = self.normal_fp = 0
        self.positive_dice, self.positive_iou = [], []

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        if prediction.shape != target.shape:
            raise ValueError("Mask shape mismatch")
        tp, fp, fn = (int(np.count_nonzero(a)) for a in
                      (prediction & target, prediction & ~target, ~prediction & target))
        self.tp += tp
        self.fp += fp
        self.fn += fn
        positive = bool(target.any())
        dice = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 1.0
        if positive:
            self.positive_dice.append(dice)
            self.positive_iou.append(iou)
        else:
            self.normal += 1
            self.normal_fp += int(prediction.any())
        return {"positive": positive, "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
                "dice": dice, "iou": iou, "predicted_pixels": int(prediction.sum())}

    def compute(self):
        return {
            "foreground_dice_micro": 2*self.tp/(2*self.tp+self.fp+self.fn) if 2*self.tp+self.fp+self.fn else 1.0,
            "foreground_iou_micro": self.tp/(self.tp+self.fp+self.fn) if self.tp+self.fp+self.fn else 1.0,
            "positive_dice_macro": float(np.mean(self.positive_dice)) if self.positive_dice else None,
            "positive_iou_macro": float(np.mean(self.positive_iou)) if self.positive_iou else None,
            "normal_false_positive_image_rate": self.normal_fp/self.normal if self.normal else None,
            "positive_images": len(self.positive_dice), "normal_images": self.normal,
            "normal_false_positive_images": self.normal_fp,
            "tp_pixels": self.tp, "fp_pixels": self.fp, "fn_pixels": self.fn,
        }


def encode_mask(mask):
    rle = mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def coco_object(images, annotations):
    result = COCO()
    result.dataset = {"info": {"description": "Fixed KSDD2 connected-component instances"},
                      "images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "defect"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        result.createIndex()
    return result


def coco_metrics(gt, predictions, kind):
    with contextlib.redirect_stdout(io.StringIO()):
        # loadRes mutates annotations (including adding NumPy uint32 mask areas).
        # Preserve our JSON-ready prediction payload and category fields.
        dt = gt.loadRes(copy.deepcopy(predictions)) if predictions else coco_object(gt.dataset["images"], [])
        evaluator = COCOeval(gt, dt, kind)
        evaluator.params.imgIds = sorted(gt.getImgIds())
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"mAP50_95": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1]),
            "mAP75": float(evaluator.stats[2]), "AR100": float(evaluator.stats[8])}


def box_counts(boxes, scores, target_boxes, threshold, iou_threshold=0.5):
    keep = scores >= threshold
    boxes, scores = boxes[keep], scores[keep]
    if not len(boxes):
        return 0, 0, len(target_boxes)
    overlaps = box_iou(boxes, target_boxes) if len(target_boxes) else torch.empty((len(boxes), 0))
    matched, tp = set(), 0
    for i in torch.argsort(scores, descending=True).tolist():
        candidates = [(float(overlaps[i, j]), j) for j in range(len(target_boxes)) if j not in matched]
        overlap, j = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            matched.add(j)
            tp += 1
    return tp, len(boxes)-tp, len(target_boxes)-tp


def proposal_matches(proposals, target_boxes, top_k, threshold):
    if not len(target_boxes) or not len(proposals):
        return 0
    return int((box_iou(proposals[:top_k], target_boxes).max(dim=0).values >= threshold).sum())


def forward_with_proposals(model, image):
    """Capture learned RPN boxes and undo INTERNAL resize for recall diagnostics."""
    captured = {}
    def transform_hook(module, inputs, output):
        captured["hw"] = output[0].image_sizes[0]
    def rpn_hook(module, inputs, output):
        captured["proposals"] = output[0][0].detach().float().cpu()
    handles = [model.transform.register_forward_hook(transform_hook), model.rpn.register_forward_hook(rpn_hook)]
    try:
        prediction = model([image])[0]
    finally:
        for handle in handles:
            handle.remove()
    rh, rw = captured["hw"]
    h, w = image.shape[-2:]
    proposals = captured["proposals"] / torch.tensor([rw/w, rh/h, rw/w, rh/h])
    return prediction, proposals


def score_unions(masks, scores, thresholds):
    """Incremental per-pixel maxima; each retained instance is reduced only once."""
    h, w = masks.shape[-2:]
    current = np.zeros((h, w), np.float32)
    order = np.argsort(-scores, kind="stable")
    cursor, output = 0, {}
    for threshold in sorted(set(thresholds), reverse=True):
        while cursor < len(order) and scores[order[cursor]] >= threshold:
            np.maximum(current, masks[order[cursor]], out=current)
            cursor += 1
        output[threshold] = current.copy()
    return output


def save_preview(folder, prepared, record, boxes, scores, predicted, threshold):
    original = read_image(prepared / portable_relative(record["image"]))
    gt = read_image(prepared / portable_relative(record["mask"]), grayscale=True) > 0
    detected, overlay = original.copy(), original.copy()
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(detected, (x1, y1), (x2-1, y2-1), (0, 230, 0), 1)
            cv2.putText(detected, f"{score:.2f}", (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 230, 0), 1)
    overlay[predicted] = (.55*original[predicted] + .45*np.array([0, 0, 255])).astype(np.uint8)
    gt_overlay = original.copy()
    gt_overlay[gt] = (.55*original[gt] + .45*np.array([0, 220, 0])).astype(np.uint8)
    canvas = cv2.copyMakeBorder(np.hstack((original, gt_overlay, detected, overlay)), 30, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(32, 32, 32))
    for i, label in enumerate(("Image", "Ground truth", "Detection", "Segmentation")):
        cv2.putText(canvas, label, (i*original.shape[1]+5, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    write_image(folder / "previews" / f"{record['id']}.jpg", canvas)


@torch.inference_mode()
def evaluate(model, dataset, config, device, output=None, calibrate=False, save_masks=False):
    model.eval()
    options = config["evaluation"]
    pairs = {(options["score_threshold"], options["mask_threshold"])}
    if calibrate:
        pairs.update((s, m) for s in options["score_grid"] for m in options["mask_grid"])
    aggregates = {pair: ForegroundMetrics() for pair in pairs}
    gt_images, annotations, box_predictions, mask_predictions, rows = [], [], [], [], []
    box_tp = box_fp = box_fn = 0
    proposal_totals = {(k, t): 0 for k in (100, 1000) for t in (.5, .7)}
    selected = {r["id"] for r in dataset.records if r["positive"]}
    previews = set(sorted(selected)[:4] + [r["id"] for r in dataset.records if not r["positive"]][:2])
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config["train"]["seed"]+100))
    for images, targets, records in tqdm(loader, desc="Joint validation", mininterval=5):
        target, record = targets[0], records[0]
        raw, proposals = forward_with_proposals(model, images[0].to(device))  # FP32 evaluation.
        boxes = raw["boxes"].detach().float().cpu()
        scores = raw["scores"].detach().float().cpu()
        masks = raw["masks"][:, 0].detach().float().cpu().numpy()
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not np.isfinite(masks).all():
            raise RuntimeError("Non-finite inference output")
        image_id = int(record["id"])
        gt_images.append({"id": image_id, "width": record["width"], "height": record["height"]})
        for j, box in enumerate(target["boxes"].tolist()):
            x1, y1, x2, y2 = box
            annotations.append({"id": len(annotations)+1, "image_id": image_id, "category_id": 1,
                                "bbox": [x1, y1, x2-x1, y2-y1], "area": float(target["area"][j]),
                                "iscrowd": 0, "segmentation": encode_mask(target["masks"][j].numpy())})
        for j, (box, score) in enumerate(zip(boxes.tolist(), scores.tolist())):
            x1, y1, x2, y2 = box
            box_predictions.append({"image_id": image_id, "category_id": 1,
                                    "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
            # No bbox field: COCO mask area must be derived from mask, not rectangle.
            mask_predictions.append({"image_id": image_id, "category_id": 1,
                                     "segmentation": encode_mask(masks[j] >= .5), "score": score})
        target_union = target["masks"].any(dim=0).numpy()
        proposal_row = {}
        for (k, threshold) in proposal_totals:
            matched = proposal_matches(proposals, target["boxes"], k, threshold)
            proposal_totals[(k, threshold)] += matched
            proposal_row[f"rpn_matched_at{k}_iou{int(threshold*100)}"] = matched
        unions = score_unions(masks, scores.numpy(), [p[0] for p in pairs])
        default_prediction = unions[options["score_threshold"]] >= options["mask_threshold"]
        row = None
        for pair, metric in aggregates.items():
            value = metric.update(unions[pair[0]] >= pair[1], target_union)
            if pair == (options["score_threshold"], options["mask_threshold"]):
                row = value
        tp, fp, fn = box_counts(boxes, scores, target["boxes"], options["score_threshold"])
        box_tp += tp
        box_fp += fp
        box_fn += fn
        rows.append({"id": record["id"], **row, "box_tp": tp, "box_fp": fp, "box_fn": fn,
                     "gt_instances": len(target["boxes"]), **proposal_row})
        if output is not None:
            output = Path(output)
            if record["id"] in previews:
                save_preview(output, dataset.prepared, record, boxes.numpy(), scores.numpy(), default_prediction, options["score_threshold"])
            if save_masks:
                write_image(output / "predicted_masks" / f"{record['id']}.png", default_prediction.astype(np.uint8)*255)
        del raw
    if not gt_images:
        raise ValueError("Empty evaluation split")
    gt = coco_object(gt_images, annotations)
    if not annotations:
        raise ValueError("AP evaluation requires at least one annotated defect instance")
    foreground = aggregates[(options["score_threshold"], options["mask_threshold"])].compute()
    box_ap, mask_ap = coco_metrics(gt, box_predictions, "bbox"), coco_metrics(gt, mask_predictions, "segm")
    result = {
        "images": len(gt_images), "instances": len(annotations), "box": box_ap, "mask": mask_ap,
        "foreground": foreground,
        "operating_point": {"score_threshold": options["score_threshold"], "mask_threshold": options["mask_threshold"],
                            "box_iou_threshold": .5, "box_precision": box_tp/(box_tp+box_fp) if box_tp+box_fp else 0.,
                            "box_recall": box_tp/(box_tp+box_fn) if box_tp+box_fn else 0.,
                            "tp": box_tp, "fp": box_fp, "fn": box_fn},
        "joint_score": math.sqrt(max(0., box_ap["mAP50_95"])*foreground["foreground_dice_micro"]),
        "rpn_proposal_recall": {f"at{k}_iou{int(t*100)}": count/len(annotations)
                                for (k, t), count in proposal_totals.items()},
        "ap_protocol": "pycocotools COCO, maxDets=100, mask AP pixel threshold=0.5; full original resolution",
    }
    if calibrate:
        candidates = [{"score_threshold": s, "mask_threshold": m, **metric.compute()}
                      for (s, m), metric in sorted(aggregates.items())]
        feasible = [r for r in candidates if r["normal_false_positive_image_rate"] is not None
                    and r["normal_false_positive_image_rate"] <= options["max_normal_fp_rate"]]
        winner = max(feasible, key=lambda r: (r["foreground_dice_micro"], r["score_threshold"], r["mask_threshold"])) if feasible else None
        result["calibration"] = {"max_normal_fp_rate": options["max_normal_fp_rate"], "constraint_met": bool(feasible),
                                 "selected": winner, "candidates": candidates}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "metrics.json", result)
        write_json(output / "predictions_bbox.json", box_predictions)
        write_json(output / "predictions_mask.json", mask_predictions)
        with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result
