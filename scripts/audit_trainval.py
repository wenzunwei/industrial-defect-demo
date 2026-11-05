"""Read-only train/val geometry audit. No test masks, GPU use or training.

Run with the existing industrial_cv Python. Outputs are limited to this workspace.
Anchor coverage is BEFORE regression, not learned RPN proposal recall.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import box_iou


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parents[1] / "reports"
MANIFEST = ROOT / "data/processed/ksdd2/manifest.json"


def percentiles(values):
    a = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(a, p)) for p in (0, 10, 25, 50, 75, 90, 100)}


def resize_hw(h, w, short=384, long=1024):
    scale = min(short / min(h, w), long / max(h, w))
    return int(h * scale), int(w * scale)


def anchor_audit(records, sizes, ratios):
    groups = defaultdict(list)
    for r in records:
        nh, nw = resize_hw(r["height"], r["width"])
        for x, y, w, h in r["boxes_xywh"]:
            groups[(nh, nw)].append([
                x * nw / r["width"], y * nh / r["height"],
                (x + w) * nw / r["width"], (y + h) * nh / r["height"],
            ])
    generator = AnchorGenerator(tuple((s,) for s in sizes), (tuple(ratios),) * 5)
    best = []
    for (nh, nw), gt in groups.items():
        # Single-image batch padding to stride 32, as in GeneralizedRCNNTransform.
        ph, pw = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
        # P2-P5 are strides 4-32. P6 is max_pool2d(P5, kernel=1, stride=2).
        grid = [[math.ceil(ph / s), math.ceil(pw / s)] for s in (4, 8, 16, 32, 64)]
        strides = [[torch.tensor(ph // gh), torch.tensor(pw // gw)] for gh, gw in grid]
        anchors = torch.cat(generator.grid_anchors(grid, strides))
        best.extend(box_iou(torch.tensor(gt, dtype=torch.float32), anchors).max(dim=1).values.tolist())
    return {
        "anchor_sizes_per_level": sizes, "ratios_height_over_width": ratios,
        "gt_boxes": len(best), "best_raw_anchor_iou": percentiles(best),
        "fraction_best_iou_ge": {str(t): sum(x >= t for x in best) / len(best) for t in (.3, .5, .7)},
    }


def main():
    torch.set_num_threads(2)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [r for r in manifest["samples"] if r["split"] in ("train", "val")]
    report = {
        "date": "2026-11-04", "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "test_masks_read": False, "training_started": False,
        "anchor_geometry": {"short_side": 384, "max_long_side": 1024, "padding_stride": 32},
        "interpretation": "Raw anchor geometry only, before regression and learned objectness. Not model AP or proposal recall. RPN low-quality matching can still assign positives below 0.7 IoU.",
        "splits": {},
    }
    candidates = {
        "torchvision_default": ([32, 64, 128, 256, 512], [.5, 1., 2.]),
        "smaller_scales": ([16, 32, 64, 128, 256], [.5, 1., 2.]),
        "smaller_scales_wider_ratios": ([16, 32, 64, 128, 256], [.2, 1., 5.]),
    }
    for split in ("train", "val"):
        rows = [r for r in records if r["split"] == split]
        short_sides, long_sides, elongations, occupancies, areas, fractions, short640 = ([] for _ in range(7))
        for r in rows:
            if not r["positive"]:
                continue
            path = ROOT / r["mask_source"]
            mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (r["height"], r["width"]):
                raise ValueError(f"Invalid mask: {path}")
            count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
            actual = sorted(stats[1:count, :4].tolist(), key=lambda b: (b[1], b[0], b[2], b[3]))
            if actual != r["boxes_xywh"]:
                raise ValueError(f"Manifest and mask disagree: {r['key']}")
            fractions.append(float(np.count_nonzero(mask) / mask.size))
            for _, _, w, h, area in stats[1:count]:
                short_sides.append(min(w, h))
                long_sides.append(max(w, h))
                elongations.append(float(max(w, h) / min(w, h)))
                occupancies.append(area / (w * h))
                areas.append(int(area))
                short640.append(float(min(w, h) * 640 / max(r["height"], r["width"])))
        report["splits"][split] = {
            "images": len(rows), "positive_images": sum(r["positive"] for r in rows),
            "normal_images": sum(not r["positive"] for r in rows), "components": len(areas),
            "bbox_short_side_px_original": percentiles(short_sides),
            "bbox_long_side_px_original": percentiles(long_sides),
            "bbox_elongation_long_over_short": percentiles(elongations),
            "component_area_px": percentiles(areas),
            "foreground_fraction_within_component_bbox": percentiles(occupancies),
            "foreground_fraction_of_positive_image": percentiles(fractions),
            "bbox_short_side_lt8_at_long640": sum(s < 8 for s in short640),
            "bbox_elongation_gt4": sum(s > 4 for s in elongations),
            "anchor_candidates": {name: anchor_audit(rows, sizes, ratios) for name, (sizes, ratios) in candidates.items()},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trainval_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
