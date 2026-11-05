"""KSDD2 integrity checks, fixed splits and mask-derived detection labels."""

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from .preparation_io import read_image, write_image


def mask_to_boxes(mask: np.ndarray, connectivity: int = 8) -> list[list[int]]:
    """One box per foreground component. x/y/w/h use exclusive right/bottom edges.

    Keep every foreground pixel, including one-pixel components. No dilation,
    component filtering or merging is applied to the ground-truth mask.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be two-dimensional")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=connectivity
    )
    boxes = [row[:4].tolist() for row in stats[1:count]]
    return sorted(boxes, key=lambda box: (box[1], box[0], box[2], box[3]))


def yolo_labels(boxes: list[list[int]], width: int, height: int) -> str:
    lines = []
    for x, y, w, h in boxes:
        if not (w > 0 and h > 0 and x >= 0 and y >= 0
                and x + w <= width and y + h <= height):
            raise ValueError(f"Invalid box: {(x, y, w, h)} in {(width, height)}")
        lines.append(
            f"0 {(x + w / 2) / width:.10f} {(y + h / 2) / height:.10f} "
            f"{w / width:.10f} {h / height:.10f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def stratified_split(records: list[dict], fraction: float, seed: int) -> dict[str, str]:
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = random.Random(seed)
    assignments = {}
    for positive in (False, True):
        group = sorted(
            (r for r in records if r["original_split"] == "train"
             and r["positive"] == positive), key=lambda r: r["id"]
        )
        if len(group) < 2:
            raise ValueError("Need at least two images of each class in training data")
        rng.shuffle(group)
        size = max(1, min(len(group) - 1, round(len(group) * fraction)))
        for index, record in enumerate(group):
            assignments[record["key"]] = "val" if index < size else "train"
    for record in records:
        if record["original_split"] == "test":
            assignments[record["key"]] = "test"
    return assignments


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_data(config: dict, root: Path) -> tuple[list[dict], dict]:
    source = (root / config["data"]["raw_dir"]).resolve()
    records, summary = [], {}
    for split in ("train", "test"):
        folder = source / split
        if not folder.is_dir():
            raise ValueError(f"Missing dataset directory: {folder}")
        images = {p.stem: p for p in folder.glob("*.png") if p.stem.isdigit()}
        masks = {p.stem[:-3]: p for p in folder.glob("*_GT.png")
                 if p.stem[:-3].isdigit()}
        if images.keys() != masks.keys():
            raise ValueError(
                f"{split}: missing masks={sorted(images.keys() - masks.keys())[:10]}, "
                f"missing images={sorted(masks.keys() - images.keys())[:10]}"
            )
        recognized = set(images.values()) | set(masks.values())
        ignored = sorted(p.name for p in folder.iterdir() if p.is_file()
                         and p not in recognized)
        positive, component_histogram = 0, Counter()
        for sample_id in tqdm(sorted(images), desc=f"Audit {split}"):
            image_path, mask_path = images[sample_id], masks[sample_id]
            image = read_image(image_path)
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask shape mismatch: {image_path}")
            if not set(np.unique(mask).tolist()).issubset({0, 255}):
                raise ValueError(f"Expected a binary 0/255 mask: {mask_path}")
            boxes = mask_to_boxes(mask, config["data"]["connectivity"])
            is_positive = bool(boxes)
            positive += int(is_positive)
            if is_positive:
                component_histogram[len(boxes)] += 1
            records.append({
                "id": sample_id, "key": f"{split}/{sample_id}",
                "original_split": split, "positive": is_positive,
                "width": image.shape[1], "height": image.shape[0],
                "foreground_pixels": int(np.count_nonzero(mask)),
                "boxes_xywh": boxes, "image_source": str(image_path.relative_to(root)),
                "mask_source": str(mask_path.relative_to(root)),
                "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
            })
        expected = config["data"].get("expected", {}).get(split)
        if expected and (len(images) != expected["images"] or positive != expected["positive"]):
            raise ValueError(f"{split}: got {len(images)} images / {positive} positives; expected {expected}")
        summary[split] = {
            "images": len(images), "positive": positive, "negative": len(images) - positive,
            "components_per_positive_image": dict(sorted(component_histogram.items())),
            "ignored_files": ignored,
        }
    return records, summary


def ensure_no_leakage(records: list[dict]) -> None:
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["image_sha256"]].append(record)
    for group in by_hash.values():
        if len({r["split"] for r in group}) > 1:
            raise ValueError(
                "Identical image bytes found across dataset splits: "
                + ", ".join(r["key"] for r in group)
            )


def save_preview(record: dict, root: Path, destination: Path) -> None:
    image = read_image(root / record["image_source"])
    mask = read_image(root / record["mask_source"], cv2.IMREAD_GRAYSCALE)
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.55 * image[foreground] + 0.45 * np.array([0, 0, 255])
    ).astype(np.uint8)
    boxed = image.copy()
    for x, y, w, h in record["boxes_xywh"]:
        cv2.rectangle(boxed, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
    panels = [image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay, boxed]
    headers = ["Image", "GT mask", "GT overlay", "Derived boxes"]
    result = np.hstack(panels)
    result = cv2.copyMakeBorder(result, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
    for index, label in enumerate(headers):
        cv2.putText(result, label, (index * record["width"] + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    write_image(destination / f"{record['id']}.jpg", result)


def prepare_data(config: dict, root: Path, output: Path | None = None) -> dict:
    output = (output or root / config["data"]["prepared_dir"]).resolve()
    raw = (root / config["data"]["raw_dir"]).resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("Prepared data directory must be separate from the raw dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists. Choose a new --output directory: {output}")
    records, audit = audit_data(config, root)
    assignments = stratified_split(records, config["data"]["val_fraction"], config["data"]["seed"])
    for record in records:
        record["split"] = assignments[record["key"]]
    ensure_no_leakage(records)
    output.mkdir(parents=True)
    # Write the completion manifest last. An interrupted export is never accepted as complete.
    (output / "INCOMPLETE").write_text("Export in progress. Do not train on this directory.\n", encoding="utf-8")
    summary = {}
    for split in ("train", "val", "test"):
        for category in ("images", "masks", "labels"):
            (output / category / split).mkdir(parents=True)
        subset = [r for r in records if r["split"] == split]
        for record in tqdm(subset, desc=f"Export {split}"):
            name = record["id"]
            shutil.copy2(root / record["image_source"], output / "images" / split / f"{name}.png")
            shutil.copy2(root / record["mask_source"], output / "masks" / split / f"{name}_GT.png")
            text = yolo_labels(record["boxes_xywh"], record["width"], record["height"])
            (output / "labels" / split / f"{name}.txt").write_text(text, encoding="utf-8")
            record["image"] = f"images/{split}/{name}.png"
            record["mask"] = f"masks/{split}/{name}_GT.png"
        summary[split] = {
            "images": len(subset), "positive": sum(r["positive"] for r in subset),
            "negative": sum(not r["positive"] for r in subset),
            "boxes": sum(len(r["boxes_xywh"]) for r in subset),
        }
    # Only training/validation images are previewed during development.
    positives = sorted((r for r in records if r["split"] == "train" and r["positive"]),
                       key=lambda r: (r["foreground_pixels"], r["id"]))
    count = min(config["data"]["preview_count"], len(positives))
    if count:
        indices = np.linspace(0, len(positives) - 1, count, dtype=int)
        for index in indices:
            save_preview(positives[index], root, output / "previews")
    negatives = [r for r in records if r["split"] == "train" and not r["positive"]]
    if negatives:
        save_preview(negatives[0], root, output / "previews")
    yolo_config = {
        "path": output.as_posix(), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": {0: "defect"},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(yolo_config, sort_keys=False), encoding="utf-8")
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "original_split", "split", "positive", "width", "height", "foreground_pixels"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema_version": 1, "seed": config["data"]["seed"],
        "val_fraction": config["data"]["val_fraction"],
        "annotation_rule": {"class": "defect", "connectivity": config["data"]["connectivity"],
                            "min_area_pixels": 1, "dilation": False, "merge": False},
        "raw_audit": audit, "splits": summary,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"report": report, "samples": records},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "INCOMPLETE").unlink()
    print(json.dumps({"output": str(output), "splits": summary}, ensure_ascii=False, indent=2))
    return report
