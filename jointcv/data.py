import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import WORKSPACE, portable_relative, read_image, sha256


def load_manifest(config):
    path = (WORKSPACE / config["data"]["manifest"]).resolve()
    if sha256(path) != config["data"]["manifest_sha256"]:
        raise ValueError("Dataset manifest SHA256 changed; review the split before training")
    if (path.parent / "INCOMPLETE").exists():
        raise ValueError("Prepared dataset is incomplete")
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document["samples"]
    for split, count in (("train", 1865), ("val", 466), ("test", 1004)):
        subset = [r for r in records if r["split"] == split]
        if len(subset) != count or len({r["id"] for r in subset}) != count:
            raise ValueError(f"Unexpected split size or duplicate IDs: {split}")
    return path.parent, records


def instances_from_mask(mask):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ids = sorted(range(1, count), key=lambda i: (stats[i, 1], stats[i, 0], stats[i, 2], stats[i, 3]))
    boxes, masks, areas = [], [], []
    for i in ids:
        x, y, w, h, area = stats[i].tolist()
        boxes.append([x, y, x + w, y + h])
        masks.append(labels == i)
        areas.append(area)
    h, w = mask.shape
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(ids), dtype=torch.int64),
        "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if ids else torch.zeros((0, h, w), dtype=torch.uint8),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros(len(ids), dtype=torch.int64),
    }


class DefectDataset(Dataset):
    def __init__(self, prepared, records, augment=None):
        self.prepared = Path(prepared)
        self.records = list(records)
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image(self.prepared / portable_relative(record["image"]))
        mask = read_image(self.prepared / portable_relative(record["mask"]), grayscale=True)
        if image.shape[:2] != mask.shape or mask.shape != (record["height"], record["width"]):
            raise ValueError(f"Image/mask dimensions changed: {record['id']}")
        target = instances_from_mask(mask)
        expected = [[x, y, x+w, y+h] for x, y, w, h in record["boxes_xywh"]]
        if target["boxes"].tolist() != expected or bool(len(expected)) != record["positive"]:
            raise ValueError(f"Mask and fixed manifest disagree: {record['id']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.augment:
            if random.random() < self.augment["horizontal_flip"]:
                image = image[:, ::-1].copy()
                target["masks"] = target["masks"].flip(-1)
                b = target["boxes"].clone()
                target["boxes"][:, 0] = image.shape[1] - b[:, 2]
                target["boxes"][:, 2] = image.shape[1] - b[:, 0]
            brightness = random.uniform(1-self.augment["brightness"], 1+self.augment["brightness"])
            contrast = random.uniform(1-self.augment["contrast"], 1+self.augment["contrast"])
            center = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip(((image-center)*contrast+center)*brightness, 0, 1)
        target["image_id"] = torch.tensor(int(record["id"]), dtype=torch.int64)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))), target, record


def collate(batch):
    return tuple(list(x) for x in zip(*batch))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


def make_train_loader(dataset, config, generator):
    options = config["train"]
    sampler = None
    if config["experiment"] == "M1":
        n_pos = sum(r["positive"] for r in dataset.records)
        n_neg = len(dataset) - n_pos
        if not n_pos or not n_neg:
            raise ValueError("M1 requires positive and normal images")
        fraction = options["positive_fraction"]
        weights = [fraction / n_pos if r["positive"] else (1-fraction)/n_neg for r in dataset.records]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=options["batch"], shuffle=sampler is None,
                      sampler=sampler, generator=generator, num_workers=options["workers"],
                      worker_init_fn=seed_worker, collate_fn=collate,
                      pin_memory=str(options["device"]).startswith("cuda"), persistent_workers=False)


def smoke_subset(records, split, positives=4, normals=4):
    subset = [r for r in records if r["split"] == split]
    return ([r for r in subset if r["positive"]][:positives]
            + [r for r in subset if not r["positive"]][:normals])
