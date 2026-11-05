import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHT_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Python opens Unicode Windows paths reliably; the C++ filename writer does not.
    with tmp.open("wb") as stream:
        torch.save(value, stream)
    tmp.replace(path)


def read_image(path, grayscale=False):
    path = Path(path)
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Cannot read image: {path}")
    return result


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    t, m, e = config["train"], config["model"], config["evaluation"]
    if config["experiment"] not in ("M0", "M1"):
        raise ValueError("Only the reviewed M0 and M1 experiments are implemented")
    if m["name"] != "maskrcnn_resnet50_fpn_v2" or m["mask_roi_size"] not in (14, 28):
        raise ValueError("Unsupported model or mask ROI size")
    if min(m["min_size"], m["max_size"], t["batch"], t["accumulate"], t["epochs"], t["cpu_threads"], t["patience"]) < 1:
        raise ValueError("Sizes, batch, accumulation, epochs, threads and patience must be positive")
    if m["max_size"] < m["min_size"] or t["workers"] < 0 or t["warmup_updates"] < 0:
        raise ValueError("Invalid image geometry, workers or warmup")
    if not 0 < t["lr"] or not 0 <= t["min_lr"] <= t["lr"]:
        raise ValueError("Invalid learning rate")
    if not 0 < e["score_floor"] <= e["score_threshold"] < 1 or not 0 < e["mask_threshold"] < 1:
        raise ValueError("Invalid evaluation thresholds")
    if e["batch"] != 1:
        raise ValueError("Evaluate one image at a time to bound full-resolution mask memory")
    if not 0 < t["positive_fraction"] < 1:
        raise ValueError("positive_fraction must be in (0,1)")
    return config


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(value):
    value = str(value)
    if value.isdigit():
        value = "cuda:" + value
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check industrial_cv or explicitly choose --device cpu")
        torch.cuda.set_device(device)
    return device


def capture_rng():
    n = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": (n[0], torch.tensor(n[1].astype(np.int64)), int(n[2]), int(n[3]), float(n[4])),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    n = state["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("GPU count changed; exact RNG resume is unsupported")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def checkpoint_load(path):
    # Our checkpoints only contain tensors and primitive types, including RNG state.
    return torch.load(path, map_location="cpu", weights_only=True)


def code_hashes():
    paths = [WORKSPACE / "run.py", *sorted((WORKSPACE / "jointcv").glob("*.py"))]
    return {p.relative_to(WORKSPACE).as_posix(): sha256(p) for p in paths}


def portable_relative(value):
    # The inherited manifest was produced on Windows; it is also usable on Linux.
    path = Path(str(value).replace("\\", "/"))
    if path.anchor or ".." in path.parts or ":" in str(path):
        raise ValueError(f"Expected a relative dataset path: {value}")
    return path
