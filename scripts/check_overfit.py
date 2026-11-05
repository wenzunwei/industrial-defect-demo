"""Bounded two-image TRAIN-only learning check; discard model afterwards."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jointcv.common import WORKSPACE, get_device, load_config, seed_everything, write_json
from jointcv.data import DefectDataset, load_manifest, smoke_subset
from jointcv.metrics import evaluate
from jointcv.model import build_model, freeze_bn


def main():
    config = load_config(WORKSPACE / "configs/m0.yaml")
    seed_everything(42, 2)
    device = get_device("cuda:0")
    prepared, records = load_manifest(config)
    dataset = DefectDataset(prepared, smoke_subset(records, "train", 1, 1))
    samples = [dataset[i] for i in range(2)]
    model = build_model(config, WORKSPACE / config["model"]["weights"]).to(device).train()
    freeze_bn(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", init_scale=config["train"]["amp_init_scale"])
    images = [s[0].to(device) for s in samples]
    targets = [{k: v.to(device) for k, v in s[1].items()} for s in samples]
    losses = []
    started = time.monotonic()
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite overfit-check loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if (step+1) % 10 == 0:
            print(f"Train-only learning check {step+1}/50: loss={losses[-1]:.4f}", flush=True)
    metrics = evaluate(model, dataset, config, device)
    first, last = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
    report = {"status": "passed" if last < first else "needs_investigation", "formal_experiment": False,
              "train_ids": [s[2]["id"] for s in samples], "steps": 50, "lr": 0.0003,
              "augmentation": False, "warmup": False, "seconds": time.monotonic()-started,
              "first_10_loss_mean": first, "last_10_loss_mean": last, "losses": losses,
              "train_only_metrics": metrics,
              "note": "Two reused train images; metrics measure memorization only. Model discarded, no checkpoint for deployment or comparison."}
    write_json(WORKSPACE / "reports/overfit_check.json", report)
    print({"status": report["status"], "first_10_loss_mean": first, "last_10_loss_mean": last,
           "box_ap50_train_only": metrics["box"]["mAP50"], "dice_train_only": metrics["foreground"]["foreground_dice_micro"]})
    if last >= first:
        raise RuntimeError("The bounded learning check did not reduce loss")


if __name__ == "__main__":
    main()
