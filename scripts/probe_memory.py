"""Bounded real-data GPU batch probe; isolated random task heads, no checkpoints.

Uses high-instance-count TRAIN positives only. This is a resource check, not a
formal experiment. No validation/test images are used to choose model weights.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from jointcv.common import WORKSPACE, get_device, load_config, seed_everything, write_json
from jointcv.data import DefectDataset, load_manifest
from jointcv.model import build_model, freeze_bn


def main():
    config = load_config(WORKSPACE / "configs/m0.yaml")
    device = get_device(config["train"]["device"])
    prepared, records = load_manifest(config)
    rows = sorted((r for r in records if r["split"] == "train" and r["positive"]),
                  key=lambda r: (len(r["boxes_xywh"]), r["foreground_pixels"]), reverse=True)[:4]
    dataset = DefectDataset(prepared, rows)
    results = []
    for batch_size in (1, 2, 4):
        seed_everything(42, 2)
        model = build_model(config, WORKSPACE / config["model"]["weights"]).to(device).train()
        freeze_bn(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        samples = [dataset[i] for i in range(batch_size)]
        images = [s[0].to(device) for s in samples]
        targets = [{k: v.to(device) for k, v in s[1].items()} for s in samples]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.monotonic()
        try:
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            row = {"batch": batch_size, "status": "passed", "optimizer_step_attempts": 3,
                   "seconds": time.monotonic()-started,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20,
                   "train_ids": [s[2]["id"] for s in samples], "last_loss": float(loss.detach())}
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch_size, "status": "out_of_memory"}
        results.append(row)
        print(row, flush=True)
        del model, optimizer, scaler, images, targets, samples
        if "loss" in locals():
            del loss, losses
        torch.cuda.empty_cache()
        if row["status"] != "passed":
            break
    write_json(WORKSPACE / "reports/memory_probe.json", {
        "status": "completed", "purpose": "Resource probe only, all-positive train batches; no formal training checkpoints saved",
        "device": torch.cuda.get_device_name(device), "image_transform": {"min_size": 384, "max_size": 1024},
        "amp": True, "results": results,
    })


if __name__ == "__main__":
    main()
