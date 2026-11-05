import copy
import csv
import json
import math
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from .common import (WORKSPACE, capture_rng, checkpoint_load, code_hashes, get_device,
                     load_config, restore_rng, save_checkpoint, seed_everything, sha256, write_json)
from .data import DefectDataset, load_manifest, make_train_loader, smoke_subset
from .metrics import evaluate
from .model import build_model, freeze_bn, learning_rate


def save_history(run, history):
    write_json(run / "history.json", history)
    rows = []
    for h in history:
        rows.append({"epoch": h["epoch"], "lr": h["lr"], "train_loss": h["train_loss"], **h["losses"],
                     "val_box_mAP50": h["validation"]["box"]["mAP50"],
                     "val_box_mAP50_95": h["validation"]["box"]["mAP50_95"],
                     "val_mask_mAP50": h["validation"]["mask"]["mAP50"],
                     "val_dice": h["validation"]["foreground"]["foreground_dice_micro"],
                     "val_iou": h["validation"]["foreground"]["foreground_iou_micro"],
                     "joint_score": h["validation"]["joint_score"],
                     "seconds": h["seconds"], "peak_gpu_allocated_mib": h["peak_gpu_allocated_mib"],
                     "positive_exposures": h["positive_exposures"], "normal_exposures": h["normal_exposures"]})
    if rows:
        with (run / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train_epoch(model, loader, optimizer, scaler, config, device, updates, total_updates):
    model.train()
    freeze_bn(model)
    options = config["train"]
    optimizer.zero_grad(set_to_none=True)
    totals, seen, positive, accumulated, microsteps, skipped = {}, 0, 0, 0, 0, 0
    last_lr = optimizer.param_groups[0]["lr"]
    bar = tqdm(loader, desc="Joint training", mininterval=5)
    for index, (images, targets, records) in enumerate(bar):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=options["amp"] and device.type == "cuda"):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not all(torch.isfinite(v).all().item() for v in losses.values()):
            raise RuntimeError(f"Non-finite training losses at {[r['id'] for r in records]}")
        batch = len(images)
        scaler.scale(loss * batch).backward()
        accumulated += batch
        microsteps += 1
        seen += batch
        positive += sum(r["positive"] for r in records)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.) + float(value.detach()) * batch
        if microsteps == options["accumulate"] or index+1 == len(loader):
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], options["clip_grad_norm"])
            last_lr = learning_rate(updates, total_updates, options["warmup_updates"], options["lr"], options["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped += 1
            else:
                updates += 1
            optimizer.zero_grad(set_to_none=True)
            accumulated = microsteps = 0
        bar.set_postfix(loss=f"{float(loss.detach()):.3f}", lr=f"{last_lr:.2g}")
    values = {name: value/seen for name, value in totals.items()}
    return {"train_loss": sum(values.values()), "losses": values, "lr": last_lr,
            "positive_exposures": positive, "normal_exposures": seen-positive,
            "skipped_amp_updates": skipped}, updates


def train(config_path, name=None, resume=None, smoke=False, device_override=None, stop_after_epoch=None):
    state = checkpoint_load(resume) if resume else None
    config = copy.deepcopy(state["config"]) if state else load_config(config_path)
    if device_override:
        config["train"]["device"] = device_override
    if state and smoke:
        raise ValueError("Resume uses the checkpoint's mode; omit --smoke")
    if state:
        smoke = state["mode"] == "smoke"
    if smoke and not state:
        config["train"].update(epochs=2, minimum_epochs=2, warmup_updates=0, accumulate=2)
    seed_everything(config["train"]["seed"], config["train"]["cpu_threads"])
    device = get_device(config["train"]["device"])
    prepared, records = load_manifest(config)
    if smoke:
        train_rows, val_rows = smoke_subset(records, "train"), smoke_subset(records, "val", 3, 3)
    else:
        train_rows = [r for r in records if r["split"] == "train"]
        val_rows = [r for r in records if r["split"] == "val"]
    dataset = DefectDataset(prepared, train_rows, config["train"]["augmentation"])
    val_dataset = DefectDataset(prepared, val_rows)
    generator = torch.Generator().manual_seed(config["train"]["seed"])
    loader = make_train_loader(dataset, config, generator)
    updates_per_epoch = math.ceil(len(loader)/config["train"]["accumulate"])
    total_updates = updates_per_epoch*config["train"]["epochs"]
    mode = "smoke" if smoke else "formal"
    source_hashes = code_hashes()
    if state:
        if state.get("kind") != "training_state" or state.get("schema_version") != 1:
            raise ValueError("Resume requires our last.pt full training state")
        if state["code_hashes"] != source_hashes:
            raise ValueError("Training code changed since checkpoint; restore the run/source version before exact resume")
        run = Path(resume).resolve().parent.parent
        if state["config"]["train"]["device"] != config["train"]["device"]:
            raise ValueError("Exact resume requires the same device setting")
    else:
        name = name or f"maskrcnn_{config['experiment'].lower()}{'_smoke' if smoke else ''}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Run name must contain letters, numbers, dots, underscores or hyphens")
        # Check initialization BEFORE reserving a run directory.
        weight_path = (WORKSPACE / config["model"]["weights"]).resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        run = WORKSPACE / "runs" / name
        run.mkdir(parents=True, exist_ok=False)
        (run / "weights").mkdir()
        for relative in source_hashes:
            destination = run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WORKSPACE / relative, destination)
        write_json(run / "request.json", {"config": config, "mode": mode, "manifest": config["data"],
                                           "train_ids": [r["id"] for r in train_rows], "val_ids": [r["id"] for r in val_rows],
                                           "initial_weights_sha256": sha256(weight_path), "code_hashes": source_hashes,
                                           "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                                           "python": platform.python_version(), "device": str(device)})
    start = time.monotonic()
    history, best = (state["history"], state["best"]) if state else ([], {"box": -1., "dice": -1., "joint": -1.})
    start_epoch = state["epoch"]+1 if state else 1
    bad_epochs, updates = (state["bad_epochs"], state["optimizer_updates"]) if state else (0, 0)
    write_json(run / "summary.json", {"status": "initializing", "mode": mode, "run_dir": str(run), "epochs_completed": len(history)})
    try:
        model = build_model(config, None if state else weight_path)
        if state:
            model.load_state_dict(state["model"], strict=True)
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=config["train"]["amp_init_scale"],
                                     enabled=config["train"]["amp"] and device.type == "cuda")
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            generator.set_state(state["loader_rng"].cpu())
            restore_rng(state["rng"])
        state = None
        write_json(run / "summary.json", {"status": "running", "mode": mode, "run_dir": str(run), "epochs_completed": len(history)})
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, config["train"]["epochs"]+1):
            epoch_start = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training, updates = train_epoch(model, loader, optimizer, scaler, config, device, updates, total_updates)
            train_peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.
            validation = evaluate(model, val_dataset, config, device)
            peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.
            record = {"epoch": epoch, **training, "validation": validation,
                      "seconds": time.monotonic()-epoch_start, "train_peak_gpu_allocated_mib": train_peak,
                      "peak_gpu_allocated_mib": peak,
                      "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else 0.}
            history.append(record)
            scores = {"box": validation["box"]["mAP50_95"], "dice": validation["foreground"]["foreground_dice_micro"],
                      "joint": validation["joint_score"]}
            improved = scores["joint"] > best["joint"] + 1e-8
            bad_epochs = 0 if improved else bad_epochs+1
            for key, value in scores.items():
                if value > best[key] + 1e-8:
                    best[key] = value
                    save_checkpoint(run / "weights" / f"best_{key}.pt", {
                        "schema_version": 1, "kind": "inference_weights", "mode": mode, "epoch": epoch,
                        "config": config, "model": model.state_dict(), "metrics": validation,
                        "code_hashes": source_hashes,
                    })
            full_state = {"schema_version": 1, "kind": "training_state", "mode": mode,
                          "epoch": epoch, "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "rng": capture_rng(), "loader_rng": generator.get_state(),
                          "optimizer_updates": updates, "total_updates": total_updates, "bad_epochs": bad_epochs,
                          "history": history, "best": best, "code_hashes": source_hashes}
            save_checkpoint(run / "weights" / "last.pt", full_state)
            del full_state
            save_history(run, history)
            print(f"Epoch {epoch}/{config['train']['epochs']} | loss {training['train_loss']:.4f} | "
                  f"Box AP50 {validation['box']['mAP50']:.4f} AP {validation['box']['mAP50_95']:.4f} | "
                  f"Mask AP50 {validation['mask']['mAP50']:.4f} | Dice {scores['dice']:.4f} | "
                  f"GPU peak {peak:.0f} MiB | {record['seconds']:.1f}s", flush=True)
            write_json(run / "summary.json", {"status": "running", "mode": mode, "run_dir": str(run),
                                               "epochs_completed": epoch, "best": best, "last_epoch": record})
            if epoch >= config["train"]["minimum_epochs"] and bad_epochs >= config["train"]["patience"]:
                stop_reason = "early_stopping"
                break
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                stop_reason = "requested_epoch_boundary"
                break
        if updates == 0 and stop_reason != "requested_epoch_boundary":
            raise RuntimeError("No optimizer update succeeded; inspect AMP overflow before formal training")
        summary = {"status": "stopped" if stop_reason == "requested_epoch_boundary" else "completed", "mode": mode, "run_dir": str(run), "epochs_completed": len(history),
                   "stop_reason": stop_reason, "best": best, "optimizer_updates": updates,
                   "elapsed_seconds_this_invocation": time.monotonic()-start,
                   "peak_gpu_allocated_mib": max((h["peak_gpu_allocated_mib"] for h in history), default=0),
                   "evaluation_split": "val", "official_test_evaluated": False,
                   "note": "Smoke results are pipeline checks, not representative performance." if smoke else "Validation selection only; test is never evaluated automatically."}
        write_json(run / "summary.json", summary)
        return summary
    except BaseException as error:
        write_json(run / "summary.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                           "mode": mode, "run_dir": str(run), "epochs_completed": len(history),
                                           "error": f"{type(error).__name__}: {error}",
                                           "resume": "last.pt resumes from its last completed epoch; incomplete epoch is repeated"})
        raise
