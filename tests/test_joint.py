import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointcv.common import capture_rng, checkpoint_load, restore_rng, save_checkpoint, portable_relative
from jointcv.data import collate, instances_from_mask
from jointcv.metrics import ForegroundMetrics, box_counts, coco_metrics, coco_object, encode_mask, proposal_matches, score_unions
from jointcv.model import freeze_bn, learning_rate
from jointcv.training import train_epoch


class InstanceTests(unittest.TestCase):
    def test_empty_target_and_single_pixel_are_valid(self):
        target = instances_from_mask(np.zeros((7, 9), np.uint8))
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["masks"].shape), (0, 7, 9))
        mask = np.zeros((7, 9), np.uint8)
        mask[6, 8] = 255
        target = instances_from_mask(mask)
        self.assertEqual(target["boxes"].tolist(), [[8., 6., 9., 7.]])
        self.assertEqual(target["area"].tolist(), [1.])

    def test_diagonal_connectivity_and_holes_preserved(self):
        mask = np.zeros((7, 9), np.uint8)
        mask[1:6, 1:6] = 255
        mask[2:5, 2:5] = 0
        mask[6, 6] = 255  # diagonal connection to the ring
        target = instances_from_mask(mask)
        self.assertEqual(len(target["boxes"]), 1)
        np.testing.assert_array_equal(target["masks"].any(0).numpy(), mask > 0)
        self.assertFalse(bool(target["masks"][0, 3, 3]))

    def test_manifest_paths_are_cross_platform_and_cannot_escape(self):
        self.assertEqual(portable_relative(r"images\train\100.png").as_posix(), "images/train/100.png")
        for path in ("../secret", "C:/secret", "/secret"):
            with self.assertRaises(ValueError):
                portable_relative(path)


class MetricTests(unittest.TestCase):
    def test_normal_images_do_not_inflate_dice(self):
        target = np.array([[True, False]])
        metric = ForegroundMetrics()
        metric.update(target, target)
        metric.update(target, np.zeros_like(target))
        for _ in range(20):
            metric.update(np.zeros_like(target), np.zeros_like(target))
        result = metric.compute()
        self.assertAlmostEqual(result["foreground_dice_micro"], 2/3)
        self.assertEqual(result["positive_dice_macro"], 1.)
        self.assertEqual(result["normal_false_positive_image_rate"], 1/21)

    def test_duplicate_boxes_cannot_match_same_target_twice(self):
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]])
        self.assertEqual(box_counts(boxes, torch.tensor([.9, .8]), boxes[:1], .25), (1, 1, 0))
        self.assertEqual(box_counts(boxes, torch.tensor([.1, .1]), boxes[:1], .25), (0, 0, 1))

    def test_proposal_recall_respects_top_k_and_is_not_detection_precision(self):
        proposals = torch.tensor([[6., 6., 9., 9.], [0., 0., 4., 4.], [0., 0., 4., 4.]])
        target = torch.tensor([[0., 0., 4., 4.]])
        self.assertEqual(proposal_matches(proposals, target, 1, .5), 0)
        self.assertEqual(proposal_matches(proposals, target, 3, .5), 1)

    def test_low_score_masks_do_not_leak_into_union(self):
        masks = np.array([[[.9, .0]], [[.0, .8]]], np.float32)
        unions = score_unions(masks, np.array([.8, .1]), [.05, .25, .9])
        np.testing.assert_array_equal(unions[.25] > .5, [[True, False]])
        np.testing.assert_array_equal(unions[.05] > .5, [[True, True]])
        self.assertFalse(unions[.9].any())
        self.assertFalse(score_unions(np.empty((0, 1, 2)), np.array([]), [.25])[.25].any())

    def test_coco_perfect_and_no_predictions_for_both_tasks(self):
        mask = np.zeros((8, 8), np.uint8)
        mask[1:5, 2:6] = 1
        rle = encode_mask(mask)
        gt = coco_object([{"id": 1, "height": 8, "width": 8}, {"id": 2, "height": 8, "width": 8}],
                         [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 4],
                           "area": 16, "iscrowd": 0, "segmentation": rle}])
        predictions = {"bbox": [{"image_id": 1, "category_id": 1, "bbox": [2, 1, 4, 4], "score": .9}],
                       "segm": [{"image_id": 1, "category_id": 1, "segmentation": rle, "score": .9}]}
        for task in ("bbox", "segm"):
            before = copy.deepcopy(predictions[task])
            self.assertAlmostEqual(coco_metrics(gt, predictions[task], task)["mAP50_95"], 1.)
            self.assertEqual(coco_metrics(gt, [], task)["mAP50"], 0.)
            self.assertEqual(predictions[task], before)
            json.dumps(predictions[task], allow_nan=False)


class TrainingTests(unittest.TestCase):
    def test_safe_checkpoint_roundtrip_restores_rng(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(2))
        cache = Path(__file__).resolve().parents[1] / ".cache" / "tests"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(cache.resolve()))
            path = Path(folder)/"checkpoint.pt"
            save_checkpoint(path, {"rng": state, "epoch": 1})
            restored = checkpoint_load(path)
        restore_rng(restored["rng"])
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(2), expected[2])

    def test_warmup_and_cosine_use_optimizer_updates(self):
        values = [learning_rate(i, 10, 3, .001, .00001) for i in range(10)]
        self.assertAlmostEqual(values[0], .0001)
        self.assertAlmostEqual(values[2], .001)
        self.assertAlmostEqual(values[-1], .00001)
        self.assertTrue(all(values[i] >= values[i+1] for i in range(3, 9)))

    def test_batchnorm_stays_frozen_after_train(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.Conv2d(2, 1, 1))
        model.train()
        freeze_bn(model)
        self.assertFalse(model[0].training)
        self.assertFalse(model[0].weight.requires_grad)
        self.assertTrue(model[1].weight.requires_grad)

    def test_partial_accumulation_matches_full_sample_mean(self):
        class ToyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.3))
            def forward(self, images, targets):
                return {"loss": torch.stack([(x*self.weight-1).square().mean() for x in images]).mean()}
        dataset = [(torch.tensor([float(i)]), {}, {"id": str(i), "positive": i%2 == 1}) for i in (1, 2, 3)]
        actual = ToyDetector()
        reference = copy.deepcopy(actual)
        optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        loss = reference([r[0] for r in dataset], [])['loss']
        loss.backward()
        ref_optimizer.step()
        cfg = {"train": {"amp": False, "accumulate": 8, "clip_grad_norm": 1000.,
                         "warmup_updates": 0, "lr": .01, "min_lr": .01}}
        result, updates = train_epoch(actual, DataLoader(dataset, batch_size=2, collate_fn=collate), optimizer,
                                      torch.amp.GradScaler("cuda", enabled=False), cfg, torch.device("cpu"), 0, 1)
        self.assertEqual(updates, 1)
        self.assertEqual(result["positive_exposures"]+result["normal_exposures"], 3)
        torch.testing.assert_close(actual.weight, reference.weight)


if __name__ == "__main__":
    unittest.main()
