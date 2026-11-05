import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from .common import EXPECTED_WEIGHT_SHA256, checkpoint_load, sha256


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def build_model(config, initial_weights=None):
    options, evaluation = config["model"], config["evaluation"]
    # Explicit None for both weight arguments guarantees no network/download calls.
    model = maskrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=options["min_size"], max_size=options["max_size"],
        box_score_thresh=evaluation["score_floor"], box_nms_thresh=evaluation["nms_iou"],
        box_detections_per_img=evaluation["max_detections"],
    )
    if initial_weights is not None:
        if sha256(initial_weights) != EXPECTED_WEIGHT_SHA256:
            raise ValueError("Unexpected COCO initialization SHA256")
        model.load_state_dict(checkpoint_load(initial_weights), strict=True)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    # Preserve pretrained mask convolutions and transposed convolution.
    old = model.roi_heads.mask_predictor.mask_fcn_logits
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(old.in_channels, 2, kernel_size=1)
    if options["mask_roi_size"] != 14:
        model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], options["mask_roi_size"], 2)
    # weights=None makes torchvision internally choose five trainable layers;
    # freeze explicitly AFTER loading our local COCO file.
    layers = ["layer4", "layer3", "layer2", "layer1", "conv1"][:options["trainable_backbone_layers"]]
    if options["trainable_backbone_layers"] == 5:
        layers.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad_(any(name.startswith(layer + ".") for layer in layers))
    freeze_bn(model)
    return model


def learning_rate(update, total, warmup, base, minimum, start_factor=0.1):
    import math
    warmup = min(warmup, max(total-1, 0))
    if warmup and update < warmup:
        # Update is zero based. The first update is start_factor * base.
        fraction = update / max(warmup-1, 1)
        return base * (start_factor + (1-start_factor)*fraction)
    fraction = (update-warmup) / max(total-warmup-1, 1)
    fraction = min(1.0, max(0.0, fraction))
    return minimum + .5*(base-minimum)*(1+math.cos(math.pi*fraction))
