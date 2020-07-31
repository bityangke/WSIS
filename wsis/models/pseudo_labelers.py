import torch
import logging
import numpy as np
import torch.nn.functional as F

from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.modeling.matcher import Matcher
from detectron2.utils.events import get_event_storage

PSEUDO_LABELER_REGISTRY = Registry("PSEUDO_LABELER")
PSEUDO_LABELER_REGISTRY.__doc__ = """
Registry for pseudo_labeler, which takes as input the network prediction 
and generates target for networks.
"""

#
# ==== pseudo labelers for weakly labeled data  ===========
#
@PSEUDO_LABELER_REGISTRY.register()
class MaxLabeler:
    def __init__(self, cfg):
        self.cfg = cfg

    def update_box_targets(self, boxes, prop_scores, targets):
        # TODO: support accumulated
        for boxes_per_image, prop_scores_per_image, targets_per_image in zip(boxes, prop_scores, targets):
            scores = prop_scores_per_image.clone()
            gt_boxes = []
            for cls in targets_per_image.gt_classes:
                max_score, max_idx = torch.max(scores[:, cls], dim=0)
                gt_boxes.append(boxes_per_image[None, max_idx])
                scores[max_idx, :] = -float('inf')  # not allow same bbox for different gt classes
            targets_per_image.gt_boxes = Boxes.cat(gt_boxes)
            # targets_per_image.gt_boxes.clip(targets_per_image.image_size)
        return targets

    def update_mask_targets(self, soft_cams, targets):
        for cams_per_image, targets_per_image in zip(soft_cams, targets):
            device = cams_per_image.device
            num_trg = len(targets_per_image)
            H, W = targets_per_image.image_size
            bitcam_per_image = torch.zeros((num_trg, H, W), dtype=torch.bool, device=device)
            for i, box in enumerate(targets_per_image.gt_boxes):
                rx1, ry1, rx2, ry2 = box.long()
                cam = cams_per_image[i]
                cam = F.interpolate(cam[None, None], (ry2 - ry1, rx2 - rx1), mode='bilinear')
                bitcam_per_image[i, ry1: ry2, rx1: rx2] = (cam > 0.5)  # cam is linearly transformed to make 0.5 as threshold
            targets_per_image.gt_masks = BitMasks(bitcam_per_image)
        return targets


# TODO: implement PCL labeler


#
# ==== building  ===========
#
def build_pseudo_labeler(cfg):
    pseudo_labeler = cfg.MODEL.PSEUDO_LABELER.NAME
    return PSEUDO_LABELER_REGISTRY.get(pseudo_labeler)(cfg)
