# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling import META_ARCH_REGISTRY, GeneralizedRCNN


__all__ = ["GeneralizedRCNN_T"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_T(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        # add teaching mode, disable teaching mode by default
        self.teaching = False
        self.roi_heads.teaching = False

    def teach(self):
        self.teaching = True
        self.roi_heads.teach()

    def forward(self, batched_inputs):
        if not self.teaching:
            return super().forward(batched_inputs)
        else:
            return self._teaching_forward(batched_inputs)

    def _teaching_forward(self, batched_inputs):
        assert self.training
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        det_targets, msk_targets = self.roi_heads(images, features, proposals, gt_instances)
        return det_targets, msk_targets


