import os
import logging
import time
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.engine import DefaultTrainer
from detectron2.utils.events import JSONWriter, TensorboardXWriter
from detectron2.evaluation import (
    PascalVOCDetectionEvaluator,
)
from detectron2.utils import comm
from detectron2.data import build_detection_train_loader

from .dataset import WSDatasetMapper
from .pascal_voc_evaluation import PascalVOCSDSEvaluator


__all__ = ["WSTrainer", "WSPLTrainer"]


class WSTrainer(DefaultTrainer):
    """
    Considering the specificity of weakly-supervised settings, following method
    should be overridden:
    - build_train_loader: a specific dataset mapper that conceals the per-instance
        annotation is needed.
    - build_
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    # def build_writers(self):
    #     """
    #     Please reference super method. Here, only CommonMetricPrinter is replaced
    #     with SemiSupMetricPrinter
    #     """
    #     # Assume the default print/log frequency.
    #     return [
    #         # It may not always print what you want to see, since it prints "common" metrics only.
    #         WSSupMetricPrinter(self.max_iter),
    #         JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
    #         TensorboardXWriter(self.cfg.OUTPUT_DIR),
    #     ]

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Two data loaders would be built, one for labeled data and the
        other for unlabeled data.
        - The weakly labeled data loader would be built by a custom method.
          A particular DatasetMapper is applied to conceal the bounding box
          annotations while reserves image class label annotations.

        Then a semi- data loader is used to integrate this two loaders
        """
        return build_detection_train_loader(cfg, mapper=WSDatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCSDSEvaluator(cfg, dataset_name)

    def only_load_model(self):
        checkpoint = self.checkpointer._load_file(self.cfg.MODEL.WEIGHTS)
        self.checkpointer._load_model(checkpoint)
        logger = logging.getLogger(__name__)
        logger.info("Loaded model from {}".format(self.cfg.MODEL.WEIGHTS))


class WSPLTrainer(DefaultTrainer):
    """
    Considering the specificity of weakly-supervised settings, following method
    should be overridden:
    - build_train_loader: a specific dataset mapper that conceals the per-instance
        annotation is needed.
    - build_
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # max proposals per image
        self.max_proposals_per_image    = cfg.MODEL.MAX_PROPOSALS_PER_IMAGE_TRAIN

        # build teacher network
        self.teacher = self.build_model(cfg)
        self.teacher.teach()
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            self.teacher = DistributedDataParallel(self.teacher, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
        self.teacher.train()
        for param_stu, param_tch in zip(self.model.parameters(), self.teacher.parameters()):
            param_tch.data.copy_(param_stu.data)  # initialize
            param_tch.requires_grad = False  # not update by gradient

    # def build_writers(self):
    #     """
    #     Please reference super method. Here, only CommonMetricPrinter is replaced
    #     with SemiSupMetricPrinter
    #     """
    #     # Assume the default print/log frequency.
    #     return [
    #         # It may not always print what you want to see, since it prints "common" metrics only.
    #         WSSupMetricPrinter(self.max_iter),
    #         JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
    #         TensorboardXWriter(self.cfg.OUTPUT_DIR),
    #     ]

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Two data loaders would be built, one for labeled data and the
        other for unlabeled data.
        - The weakly labeled data loader would be built by a custom method.
          A particular DatasetMapper is applied to conceal the bounding box
          annotations while reserves image class label annotations.

        Then a semi- data loader is used to integrate this two loaders
        """
        return build_detection_train_loader(cfg, mapper=WSDatasetMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCSDSEvaluator(cfg, dataset_name)

    def only_load_model(self):
        checkpoint = self.checkpointer._load_file(self.cfg.MODEL.WEIGHTS)
        self.checkpointer._load_model(checkpoint)
        logger = logging.getLogger(__name__)
        logger.info("Loaded model from {}".format(self.cfg.MODEL.WEIGHTS))

    def pre_sample_proposals(self, batched_inputs):
        """

        proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
        """
        # random sample. Note that these proposals are topk proposals w.r.t. objectness
        # which are selected in DatasetMapper
        for inputs_per_image in batched_inputs:
            proposals_per_image = inputs_per_image["proposals"]
            num_prop_this_image = len(proposals_per_image)
            max_prop_this_image = min(num_prop_this_image, self.max_proposals_per_image)
            device = proposals_per_image.proposal_boxes.device
            sampled_idx = torch.randperm(num_prop_this_image, device=device)[:max_prop_this_image]
            inputs_per_image["proposals"] = proposals_per_image[sampled_idx]

    def update_teacher(self):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.iter + 1), 0.99)  # TODO: configure this momentum
        for param_stu, param_tch in zip(self.model.parameters(), self.teacher.parameters()):
            param_tch.data = param_tch.data * alpha + param_stu.data * (1. - alpha)

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        """load data"""
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # presample proposals here, making the proposal pool for teacher and student consistent
        self.pre_sample_proposals(data)

        """get pseudo label, and update the targets in data"""
        self.update_teacher()
        with torch.no_grad():
            det_targets, msk_targets = self.teacher(data)
            for input_per_im, det_target_per_im, msk_target_per_im in zip(data, det_targets, msk_targets):
                if det_target_per_im is not None:
                    input_per_im["instances"].det_gt_boxes = det_target_per_im.gt_boxes
                if msk_target_per_im is not None:
                    input_per_im["instances"].msk_gt_boxes = msk_target_per_im.gt_boxes
                    input_per_im["instances"].msk_gt_masks = msk_target_per_im.gt_masks
        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()


