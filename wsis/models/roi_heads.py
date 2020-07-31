import torch
from torch import nn
from typing import Dict
from copy import deepcopy
import torch.nn.functional as F

import numpy as np

from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads, build_mask_head
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import pairwise_iou, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.matcher import Matcher

from .mil_heads import build_mil_head, multi_class_cross_entropy_loss
from .pseudo_labelers import build_pseudo_labeler
from .fast_rcnn import FastRCNNOutputs, FastRCNNOutputLayers


__all__ = ["Res5WSROIHeads", "Res5WSROIHeads_T"]


class WSROIHeads(ROIHeads):
    """
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

        # Matcher to assign box proposals to gt boxes
        # self.proposal_matcher = Matcher(
        #     cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
        #     cfg.MODEL.ROI_HEADS.IOU_LABELS,
        #     allow_low_quality_matches=True,
        # )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        # -- original faster rcnn:
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        # --- our weakly supervised learning:
        # however, the pseudo ground-truth boxes are selected from proposals :(
        # this step is skipped
        # if self.proposal_append_gt:
        #     proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_all = []
        selection_masks = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # sample proposals
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            device = proposals_per_image.proposal_boxes.device
            selection_mask = torch.zeros(len(proposals_per_image), device=device, dtype=torch.bool)
            selection_mask[sampled_idxs] = 1

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_all.append(proposals_per_image)
            selection_masks.append(selection_mask)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_all, selection_masks


@ROI_HEADS_REGISTRY.register()
class Res5WSROIHeads(WSROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION            # default: 14
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE                  # default: ROIAlignV2
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], ) #
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO        # default: 0
        self.box_cls_on   = cfg.MODEL.BOX_CLS_ON
        self.box_reg_on   = cfg.MODEL.BOX_REG_ON
        self.mask_on      = cfg.MODEL.MASK_ON
        # self.max_proposals_per_image    = cfg.MODEL.MAX_PROPOSALS_PER_IMAGE_TRAIN
        # loss weight
        self.loss_mil_weight            = cfg.MODEL.MIL_LOSS_WEIGHT
        self.loss_box_cls_weight        = cfg.MODEL.BOX_CLS_LOSS_WEIGHT
        self.loss_box_reg_weight        = cfg.MODEL.BOX_REG_LOSS_WEIGHT
        self.loss_mask_weight           = cfg.MODEL.MASK_LOSS_WEIGHT
        # recurrent steps
        self.recurrent_steps            = cfg.MODEL.RECURRENT_STEPS
        #
        self.test_latent_predictions    = cfg.MODEL.TEST_LATENT_PREDICTIONS
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        # TODO: separate pooler for cls/loc&det/mask
        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.recurrent_on = self.recurrent_steps > 1
        if self.recurrent_on:
            assert self.box_cls_on
            self.num_tasks = 2 + self.mask_on
            out_channels += self.num_tasks * self.num_classes

        # three branches for image classification, object detection, instance segmentation respectively
        self.im_predictor = build_mil_head(
            cfg,
            ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
        )

        if self.box_cls_on:
            self.box_predictor = FastRCNNOutputLayers(
                out_channels, self.num_classes, self.box_reg_on, self.cls_agnostic_bbox_reg
            )  # this include a bbox cls branch and a bbox reg branch

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        # build pseudo labeler
        self.pseudo_labeler = build_pseudo_labeler(cfg)

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.

        proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".

        targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                Since we are weakly supervised learning, it only have the following fields:
                - gt_classes: the label for the whole image with categories ranging in [0, #class).
        """
        del images

        # pre-process targets
        if targets is not None:
            if hasattr(targets[0], "msk_gt_boxes"):
                msk_targets = [
                    Instances(
                        x.image_size,
                        gt_classes=x.gt_classes,
                        gt_boxes=x.msk_gt_boxes,
                        gt_masks=x.msk_gt_masks
                    ) for x in targets
                ]
                for x in targets:
                    x.remove("msk_gt_boxes")
                    x.remove("msk_gt_masks")
            else:
                msk_targets = None
            if hasattr(targets[0], "det_gt_boxes"):
                det_targets = [
                    Instances(
                        x.image_size,
                        gt_classes=x.gt_classes,
                        gt_boxes=x.det_gt_boxes
                    ) for x in targets
                ]
                for x in targets:
                    x.remove("det_gt_boxes")
            else:
                det_targets = None

        # if self.training:
        #     # we have to do this because of the limitation of computation
        #     proposals = self.pre_sample_proposals(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        bag_scores = None
        pred_instances = None
        for k in range(self.recurrent_steps):
            # embedding for recurrent structure
            if self.recurrent_on:
                # embed bag_scores, and pred_instances
                # TODO: only embed instance segmentation
                embeded_features = self.embed_predictions(
                    bag_scores, pred_instances, features[self.in_features[0]].shape[2:]
                )
                extra_features = self._shared_roi_transform(
                    [embeded_features], proposal_boxes
                )
                extended_features = torch.cat((box_features, extra_features), dim=1)
                del extra_features
            else:
                # non-recurrent
                embeded_features = None
                extended_features = box_features
            feature_pooled = extended_features.mean(dim=[2, 3])  # pooled to 1x1

            # image classification task
            inst_scores, bag_scores, pred_instances, losses = self.forward_im_cls(
                feature_pooled, proposals, targets, step=k
            )

            # object detection task
            if self.box_cls_on:
                # generate pseudo label for detection task
                if self.training:
                    targets = self.pseudo_labeler.update_box_targets(proposal_boxes, inst_scores, targets) \
                        if det_targets is None else det_targets
                    del inst_scores, bag_scores

                if self.training or not self.test_latent_predictions:
                    # forward detection branch
                    det_outputs, pred_instances, losses = self.forward_detection(
                        feature_pooled, proposals, targets, losses, step=k
                    )
            del feature_pooled

            # instance segmentation task
            if self.mask_on:
                assert self.box_cls_on
                # update pseudo label for mask prediction
                if self.training:
                    targets = self._update_mask_pseudo_label(features, embeded_features, det_outputs, det_targets) \
                        if msk_targets is None else msk_targets

                # forward mask branch
                pred_instances, losses = self.forward_instseg(
                    extended_features, proposals, targets, losses, pred_instances, step=k
                )
            elif (not self.training) and self.test_latent_predictions:
                # forward mask branch
                pred_instances, losses = self.forward_instseg(
                    features, proposals, targets, losses, pred_instances, step=k
                )
            del extended_features

        if self.training:
            # average over multiple steps
            losses = {k: 1.0 / self.recurrent_steps * v for k, v in losses.items()}
            return [], losses
        else:
            return pred_instances, {}

    def embed_predictions(self, im_cls_scores, instances, spatial_shape):
        H, W = spatial_shape
        B = im_cls_scores.size(0)

        # step 0, where 0 tensor is initialized
        if instances is None:
            num_tasks = 1 + self.box_cls_on + self.mask_on
            return torch.zeros((B, num_tasks * self.num_classes, H, W)).type_as(im_cls_scores)

        embeded_features = []
        # embed image classification
        embeded_im_cls = im_cls_scores[:, :, None, None].expand(-1, -1, H, W).contiguous().clone()
        embeded_features.append(embeded_im_cls)

        # embed detection
        spatial_ratio = 1.0 / self.feature_strides[self.in_features[0]]
        embeded_bbox = torch.zeros((B, self.num_classes, H, W)).type_as(embeded_im_cls)
        for idx, instances_per_image in enumerate(instances):
            boxes = spatial_ratio * instances_per_image.pred_boxes.tensor
            classes = instances_per_image.pred_classes
            scores = instances_per_image.pred_scores
            for box, cls, score in zip(boxes, classes, scores):
                xmin, ymin, xmax, ymax = box.long()
                embeded_bbox[idx, cls, ymin: ymax, xmin: xmax] = torch.max(
                    embeded_bbox[idx, cls, ymin: ymax, xmin: xmax], score
                )
        embeded_features.append(embeded_bbox)

        # embed mask
        if self.mask_on:
            embeded_mask = torch.zeros((B, self.num_classes, H, W)).type_as(embeded_im_cls)
            for idx, instances_per_image in enumerate(instances):
                boxes = spatial_ratio * instances_per_image.pred_boxes.tensor
                classes = instances_per_image.pred_classes
                masks = instances_per_image.pred_masks
                for box, cls, mask in zip(boxes, classes, masks):
                    xmin, ymin, xmax, ymax = box.long()
                    resized_mask = F.interpolate(
                        mask[None], size=(ymax - ymin, xmax - xmin), mode="bilinear"
                    )
                    embeded_mask[idx, cls, ymin: ymax, xmin: xmax] = torch.max(
                        embeded_mask[idx, cls, ymin: ymax, xmin: xmax], resized_mask[0, 0]
                    )
            embeded_features.append(embeded_mask)
        return torch.cat(embeded_features, dim=1)

    def forward_im_cls(self, features, proposals, targets, step=0):
        # image classification by mil head
        inst_scores, bag_scores = self.im_predictor(features, [len(p) for p in proposals])

        if self.training:
            # TODO: log the classification mAP
            losses = {}
            losses[f"loss_mil_iter{step}"] = self.loss_mil_weight * multi_class_cross_entropy_loss(bag_scores, targets)
            pred_instances = None
        else:
            losses = None
            if self.test_latent_predictions:
                # append an artificial background dimension
                inst_scores = [
                    torch.cat([x, torch.zeros((x.size(0), 1), device=x.device)], dim=1)
                    for x in inst_scores
                ]
                boxes = [x.proposal_boxes.tensor for x in proposals]
                image_shapes = [x.image_size for x in proposals]

                # normal inference
                pred_instances, _ = fast_rcnn_inference(
                    boxes, inst_scores, image_shapes, self.test_score_thresh,
                    self.test_nms_thresh, self.test_detections_per_img
                )

                # if non-prediction
                for pred_inst_per_image, boxes_per_image, inst_scores_per_image in zip(
                        pred_instances, boxes, inst_scores
                ):
                    if len(pred_inst_per_image) == 0:
                        # pick max scored boxes
                        max_score, max_i = torch.max(inst_scores_per_image[:, :-1].flatten(), dim=0)
                        box_idx = max_i // self.num_classes
                        cls_idx = max_i % self.num_classes
                        pred_inst_per_image.remove("pred_boxes")
                        pred_inst_per_image.remove("pred_classes")
                        pred_inst_per_image.remove("scores")
                        pred_inst_per_image.pred_boxes = Boxes(boxes_per_image[box_idx, None])
                        pred_inst_per_image.pred_classes = cls_idx.view(1)
                        pred_inst_per_image.scores = max_score.view(1)
            else:
                pred_instances = None

        return inst_scores, bag_scores, pred_instances, losses

    def forward_detection(self, features, proposals, targets, losses, step=0):
        if self.training:
            # object detection
            det_tr_proposals, det_tr_selection_masks = self.label_and_sample_proposals(proposals, targets)
            # del targets
            pred_class_logits, pred_proposal_deltas = self.box_predictor(features)
            losses.update(
                {
                    k + f"_iter{step}": v
                    for k, v in FastRCNNOutputs(
                        self.box2box_transform,
                        pred_class_logits[torch.cat(det_tr_selection_masks, dim=0)],
                        self.box_reg_on,
                        pred_proposal_deltas[torch.cat(det_tr_selection_masks, dim=0)],
                        det_tr_proposals,
                        self.smooth_l1_beta,
                    ).losses().items()
                }
            )
            losses[f"loss_box_cls_iter{step}"] = self.loss_box_cls_weight * losses[f"loss_box_cls_iter{step}"]
            if f"loss_box_reg_iter{step}" in losses:
                losses[f"loss_box_reg_iter{step}"] = self.loss_box_reg_weight * losses[f"loss_box_reg_iter{step}"]

            # make predictions for recurrent embedding
            if self.mask_on or self.recurrent_on:
                det_outputs = FastRCNNOutputs(
                    self.box2box_transform,
                    pred_class_logits,
                    self.box_reg_on,
                    pred_proposal_deltas,
                    proposals,
                    self.smooth_l1_beta,
                )
                pred_instances, _ = det_outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
                )
            else:
                det_outputs = None
                pred_instances = None
        else:
            det_outputs = None
            if not self.test_latent_predictions:
                # object detection
                pred_class_logits, pred_proposal_deltas = self.box_predictor(features)
                outputs = FastRCNNOutputs(
                    self.box2box_transform,
                    pred_class_logits,
                    self.box_reg_on,
                    pred_proposal_deltas,
                    proposals,
                    self.smooth_l1_beta,
                )
                pred_instances, _ = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
                )
        return det_outputs, pred_instances, losses

    def _update_mask_pseudo_label(self, features, embeded_features, det_outputs, targets):
        det_scores = [x[:, :-1] for x in det_outputs.predict_probs()]  # exclude background
        det_boxes = [Boxes(x) for x in det_outputs.predict_boxes()]
        targets = self.pseudo_labeler.update_box_targets(det_boxes, det_scores, targets)

        # produce pseudo foreground mask by cam
        with torch.no_grad():
            target_boxes = [x.gt_boxes for x in targets]
            cam_features = self._shared_roi_transform(
                [features[f] for f in self.in_features], target_boxes
            )
            if self.recurrent_on:
                # extend cam_features
                cam_extra_features = self._shared_roi_transform(
                    [embeded_features], target_boxes
                )
                cam_features = torch.cat((cam_features, cam_extra_features), dim=1)
            binary_cam = self.im_predictor.forward_for_cam(cam_features, targets)
            targets = self.pseudo_labeler.update_mask_targets(binary_cam, targets)
        return targets

    def forward_instseg(self, features, proposals, targets, losses, pred_instances, step=0):
        if self.training:
            # re-assign targets
            msk_tr_proposals, msk_tr_selection_masks = self.label_and_sample_proposals(proposals, targets)
            mask_features = features[torch.cat(msk_tr_selection_masks, dim=0)]
            msk_tr_proposals, fg_selection_masks = select_foreground_proposals(
                msk_tr_proposals, self.num_classes
            )
            # Since the ROI feature transform is shared between boxes and masks,
            # we don't need to recompute features. The mask loss is only defined
            # on foreground proposals, so we need to select out the foreground
            # features.
            mask_features = mask_features[torch.cat(fg_selection_masks, dim=0)]
            mask_logits = self.mask_head(mask_features)
            losses[f"loss_mask_iter{step}"] = mask_rcnn_loss(mask_logits, msk_tr_proposals)

            # make predictions for recurrent embedding
            if self.recurrent_on:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
        else:
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
        return pred_instances, losses

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        # assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on and not self.test_latent_predictions:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            mask_logits = self.mask_head(x)
            mask_rcnn_inference(mask_logits, instances)
        else:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            cam_soft = self.im_predictor.forward_for_cam(x, instances)
            for cam_per_image, instances_per_image in zip(cam_soft, instances):
                instances_per_image.pred_masks = cam_per_image[:, None]  # (1, Hmask, Wmask)

        return instances

    def pre_sample_proposals(self, proposals):
        """

        proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
        """
        # random sample. Note that these proposals are topk proposals w.r.t. objectness
        # which are selected in DatasetMapper
        sampled_proposals = []
        for proposals_per_image in proposals:
            num_prop_this_image = len(proposals_per_image)
            max_prop_this_image = min(num_prop_this_image, self.max_proposals_per_image)
            device = proposals_per_image.proposal_boxes.device
            sampled_idx = torch.randperm(num_prop_this_image, device=device)[:max_prop_this_image]
            sampled_proposals.append(proposals_per_image[sampled_idx])
        return sampled_proposals


@ROI_HEADS_REGISTRY.register()
class Res5WSROIHeads_T(Res5WSROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.teaching = False

    def teach(self):
        self.teaching = True

    def forward(self, images, features, proposals, targets=None):
        if not self.teaching:
            return super().forward(images, features, proposals, targets)
        else:
            assert self.training
            return self._teaching_forward(images, features, proposals, targets)

    def _teaching_forward(self, images, features, proposals, targets):
        del images

        # # we have to do this because of the limitation of computation
        # proposals = self.pre_sample_proposals(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        bag_scores = None
        pred_instances = None
        # TODO: support recurrent
        for k in range(self.recurrent_steps):
            # embedding for recurrent structure
            if self.recurrent_on:
                # embed bag_scores, and pred_instances
                # TODO: only embed instance segmentation
                embeded_features = self.embed_predictions(
                    bag_scores, pred_instances, features[self.in_features[0]].shape[2:]
                )
                extra_features = self._shared_roi_transform(
                    [embeded_features], proposal_boxes
                )
                extended_features = torch.cat((box_features, extra_features), dim=1)
                del extra_features
            else:
                # non-recurrent
                embeded_features = None
                extended_features = box_features
            feature_pooled = extended_features.mean(dim=[2, 3])  # pooled to 1x1

            # image classification task
            inst_scores, bag_scores = self.im_predictor(feature_pooled, [len(p) for p in proposals])

            # object detection task
            if self.box_cls_on:
                # generate pseudo label for detection task
                det_targets = self.pseudo_labeler.update_box_targets(proposal_boxes, inst_scores, deepcopy(targets))
                del inst_scores, bag_scores

                # forward detection branch
                det_outputs, pred_instances, _ = self.forward_detection(
                    feature_pooled, proposals, det_targets, {}, step=k
                )
            else:
                det_targets = [None] * len(targets)
            del feature_pooled

            # instance segmentation task
            if self.mask_on:
                assert self.box_cls_on
                # update pseudo label for mask prediction
                msk_targets = self._update_mask_pseudo_label(features, embeded_features, det_outputs, deepcopy(det_targets))
            else:
                msk_targets = [None] * len(targets)
            del extended_features

        return det_targets, msk_targets



