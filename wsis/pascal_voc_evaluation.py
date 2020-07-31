# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import six
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
import pycocotools.mask as mask_util

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.pascal_voc_evaluation import voc_eval

import chainercv
from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.evaluations import calc_detection_voc_ap
from chainercv.utils.mask.mask_iou import mask_iou


class PascalVOCSDSEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, cfg, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.eval_mask_on = cfg.MODEL.TEST_LATENT_PREDICTIONS or cfg.MODEL.MASK_ON

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings
        self._inst_seg_pred = {
            "pred_class": [],
            "pred_mask": [],
            "pred_score": []
        }

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()

            # add masks
            if instances.has("pred_masks"):
                self._inst_seg_pred["pred_class"].append(np.asarray(classes))
                self._inst_seg_pred["pred_mask"].append(instances.pred_masks.numpy())
                self._inst_seg_pred["pred_score"].append(np.asarray(scores))

            # append the predicted results to self._predictions for detection evaluation
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        if self.eval_mask_on:
            # evaluate mask
            dataset = VOCInstanceSegmentationDataset(split="val", data_dir="./datasets/VOC2012")
            gt_masks = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
            gt_labels = [dataset.get_example_by_keys(i, (2,))[0] for i in range(len(dataset))]
            mask_aps = {}
            for thresh in range(50, 100, 5):
                ap_dict = chainercv.evaluations.eval_instance_segmentation_voc(
                    self._inst_seg_pred["pred_mask"],
                    self._inst_seg_pred["pred_class"],
                    self._inst_seg_pred["pred_score"],
                    gt_masks,
                    gt_labels,
                    iou_thresh=thresh / 100.0
                )
                mask_aps[thresh] = ap_dict["ap"] * 100

            mask_mAP = {iou: np.mean(x) for iou, x in mask_aps.items()}
            ret["mask"] = {"AP": np.mean(list(mask_mAP.values())), "AP50": mask_mAP[50], "AP75": mask_mAP[75]}

        return ret


def eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted masks obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in `FCIS`_.

    .. _`FCIS`: https://arxiv.org/abs/1611.07709

    Args:
        pred_masks (iterable of numpy.ndarray): See the table below.
        pred_labels (iterable of numpy.ndarray): See the table below.
        pred_scores (iterable of numpy.ndarray): See the table below.
        gt_masks (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`pred_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`pred_scores`, ":math:`[(R,)]`", :obj:`float32`, \
        --
        :obj:`gt_masks`, ":math:`[(R, H, W)]`", :obj:`bool`, --
        :obj:`gt_labels`, ":math:`[(R,)]`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_instance_segmentation_voc_prec_rec(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_instance_segmentation_voc_prec_rec(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, iou_thresh):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted masks obtained from a dataset which has :math:`N` images.
    The code is based on the evaluation code used in `FCIS`_.

    .. _`FCIS`: https://arxiv.org/abs/1611.07709

    Args:
        pred_masks (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of masks. Its index corresponds to an index for the base
            dataset. Each element of :obj:`pred_masks` is an object mask
            and is an array whose shape is :math:`(R, H, W)`,
            where :math:`R` corresponds
            to the number of masks, which may vary among images.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_masks`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted masks. Similar to :obj:`pred_masks`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_masks (iterable of numpy.ndarray): An iterable of ground truth
            masks whose length is :math:`N`. An element of :obj:`gt_masks` is
            an object mask whose shape is :math:`(R, H, W)`. Note that the
            number of masks :math:`R` in each image does not need to be
            same as the number of corresponding predicted masks.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_masks`. Its
            length is :math:`N`.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_mask, pred_label, pred_score, gt_mask, gt_label in \
            six.moves.zip(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels):

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_keep_l = pred_label == l
            pred_mask_l = pred_mask[pred_keep_l]
            pred_score_l = pred_score[pred_keep_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_mask_l = pred_mask_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_mask_l = gt_mask[gt_keep_l]

            n_pos[l] += gt_keep_l.sum()
            score[l].extend(pred_score_l)

            if len(pred_mask_l) == 0:
                continue
            if len(gt_mask_l) == 0:
                match[l].extend((0,) * pred_mask_l.shape[0])
                continue

            iou = mask_iou(pred_mask_l, gt_mask_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_mask_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (pred_masks, pred_labels, pred_scores, gt_masks, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(
            match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

