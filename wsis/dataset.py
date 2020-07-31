import logging
import copy
import numpy as np
from fvcore.common.file_io import PathManager
import os
import xml.etree.ElementTree as ET
from PIL import Image

import torch.utils.data

from detectron2.utils.comm import get_world_size
from detectron2.data import (
    samplers,
    get_detection_dataset_dicts,
    DatasetCatalog,
    MetadataCatalog
)
from detectron2.structures import BoxMode
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_batch_data_sampler, trivial_batch_collator, worker_init_reset_seed
from detectron2.data.datasets.pascal_voc import CLASS_NAMES
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T


class WSDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Differing from its super class, this mapper deduces and reserves image
        class label from instance annotations, and then conceals the instance
        annotation for the subsequent model training.

        :param dataset_dict: has keys "file_name", "image_id", "height", "height",
            and "annotation", but this mapper is only concerned with the "annotation".
            dataset_dict["annotation"] is a list[dict] whose element has the keys
            "category_id", "bbox", "bbox_mode". Assuming that anno is one element,
            it is like:
                - anno["category_id"]: int range [0, C-1] indicating class
                - anno["bbox"]: list[float] either in the format of [x1, y1, x2, y2]
                    or the other, format indicated by anno["bbox_mode"]
                - anno["bbox_mode"]: one of BoxMode.XYXY_ABS, BoxMode.XYWH_ABS, etc.
            The content of "annotation" would be first transformed into "instances", which
            is an instance of class 'Instance' representing the set of per-instance ground
            truth. Its attributes gt_boxes and gt_classes are concerned in this method.
            The bounding box annotation is concealed by setting gt_boxes None, and image class
            label is given by deduplicating the gt_classes and reserving it in gt_classes,
            thus instance number information is concealed. Keeping a incomplete the Instance
            has a lot convenience.
        :return:
        """
        # use the super class method to wrap image into tensor
        dataset_dict = super().__call__(dataset_dict)

        # conceal the instance number information and bounding box annotation
        gt_classes = dataset_dict["instances"].gt_classes
        dataset_dict["instances"].remove("gt_classes")
        dataset_dict["instances"].remove("gt_boxes")
        dataset_dict["instances"].gt_classes = torch.unique(gt_classes)

        return dataset_dict


def build_detection_multi_test_loader(cfg, dataset_names, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_names[0])]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# TODO: register voc_sds_train, voc_sds_val
def load_voc_seg_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        smsg_file = os.path.join(dirname, "SegmentationClass", fileid + ".png")
        # anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        # read the segmentation annotation
        with PathManager.open(smsg_file, "rb") as f:
            sem_seg_gt = Image.open(f)
            sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        gt_classes = np.unique(sem_seg_gt)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": sem_seg_gt.shape[0],
            "width": sem_seg_gt.shape[1],
        }
        instances = []
        for cls in gt_classes:
            if cls == 0 or cls > len(CLASS_NAMES):
                continue
            # fake ground truth box
            bbox = [0.0, 0.0, float(sem_seg_gt.shape[1] - 1), float(sem_seg_gt.shape[0] - 1)]
            instances.append(
                {"category_id": cls - 1, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc_seg(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_seg_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )


register_pascal_voc_seg('voc_2012_seg_trainaug', './datasets/VOC2012', 'Segmentation/train_aug', 2012)
register_pascal_voc_seg('voc_2012_seg_val', './datasets/VOC2012', 'Segmentation/val', 2012)

