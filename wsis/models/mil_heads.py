import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_MIL_HEAD_REGISTRY = Registry("ROI_MIL_HEAD")
ROI_MIL_HEAD_REGISTRY.__doc__ = """
Registry for mil heads, which predicts image class scores given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def multi_class_cross_entropy_loss(pred_bag_scores, instances):
    pred_bag_scores = torch.clamp(pred_bag_scores, min=1e-8, max=1.0-1e-8)
    # pred_bag_scores = torch.clamp(pred_bag_scores, min=0.0, max=1.0)

    # prepare label
    labels = torch.zeros_like(pred_bag_scores)
    for i, instance_per_image in enumerate(instances):
        labels[i, instance_per_image.gt_classes] = 1.0

    # TODO: consider focal loss or something
    mil_loss = F.binary_cross_entropy(
        pred_bag_scores.flatten(), labels.flatten(), reduction="mean"
    )
    # mil_loss /= pred_bag_scores.size(0)
    return mil_loss


@ROI_MIL_HEAD_REGISTRY.register()
class WSDNHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        # fmt: off
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # conv_dims = cfg.MODEL.ROI_MIL_HEAD.CONV_DIM
        # self.norm = cfg.MODEL.ROI_MIL_HEAD.NORM
        # num_conv = cfg.MODEL.ROI_MIL_HEAD.NUM_CONV
        input_channels = input_shape.channels
        self.cam_threshold = cfg.MODEL.ROI_MIL_HEAD.THRESHOLD
        # fmt: on

        # TODO: consider more layers for each stream
        # TODO: add a norm layer for CAM
        # TODO: add relation module to loc_stream
        self.cls_predictor = nn.Linear(input_channels, num_classes)
        self.loc_predictor = nn.Linear(input_channels, num_classes)

        # use normal distribution initialization for cls and loc prediction layer
        nn.init.normal_(self.cls_predictor.weight, std=0.01)
        nn.init.constant_(self.cls_predictor.bias, 0)
        nn.init.normal_(self.loc_predictor.weight, std=0.01)
        nn.init.constant_(self.loc_predictor.bias, 0)

    def forward(self, x, num_insts_per_bag):
        cls = self.cls_predictor(x)
        cls = [F.softmax(cls_i, dim=1) for cls_i in cls.split(num_insts_per_bag, dim=0)]
        loc = self.loc_predictor(x)
        loc = [F.softmax(loc_i, dim=0) for loc_i in loc.split(num_insts_per_bag, dim=0)]
        inst_scores = [cls_i * loc_i for cls_i, loc_i in zip(cls, loc)]
        bag_scores = torch.stack([inst_scores_i.sum(dim=0) for inst_scores_i in inst_scores])
        return inst_scores, bag_scores

    @torch.no_grad()
    def forward_for_cam(self, x, instances):
        cls_actv_logits = F.conv2d(
            input=x,
            weight=self.cls_predictor.weight[:, :, None, None],
            bias=self.cls_predictor.bias
        )  # Tensor(N, C, H, W)

        soft_cams = []
        num_inst_per_image = [len(instance_per_image) for instance_per_image in instances]
        for cls_actv_logits_per_image, instance_per_image in zip(
            cls_actv_logits.split(num_inst_per_image, dim=0), instances
        ):
            if len(instance_per_image) == 0:
                soft_cams.append([])
                continue
            device = cls_actv_logits_per_image.device
            classes = instance_per_image.gt_classes if instance_per_image.has("gt_classes") \
                else instance_per_image.pred_classes
            num_inst = len(instance_per_image)
            assert num_inst == cls_actv_logits_per_image.size(0)
            cal = cls_actv_logits[torch.arange(num_inst, device=device), classes, :, :]
            cal = cal[None]
            # normalize using PSA[Jiwoon et al 2018]
            cam = F.relu(cal) / F.adaptive_max_pool2d(cal, (1, 1)).clamp(min=1e-8)
            # linear transform the cam values from [0, 1] to [0, 0.5/cam_threshold] to
            # coincide with binarization threshold 0.5
            cam = cam * 0.5 / self.cam_threshold
            soft_cams.append(cam[0])
        return soft_cams


# TODO: implement CMIL head


def build_mil_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MIL_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MIL_HEAD.NAME
    return ROI_MIL_HEAD_REGISTRY.get(name)(cfg, input_shape)
