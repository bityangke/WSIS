from detectron2.config import CfgNode as CN


def add_wsis_config(cfg):
    """
    Add config for weakly supervised instance segmentation.
    """
    _C = cfg

    # multiple heads and loss weight
    _C.MODEL.BOX_CLS_ON = False
    _C.MODEL.BOX_REG_ON = False
    _C.MODEL.MIL_LOSS_WEIGHT = 1.0
    _C.MODEL.BOX_CLS_LOSS_WEIGHT = 0.1
    _C.MODEL.BOX_REG_LOSS_WEIGHT = 0.1
    _C.MODEL.MASK_LOSS_WEIGHT = 0.1
    _C.MODEL.TEST_LATENT_PREDICTIONS = False
    # proposals
    _C.MODEL.MAX_PROPOSALS_PER_IMAGE_TRAIN = 500
    # MIL head
    _C.MODEL.ROI_MIL_HEAD = CN()
    _C.MODEL.ROI_MIL_HEAD.NAME = ""
    _C.MODEL.ROI_MIL_HEAD.THRESHOLD = 0.15
    # pseudo labeler
    _C.MODEL.PSEUDO_LABELER = CN()
    _C.MODEL.PSEUDO_LABELER.NAME = ""
    # recurrent structure for multi-task representation learning
    # default: non-recurrent
    _C.MODEL.RECURRENT_STEPS = 1
