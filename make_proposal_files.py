import pickle
import numpy as np
import os.path as osp
from tqdm import tqdm

# make following files
# - voc_2007_trainval_slc_srch.pkl
# - voc_2007_test_slc_srch.pkl
# - voc_2012_trainval_slc_srch.pkl
# - voc_2012_test_slc_srch.pkl
# - voc_2012_trainaug_slc_srch.pkl
# - voc_2012_val_slc_srch.pkl


def make_slc_srch_proposal_file(year, imgset_file, file_name):
    # load the image set
    img_id_list = [x for x in np.loadtxt(imgset_file, dtype=str)]

    # load the boxes and objectness_logits (set as random number)
    proposals_dir = osp.join(f"./datasets/VOC{year}/SelectiveSearch/")
    boxes = []
    logits = []
    for index in tqdm(img_id_list):
        box = np.loadtxt(osp.join(proposals_dir, index + ".txt"), dtype=np.float)
        prob = np.clip(np.random.rand(box.shape[0]), 1e-8, 1.0)
        logit = np.log(prob) - np.log(1 - prob)
        boxes.append(box)
        logits.append(logit)

    proposals = {
        "indexes": img_id_list,
        "boxes": boxes,
        "scores": logits
    }

    #
    with open(file_name, "wb") as f:
        pickle.dump(proposals, f)


if __name__ == '__main__':
    make_slc_srch_proposal_file("2012", "./datasets/VOC2012/ImageSets/Segmentation/val.txt",
                                "./datasets/voc_2012_val_slc_srch.pkl")
    make_slc_srch_proposal_file("2012", "./datasets/VOC2012/ImageSets/Segmentation/train_aug.txt",
                                "./datasets/voc_2012_trainaug_slc_srch.pkl")
    make_slc_srch_proposal_file("2012", "./datasets/VOC2012/ImageSets/Main/trainval.txt",
                                "./datasets/voc_2012_trainval_slc_srch.pkl")
    make_slc_srch_proposal_file("2012", "./datasets/VOC2012/ImageSets/Main/test.txt",
                                "./datasets/voc_2012_test_slc_srch.pkl")
