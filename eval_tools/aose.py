import os
import cv2
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import argparse

import sys
sys.path.append('../')
from voc_eval_offical import voc_evaluate

def set_up_parse():
    parser = argparse.ArgumentParser(
        description='Evaluatation.')
    parser.add_argument('predict_json', help='path to json file of predictions')
    parser.add_argument('target_json', help='path to json file of target')
    parser.add_argument('known_select_thre', type=float, default=0, help='select threshold of known classes for mAP')
    args = parser.parse_args()
    return args



class Draw:
    def __init__(self):
        self.classes = range(1, 7)
        self.args = set_up_parse()
        # load groundtruth
        self.gt_coda_api = COCO(self.args.target_json)
        self.res_coda_api = self.gt_coda_api.loadRes(self.args.predict_json)
        imgidlist = list(self.gt_coda_api.imgs.keys())
        self.imgidlist = []
        for imgID in imgidlist:
            gt_list_this_img = self.gt_coda_api.loadAnns(self.gt_coda_api.getAnnIds(imgIds=imgID))
            if len(gt_list_this_img) == 0:
                continue
            self.imgidlist.append(imgID)
        print("img num: {}\n".format(len(self.imgidlist)))


    def readdata(self, gt_coco_api, res_coco_api, select_thre=0.):
        self.res = {7: {}}
        self.OOD_gt = {}
        for imgID in self.imgidlist:
            res_list_this_img = res_coco_api.loadAnns(res_coco_api.getAnnIds(imgIds=[imgID]))
            res_list_this_img = [res for res in res_list_this_img if res["category_id"]!=7] # known predictions
            if len(res_list_this_img)==0:
                img_res = np.array([])
            else:
                # xyhw
                img_res = np.array([res['bbox'] + [res[self.sort_scores_name]]  for res in res_list_this_img])
                img_res = img_res[img_res[:, -1] > select_thre]
                img_res[:, 2] = img_res[:, 0] + img_res[:, 2]
                img_res[:, 3] = img_res[:, 1] + img_res[:, 3]
            self.res[7].update({imgID: img_res})

            gt_list_this_img = gt_coco_api.loadAnns(gt_coco_api.getAnnIds(imgIds=[imgID]))
            gt_list_this_img = [gt for gt in gt_list_this_img if gt["category_id"]==7] # unknown groundtruths
            if len(gt_list_this_img)==0:
                img_gt = np.array([])
            else:
                img_gt = np.array([gti['bbox'] + [7] for gti in gt_list_this_img])
                img_gt[:, 2] = img_gt[:, 0] + img_gt[:, 2]
                img_gt[:, 3] = img_gt[:, 1] + img_gt[:, 3]
            self.OOD_gt.update({imgID: img_gt})

    def run(self):
        self.sort_scores_name = "score" 

        self.readdata(self.gt_coda_api, self.res_coda_api, self.args.known_select_thre)
        recall, precision, ap, rec, prec, state, det_image_files = voc_evaluate(self.res, self.OOD_gt, 7)
        print("AOSE = {}".format(state[0].sum()))
        


eva = Draw()
eva.run()