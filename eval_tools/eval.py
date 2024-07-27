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
    parser.add_argument('dataset', default="soda", help='the type of the training dataset')
    args = parser.parse_args()
    return args



class Eval:
    def __init__(self):
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

    def readdata(self, gt_coda_api, res_coda_api, cid, select_thre=0., select_num=100):
        if cid == "unknown":
            self.res = {self.res_known_cid_list[-1] + 1: {}}
        elif cid == "agnostic":
            self.res = {-1: {}}
        else:
            self.res = {cid: {}}
        self.gt = {}

        for imgID in self.imgidlist:
            # read prediction
            res_list_this_img = res_coda_api.loadAnns(res_coda_api.getAnnIds(imgIds=[imgID]))
            if cid == "agnostic":
                if len(res_list_this_img)==0:
                    img_res = np.array([])
                else:
                    # xyhw -> xyxy
                    img_res = np.array([res['bbox'] + [res[self.sort_scores_name]] for res in res_list_this_img])
                    if select_thre != 0:
                        img_res = img_res[img_res[:, -1] > select_thre]
                    if select_num < 100:
                        img_res = img_res[:select_num]
                    img_res[:, 2] = img_res[:, 0] + img_res[:, 2]
                    img_res[:, 3] = img_res[:, 1] + img_res[:, 3]
            else:
                if cid == "unknown":
                    res_list_this_img = [res for res in res_list_this_img if res["category_id"] not in self.res_known_cid_list] 
                    # res_list_this_img = [res for res in res_list_this_img if res["category_id"] >10] 
                else:
                    res_list_this_img = [res for res in res_list_this_img if res["category_id"]==cid] 
                if len(res_list_this_img)==0:
                    img_res = np.array([])
                else:
                    # xyhw -> xyxy
                    img_res = np.array([res['bbox'] + [res[self.sort_scores_name]] for res in res_list_this_img])
                    if select_thre != 0:
                        img_res = img_res[img_res[:, -1] > select_thre]
                    if select_num < 100:
                        img_res = img_res[:select_num]
                    img_res[:, 2] = img_res[:, 0] + img_res[:, 2]
                    img_res[:, 3] = img_res[:, 1] + img_res[:, 3]
            if cid == "unknown":
                self.res[self.res_known_cid_list[-1] + 1].update({imgID: img_res})
            elif cid == "agnostic":
                self.res[-1].update({imgID: img_res})
            else:
                self.res[cid].update({imgID: img_res})
            # read groundtruth
            gt_list_this_img = gt_coda_api.loadAnns(gt_coda_api.getAnnIds(imgIds=[imgID]))
            if cid == "agnostic":
                img_gt = np.array([gti['bbox'] + [-1] for gti in gt_list_this_img]).astype(int)
            else:
                img_gt = []
                for gti in gt_list_this_img:
                    if gti["category_id"] in self.gt_known_cid_list:
                        img_gt.append(gti['bbox'] + [self.gt_res_cid_dict[gti["category_id"]]])
                    else:
                        img_gt.append(gti['bbox'] + [self.res_known_cid_list[-1] + 1])
                img_gt = np.array(img_gt).astype(int)
            img_gt[:, 2] = img_gt[:, 0] + img_gt[:, 2]
            img_gt[:, 3] = img_gt[:, 1] + img_gt[:, 3]
            self.gt.update({imgID: img_gt})

    def run(self):
        self.sort_scores_name = "score"
        if self.args.dataset == "soda":
            # cid_list = [0, 1, 2, 3, 4, 5, 6, -1] # 0-5 for known metric, 6 for unknown metirc, and -1 for agnostic metric
            # cid_list = [1, 2, 3, 4, 5, 6, 7, -1] # 1-6 for known metric, 7 for unknown metirc, and -1 for agnostic metric
            self.res_known_cid_list = [1, 2, 3, 4, 5, 6]
            self.gt_known_cid_list = [1, 2, 3, 4, 7, 6]
            self.gt_res_cid_dict = {}
            for  res_cid, gt_cid in zip(self.res_known_cid_list, self.gt_known_cid_list):
                self.gt_res_cid_dict[gt_cid] = res_cid
            cid_list = self.res_known_cid_list + ["unknown", "agnostic"]
        elif self.args.dataset == "cityscapes":
            self.res_known_cid_list = [1, 3, 4, 5]
            self.gt_known_cid_list = [1, 3, 4, 7]
            self.gt_res_cid_dict = {}
            for  res_cid, gt_cid in zip(self.res_known_cid_list, self.gt_known_cid_list):
                self.gt_res_cid_dict[gt_cid] = res_cid
            cid_list = self.res_known_cid_list + ["unknown", "agnostic"]
        elif self.args.dataset == "bdd":
            self.res_known_cid_list = [1, 2, 3, 5, 4] # for other methods
            # self.res_known_cid_list = [1, 2, 3, 4, 5] # for Unsniffer and VOS
            self.gt_known_cid_list = [1, 2, 3, 4, 7]
            self.gt_res_cid_dict = {}
            for  res_cid, gt_cid in zip(self.res_known_cid_list, self.gt_known_cid_list):
                self.gt_res_cid_dict[gt_cid] = res_cid
            cid_list = self.res_known_cid_list + ["unknown", "agnostic"]
        else:
            print("Dataset is not implemented yet!")
        ap_list = []
        precision_list = []
        recall_list = []
        recall_list_10 = []
        recall_list_20 = []
        recall_list_30 = []
        f1_list = []
        for cid in cid_list:
            self.readdata(self.gt_coda_api, self.res_coda_api, cid, select_thre=self.args.known_select_thre, select_num=100)
            if cid == "unknown":
                cid = self.res_known_cid_list[-1] + 1
            if cid == "agnostic":
                cid = -1
            recall, precision, ap, _, _, _, _ = voc_evaluate(self.res, self.gt, cid)
            ap_list.append(ap)
            recall_list.append(recall)
            precision_list.append(precision)
        for cid in cid_list:
            self.readdata(self.gt_coda_api, self.res_coda_api, cid, select_thre=0, select_num=10)
            if cid == "unknown":
                cid = self.res_known_cid_list[-1] + 1
            if cid == "agnostic":
                cid = -1
            recall, _, _, _, _, _, _ = voc_evaluate(self.res, self.gt, cid)
            recall_list_10.append(recall)
        for cid in cid_list:
            self.readdata(self.gt_coda_api, self.res_coda_api, cid, select_thre=0, select_num=20)
            if cid == "unknown":
                cid = self.res_known_cid_list[-1] + 1
            if cid == "agnostic":
                cid = -1
            recall, _, _, _, _, _, _ = voc_evaluate(self.res, self.gt, cid)
            recall_list_20.append(recall)
        for cid in cid_list:
            self.readdata(self.gt_coda_api, self.res_coda_api, cid, select_thre=0, select_num=30)
            if cid == "unknown":
                cid = self.res_known_cid_list[-1] + 1
            if cid == "agnostic":
                cid = -1
            recall, _, _, _, _, _, _ = voc_evaluate(self.res, self.gt, cid)
            recall_list_30.append(recall)
            # f1_list.append(2 * (precision * recall) / (precision + recall))
        print("ap_list:", ap_list)
        print("recall_list:", recall_list)
        # print("recall_list_10:", recall_list_10)
        # print("recall_list_20:", recall_list_20)
        # print("recall_list_30:", recall_list_30)
        known_ap = np.mean(ap_list[:-2])
        # known_recall =np.mean(recall_list[:-2])
        print("unknown_ap:", ap_list[-2])
        print("known_ap:", known_ap)
        # print("known_recall:", known_recall)
        print("agnostic_ap:", ap_list[-1])
        print("agnostic_recall:", recall_list[-1])
        print("unknown_recall:", recall_list[-2])

        # print("agnostic_recall 10:", recall_list_10[-1])
        print("unknown_recall 10:", recall_list_10[-2])
        # print("agnostic_recall 20:", recall_list_20[-1])
        print("unknown_recall 20:", recall_list_20[-2])
        # print("agnostic_recall 30:", recall_list_30[-1])
        print("unknown_recall 30:", recall_list_30[-2])
        
        print_list = [round(known_ap, 3), round(ap_list[-1], 3), round(recall_list[-1], 3), round(recall_list[-2], 3), 
                    round(recall_list_10[-2], 3), round(recall_list_20[-2], 3), round(recall_list_30[-2], 3)]
        print(print_list)

        # print("Ap: ", ap_list)
        # print("precision: ", precision_list)
        # print("recall: ", recall_list)
        # print("f1: ", f1_list)


eva = Eval()
eva.run()