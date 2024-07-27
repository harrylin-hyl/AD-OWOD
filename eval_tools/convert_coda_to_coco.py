# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmengine.fileio import list_from_file
from mmengine.utils import mkdir_or_exist, track_progress
import json

coda_classes = [
    'pedestrian', 'cyclist', 'car', 'truck', 'bus', 'tricycle', 'corner_case'
]

label_ids = {name: i+1 for i, name in enumerate(coda_classes)}


def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    names = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in coda_classes:
            name = "corner_case"
        # if obj.find("corner_case").text == "True":
        #     name = "corner_case"
        # else:
        #     name = obj.find('name').text
        names.append(name)
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'names': names,
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, split, out_file):
    annotations = []
    filelist = osp.join(devkit_path,
                            f'ImageSets/CODA/{split}.txt')
    if not osp.isfile(filelist):
        print(f'filelist does not exist: {filelist}')
        return
    img_names = list_from_file(filelist)
    xml_paths = [
        osp.join(devkit_path, f'Annotations/{img_name}.xml')
        for img_name in img_names
    ]
    img_paths = [
        f'JPEGImages/{img_name}.jpg' for img_name in img_names
    ]
    part_annotations = track_progress(parse_xml,
                                        list(zip(xml_paths, img_paths)))
    annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)

    with open(out_file, 'w') as fp:
        json.dump(annotations, fp, indent=4, separators=(',', ': '))
    return 


def cvt_to_coco_json(annotations):
    annotation_id = 0
    coco = dict()
    coco['categories'] = []
    coco['annotations'] = []
    coco['images'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, name):
        annotation_item = dict()
        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = xywh.tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['name'] = name
        annotation_item['id'] = int(annotation_id)
        annotation_id += 1

        return annotation_item, annotation_id

    for category_id, name in enumerate(coda_classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id) + 1
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(file_name.split('/')[-1].split('.')[0])
        image_item['file_name'] = str(file_name.split("/")[-1])
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)
        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        names = ann['names']
        img_results = []
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            name = names[bbox_id]
            image_id = int(file_name.split('/')[-1].split('.')[0])
            annotation_item, annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, name)
            img_results.append(annotation_item)
        coco['annotations'].extend(img_results)


    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('-d', '--dataset-name', help='output path')
    parser.add_argument(
        '--out-format',
        default='pkl',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    cvt_annotations(devkit_path, args.dataset_name,
                    osp.join(out_dir, args.dataset_name + out_fmt))

    print('Done!')


if __name__ == '__main__':
    main()
