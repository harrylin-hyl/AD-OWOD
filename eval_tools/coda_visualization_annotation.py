import os
import xml.etree.ElementTree as ET
import collections
import cv2

classes = ('pedestrian', 'cyclist', 'car', 'truck', 'bus', 'tricycle')

# common and corner case
palette = [(0, 255, 0), (0, 0, 255)]

def extract_fns(image_set, voc_root):
    splits_dir = os.path.join(voc_root, 'ImageSets')
    split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    return file_names

def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                    for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

image_set = "test"
root = "/media/heyulin/10ebc972-525f-4aca-b905-281ecbb3c5e4/datasets/CODA/CODA2022-val"
annotation_dir = os.path.join(root, 'Annotations')
image_dir = os.path.join(root, 'JPEGImages')
file_names = extract_fns(image_set, root)

for file in file_names:
    annot = os.path.join(annotation_dir, file + ".xml")
    img_file = os.path.join(image_dir, file + ".jpg")
    img = cv2.imread(img_file)
    tree = ET.parse(annot)
    target = parse_voc_xml(tree.getroot())
    instances = []
    for obj in target['annotation']['object']:
        cls = obj["name"]
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        if obj['corner_case'] == str(True) and cls in classes:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), palette[1], 2)
        else:
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), palette[0], 2)
        #  if obj['corner_case'] == str(True):
        #     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), palette[1], 2)
        # else:
        #     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), palette[0], 2)
        img = cv2.putText(
            img,
            cls, (xmin, ymin + len(cls) - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 0, 255),
            2)
    print("file:", file)
    cv2.imshow("draw img", img)
    cv2.waitKey(0)
    # cv2.imwrite("0.jpg", img)
    # print(a)
        