import os
import xml.etree.ElementTree as ET
import collections
import cv2

classes = ('pedestrian', 'cyclist', 'car', 'truck', 'bus', 'tricycle')

# palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
#          (0, 60, 100)]

palette = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255),
         (255, 0, 255)]

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

image_set = "train"
root = "./PROB/data/OWOD"
annotation_dir = os.path.join(root, 'Annotations')
image_dir = os.path.join(root, 'JPEGImages')
file_names = extract_fns(image_set, root)
cls2id_dict = {}
for i, cls in enumerate (classes):
    cls2id_dict[cls]=i 
for file in file_names:
    annot = os.path.join(annotation_dir, file + ".xml")
    img_file = os.path.join(image_dir, file + ".jpg")
    img = cv2.imread(img_file)
    tree = ET.parse(annot)
    target = parse_voc_xml(tree.getroot())
    instances = []
    for obj in target['annotation']['object']:
        cls = obj["name"]
        cls_id = cls2id_dict[cls]
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), palette[cls_id], 2)
        img = cv2.putText(
            img,
            cls, (xmin, ymin + len(cls) - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            palette[cls_id],
            2)
    print("file:", file)
    # cv2.imshow("draw img", img)
    # cv2.waitKey(0)
    cv2.imwrite("0.jpg", img)
    print(a)
        