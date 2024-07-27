import xml.etree.cElementTree as ET
import os
from pycocotools.coco import COCO

category_dict = {
    'Pedestrian': 'pedestrian', 'Cyclist':'cyclist', 'Car':'car', 'Truck': 'truck', 'Bus':'bus', 'Tricycle':'tricycle'
}

def imagesets(coco_train_annotation_file, coco_val_annotation_file, target_folder):
    # os.makedirs(os.path.join(target_folder, 'Main'), exist_ok=True)
    # import pdb;pdb.set_trace()
    coco_train_instance = COCO(coco_train_annotation_file)
    coco_val_instance = COCO(coco_val_annotation_file)
    with open(target_folder+"/ImageSets/train.txt", "a") as myfile:
        for index, image_id in enumerate(coco_train_instance.imgToAnns):
            image_details = coco_train_instance.imgs[image_id]
            myfile.write(image_details['file_name'].split('.')[0])
            myfile.write('\n')

        for index, image_id in enumerate(coco_val_instance.imgToAnns):
            image_details = coco_val_instance.imgs[image_id]
            myfile.write(image_details['file_name'].split('.')[0])
            myfile.write('\n')


def coco_to_voc_detection(coco_annotation_file, target_folder):
    os.makedirs(os.path.join(target_folder, 'Annotations'), exist_ok=True)
    coco_instance = COCO(coco_annotation_file)
    for index, image_id in enumerate(coco_instance.imgToAnns):
        # import pdb;pdb.set_trace()
        image_details = coco_instance.imgs[image_id]
        annotation_el = ET.Element('annotation')
        ET.SubElement(annotation_el, 'filename').text = image_details['file_name']

        size_el = ET.SubElement(annotation_el, 'size')
        ET.SubElement(size_el, 'width').text = str(image_details['width'])
        ET.SubElement(size_el, 'height').text = str(image_details['height'])
        ET.SubElement(size_el, 'depth').text = str(3)

        for annotation in coco_instance.imgToAnns[image_id]:
            object_el = ET.SubElement(annotation_el, 'object')
            # import pdb;pdb.set_trace()
            if coco_instance.cats[annotation['category_id']]['name'] == "Tram":
                cat_name = "bus"
            else:
                cat_name = category_dict[coco_instance.cats[annotation['category_id']]['name']]
            ET.SubElement(object_el,'name').text = cat_name
            # ET.SubElement(object_el, 'name').text = 'unknown'
            ET.SubElement(object_el, 'difficult').text = '0'
            bb_el = ET.SubElement(object_el, 'bndbox')
            ET.SubElement(bb_el, 'xmin').text = str(int(annotation['bbox'][0] + 1.0))
            ET.SubElement(bb_el, 'ymin').text = str(int(annotation['bbox'][1] + 1.0))
            ET.SubElement(bb_el, 'xmax').text = str(int(annotation['bbox'][0] + annotation['bbox'][2] + 1.0))
            ET.SubElement(bb_el, 'ymax').text = str(int(annotation['bbox'][1] + annotation['bbox'][3] + 1.0))

        ET.ElementTree(annotation_el).write(os.path.join(target_folder, 'Annotations', image_details['file_name'].split('.')[0] + '.xml'))

    print('Processed ' + str(len(coco_instance.imgToAnns)) + ' images.')

if __name__ == '__main__':
    target_folder = '/media/heyulin/10ebc972-525f-4aca-b905-281ecbb3c5e4/datasets/CODA/labeled_trainval/SSLAD-2D/labeled'
    soda_train_annotation_file = target_folder + '/' + "annotations" + '/' + "instances_train.json" 
    soda_val_annotation_file =  target_folder + '/' + "annotations" + '/' + "instances_val.json" 

    coco_to_voc_detection(soda_train_annotation_file, target_folder)
    coco_to_voc_detection(soda_val_annotation_file, target_folder)
    imagesets(soda_train_annotation_file, soda_val_annotation_file, target_folder)
