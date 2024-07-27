import random

image_set_file = './PROB/data/OWOD/ImageSets/CODA/train.txt'
id_list = []
with open(image_set_file, 'r') as f:
    for id in f.readlines():
        id = id.strip('\n')
        id_list.append(id)
val_list = random.sample(id_list, 200)
test_list = list(set(id_list) - set(val_list))

target_folder = "./PROB/data/OWOD"
with open(target_folder+"/ImageSets/CODA/pretest.txt", "a") as myfile:
    for val_id in val_list:
        myfile.write(val_id)
        myfile.write('\n')

with open(target_folder+"/ImageSets/CODA/vos_train.txt", "a") as myfile:
    for test_id in test_list:
        myfile.write(test_id)
        myfile.write('\n')