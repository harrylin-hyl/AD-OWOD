import json
import os 

json_dir_path = "/media/heyulin/10ebc972-525f-4aca-b905-281ecbb3c5e4/datasets/CODA/labeled_trainval/SSLAD-2D/labeled/annotations"
final_output_list = []

file_list = os.listdir(json_dir_path)
for file in file_list:
    with open(json_dir_path+"/"+file, "r") as fp:
        json_data = json.load(fp)
    final_output_list.append(json_data)

with open(os.path.join('/media/heyulin/10ebc972-525f-4aca-b905-281ecbb3c5e4/datasets/CODA/labeled_trainval/SSLAD-2D/labeled', 'train.json'), 'w') as fp:
    json.dump(final_output_list, fp, indent=4, separators=(',', ': '))