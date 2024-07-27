#!/bin/bash

config=$1
# exp=$2
weights=$2
text_prompts=$3
save_json=$4
data=$5

echo $save_json
# python demo/image_demo.py ./data/CODA/val/  $config  --weights ./work_dirs/$exp/epoch_12.pth   --texts "$text_prompts" --no-save-pred --no-save-vis --save-json-path ./json_output_dir/$save_json
python demo/image_demo.py ./data/CODA/val/  $config  --weights $weights  --texts "$text_prompts" --no-save-pred --no-save-vis --save-json-path ./val_json_output_dir/$save_json
wait
python eval_tools/eval.py val_json_output_dir/$save_json   json_output_dir/annotations.json 0. $data
# wait
# python eval_tools/WI.py json_output_dir/$save_json   json_output_dir/val.json 0.
# wait
# python eval_tools/aose.py json_output_dir/$save_json   json_output_dir/val.json 0.

# python demo/image_demo.py ./data/CODA/val/0008.jpg  $config  --weights ./work_dirs/$exp/epoch_12.pth   --texts "$text_prompts" --no-save-pred --save-json-path ./json_output_dir/test.json