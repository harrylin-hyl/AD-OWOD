# Sniffing Threatening Open-World Objects in Autonomous Driving by Open-Vocabulary Models (ACMMM 2024)

<!-- [`Paper`](https://arxiv.org/abs/2212.01424)  -->

#### Yulin He, Siqi Wang, Wei Chen, Tianci Xun, and Yusong Tan


# Abstract

Autonomous driving (AD) is a typical application that requires effectively exploiting multimedia information. 
For AD, it is critical to ensure safety by detecting unknown objects in an open world, driving the demand for open world object detection (OWOD).
However, existing OWOD methods treat generic objects beyond known classes in the train set as unknown objects and prioritize recall in evaluation.
This encourages excessive false positives and endangers safety of AD. To address this issue, we restrict the definition of unknown objects to threatening objects in AD, and introduce a new evaluation protocol, which is built upon a new metric named U-ARecall, to alleviate biased evaluation caused by neglecting false positives.
Under the new evaluation protocol, we re-evaluate existing OWOD methods and discover that they typically perform poorly in AD.
Then, we propose a novel OWOD paradigm for AD based on fine-tuning foundational open-vocabulary models (OVMs), as they can exploit rich linguistic and visual prior knowledge for OWOD. 
Following this new paradigm, we propose a brand-new OWOD solution, which effectively addresses two core challenges of fine-tuning OVMs via two novel techniques: 1) the maintenance of open-world generic knowledge by a dual-branch architecture; 2) the acquisition of scenario-specific knowledge by the visual-oriented contrastive learning scheme.
Besides, a dual-branch prediction fusion module is proposed to avoid post-processing and hand-crafted heuristics.
Extensive experiments show that our proposed method not only surpasses classic OWOD methods in unknown object detection by a large margin ($\sim$ 3 $\times$ U-ARecall), but also notably outperforms OVMs without fine-tuning in known object detection ($\sim$ 20\% K-mAP).

# Overview

- We devise a more suitable evaluation protocol for AD-oriented OWOD, which includes restricting unknown objects to threatening objects in AD and introducing a new evaluation metric to alleviate the biased evaluation in OWOD. Based on this evaluation protocol, we re-evaluate existing OWOD methods and establish the AD-oriented OWOD benchmark.
- We propose a new OWOD paradigm for AD based on fine-tuning OVMs, exploiting rich prior knowledge, including high-level language semantic and diverse visual patterns, to distinguish between threatening objects and false positives.  
- We identify and address two core challenges when fine-tuning OVMs: preserving open-world generic knowledge by dual-branch architecture and acquiring scenario-specific knowledge by visual-oriented contrastive learning.
- We propose a prediction fusion module that can integrate predictions from multiple branches without the need of post-processing and hand-crafted heuristics, serving as a generalized method applicable to transformer-based detectors.

# Installation
Our code is based on the dev-3.x branch of [mmdetection](http://github.com/open-mmlab/mmdetection/tree/dev-3.x), which releases the implementation of [GroundingDino](https://github.com/IDEA-Research/GroundingDINO.git).

Note: using the ``**dev-3.x**'' branch.
### Requirements

```bash
conda create --name ad-owod python==3.8 -y
conda activate ad-owod
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .

optional:
pip install peft # for lora finetuning
```

### Backbone features

Download the GroundingDINO pretrained weight from [mmdetection](http://github.com/open-mmlab/mmdetection/tree/dev-3.x) and add in `weights` folder.

## Code Structure
```
AD-OWOD /
└── configs
        └── auto_driving_grounding_dino (fine-tuning methods and our method)
└── mmdet 
        └── datasets
                ├── soda.py
                └── bdd.py
        └── hooks
                └── load_weight_hook.py (load the weights of dual-branch network)
        └── models
                └── detectors
                        ├── auto_driving_grounding_dino.py
                        ├── adapter_grounding_dino.py linear_prob_grounding_dino.py 
                        ├── linear_prob_grounding_dino.py
                        └── lora_grounding_dino.py
                    └── dense_heads 
                            └── auto_driving_grounding_dino_head.py
└── eval_tools (data conversion tools, visualization tools, and evaluation tools)
    ├── soda_formnat_to_voc.py 
    ├── bdd_formnat_to_voc.py 
    ├── coda_formnat_to_voc.py 
    ├── convert_to_pretest.py 
    ├── convert_coda_to_pretest.py 
    ├── soda_visualization.py 
    ├── bdd_visualization.py 
    ├── coda_visualization.py 
    ├── output_visualization.py 
    ├── eval.py (evaluation) 
    └── voc_eval_offical.py (evaluation)

```
Note: If you can not access the website of huggingface online, you need to download the `bert-base-cased` from https://huggingface.co/bert-base-uncased/tree/main and change the path of `lang_model_name` in the config file.
## Data Structure
```
AD-OWOD/
└── data/
    └── CODA/
        ├── train (train and val set of SODA)
        ├── val (val set of CODA)
        └── annotations
            ├── train.json
            ├── val.json
            └── annotation.json (CODA)
    └── BDD/
        ├── train
        ├── val
        └── annotations
            ├── bdd100k_labels_images_det_coco_train.json
            ├── bdd100k_labels_images_det_coco_val.json
            └── annotation.json (CODA)
```
### Dataset Preparation

CODA:
- Download the train and val set of SODA from https://soda-2d.github.io/download.html.
- Download the val set of CODA from https://coda-dataset.github.io/.
- Prepare the annotation files. (1) Get the train.json file by running `eval_tools/merge_json.py`. (2) Get the val.json by running `eval_tools/coda_format_to_voc.py` and `eval_tools/convert_coda_to_coco.py`. You can also directly download the annotation files from [here](https://drive.google.com/file/d/1YskzEbtRqrjYzic5roWPEGqtDH7cXFJS)

BDD:
- Download the train and val set of BDD0100K from https://doc.bdd100k.com/download.html.


# Training

#### Training on single node

To train SGROD on a single node with 4 GPUS, run
```bash
bash tools/dist_train.sh configs/auto_driving_grounding_dino/auto_driving_grounding_dino_swin-t_16xb2_1x_soda.py    4
```
you can also decide to run each one of the configurations defined in ``configs/auto_driving_grounding_dino``.


# Evaluation & Result Reproduction

For reproducing any of the aforementioned results, please download our [pretrain weights](https://drive.google.com/file/d/1YI33ZGUfNHT0dKS3LwmpsbNHphGdxRbL) and place them in the 
'weights' directory. 

1. Run the `demo/image_demo.py` file to output json file for testing.
```bash
python demo/image_demo.py ./data/CODA/val/  $config  --weights $weights  --texts "$text_prompts" --no-save-pred --no-save-vis --save-json-path ./val_json_output_dir/$save_json
```
For example:
```bash
# SODA
python demo/image_demo.py ./data/CODA/val/  configs/auto_driving_grounding_dino/auto_driving_grounding_dino_swin-t_16xb2_1x_soda.py  --weights weights/auto_gd_soda.pth   --texts "pedestrian . cyclist . car . truck . bus . tricycle . vehicle . roadblock . obstacle ." --no-save-pred --no-save-vis --save-json-path ./val_json_output_dir/auto_gd_soda.json
# BDD
python demo/image_demo.py ./data/CODA/val/  configs/auto_driving_grounding_dino/auto_driving_grounding_dino_swin-t_16xb2_1x_bdd.py  --weights weights/auto_gd_bdd.pth   --texts "person . rider . car . bus . truck . bike . motor . traffic light . traffic sign . train . vehicle . roadblock . obstacle ." --no-save-pred --no-save-vis --save-json-path ./val_json_output_dir/auto_gd_bdd.json
```

2. Run the `eval_tools/eval.py` file to output the metrics.
```bash
python eval_tools/eval.py val_json_output_dir/$save_json   json_output_dir/annotations.json 0. $data
```
For example:
```bash
# SODA
python eval_tools/eval.py val_json_output_dir/auto_gd_soda.json   val_json_output_dir/annotations.json 0. soda
# BDD
python eval_tools/eval.py val_json_output_dir/auto_gd_bdd.json   val_json_output_dir/annotations.json 0. bdd
```

<!-- ```
SGROD/
└── checkpoints/
    ├── MOWODB/
    |   └── t1 checkpoint0040.pth
        └── t2_ft checkpoint0110.pth
        └── t3_ft checkpoint0180.pth
        └── t4_ft checkpoint0260.pth
    └── SOWODB/
        └── t1 checkpoint0040.pth
        └── t2_ft checkpoint0120.pth
        └── t3_ft checkpoint0200.pth
        └── t4_ft checkpoint0300.pth
``` -->


**Note:**
For more training and evaluation details please check the [mmdetection](http://github.com/open-mmlab/mmdetection/tree/dev-3.x) reposistory.




<!-- # Citation

If you use PROB, please consider citing:

```
@misc{zohar2022prob,
  author = {Zohar, Orr and Wang, Kuan-Chieh and Yeung, Serena},
  title = {PROB: Probabilistic Objectness for Open World Object Detection},
  publisher = {arXiv},
  year = {2022}
}
``` -->

# Contact

Should you have any question, please contact: heyulin@nudt.edu.cn

**Acknowledgments:**

This work builds on previous works' code base such as [mmdetection](http://github.com/open-mmlab/mmdetection/tree/dev-3.x), [UnSniffer](https://github.com/Went-Liang/UnSniffer), [GroundingDino](https://github.com/IDEA-Research/GroundingDINO.git). Please consider citing these works as well.

"# AD-OWOD" 
