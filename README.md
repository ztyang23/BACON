# BACON: Supercharge Your VLM with Bag-of-Concept Graph to Mitigate Hallucinations

[📢 [[Project Page](https://ztyang23.github.io/bacon-page/)] [[Model](https://huggingface.co/ztyang196/bacon-captioner/)] [[Paper](https://arxiv.org/abs/2407.03314)]]

## Release

- [2024/7/10] 🔥 The code of BACON is released!

## Contents

- [Install](#install)
- [Model](#model)
- [Dataset](#dataset)
- [Train](#train)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Install

Currently, we only provide installation guides for Linux.

1. Clone this repository and navigate to BACON folder

```bash
git clone https://github.com/ztyang23/BACON.git
cd BACON
```

2. Install Package

```Shell
conda create -n bacon python=3.10 -y
conda activate bacon
pip install -r requirements.txt
```
## Model

We strongly recommend downloading all the weights locally. Please create a folder named `ckpt` under the `BACON` folder and place all the downloaded weights into the `ckpt` folder. The file structure is as follows:

```
├── ckpt
│   ├── captioner (dir)
│   ├── llava-v1.5-13b (dir)
│   ├── groundingdino_swint_ogc.pth
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── sam_vit_h_4b8939.pth
│   ├── ViT-B-32.pt
│   ├── ViT-L-14.pt
```

First, download the checkpoints for [LLaVA](https://huggingface.co/liuhaotian/llava-v1.5-13b), and [BACON-Captioner](https://huggingface.co/ztyang196/bacon-captioner/)

Second, download the checkpoint for [GroundingDINO](https://huggingface.co/ShilongLiu/GroundingDINO/tree/main) (only the `groundingdino_swint_ogc.pth` and `GroundingDINO_SwinT_OGC.py` are needed) used by BACON, as well as the checkpoint for [SAM](https://huggingface.co/spaces/abhishek/StableSAM/tree/main) (only the `sam_vit_h_4b8939.pth` is needed)

Finally, download the checkpoint for CLIP used for evaluation; both [CLIP-B-32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) and [CLIP-L-14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) are utilized in our project.

## Dataset

For the training set, we used images from [Unsplash](https://unsplash.com/) and [MSCOCO](https://cocodataset.org/#home). For convenience, we have renumbered these images, and you can download them here. For the test set, we used the COCO2017 test set, so we do not provide these images; please download them from the [official COCO website](https://cocodataset.org/#home). Additionally, in our evaluation, we used the [COCO2014 validation set](https://cocodataset.org/#home), [COCO2015 test set](https://cocodataset.org/#home), and [NLVR2](https://lil.nlp.cornell.edu/nlvr/), [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html). Please download all these datasets and create a folder named `data` under the `BACON` folder, placing all the datasets in the`data`folder. Additionally, we provide all [annotations](https://drive.google.com/file/d/1sSBTxkRsyL_bk0XfrXOIdGpTY2XzA_bK/view?usp=drive_link); please download them and place them in the `data` folder. The directory structure should be as follows.

```
├── data
│   ├── coco2014_val
│   │   ├── COCO_val2014_000000000042.jpg
│   │   ├── COCO_val2014_000000000073.jpg
│   ├── coco2015_test
│   │   ├── COCO_test2015_000000000001.jpg
│   │   ├── COCO_test2015_000000000014.jpg
│   ├── coco2017_test
│   │   ├── 000000000001.jpg
│   │   ├── 000000000016.jpg
│   ├── coco2017_val
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│   ├── nlvr2_test1
│   │   ├── test1-0-2-img0.png
│   │   ├── test1-0-2-img1.png
│   ├── training_data
│   │   ├── 000000000000.jpg
│   │   ├── 000000000001.jpg
│   ├── visual_genome
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   ├── bacondata_image_ids.txt
│   ├── coco_image_ids.txt
│   ├── nlvr2_test1.json
│   ├── okvqa_mscoco_val2014_annotations.json
│   ├── okvqa_OpenEnded_mscoco_val2014_questions.json
│   ├── pointqa_local_test.jsonl
│   ├── test_dataset.jsonl
│   ├── test.jpg
│   ├── train.json
│   ├── training_dataset.jsonl
│   ├── v7w_pointing_test.jsonl
│   ├── vg_attributes.json
│   ├── vg_object_list.txt
│   ├── vg_question_answers.json
│   ├── vg_relationship_list.txt
│   ├── vg_scene_graphs.json
│   ├── vg150.json
│   ├── vqav1_vqa_E_val.jsonl
│   ├── vqav2_OpenEnded_mscoco_test-dev2015_questions.jsonl
│   ├── vqav2_OpenEnded_mscoco_test2015_questions.jsonl
```

## Train

We have generally followed LLaVA's training code. To train a BACON-Captioner, you first need to convert the training data into the format required by LLaVA. (We have provided the result of running this code, so you don't need to perform this step.)

```Shell
python construct_training_data.py
```

Then, simply run the training script."

```Shell
sh train.sh
```

## Inference

Run the inference script, and the results will be output to a file named `result/inference.json`.

```Shell
sh inference.sh
```

## Evaluation

We provide the code of evalution for multiple downstream tasks including Open-vocabulary detection, Open-vocabulary scene graph generation, PointQA, PointingQA, VQA, Plan.

Complete evaluation requires running a large amount of baseline code; therefore, we only provide the code for calculating metrics. For the inference part of the baselines, please refer to the official code of each respective baseline. For convenience, we have provided the result files of all baselines we ran [here](https://drive.google.com/file/d/1bXoya2Ca-zcUJVdYuu6JGOJjxlg_6UBl/view?usp=drive_link), please download them and place them in the `results` folder. If you need to run the inference yourself, please format the output according to these files. To calculate metrics, modify `cfg.task` in `eval.py` to the desired task and then run `eval.py`.

```Shell
python eval.py
```
