import json
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from bacon_modules.utils.dataloader import Bacon_benchmark, CoCoplan
from bacon_modules.utils.tools import prepare_clip, extract_clip_feature, drop_index


def preprocess_dataset_for_plan(cfg, dataset):
    dataset_plan = {}
    for image_id in tqdm(dataset.dataset):
        data_obj = dataset.dataset[image_id]
        organized_caption = data_obj["organized_caption"]
        name_list = []
        position_list = []
        for obj in organized_caption["object_list"]:
            obj_name = obj["name"].strip()
            position = obj["position"]
            name_list.append(obj_name)
            position_list.append(position)
        
        dataset_plan[image_id] = {}
        dataset_plan[image_id]["name"] = name_list
        dataset_plan[image_id]["position"] = position_list

    return dataset_plan


def compute_iou(bbox_1, bbox_2):
    x_min1, y_min1, x_max1, y_max1 = bbox_1
    x_min2, y_min2, x_max2, y_max2 = bbox_2

    inter_x_min, inter_y_min, inter_x_max, inter_y_max = max(x_min1, x_min2), max(y_min1, y_min2), min(x_max1, x_max2), min(y_max1, y_max2)

    inter_width = max(inter_x_max - inter_x_min, 0)
    inter_height = max(inter_y_max - inter_y_min, 0)

    intersection_area = inter_width * inter_height

    bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area

    return iou


def get_predictions(cfg, dataset, image_dir):
    predictions = {}
    predictions["bacon"] = {}
    predictions["layoutgpt"] = {}
    if dataset == "bacon benchmark":
        with open (cfg.plan.bacondata_layoutgpt_result, "r") as inputfile:
            data = json.load(inputfile)
            for data_obj in data:
                image_id = data_obj["image_id"]
                image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
                image = Image.open(image_path)
                w, h = image.size
                predictions["layoutgpt"][image_id] = {}
                predictions["layoutgpt"][image_id]["name"] = []
                predictions["layoutgpt"][image_id]["bbox"] = []
                object_list = data_obj["object_list"]
                for obj in object_list:
                    name, bbox = obj
                    x_min, y_min, x_max, y_max = bbox
                    bbox_ = (x_min * w, y_min * h, x_max * w, y_max * h)
                    predictions["layoutgpt"][image_id]["name"].append(name)
                    predictions["layoutgpt"][image_id]["bbox"].append(bbox_)
        with open (cfg.plan.bacondata_bacon_result, "r") as inputfile:
            data = json.load(inputfile)
            for data_obj in data:
                image_id = data_obj["image_id"]
                image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
                image = Image.open(image_path)
                w, h = image.size
                predictions["bacon"][image_id] = {}
                predictions["bacon"][image_id]["name"] = []
                predictions["bacon"][image_id]["bbox"] = []
                object_list = data_obj["object_list"]
                for obj in object_list:
                    name, bbox = obj
                    x_min, y_min, x_max, y_max = bbox
                    bbox_ = (x_min * w, y_min * h, x_max * w, y_max * h)
                    predictions["bacon"][image_id]["name"].append(name)
                    predictions["bacon"][image_id]["bbox"].append(bbox_)
    elif dataset == "coco":
        with open (cfg.plan.coco_layoutgpt_result, "r") as inputfile:
            data = json.load(inputfile)
            for data_obj in data:
                image_id = data_obj["image_id"]
                image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
                image = Image.open(image_path)
                w, h = image.size
                predictions["layoutgpt"][image_id] = {}
                predictions["layoutgpt"][image_id]["name"] = []
                predictions["layoutgpt"][image_id]["bbox"] = []
                object_list = data_obj["object_list"]
                for obj in object_list:
                    name, bbox = obj
                    x_min, y_min, x_max, y_max = bbox
                    bbox_ = (x_min * w, y_min * h, x_max * w, y_max * h)
                    predictions["layoutgpt"][image_id]["name"].append(name)
                    predictions["layoutgpt"][image_id]["bbox"].append(bbox_)
        with open (cfg.plan.coco_bacon_result, "r") as inputfile:
            data = json.load(inputfile)
            for data_obj in data:
                image_id = data_obj["image_id"]
                image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
                image = Image.open(image_path)
                w, h = image.size
                predictions["bacon"][image_id] = {}
                predictions["bacon"][image_id]["name"] = []
                predictions["bacon"][image_id]["bbox"] = []
                object_list = data_obj["object_list"]
                for obj in object_list:
                    name, bbox = obj
                    x_min, y_min, x_max, y_max = bbox
                    bbox_ = (x_min * w, y_min * h, x_max * w, y_max * h)
                    predictions["bacon"][image_id]["name"].append(name)
                    predictions["bacon"][image_id]["bbox"].append(bbox_)
    
    return predictions


def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=0)


def max_softmax_value_dict(input_dict):
    keys = list(input_dict.keys())
    values = list(input_dict.values())
    
    values = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item for item in values]
    
    values = np.array(values, dtype=np.float32)
    
    softmax_values = softmax(values)
    
    max_index = np.argmax(softmax_values)
    max_key = keys[max_index]
    max_value = softmax_values[max_index]
    
    return max_key, max_value


def max_softmax_value_list(input_list):
    values = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item for item in input_list]
    
    values = np.array(values, dtype=np.float32)
    
    softmax_values = softmax(values)
    
    max_index = np.argmax(softmax_values)
    max_value = softmax_values[max_index]
    
    return max_index, max_value


def map_function(cfg, input_list, label_list, label_count_dict, pre_pos_list, threshold, mode=1, label_feature_dict=None):
    if mode == 1 and label_feature_dict is None:
        label_feature_dict = {}
    elif mode == 2:
        label_feature_list = []
    map_dict = {}
    map_pos_dict = {}

    if mode == 2 or (mode == 1 and label_feature_dict == {}):
        if mode == 1:
            for label in label_count_dict:
                feature = extract_clip_feature(label, cfg.clip, cfg.preprocess, mode='t')
                label_feature_dict[label] = feature
        elif mode == 2:
            for label in label_list:
                feature = extract_clip_feature(label, cfg.clip, cfg.preprocess, mode='t')
                label_feature_list.append(feature)

    for pre_idx in range(len(input_list)):
        pre_name = input_list[pre_idx]
        bbox = pre_pos_list[pre_idx]
        if mode == 1:
            similarity_dict = {}
        elif mode == 2:
            similarity_list = []
        pre_name_feature = extract_clip_feature(pre_name, cfg.clip, cfg.preprocess, mode='t')
        if mode == 1:
            for label in label_feature_dict:
                similarity_score = 100 * pre_name_feature @ label_feature_dict[label].T
                similarity_dict[label] = similarity_score
            max_smname, max_smvalue = max_softmax_value_dict(similarity_dict)
        elif mode == 2:
            for label_feature in label_feature_list:
                similarity_score = 100 * pre_name_feature @ label_feature.T
                similarity_list.append(similarity_score)
            max_smindex, max_smvalue = max_softmax_value_list(similarity_list)
        if max_smvalue > threshold:
            if mode == 1:
                if max_smname not in map_dict:
                    map_dict[max_smname] = 1
                    map_pos_dict[max_smname] = [bbox]
                else:
                    map_dict[max_smname] += 1
                    map_pos_dict[max_smname].append(bbox)
            elif mode == 2:
                if label_list[max_smindex] not in map_dict:
                    map_dict[label_list[max_smindex]] = 1
                    map_pos_dict[label_list[max_smindex]] = [bbox]
                else:
                    map_dict[label_list[max_smindex]] += 1
                    map_pos_dict[label_list[max_smindex]].append(bbox)
        elif pre_name in label_list:
            if mode == 1:
                if pre_name not in map_dict:
                    map_dict[pre_name] = 1
                    map_pos_dict[pre_name] = [bbox]
                else:
                    map_dict[pre_name] += 1
                    map_pos_dict[pre_name].append(bbox)
            elif mode == 2:
                if pre_name not in map_dict:
                    map_dict[pre_name] = 1
                    map_pos_dict[pre_name] = [bbox]
                else:
                    map_dict[pre_name] += 1
                    map_pos_dict[pre_name].append(bbox)
    return map_dict, map_pos_dict


def calculate_score(cfg, predictions, datasets, dataset_name, threshold):
    cfg.clip, cfg.preprocess = prepare_clip("ckpt/ViT-L-14.pt")

    dataid_list = []
    if dataset_name == "bacon benchmark":
        with open(cfg.plan.bacondata_imageid, "r") as inputfile:
            for line in inputfile:
                dataid_list.append(int(line))
    elif dataset_name == "coco":
        with open(cfg.plan.coco_imageid, "r") as inputfile:
            for line in inputfile:
                dataid_list.append(int(line))
        label_list = []
        for cate_id in datasets.categories:
            label_list.append(datasets.categories[cate_id])
        datasets = datasets.dataset
        label_feature_dict = {}
        for label in label_list:
            feature = extract_clip_feature(label, cfg.clip, cfg.preprocess, mode='t')
            label_feature_dict[label] = feature

    precision_score = 0
    recall_score = 0
    iou_score = 0
    total_sample_number = 0
    total_iou_number = 0
    for image_id in tqdm(dataid_list):
        if image_id in datasets and image_id in predictions["layoutgpt"]:
            pre = predictions["layoutgpt"][image_id]["name"]
            gt = datasets[int(image_id)]["name"]
            positions = datasets[int(image_id)]["position"]
            pre_positions = predictions["layoutgpt"][image_id]["bbox"]
            new_gt = [drop_index(item) for item in gt]
            gt_dict = {}
            gt_pos_dict = {}
            for obj_idx in range(len(new_gt)):
                obj_name = new_gt[obj_idx]
                if obj_name not in gt_dict:
                    gt_dict[obj_name] = 1
                    gt_pos_dict[obj_name] = [positions[obj_idx]]
                else:
                    gt_dict[obj_name] += 1
                    gt_pos_dict[obj_name].append(positions[obj_idx])
            if dataset_name == "bacon benchmark":
                map_dict, map_pos_dict = map_function(cfg, pre, new_gt, gt_dict, pre_positions, threshold)
            elif dataset_name == "coco":
                map_dict, map_pos_dict = map_function(cfg, pre, label_list, gt_dict, pre_positions, threshold, label_feature_dict=label_feature_dict)

            for obj_name in map_pos_dict:
                if obj_name in gt_pos_dict:
                    pre_pos = map_pos_dict[obj_name]
                    gt_pos = gt_pos_dict[obj_name]
                    for pos1 in pre_pos:
                        for pos2 in gt_pos:
                            iou_score += compute_iou(pos1, pos2)
                            total_iou_number += 1

            gt_number = len(new_gt)
            correct_number = 0
            if dataset_name == "bacon benchmark":
                for obj_name in map_dict:
                    correct_number += min(map_dict[obj_name], gt_dict[obj_name])
                predict_number = len(pre)
            elif dataset_name == "coco":
                predict_number = 0
                for obj_name in map_dict:
                    if obj_name in gt_dict:
                        correct_number += min(map_dict[obj_name], gt_dict[obj_name])
                    predict_number += map_dict[obj_name]
            if predict_number > 0:
                precision_score += (correct_number / predict_number)
            recall_score += (correct_number / gt_number)
            total_sample_number += 1
    
    precision = precision_score / total_sample_number
    recall = recall_score / total_sample_number
    IOU = iou_score / total_iou_number
    print (f"The precision of layoutgpt on {dataset_name} is {precision}, the recall is {recall}, IOU is {IOU}")

    precision_score = 0
    recall_score = 0
    iou_score = 0
    total_sample_number = 0
    total_iou_number = 0
    for image_id in tqdm(dataid_list):
        if image_id in datasets and image_id in predictions["bacon"]:
            pre = predictions["bacon"][image_id]["name"]
            gt = datasets[int(image_id)]["name"]
            positions = datasets[int(image_id)]["position"]
            pre_positions = predictions["bacon"][image_id]["bbox"]
            new_gt = [drop_index(item) for item in gt]
            gt_dict = {}
            gt_pos_dict = {}
            for obj_idx in range(len(new_gt)):
                obj_name = new_gt[obj_idx]
                if obj_name not in gt_dict:
                    gt_dict[obj_name] = 1
                    gt_pos_dict[obj_name] = [positions[obj_idx]]
                else:
                    gt_dict[obj_name] += 1
                    gt_pos_dict[obj_name].append(positions[obj_idx])
            if dataset_name == "bacon benchmark":
                map_dict, map_pos_dict = map_function(cfg, pre, new_gt, gt_dict, pre_positions, threshold)
            elif dataset_name == "coco":
                map_dict, map_pos_dict = map_function(cfg, pre, label_list, gt_dict, pre_positions, threshold, label_feature_dict=label_feature_dict)

            iou = 0
            iou_count = 0
            for obj_name in map_pos_dict:
                if obj_name in gt_pos_dict:
                    pre_pos = map_pos_dict[obj_name]
                    gt_pos = gt_pos_dict[obj_name]
                    for pos1 in pre_pos:
                        for pos2 in gt_pos:
                            iou_score += compute_iou(pos1, pos2)
                            total_iou_number += 1

            gt_number = len(new_gt)
            correct_number = 0
            if dataset_name == "bacon benchmark":
                for obj_name in map_dict:
                    correct_number += min(map_dict[obj_name], gt_dict[obj_name])
                predict_number = len(pre)
            elif dataset_name == "coco":
                predict_number = 0
                for obj_name in map_dict:
                    if obj_name in gt_dict:
                        correct_number += min(map_dict[obj_name], gt_dict[obj_name])
                    predict_number += map_dict[obj_name]
            if predict_number > 0:
                precision_score += (correct_number / predict_number)
            recall_score += (correct_number / gt_number)
            total_sample_number += 1
    
    precision = precision_score / total_sample_number
    recall = recall_score / total_sample_number
    IOU = iou_score / total_iou_number
    print (f"The precision of bacon on {dataset_name} is {precision}, the recall is {recall}, IOU is {IOU}")


def main(cfg):

    cfg.clip, cfg.preprocess = prepare_clip(cfg.clip_path)

    coco_benchmark = CoCoplan(cfg.plan.coco_annotations)
    coco_benchmark.get_gt()

    predictions = get_predictions(cfg, "coco", cfg.plan.coco_image_dir)
    calculate_score(cfg, predictions, coco_benchmark, "coco", cfg.plan.coco_threshold)

    bacon_benchmark = Bacon_benchmark(cfg.test_benchmark_path)
    bacon_benchmark.organize_dataset()
    bacon_benchmark.statistical_info()
    bacon_benchmark_plan = preprocess_dataset_for_plan(cfg, bacon_benchmark)

    predictions = get_predictions(cfg, "bacon benchmark", cfg.plan.bacondata_image_dir)
    calculate_score(cfg, predictions, bacon_benchmark_plan, "bacon benchmark", cfg.plan.bacondata_threshold)
