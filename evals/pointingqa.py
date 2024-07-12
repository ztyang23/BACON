import json
import os
import torch
import random
from tqdm import tqdm
from PIL import Image
from bacon_modules.utils.dataloader import V7W
from bacon_modules.utils.tools import prepare_clip, extract_clip_feature


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


def evaluate(cfg, dataset, result_file, mode):
    cfg.clip, cfg.preprocess = prepare_clip(cfg.clip_path)

    result_dict = {}
    position_description_dict = {}
    if mode == "bacon":
        with open(result_file, "r") as file:
            for line in tqdm(file):
                data = json.loads(line)
                image_path = data["path"]
                image_path = os.path.join(dataset.image_dir, image_path.split("/")[-1])
                result = data["caption"]
                result_dict[image_path] = result
                position_list = []
                description_list = []
                for obj in result["object_list"]:
                    if "position" in obj.keys():
                        position = obj["position"]
                        obj_name = obj["name"]
                        obj_description = '.'.join(obj["description"]["content"])
                        description = f"This is a {obj_name}. {obj_description}"
                        position_list.append(position)
                        description_list.append(description)
                position_description_dict[image_path] = (position_list, description_list)
    else:
        with open(result_file, "r") as file:
            for line in tqdm(file):
                data = json.loads(line)
                image_path = data["path"]
                if mode == "kosmos" or mode == "next":
                    image_path = os.path.join(dataset.image_dir, f"{image_path}.jpg")
                    image = Image.open(image_path)
                    w, h = image.size
                elif mode == "glamm":
                    image_path = os.path.join(dataset.image_dir, image_path)
                result = data["caption"]
                result_dict[image_path] = result
                position_list = []
                description_list = []
                for idx in range(len(data["phrases"])):
                    if mode == "kosmos" or mode == "next":
                        x_min, y_min, x_max, y_max = data["bboxes"][idx]
                        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
                        position_list.append((x_min, y_min, x_max, y_max))
                    else:
                       position_list.append(data["bboxes"][idx])
                    description = data["phrases"][idx]
                    description_list.append(description)
                position_description_dict[image_path] = (position_list, description_list)

    total_score = 0
    total_number = 0
    for qa_obj in tqdm(dataset.qa_list):
        image_path = qa_obj["image_path"]
        if image_path not in position_description_dict:
            continue
        position_list, description_list = position_description_dict[image_path]
        candidates = qa_obj["candidates"]
        prompt = qa_obj["prompt"]

        if cfg.align_method == "weighted_average":
            similarity_list = []
            prompt_feature = extract_clip_feature(prompt, cfg.clip, cfg.preprocess, mode='t')
            for description in description_list:
                des_feature = extract_clip_feature(description, cfg.clip, cfg.preprocess, mode='t')
                similarity_list.append(prompt_feature @ des_feature.T)

            max_similarity = 0
            selected_bbox = None
            for candidate in candidates:
                total_iou = 0
                total_similarity = 0
                for idx in range(len(position_list)):
                    bbox = position_list[idx]
                    iou = compute_iou(candidate, bbox) 
                    total_iou += iou
                    total_similarity += iou * similarity_list[idx]
                if total_iou == 0:
                    similarity = 0
                else:
                    similarity = total_similarity / total_iou
                if similarity > max_similarity:
                    max_similarity = similarity
                    selected_bbox = candidate

            answer = selected_bbox

        elif cfg.align_method == "random":
            answer = random.choice(candidates)

        if answer == qa_obj["answer"]:
            total_score += 1

        total_number += 1

    score = total_score / total_number
    if cfg.align_method == "weighted_average":
        print (f"The accuracy of {mode} is {score}")
    elif cfg.align_method == "random":
        print (f"The accuracy of random is {score}")

    return score


def main(cfg):

    cfg.clip, cfg.preprocess = prepare_clip(cfg.clip_path)

    v7w_dataset = V7W(cfg.pointingqa.question_answer_file, cfg.pointingqa.image_dir)
    v7w_dataset.get_qa_list()

    cfg.align_method = "random"
    evaluate(cfg, v7w_dataset, cfg.pointingqa.bacon_path, "bacon")
    cfg.align_method = "weighted_average"
    evaluate(cfg, v7w_dataset, cfg.pointingqa.bacon_path, "bacon")
    evaluate(cfg, v7w_dataset, cfg.pointingqa.glamm_path, "glamm")
    evaluate(cfg, v7w_dataset, cfg.pointingqa.next_chat_path, "next")
    evaluate(cfg, v7w_dataset, cfg.pointingqa.kosmos_path, "kosmos")
