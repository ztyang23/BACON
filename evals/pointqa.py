import json
import os
from tqdm import tqdm
from bacon_modules.utils.dataloader import LookTwiceQA
from bacon_modules.utils.tools import prepare_llava, llava_inference


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


def extract_caption_aligned_with_bbox_bacon(organized_caption, bbox, threshold = 0.3):
    region_caption = ""
    for obj in organized_caption["object_list"]:
        if "position" in obj:
            if compute_iou(obj["position"], bbox) > threshold:
                name = obj["name"]
                description = ".".join(obj["description"]["content"])
                try:
                    color = ".".join(obj["color"]["content"])
                except:
                    pass
                region_caption += (f"There is a {name}" + description + f"The color information is {color}")
    if region_caption == "":
        region_caption += ".".join(organized_caption["overall_description"]["background_global_description"]["content"])
        region_caption += ".".join(organized_caption["overall_description"]["foreground_global_description"]["content"])
        for obj in organized_caption["object_list"]:
            region_caption += "There is a" + obj["name"]
            region_caption += ".".join(obj["description"]["content"])
            try:
                region_caption += ".".join(obj["color"]["content"])
            except:
                pass

    return region_caption


def extract_caption_aligned_with_bbox_others(caption, position_list, description_list, bbox, threshold = 0.3):
    region_caption = ""
    for idx in range(len(position_list)):
        if compute_iou(position_list[idx], bbox) > threshold:
            name = description_list[idx]
            region_caption += (f"There is a {name}")
    if region_caption == "":
        region_caption = caption

    return region_caption


def obtain_vqa_answer_score(cfg, dataset, result_file, question_format_dict, mode):
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
                elif mode == "glamm":
                    image_path = os.path.join(dataset.image_dir, image_path)
                result = data["caption"]
                result_dict[image_path] = result
                position_list = []
                description_list = []
                for idx in range(len(data["phrases"])):
                    position_list.append(data["bboxes"][idx])
                    description = data["phrases"][idx]
                    description_list.append(description)
                position_description_dict[image_path] = (position_list, description_list)

    total_score = 0
    total_number = 0
    prediction_dict = {}
    for qa_obj in tqdm(dataset.qa_list):
        image_path = qa_obj["image_path"]
        core_question = qa_obj["question"]
        answer = qa_obj["answer"]
        tp = qa_obj["type"]
        bbox = qa_obj["bbox"]
        question_format = question_format_dict[tp]
        if image_path in result_dict:
            if mode == "bacon":
                description = extract_caption_aligned_with_bbox_bacon(result_dict[image_path], bbox)
            else:
                position_list, description_list = position_description_dict[image_path]
                description = extract_caption_aligned_with_bbox_others(result_dict[image_path], position_list, description_list, bbox)
            question = question_format.format(description, core_question)
            try:
                key = image_path + "$$" + core_question

                prediction = llava_inference(question, None, cfg.llava_config, cfg.tokenizer, cfg.model, cfg.image_processor, "pil")
                prediction_dict[key] = prediction
                if answer.lower() in prediction.lower():
                    total_score = total_score + 1
                total_number += 1
            except:
                print ("fail on this case")

    score = total_score / total_number
    print (f"The accuracy of {mode} is {score}")

    return score


def main(cfg):

    cfg.tokenizer, cfg.model, cfg.image_processor = prepare_llava(cfg.llava_config)

    looktwiceqa_dataset = LookTwiceQA(cfg.pointqa.question_answer_file, cfg.pointqa.image_dir)
    looktwiceqa_dataset.get_qa_list()

    obtain_vqa_answer_score(cfg, looktwiceqa_dataset, cfg.pointqa.bacon_path, cfg.pointqa.question_format, "bacon")
    obtain_vqa_answer_score(cfg, looktwiceqa_dataset, cfg.pointqa.glamm_path, cfg.pointqa.question_format, "glamm")
    obtain_vqa_answer_score(cfg, looktwiceqa_dataset, cfg.pointqa.next_chat_path, cfg.pointqa.question_format, "next")
    obtain_vqa_answer_score(cfg, looktwiceqa_dataset, cfg.pointqa.kosmos_path, cfg.pointqa.question_format, "kosmos")