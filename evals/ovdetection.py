
import json
import os
import torch
from tqdm import tqdm
from PIL import Image
from easydict import EasyDict
from bacon_modules.utils.dataloader import Bacon_benchmark
from bacon_modules.utils.tools import prepare_clip, extract_clip_feature


def preprocess_dataset_for_ovd(cfg, dataset):
    ground_truths = {}
    for image_id in tqdm(dataset):
        data_obj = dataset[image_id]
        organized_caption = data_obj["organized_caption"]
        object_list = organized_caption["object_list"]
        ground_truths[image_id] = {}
        for obj in object_list:
            obj_name = obj["name"]
            obj_bbox = obj["position"]
            obj_clip_feature = extract_clip_feature(obj_name, cfg.clip, cfg.preprocess, mode='t')
            ground_truths[image_id][obj_name] = {}
            ground_truths[image_id][obj_name]["bbox"] = obj_bbox
            ground_truths[image_id][obj_name]["clip_feature"] = obj_clip_feature

    return ground_truths


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


def get_predictions(cfg):
    predictions = {}
    next_chat_predictions = {}
    bacon_predictions = {}
    grounding_dino_predictions = {}
    glamm_predictions = {}
    kosmos_predictions = {}
    devit_predictions = {}
    ovdquo_predictions = {}

    with open(cfg.bacon_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"].split('/')[-1].split(".")[0])
            object_list = data_obj["caption"]["object_list"]
            for obj in object_list:
                if "position" in obj:
                    name = obj["name"]
                    position = obj["position"]

                if image_id not in bacon_predictions:
                    bacon_predictions[image_id] = {}
                bacon_predictions[image_id][name] = position

    with open(cfg.glamm_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"].split(".")[0])
            object_list = data_obj["phrases"]
            bboxes = data_obj["bboxes"]
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in glamm_predictions:
                    glamm_predictions[image_id] = {}
                glamm_predictions[image_id][obj_name] = position

    with open(cfg.kosmos_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"])
            object_list = data_obj["phrases"]
            bboxes = data_obj["bboxes"]
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in kosmos_predictions:
                    kosmos_predictions[image_id] = {}
                kosmos_predictions[image_id][obj_name] = position

    with open(cfg.next_chat_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"])
            object_list = data_obj["phrases"]
            try:
                bboxes = data_obj["bboxes"]
            except:
                bboxes = data_obj["bboxs"]
            if bboxes is None or len(object_list) == 0:
                continue
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in next_chat_predictions:
                    next_chat_predictions[image_id] = {}
                next_chat_predictions[image_id][obj_name] = position
    
    with open(cfg.grounding_dino_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"].split("/")[-1].split(".")[0])
            object_list = data_obj["phrases"]
            bboxes = data_obj["boxes"]
            if bboxes is None or len(object_list) == 0:
                continue
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in grounding_dino_predictions:
                    grounding_dino_predictions[image_id] = {}
                grounding_dino_predictions[image_id][obj_name] = position

    with open(cfg.devit_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"])
            object_list = data_obj["phrases"]
            bboxes = data_obj["bboxes"]
            if bboxes is None or len(object_list) == 0:
                continue
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in devit_predictions:
                    devit_predictions[image_id] = {}
                if obj_name not in devit_predictions[image_id]:
                    devit_predictions[image_id][obj_name] = []
                devit_predictions[image_id][obj_name].append(position)

    with open(cfg.ovdquo_result, "r") as inputfile:
        for line in inputfile:
            data_obj = json.loads(line)
            image_id = int(data_obj["path"])
            object_list = data_obj["phrases"]
            bboxes = data_obj["bboxes"]
            if bboxes is None or len(object_list) == 0:
                continue
            for obj_name, position in zip(object_list, bboxes):
                if image_id not in ovdquo_predictions:
                    ovdquo_predictions[image_id] = {}
                if obj_name not in ovdquo_predictions[image_id]:
                    ovdquo_predictions[image_id][obj_name] = []
                ovdquo_predictions[image_id][obj_name].append(position)

    predictions["next_chat"] = next_chat_predictions
    predictions["bacon"] = bacon_predictions
    predictions["devit"] = devit_predictions
    predictions["glamm"] = glamm_predictions
    predictions["grounding_dino"] = grounding_dino_predictions
    predictions["kosmos"] = kosmos_predictions
    predictions["ovdquo"] = ovdquo_predictions

    return predictions


def calculate_score(cfg, predictions_dict, ground_truths):

    for method in predictions_dict:
        predictions = predictions_dict[method]

        total_accuracy = 0
        total_recall = 0
        total_iou = 0
        total_number = 0
        total_iou_number = 0
        for image_id in tqdm(predictions):

            if image_id not in ground_truths:
                continue
            
            total_number += 1
            obj_number = len(list(ground_truths[image_id].keys()))
            correct_number = 0
            prediction_number = 0
            iou = 0
            iou_count = 0

            already_matched = []
            for predict_obj_name in predictions[image_id]:
                prediction_number += 1

                predict_bbox = predictions[image_id][predict_obj_name]
                if method == "bacon":
                    pre_x_min, pre_y_min, pre_x_max, pre_y_max = predict_bbox
                    predict_bbox = [pre_y_min, pre_x_min, pre_y_max, pre_x_max]
                elif method == "grounding_dino":
                    image_path = os.path.join(cfg.image_dir, f"{image_id:012d}.jpg")
                    image = Image.open(image_path)
                    h, w = image.size
                    center_x, center_y, w_, h_ = predict_bbox
                    predict_bbox = [(center_y - 0.5 * h_) * w, (center_x - 0.5 * w_) * h, (center_y + 0.5 * h_) * w, (center_x + 0.5 * w_) * h]
                elif method == "kosmos" or method == "next_chat":
                    image_path = os.path.join(cfg.image_dir, f"{image_id:012d}.jpg")
                    image = Image.open(image_path)
                    w, h = image.size
                    pre_x_min, pre_y_min, pre_x_max, pre_y_max = predict_bbox
                    predict_bbox = [pre_y_min * h, pre_x_min * w, pre_y_max * h, pre_x_max * w]
                elif method == "devit":
                    new_predict_bbox = []
                    for pb in predict_bbox:
                        x_min, y_min, w, h = pb
                        new_predict_bbox.append([y_min, x_min, y_min + h, x_min + w])
                    predict_bbox = new_predict_bbox
                elif method == "ovdquo":
                    new_predict_bbox = []
                    for pb in predict_bbox:
                        x_min, y_min, w, h = pb
                        new_predict_bbox.append([y_min, x_min, y_min + h, x_min + w])
                    predict_bbox = new_predict_bbox

                predict_obj_name_feature = extract_clip_feature(predict_obj_name, cfg.clip, cfg.preprocess, mode='t')
                
                for gt_obj_name in ground_truths[image_id]:
                    gt_obj = ground_truths[image_id][gt_obj_name]
                
                    gt_bbox = gt_obj["bbox"]

                    gt_obj_name_feature = gt_obj["clip_feature"].to(predict_obj_name_feature.device)
                    obj_name_score = predict_obj_name_feature @ gt_obj_name_feature.T
                    if method == "devit" or method == "ovdquo":
                        bbox_score = 0
                        for pb in predict_bbox:
                            score_ = compute_iou(pb, gt_bbox)
                            if score_ > bbox_score:
                                bbox_score = score_
                    else:
                        bbox_score = compute_iou(predict_bbox, gt_bbox)

                    if obj_name_score > cfg.ovdetection.clip_threshold and bbox_score > cfg.ovdetection.bbox_threshold and gt_obj_name not in already_matched:
                        correct_number += 1
                        iou += bbox_score
                        already_matched.append(gt_obj_name)
                        iou_count += 1

                        break

            total_accuracy += float(correct_number / prediction_number)
            total_recall += float(correct_number / obj_number)
            if correct_number > 0:
                total_iou += float(iou / iou_count)

                total_iou_number += 1

        accuracy_score = total_accuracy / total_number
        recall_score = total_recall / total_number
        iou_score = total_iou / total_iou_number
        
        print (f"{method}: the Accuracy is {accuracy_score}, the Recall is {recall_score}, the mIOU is {iou_score}.")


def main(cfg):

    cfg.clip, cfg.preprocess = prepare_clip(cfg.clip_path)

    bacon_benchmark = Bacon_benchmark(cfg.test_benchmark_path)
    bacon_benchmark.organize_dataset()

    print ("Preprocessing test benchmark for open-vocabulary detection task ...")
    ground_truths = preprocess_dataset_for_ovd(cfg, bacon_benchmark.dataset)
    torch.save(ground_truths, cfg.cache_path)
    # ground_truths = torch.load(cfg.cache_path)
    print ("Geting predictions ...")
    predictions = get_predictions(cfg.ovdetection)

    print ("Calculating the score ...")
    calculate_score(cfg, predictions, ground_truths)