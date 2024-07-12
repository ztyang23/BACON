import json
import os
import torch
import ast
from tqdm import tqdm
from bacon_modules.utils.dataloader import VG, Bacon_benchmark
from bacon_modules.utils.tools import prepare_clip, extract_clip_feature


def convert_bounding_box_xyxy_yxyx(bbox):
    x_min, y_min, x_max, y_max = bbox
    bbox = (y_min, x_min, y_max, x_max)
    return bbox


def preprocess_dataset_for_ovsgg(cfg, dataset):
    all_triplet_count = 0
    fail_count = 0
    dataset_sgg = {}
    for image_id in tqdm(dataset.dataset):
        data_obj = dataset.dataset[image_id]
        organized_caption = data_obj["organized_caption"]
        position_dict = {}
        for obj in organized_caption["object_list"]:
            obj_name = obj["name"].strip()
            position = obj["position"]
            position_dict[obj_name] = position
        
        relationships = organized_caption["relationships"]
        triplet_list = []
        for rel in relationships:
            if len(rel["nouns"]) == 2 and len(rel["relationship"]) == 1:
                sub, obj = rel["nouns"]
                sub, obj = sub.strip(), obj.strip()
                predicate = rel["relationship"][0]
                try:
                    pos1 = convert_bounding_box_xyxy_yxyx(position_dict[sub])
                    pos2 = convert_bounding_box_xyxy_yxyx(position_dict[obj])
                except:
                    fail_count += 1
                    continue
                triplet = (sub, predicate, obj, pos1, pos2)
                triplet_list.append(triplet)
                all_triplet_count += 1
            else:
                fail_count += 1
        dataset_sgg[image_id] = triplet_list
    # print (f"This dataset has {all_triplet_count} triplets")

    return dataset_sgg


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


def construct_predictions_for_bacondata(cfg, result_path, testset_file, mode):
    predictions = {}
    count = 0
    total_count = 0
    if mode == "bacon":
        with open(result_path, "r") as inputfile:
            for line in inputfile:
                data = json.loads(line)
                image_id = int(data["path"].split("/")[-1].split(".")[0])
                caption = data["caption"]
                try:
                    organized_caption = organize_caption(caption)
                except:
                    organized_caption = caption

                position_dict = {}
                for obj in organized_caption["object_list"]:
                    if "position" in obj.keys():
                        position_dict[obj["name"].strip()] = obj["position"]

                data_obj = []
                for item in organized_caption["relationship"]:
                    if len(item["nouns"]) == 2 and len(item["relationship"]) == 1:
                        sbj = item["nouns"][0]
                        obj = item["nouns"][1]
                        if sbj in position_dict and obj in position_dict:
                            pos1 = position_dict[sbj]
                            pos2 = position_dict[obj]
                            data_obj.append((sbj, item["relationship"][0], obj, pos1, pos2))
                            count += 1
                        total_count += 1
                    elif len(item["nouns"]) > 2 and len(item["relationship"]) == 1:
                        sbj_list = []
                        obj_list = []
                        for idx in range(len(item["nouns"])):
                            if item["nouns"][idx] in item["content"].split(item["relationship"][0])[0]:
                                sbj_list.append(item["nouns"][idx].strip())
                            elif item["nouns"][idx] in item["content"].split(item["relationship"][0])[1]:
                                obj_list.append(item["nouns"][idx].strip())
                        for sbj in sbj_list:
                            for obj in obj_list:
                                if sbj in position_dict and obj in position_dict:
                                    pos1 = position_dict[sbj]
                                    pos2 = position_dict[obj]
                                    data_obj.append((sbj, item["relationship"][0], obj, pos1, pos2))
                                    count += 1
                                total_count += 1
                predictions[image_id] = data_obj
        # print ("total_count", total_count)

    else:
        with open(testset_file, "r") as inputfile:
            input_file = json.load(inputfile)
            test_image_ids = input_file["test_image_ids"]
            thing_classes = input_file["thing_classes"]
            predicate_classes = input_file["predicate_classes"]

        work_dir = result_path

        triplet_path = os.path.join(work_dir, "pred.json")
        position_path = os.path.join(work_dir, "pred_bbox.json")
        name_path = os.path.join(work_dir, "names.pt")

        triplet_list = []
        with open(triplet_path, "r") as f:
            for line in f:
                triplet_list.append(ast.literal_eval(line))

        position_list = []
        with open(position_path, "r") as f:
            for line in f:
                position_list.append(ast.literal_eval(line))

        names = torch.load(name_path)

        predictions = {}
        for idx in range(len(triplet_list)):
            try:
                image_id = int(names[idx].split(".")[0])
                data_obj = []
                assert len(triplet_list[idx]) == len(position_list[idx])
                for idx_j in range(len(triplet_list[idx])):
                    total_count += 1
                    try:
                        triplet = triplet_list[idx][idx_j]
                        bbox = position_list[idx][idx_j]
                        x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2 = bbox
                        pos1 = (x1_1, y1_1, x2_1, y2_1)
                        pos2 = (x1_2, y1_2, x2_2, y2_2)
                        sbj, rel, obj = triplet
                        sbj = thing_classes[sbj]
                        obj = thing_classes[obj]
                        rel = predicate_classes[rel]
                        data_obj.append((sbj, rel, obj, pos1, pos2))
                        count += 1
                    except:
                        pass
            except:
                pass
            predictions[image_id] = data_obj

    key = list(predictions.keys())[0]

    return predictions


def construct_predictions_for_vg(cfg, result_path, testset_file, mode):
    predictions = {}
    count = 0
    total_count = 0
    if mode == "bacon":
        with open(result_path, "r") as inputfile:
            for line in inputfile:
                data = json.loads(line)
                image_id = int(data["path"].split("/")[-1].split(".")[0])
                caption = data["caption"]
                try:
                    organized_caption = organize_caption(caption)
                except:
                    organized_caption = caption

                position_dict = {}
                for obj in organized_caption["object_list"]:
                    if "position" in obj.keys():
                        position_dict[obj["name"].strip()] = obj["position"]

                data_obj = []
                for item in organized_caption["relationship"]:
                    if len(item["nouns"]) == 2 and len(item["relationship"]) == 1:
                        sbj = item["nouns"][0]
                        obj = item["nouns"][1]
                        if sbj in position_dict and obj in position_dict:
                            pos1 = position_dict[sbj]
                            pos2 = position_dict[obj]
                            data_obj.append((sbj, item["relationship"][0], obj, pos1, pos2))
                            count += 1
                        total_count += 1
                    elif len(item["nouns"]) > 2 and len(item["relationship"]) == 1:
                        sbj_list = []
                        obj_list = []
                        for idx in range(len(item["nouns"])):
                            if item["nouns"][idx] in item["content"].split(item["relationship"][0])[0]:
                                sbj_list.append(item["nouns"][idx].strip())
                            elif item["nouns"][idx] in item["content"].split(item["relationship"][0])[1]:
                                obj_list.append(item["nouns"][idx].strip())
                        for sbj in sbj_list:
                            for obj in obj_list:
                                if sbj in position_dict and obj in position_dict:
                                    pos1 = position_dict[sbj]
                                    pos2 = position_dict[obj]
                                    data_obj.append((sbj, item["relationship"][0], obj, pos1, pos2))
                                    count += 1
                                total_count += 1
                predictions[image_id] = data_obj

    else:
        with open(testset_file, "r") as inputfile:
            input_file = json.load(inputfile)
            test_image_ids = input_file["test_image_ids"]
            thing_classes = input_file["thing_classes"]
            predicate_classes = input_file["predicate_classes"]

        work_dir = result_path

        triplet_path = os.path.join(work_dir, "pred.json")
        position_path = os.path.join(work_dir, "pred_bbox.json")
        name_path = os.path.join(work_dir, "names.pt")

        triplet_list = []
        with open(triplet_path, "r") as f:
            for line in f:
                triplet_list.append(ast.literal_eval(line))

        position_list = []
        with open(position_path, "r") as f:
            for line in f:
                position_list.append(ast.literal_eval(line))

        names = torch.load(name_path)

        predictions = {}
        for idx in range(len(triplet_list)):
            try:
                image_id = int(names[idx].split(".")[0])
                data_obj = []
                assert len(triplet_list[idx]) == len(position_list[idx])
                for idx_j in range(len(triplet_list[idx])):
                    total_count += 1
                    try:
                        triplet = triplet_list[idx][idx_j]
                        bbox = position_list[idx][idx_j]
                        x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2 = bbox
                        pos1 = (x1_1, y1_1, x2_1, y2_1)
                        pos2 = (x1_2, y1_2, x2_2, y2_2)
                        sbj, rel, obj = triplet
                        sbj = thing_classes[sbj]
                        obj = thing_classes[obj]
                        rel = predicate_classes[rel]
                        data_obj.append((sbj, rel, obj, pos1, pos2))
                        count += 1
                    except:
                        pass
            except:
                pass
            predictions[image_id] = data_obj

    key = list(predictions.keys())[0]

    return predictions


def get_score(cfg, predictions, datasets, device, mode, dataset_name):

    with open(cfg.ovsgg.testset_file, "r") as inputfile:
        input_file = json.load(inputfile)
        test_image_ids = input_file["test_image_ids"]
    
    if dataset_name == "bacon test benchmark":
        test_image_ids = datasets

    if dataset_name == "bacon test benchmark":
        gt_dataset = datasets
    elif dataset_name == "vg dataset":
        gt_dataset = datasets.sg_data_dict

    correct_number = 0
    total_number = 0
    for image_id_ in tqdm(test_image_ids):
        image_id = int(image_id_)

        record_triplets = {}
        for triplet in gt_dataset[image_id]:
            obj1, rel, obj2, pos1, pos2 = triplet
            key = f"{obj1}_{rel}_{obj2}"
            if key in record_triplets:
                record_triplets[key] += 1
            else:
                record_triplets[key] = 1

        feature_dict = {}
        predicate_dict = {}
        pre_feature_dict = {}
        pre_predicate_dict = {}

        total_number += len(gt_dataset[image_id])
        if image_id not in predictions:
            continue
        for pred_triplet in predictions[image_id]:
            pre_obj1, pre_rel, pre_obj2, pre_pos1, pre_pos2 = pred_triplet
            for triplet in gt_dataset[image_id]:
                obj1, rel, obj2, pos1, pos2 = triplet
                record_key = f"{obj1}_{rel}_{obj2}"
                if record_triplets[record_key] > 0:
                    iou1, iou2 = compute_iou(pos1, pre_pos1), compute_iou(pos2, pre_pos2)
                    if iou1 > cfg.ovsgg.iou_threshold and iou2 > cfg.ovsgg.iou_threshold:
                        if obj1 not in feature_dict:
                            obj1_features = extract_clip_feature(obj1, cfg.model, cfg.preprocess, 't', device)
                            feature_dict[obj1] = obj1_features
                        else:
                            obj1_features = feature_dict[obj1]
                        if obj2 not in feature_dict:
                            obj2_features = extract_clip_feature(obj2, cfg.model, cfg.preprocess, 't', device)
                            feature_dict[obj2] = obj1_features
                        else:
                            obj2_features = feature_dict[obj2]
                        if rel not in predicate_dict:
                            rel_features = extract_clip_feature(rel, cfg.model, cfg.preprocess, 't', device)
                            feature_dict[rel] = rel_features
                        else:
                            rel_features = feature_dict[rel]

                        if pre_obj1 not in pre_feature_dict:
                            pre_obj1_features = extract_clip_feature(pre_obj1, cfg.model, cfg.preprocess, 't', device)
                            pre_feature_dict[pre_obj1] = pre_obj1_features
                        else:
                            pre_obj1_features = pre_feature_dict[pre_obj1]
                        if pre_obj2 not in pre_feature_dict:
                            pre_obj2_features = extract_clip_feature(pre_obj2, cfg.model, cfg.preprocess, 't', device)
                            pre_feature_dict[pre_obj2] = pre_obj2_features
                        else:
                            pre_obj2_features = pre_feature_dict[pre_obj2]
                        if pre_rel not in pre_predicate_dict:
                            pre_rel_features = extract_clip_feature(pre_rel, cfg.model, cfg.preprocess, 't', device)
                            pre_predicate_dict[pre_rel] = pre_rel_features
                        else:
                            pre_rel_features = pre_predicate_dict[pre_rel]
                        score_obj1 = obj1_features @ pre_obj1_features.T
                        score_obj2 = obj2_features @ pre_obj2_features.T
                        score_rel = rel_features @ pre_rel_features.T
                        if score_obj1 > cfg.ovsgg.clip_threshold and score_obj2 > cfg.ovsgg.clip_threshold and score_rel > cfg.ovsgg.clip_threshold:
                            record_triplets[record_key] -= 1
                            correct_number += 1
                            break

    print (f"The number of correct predictions for {mode} on {dataset_name} is {correct_number}")

    return


def main(cfg):

    cfg.device = "cuda:2"
    cfg.model, cfg.preprocess = prepare_clip(cfg.clip_path, cfg.device)

    bacon_benchmark = Bacon_benchmark(cfg.test_benchmark_path)
    bacon_benchmark.organize_dataset()
    bacon_benchmark.statistical_info()
    bacon_benchmark_sgg = preprocess_dataset_for_ovsgg(cfg, bacon_benchmark)

    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_bacon_path, cfg.ovsgg.testset_file, "bacon")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "bacon", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_gpsnet_dir, cfg.ovsgg.testset_file, "gpsnet")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "gpsnet", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_motifs_dir, cfg.ovsgg.testset_file, "motifs")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "motifs", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_psgformer_dir, cfg.ovsgg.testset_file, "psgformer")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "psgformer", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_psgtr_dir, cfg.ovsgg.testset_file, "psgtr")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "psgtr", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_vctree_dir, cfg.ovsgg.testset_file, "vctree")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "vctree", "bacon test benchmark")
    predictions = construct_predictions_for_bacondata(cfg, cfg.ovsgg.bacondata_imp_dir, cfg.ovsgg.testset_file, "imp")
    get_score(cfg, predictions, bacon_benchmark_sgg, cfg.device, "imp", "bacon test benchmark")
    
    vg_dataset = VG(cfg.ovsgg.image_dir, cfg.ovsgg.question_answer_file, cfg.ovsgg.scene_graphs_file, cfg.ovsgg.bbox_file, cfg.ovsgg.object_file, cfg.ovsgg.relationship_file)
    vg_dataset.organize_data()

    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_bacon_path, cfg.ovsgg.testset_file, "bacon")
    get_score(cfg, predictions, vg_dataset, cfg.device, "bacon", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_gpsnet_dir, cfg.ovsgg.testset_file, "gpsnet")
    get_score(cfg, predictions, vg_dataset, cfg.device, "gpsnet", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_motifs_dir, cfg.ovsgg.testset_file, "motifs")
    get_score(cfg, predictions, vg_dataset, cfg.device, "motifs", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_psgformer_dir, cfg.ovsgg.testset_file, "psgformer")
    get_score(cfg, predictions, vg_dataset, cfg.device, "psgformer", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_psgtr_dir, cfg.ovsgg.testset_file, "psgtr")
    get_score(cfg, predictions, vg_dataset, cfg.device, "psgtr", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_vctree_dir, cfg.ovsgg.testset_file, "vctree")
    get_score(cfg, predictions, vg_dataset, cfg.device, "vctree", "vg dataset")
    predictions = construct_predictions_for_vg(cfg, cfg.ovsgg.vg_imp_dir, cfg.ovsgg.testset_file, "imp")
    get_score(cfg, predictions, vg_dataset, cfg.device, "imp", "vg dataset")