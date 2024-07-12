import json
import os
from PIL import Image
from tqdm import tqdm
from bacon_modules.utils.tools import organize_caption


class Bacon_benchmark:
    def __init__(self, dataset_path, segmentation_path=None):
        self.dataset_path = dataset_path
        self.segmentation_path = segmentation_path
    
    def organize_dataset(self):
        self.dataset = {}
        with open(self.dataset_path, "r") as inputfile:
            for line in inputfile:
                data_obj = json.loads(line)
                image_id = data_obj["path"]
                image_path = f"data/coco2017_test/{image_id:012d}.jpg"
                image = Image.open(image_path)
                h, w = image.size
                text_caption = data_obj["caption"]
                organized_caption = organize_caption(text_caption, w, h)

                self.dataset[image_id] = {}
                self.dataset[image_id]["text_caption"] = text_caption
                self.dataset[image_id]["organized_caption"] = organized_caption

    def statistical_info(self):
        image_count = 0
        object_count = 0
        relationship_count = 0
        for image_id in self.dataset:
            image_count += 1
            data_obj = self.dataset[image_id]["organized_caption"]
            for obj in data_obj["object_list"]:
                object_count += 1
            for rel in data_obj["relationships"]:
                relationship_count += 1
        print (f"This dataset has {image_count} images, {object_count} objects, and {relationship_count} relationships")


class VG:
    def __init__(self, image_dir, question_answer_file, scene_graphs_file, bbox_file, object_file, relationships_file):
        self.image_dir = image_dir
        self.question_answer_file = question_answer_file
        self.scene_graphs_file = scene_graphs_file
        self.bbox_file = bbox_file
        self.object_name_list = []
        self.relationships_name_list = []
        with open(object_file, "r") as file:
            for line in file:
                self.object_name_list.append(line.replace("\n", ""))
        with open(relationships_file, "r") as file:
            for line in file:
                self.relationships_name_list.append(line.replace("\n", ""))

    def organize_data(self):

        self.attributes_dict = {}
        with open(self.bbox_file, 'r') as file:
            for line in file:
                data_list = json.loads(line) 
                for data_obj in data_list:
                    image_id = data_obj["image_id"]
                    attributes = data_obj["attributes"]
                    self.attributes_dict[image_id] = attributes

        self.sg_data_dict = {}
        self.all_triplet_count = 0
        relationships_dict = {}
        objects_dict = {}
        with open(self.scene_graphs_file, 'r') as file:
            for line in file:
                data_list = json.loads(line)
                for data in tqdm(data_list):

                    position_dict = {}
                    attributes = self.attributes_dict[data["image_id"]]
                    for attri in attributes:
                        position_dict[attri["object_id"]] = (attri["x"], attri["y"], attri["x"] + attri["w"], attri["y"] + attri["h"])

                    data_obj = []
                    object_dict = {}
                    for obj in data["objects"]:
                        object_id = obj["object_id"]
                        object_name = obj["names"][0]
                        if object_name.lower() not in objects_dict:
                            objects_dict[object_name.lower()] = 1
                        else:
                            objects_dict[object_name.lower()] += 1
                        object_dict[object_id] = object_name
                    for relation in data["relationships"]:
                        subject_id = relation["subject_id"]
                        object_id = relation["object_id"]
                        if relation["predicate"].lower() not in relationships_dict:
                            relationships_dict[relation["predicate"].lower()] = 1
                        else:
                            relationships_dict[relation["predicate"].lower()] += 1
                        triplet = (object_dict[subject_id], relation["predicate"], object_dict[object_id], position_dict[subject_id], position_dict[object_id])
                        data_obj.append(triplet)
                        self.all_triplet_count += 1
                    self.sg_data_dict[data["image_id"]] = data_obj

        # top_objects = self.visualize_common_object_relation(objects_dict, 150)
        # top_relationships = self.visualize_common_object_relation(relationships_dict, 50)
        
    def remove_open_vocabulary_data(self):
        self.nov_sg_data_dict = {}
        self.ov_sg_data_dict = {}
        self.nov_triplet_count = 0
        self.ov_triplet_count = 0
        self.total_number = 0
        for data in self.sg_data_dict:
            new_data = []
            new_data_2 = []
            for triplet in self.sg_data_dict[data]:
                self.total_number += 1
                if triplet[0] in self.object_name_list and triplet[2] in self.object_name_list and triplet[1] in self.relationships_name_list:
                    new_data.append(triplet)
                    self.nov_triplet_count += 1
                else:
                    new_data_2.append(triplet)
                    self.ov_triplet_count += 1
            self.nov_sg_data_dict[data] = new_data
            self.ov_sg_data_dict[data] = new_data_2

        # print (self.total_number, self.nov_triplet_count, self.ov_triplet_count)
        return

    def visualize_common_object_relation(self, my_dict, k=100):
        sorted_items = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)
        top_k_keys = [item[0] for item in sorted_items[:k]]

        return top_k_keys

    def visualize_bbox(self, obj, save_path, n=5):
        image_id = obj["image_id"]
        attributes = obj["attributes"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image_pil = Image.open(image_path)

        colors = generate_distinct_colors(len(attributes))
        length_split = len(attributes) // n
        for split in range(n):
            if split == n - 1:
                split_attributes = attributes[split * length_split : ]
            else:
                split_attributes = attributes[split * length_split : (split + 1) * length_split]
            image = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
            for idx in range(len(split_attributes)):
                item = split_attributes[idx]
                x_min, y_min, x_max, y_max = item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), tuple(colors[idx]), 10)
                obj_name = item["names"][0]
                cv2.putText(image, obj_name, (x_min, y_max), cv2.FONT_HERSHEY_SIMPLEX, 3, tuple(colors[idx]), 8)

            cv2.imwrite(save_path.format(split), image)


class LookTwiceQA:
    def __init__(self, question_answer_file, image_dir):
        self.question_answer_file = question_answer_file
        self.image_dir = image_dir

    def get_qa_list(self):
        self.qa_list = []

        image_path_list = os.listdir(self.image_dir)
        with open(self.question_answer_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                genome_id = data["genome_id"]
                image_path = os.path.join(self.image_dir, f"{genome_id}.jpg")
                question = data["question"]
                bbox = data["bbox"]
                answer = data["answer"]

                data_obj = {}
                data_obj["image_path"] = image_path
                data_obj["question"] = question
                data_obj["bbox"] = bbox
                data_obj["answer"] = answer
                data_obj["size"] = (data["img_h"], data["img_w"])
                data_obj["type"] = "normal"

                self.qa_list.append(data_obj)


class V7W:
    def __init__(self, question_answer_file, image_dir):
        self.question_answer_file = question_answer_file
        self.image_dir = image_dir

    def get_qa_list(self):
        self.qa_list = []

        image_path_list = os.listdir(self.image_dir)
        with open(self.question_answer_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                genome_id = data["genome_id"]
                image_path = os.path.join(self.image_dir, f"{genome_id}.jpg")
                question = data["question"]
                candidates = data["candidates"]
                answer = data["answer"]
                prompt = question.replace("Which", "The").replace("?", "")

                data_obj = {}
                data_obj["image_path"] = image_path
                data_obj["question"] = question
                data_obj["prompt"] = prompt
                data_obj["candidates"] = candidates
                data_obj["answer"] = answer
                data_obj["size"] = (data["img_h"], data["img_w"])
                data_obj["type"] = "normal"

                self.qa_list.append(data_obj)


class NLVR2:
    def __init__(self, question_answer_file, image_dir):
        self.question_answer_file = question_answer_file
        self.image_dir = image_dir

    def get_qa_list(self):
        self.qa_list = []

        image_path_list = os.listdir(self.image_dir)
        with open(self.question_answer_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                validation = data["validation"]
                answer = list(validation.values())[0]
                image_path = os.path.join(self.image_dir, ('-'.join(data["identifier"].split('-')[:-1]) + "-img" + data["identifier"].split('-')[-1] + ".png"))

                if image_path.split('/')[-1] in image_path_list:
                    data_obj = {}
                    data_obj["image_path"] = image_path
                    data_obj["question"] = data["sentence"]
                    answers = {}
                    answers[answer] = 1
                    data_obj["answers"] = answers
                    data_obj["type"] = "normal"

                    self.qa_list.append(data_obj)


class OK_VQA:
    def __init__(self, question_file, answer_file, image_dir, image_path):
        self.question_file = question_file
        self.answer_file = answer_file
        self.image_dir = image_dir
        self.image_path = image_path

    def get_qa_list(self):
        self.qa_list = []

        questions_dict = {}
        with open(self.question_file, 'r') as file:
            data = json.load(file)
            for item in data["questions"]:
                image_id = item["image_id"]
                question_id = item["question_id"]
                question = item["question"]
                questions_dict[question_id] = question


        with open(self.answer_file, 'r') as file:
            data = json.load(file)
            for item in data["annotations"]:
                data_obj = {}
                question_id = item["question_id"]
                image_id = item["image_id"]
                confidence = item["confidence"]
                question = questions_dict[question_id]
                answers = {}
                for ans in item["answers"]:
                    if ans["answer"] not in answers:
                        answers[ans["answer"]] = 1 / len(item["answers"])
                    else:
                        answers[ans["answer"]] += 1 / len(item["answers"])

                data_obj["image_path"] = self.image_path.format(f"{image_id:012d}")
                data_obj["question"] = question
                data_obj["answers"] = answers
                data_obj["type"] = "normal"
                data_obj["image_id"] = image_id
                data_obj["question_id"] = question_id
                data_obj["confidence"] = confidence

                self.qa_list.append(data_obj)


class VQAv1:
    def __init__(self, question_answer_file, image_dir):
        self.question_answer_file = question_answer_file
        self.image_dir = image_dir

    def get_qa_list(self):
        self.qa_list = []

        with open(self.question_answer_file, 'r') as file:
            for line in tqdm(file):
                data = json.loads(line)
                image_path = os.path.join(self.image_dir, data["file_path"].split("/")[-1])
                question = data["question"]
                answer_type = data["answer_type"]

                answers = {}
                for ans in data["answers"]:
                    if ans not in answers:
                        answers[ans] = 1 / len(data["answers"])
                    else:
                        answers[ans] += 1 / len(data["answers"])

                data_obj = {}
                data_obj["image_path"] = image_path
                data_obj["question"] = question
                data_obj["answers"] = answers
                data_obj["type"] = answer_type

                self.qa_list.append(data_obj)


class VQAv2:
    def __init__(self, question_answer_file, image_dir):
        self.question_answer_file = question_answer_file
        self.image_dir = image_dir

    def get_qa_list(self):
        self.qa_list = []

        with open(self.question_answer_file, 'r') as file:
            for line in tqdm(file):
                data = json.loads(line)
                image_path = data["image_path"].split("/")[-1]
                question = data["question"]

                data_obj = {}
                data_obj["image_path"] = image_path
                data_obj["question"] = question
                data_obj["type"] = "normal"
                data_obj["question_id"] = data["question_id"]

                self.qa_list.append(data_obj)


class CoCoplan:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file

    def get_gt(self):
        self.dataset = {}
        self.categories = {}
        with open(self.annotation_file, "r") as file:
            data = json.load(file)
            for cate_obj in data["categories"]:
                cate_id = cate_obj["id"]
                cate_name = cate_obj["name"]
                self.categories[cate_id] = cate_name
            for data_obj in data["images"]:
                image_id = int(data_obj["file_name"].split(".")[0])
                self.dataset[image_id] = {}
                self.dataset[image_id]["name"] = []
                self.dataset[image_id]["position"] = []
            for data_obj in data["annotations"]:
                bbox = data_obj["bbox"]
                x_min, y_min, w_bbox, h_bbox = bbox
                bbox_ = (x_min, y_min, x_min + w_bbox, y_min + h_bbox)
                image_id = int(data_obj["image_id"])
                cate_id = int(data_obj["category_id"])
                obj_name = self.categories[cate_id]
                self.dataset[image_id]["name"].append(obj_name)
                self.dataset[image_id]["position"].append(bbox_)
        self.fixed_dataset = {}
        for image_id in self.dataset:
            data_obj = self.dataset[image_id]
            if len(list(data_obj.keys())) > 0:
                self.fixed_dataset[image_id] = self.dataset[image_id]        