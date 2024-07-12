import json
import os
import re
from easydict import EasyDict
from random import choice
from tqdm import tqdm
from unidecode import unidecode
from PIL import Image
from bacon_modules.utils.tools import organize_caption


def process_text(text):
    return unidecode(text).replace('\n', '').replace('\'', "'").replace('\"', "'").replace('\f', '').replace('/', ' or ').strip()
    

def build_dataset(cfg):

    fail_count = 0
    training_data_list = []
    with open(cfg.dataset_file, "r") as inputfile:
        for line in tqdm(inputfile):
            data_obj = json.loads(line)
            image_index = int(data_obj["path"].split(".")[0])
            image_path = os.path.join("data/training_data", data_obj["path"])
            image = Image.open(image_path)
            h, w = image.size
            try:
                organized_caption = organize_caption(data_obj["caption"], w, h)
            except:
                fail_count += 1
                continue

            data_train = {}
            data_train["id"] = f"{image_index:012d}"
            data_train["image"] = image_path
            data_train["conversations"] = []
            conver_obj = {"from": "human"}
            conver_obj["value"] = process_text(choice(cfg.question_list))
            data_train["conversations"].append(conver_obj)
            conver_obj = {"from": "gpt"}
            conver_obj["value"] = process_text(data_obj["caption"])
            data_train["conversations"].append(conver_obj)
            training_data_list.append(data_train)

    with open(cfg.save_path, 'w') as file:
        json.dump(training_data_list, file, indent=4)


if __name__ == "__main__":
    cfg = EasyDict(__name__='Config: Translate')

    cfg.image_dir = "data/training_data"
    cfg.dataset_file = "data/training_dataest.jsonl"
    cfg.save_path = "data/train.json"

    cfg.question_list = [
        "Kindly provide a detailed description of this image, ensuring that your response is methodically structured.",
        "I request a thorough analysis of this image with a meticulously organized answer format.",
        "Please give an in-depth depiction of this image and structure your explanation rigorously.",
        "Can you present a comprehensive elaboration of this image with a well-organized response?",
        "Offer a full description of this image and arrange your points in a systematic manner.",
        "I would appreciate a detailed examination of this image, with your observations structured neatly.",
        "Ensure you detail this image exhaustively and adhere to a strict organizational structure in your reply.",
        "Describe this image fully, and I expect your answer to be structured in a precise manner.",
        "Provide us with an elaborate analysis of this image, and be sure to structure your answer thoroughly.",
        "I'm looking for a detailed portrayal of this image, coupled with a rigorously structured response.",
        "It's important to describe this image in-depth while keeping your answer format highly organized.",
        "I urge you to delve into the details of this image and organize your findings methodically.",
        "Please offer a comprehensive insight into this image, and follow a strict structure in your description.",
        "Can you articulate the finer details of this image, ensuring a rigorous framework for your answer?",
        "I'd like you to meticulously outline this image and present your answer with clear structure.",
        "Please proceed to dissect this image in detail and construct your answer with precise organization.",
        "Could you explicate this image thoroughly and structure your explanation in an orderly way?",
        "Offer an in-depth review of this image, and make sure your presentation is systematically structured.",
        "Please narrate the specifics of this image, and maintain a strict arrangement in your reply.",
        "Describe every aspect of this image while organizing your answer in a highly structured format.",
        "Can you conduct a detailed survey of this image with a focus on rigorous answer structuring?",
        "Please compose a detailed illustration of this image, ensuring that your description is organized effectively.",
        "I would like a descriptive analysis of this image, with a well-structured and precise answer.",
        "Can you deliver a detailed depiction of this image and provide an organized and structured response?",
        "I'm requesting a finely detailed description of this image, with an emphasis on structured answer format.",
        "Please go into great detail about this image and structure your thoughts with great care and organization.",
        "Offer a thorough and intricate description of this image, while keeping your response tightly structured.",
        "Kindly paint a detailed picture of this image, arranging your insights in a systematic structure.",
        "It's essential to describe this image with detail and answer in a rigorously organized manner.",
        "Please provide a granular description of this image with a well-planned and structured analysis.",
        "I'm looking for an exhaustive description of this image, and your answer should be structured methodically.",
        "Can you provide a detailed narrative of this image, ensuring a logically structured presentation?",
        "Detailed description is required for this image, along with a strict and orderly structure in your answer.",
        "Elaborate on this image's details and please formulate your response with a strong structural approach.",
        "Describe this image in depth, and I would like you to organize your points in a structured and rigorous manner.",
        "Please expound on this image with extensive detail and maintain a rigid structure in your explanation.",
        "Can you dissect this image with precision and organize your answer in a thoroughly structured way?",
        "I'm interested in a detailed critique of this image, with a coherent and well-structured follow-up.",
        "Offer a descriptive exploration of this image, and be rigorous in your answer's structure.",
        "Please provide an elaborate dissection of this image, and ensure your response is structured systematically.",
        "Kindly go into detail about this image, organizing your response with a highly structured approach.",
        "Can you give a detailed account of this image, with a strict emphasis on the organization of your reply?",
        "Please articulate the elements of this image in detail, while structuring your answer with precision.",
        "I'd appreciate a comprehensive description of this image, with a rigorously structured analysis to follow.",
        "Can you outline this image in elaborate detail, and then arrange your answer in a disciplined structure?",
        "Offer an extensive rendition of this image and ensure that your response follows a strict structural pattern.",
        "Please analyze this image thoroughly, and I expect a meticulously organized structure in your explanation.",
        "Provide a full-scale detail of this image, with an answer that's organized in a rigorous and structured manner.",
        "I'd like an in-depth visual explanation of this image, with a subsequent answer that is well-structured and orderly.",
        "Can you present a richly detailed study of this image, while adhering to a tightly organized structure in your response?"
    ]

    build_dataset(cfg)