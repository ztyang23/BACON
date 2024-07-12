import json
import random
import torch
import numpy as np
from easydict import EasyDict
from PIL import Image
from bacon_modules.utils.tools import prepare_clip, llava_inference, prepare_llava, llava_inference, organize_caption, get_grounding_config, init_model_grounding, grounding


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_question(cfg, description, edit_list, w, h):
    a = organize_caption(description, w, h)
    qs = cfg.description_instruction.format(description)
    qs += cfg.edit_instruction
    for idx in range(len(edit_list)):
        qs = qs + "\n"
        qs = qs + f"{idx}. {edit_list[idx]}"
    return qs


def main(cfg):
    cfg.tokenizer, cfg.model, cfg.image_processor = prepare_llava(cfg.captioner_config)
    cfg.llava_tokenizer, cfg.llava_model, cfg.llava_image_processor = prepare_llava(cfg.captioner_config)
    cfg = get_grounding_config(cfg)
    cfg.grounding_config.grounding_dino_model, cfg.grounding_config.sam_predictor = init_model_grounding(cfg.grounding_config, "cuda:0")

    output = []
    output_data = {}
    print ("********** Test Caption **********")
    print ("Obtain caption from captioner ...")
    captioner_caption = llava_inference(cfg.caption_instruction, cfg.image_path, cfg.captioner_config, cfg.tokenizer, cfg.model, cfg.image_processor, "path")
    image = Image.open(cfg.image_path)
    h, w = image.size
    organized_caption = organize_caption(captioner_caption, w, h)
    grounding_img, grounding_organized_caption = grounding(image, organized_caption, cfg, verbose=False, return_confidence=True, return_segmentation=False)
    grounding_img.save(cfg.image_save_path)

    print ("********** Test Recaption **********")
    print ("Transformate a normal caption into our structure  ...")
    recaption = llava_inference(cfg.recaption_instruction.format(cfg.caption), None, cfg.captioner_config, cfg.tokenizer, cfg.model, cfg.image_processor, "path")

    print ("********** Test Edit **********")
    print ("Edit caption using captioner ...")
    edit_instruction = get_question(cfg, captioner_caption, cfg.edit_item, w, h)
    captioner_edit_caption = llava_inference(edit_instruction, None, cfg.captioner_config, cfg.tokenizer, cfg.model, cfg.image_processor, "path")

    output_data["captioner_caption"] = captioner_caption
    output_data["recaption"] = recaption
    output_data["captioner_edit_caption"] = captioner_edit_caption

    output.append(output_data)
    with open(cfg.output_file, 'w') as file:
        json.dump(output, file, indent=4)


if __name__ == "__main__":
    cfg = EasyDict(__name__='Config: Inference')

    cfg.image_path = "data/test.jpg"
    cfg.clip_path = "ckpt/ViT-B-32.pt"
    cfg.repeat_times = 4

    cfg.caption_instruction = "Please describe this image in detail. Specifically, please list all important items and relationships in this image."
    cfg.caption = "A monkey landed over the city with a parachute."
    cfg.recaption_instruction = "There is an image of {}. Please describe this image in detail. Specifically, please list all important items and relationships in this image."
    cfg.edit_item = ["change the color of the clothes of man1 to red"]
    cfg.edit_instruction = "I hope you can make the following modifications to the description."
    cfg.description_instruction = "I will provide you a structured description of an image. \n The description is {} \n"

    cfg.captioner_config = EasyDict(__name__='Config: Bacon-Captioner')
    cfg.captioner_config.model_path = 'ckpt/captioner'
    cfg.captioner_config.model_base = 'ckpt/llava-v1.5-13b'
    cfg.captioner_config.conv_mode = "llava_v1"
    cfg.captioner_config.num_chunks = 1
    cfg.captioner_config.chunk_idx = 0
    cfg.captioner_config.temperature = 0.2
    cfg.captioner_config.top_p = None
    cfg.captioner_config.num_beams = 1

    cfg.llava_config = EasyDict(__name__='Config: LLaVA')
    cfg.llava_config.model_path = 'ckpt/llava-v1.5-13b'
    cfg.llava_config.model_base = None
    cfg.llava_config.conv_mode = "llava_v1"
    cfg.llava_config.num_chunks = 1
    cfg.llava_config.chunk_idx = 0
    cfg.llava_config.temperature = 0.2
    cfg.llava_config.top_p = None
    cfg.llava_config.num_beams = 1
    
    cfg.output_file = "results/inference.json"
    cfg.image_save_path = "results/grounding_image.png"

    cfg.seed = 22
    set_seed(cfg.seed)

    main(cfg)