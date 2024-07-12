import re
import torch
import clip
import os
import cv2
import random
import numpy as np
from PIL import Image
from easydict import EasyDict
from bacon_modules.llava_module.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from bacon_modules.llava_module.conversation import conv_templates, SeparatorStyle
from bacon_modules.llava_module.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_model_name_from_path
from bacon_modules.llava_module.model.builder import load_pretrained_model
from bacon_modules.llava_module.utils import disable_torch_init
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


def prepare_llava(config):
    disable_torch_init()
    model_path = os.path.expanduser(config.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, config.model_base, model_name)

    return tokenizer, model, image_processor


def llava_inference(qs, input_image, config, tokenizer, model, image_processor, mode="path"):
    assert mode in ["path", "pil", "tensor"]
    if model.config.mm_use_im_start_end:
        if input_image is not None:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    elif input_image is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[config.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    if input_image is not None:
        if mode in ["path", "pil"]:
            if mode == "path":
                input_image = Image.open(input_image)
            try:
                image_tensor = image_processor.preprocess(input_image, return_tensors='pt')['pixel_values'][0]
            except:
                return None
        else:
            image_tensor = input_image

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda() if input_image is not None else None, #[3, 224, 224]
            do_sample=True if config.temperature > 0 else False,
            temperature=config.temperature,
            top_p=config.top_p,
            num_beams=config.num_beams,
            max_new_tokens=4096,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs


def prepare_clip(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(path, device=device) 

    return model, preprocess


def extract_clip_feature(image, model, preprocess, mode='i', device=None, image_mode='pil'):
    assert mode in ['i', 't']
    assert image_mode in ['pil', 'path']
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == 'i':
        if image_mode == "path":
            image = Image.open(image)
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    else:
        text = clip.tokenize([image], truncate=True).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features


def organize_caption(caption, w, h):
    pattern = "<([^>]*)>"
    pattern2 = "\[([^\]]*)\]"

    out = {}
    split_1 = caption.split('%%')
    out["overall_description"] = {}
    out["object_list"] = []
    out["relationships"] = []

    split_2 = split_1[2].split('&&')
    out["overall_description"]["style"] = {}
    out["overall_description"]["style"]["content"] = re.sub(pattern, r"\1", split_2[2].replace('\n', ''))
    out["overall_description"]["style"]["nouns"] = re.findall(pattern, split_2[2].replace('\n', ''))
    out["overall_description"]["theme"] = {}
    out["overall_description"]["theme"]["content"] = re.sub(pattern, r"\1", split_2[4].replace('\n', ''))
    out["overall_description"]["theme"]["nouns"] = re.findall(pattern, split_2[4].replace('\n', ''))
    out["overall_description"]["background_global_description"] = {}
    out["overall_description"]["background_global_description"]["content"] = re.sub(pattern, r"\1", split_2[6]).replace('\n', '').split('.')
    out["overall_description"]["background_global_description"]["nouns"] = [re.findall(pattern, item) for item in split_2[6].replace('\n', '').split('.')]
    if out["overall_description"]["background_global_description"]["content"][-1] == '':
        out["overall_description"]["background_global_description"]["content"] = out["overall_description"]["background_global_description"]["content"][:-1]
        out["overall_description"]["background_global_description"]["nouns"] = out["overall_description"]["background_global_description"]["nouns"][:-1]
    out["overall_description"]["foreground_global_description"] = {}
    out["overall_description"]["foreground_global_description"]["content"] = re.sub(pattern, r"\1", split_2[8]).replace('\n', '').split('.')
    out["overall_description"]["foreground_global_description"]["nouns"] = [re.findall(pattern, item) for item in split_2[8].replace('\n', '').split('.')]
    if out["overall_description"]["foreground_global_description"]["content"][-1] == '':
        out["overall_description"]["foreground_global_description"]["content"] = out["overall_description"]["foreground_global_description"]["content"][:-1]
        out["overall_description"]["foreground_global_description"]["nouns"] = out["overall_description"]["foreground_global_description"]["nouns"][:-1]

    split_3 = split_1[4].replace('\n', '').split(')')
    for item in split_3:
        if item != '':
            item_list = item.split('(')
            if len(item_list) < 2:
                continue
            obj = {}
            obj["name"] = re.sub(pattern, r"\1", item_list[0])
            des_list = item_list[1].split(';')
            if len(des_list) < 2:
                continue
            obj["cat1"] = des_list[0]
            obj["cat2"] = des_list[1]

            obj["description"] = {}
            obj["description"]["content"] = re.sub(pattern, r"\1", des_list[2]).split('.')
            obj["description"]["nouns"] = [re.findall(pattern, item) for item in des_list[2].split('.')]
            if obj["description"]["content"][-1] == '':
                obj["description"]["content"] = obj["description"]["content"][:-1]
                obj["description"]["nouns"] = obj["description"]["nouns"][:-1]

            obj["color"] = {}
            try:
                obj["color"]["content"] = re.sub(pattern, r"\1", des_list[3].split(':')[1]).split('.')
                obj["color"]["nouns"] = [re.findall(pattern, item) for item in des_list[3].split('.')]
                if obj["color"]["content"][-1] == '':
                    obj["color"]["content"] = obj["color"]["content"][:-1]
                    obj["color"]["nouns"] = obj["color"]["nouns"][:-1]
            except:
                pass
            out["object_list"].append(obj)

    split_4 = split_1[6].replace('\n', '').split('.')
    for item in split_4:
        if item.replace(" ", "") != "":
            relation = {}
            relation["nouns"] = re.findall(pattern, item.replace('\n', ''))
            relation["relationship"] = re.findall(pattern2, item.replace('\n', ''))
            relation["content"] = re.sub(pattern2, r"\1", re.sub(pattern, r"\1", item.replace('\n', '')))
            out["relationships"].append(relation)

    if "* " in split_1[8]:
        split_5 = split_1[8].split("* ")
        position_dict = {}
        for item in split_5:
            name = item.split("(")[0].strip()
            pos_list = item.split("(")[1].split(")")[0].split(";")
            pos_list = [float(pos.split(": ")[1].split("px")[0]) for pos in pos_list]
            weight, height, x_min, y_min = pos_list
            x_max, y_max = x_min + weight, y_min + height
            x_min, y_min, x_max, y_max = float(x_min / 512), float(y_min / 512), float(x_max / 512), float(y_max / 512)
            x_min, y_min, x_max, y_max = x_min * w, y_min * h, x_max * w, y_max * h
            position = [x_min, y_min, x_max, y_max]
            position_dict[name] = position
    else:
        split_5 = split_1[8].split("positions are:")[-1].split(",")
        position_dict = {}
        for item in split_5:
            name = item.split("{")[0].strip()
            pos_list = item.split("{")[1].split("}")[0].split(";")
            pos_list_ = []
            for pos in pos_list:
                if ":" in pos:
                    pos_list_.append(float(pos.split(": ")[1].split("px")[0]))
            pos_list = pos_list_
            weight, height, x_min, y_min = pos_list
            x_max, y_max = x_min + weight, y_min + height
            x_min, y_min, x_max, y_max = float(x_min / 512), float(y_min / 512), float(x_max / 512), float(y_max / 512)
            x_min, y_min, x_max, y_max = x_min * w, y_min * h, x_max * w, y_max * h
            position = [x_min, y_min, x_max, y_max]
            position_dict[name] = position

    new_object_list = []
    for obj in out["object_list"]:
        obj_name = obj["name"]
        if obj_name.strip() in position_dict:
            obj["position"] = position_dict[obj_name.strip()]
            new_object_list.append(obj)
    out["object_list"] = new_object_list

    return out


def get_grounding_config(cfg):
    cfg.grounding_config = EasyDict(__name__='Config: Grounding config')
    cfg.grounding_config.GROUNDING_DINO_CHECKPOINT_PATH = "ckpt/groundingdino_swint_ogc.pth"
    cfg.grounding_config.GROUNDING_DINO_CONFIG_PATH = "ckpt/GroundingDINO_SwinT_OGC.py"
    cfg.grounding_config.BOX_THRESHOLD = 0.25
    cfg.grounding_config.TEXT_THRESHOLD = 0.25
    cfg.grounding_config.SAM_ENCODER_VERSION = "vit_h"
    cfg.grounding_config.SAM_CHECKPOINT_PATH = "ckpt/sam_vit_h_4b8939.pth"

    return cfg


def init_model_grounding(cfg, device):
    grounding_dino_model = Model(model_config_path=cfg.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=cfg.GROUNDING_DINO_CHECKPOINT_PATH)
    sam = sam_model_registry[cfg.SAM_ENCODER_VERSION](checkpoint=cfg.SAM_CHECKPOINT_PATH).to(device)
    sam_predictor = SamPredictor(sam)
    
    return grounding_dino_model, sam_predictor


def init_model_grounding_dino(cfg, device):
    grounding_dino_model = load_model(cfg.GROUNDING_DINO_CONFIG_PATH, cfg.GROUNDING_DINO_CHECKPOINT_PATH)
    
    return grounding_dino_model


def create_color(average_color,used_color):
    while True:
        r,g,b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        try:
            diff = abs(r - average_color[0]) + abs(g - average_color[1]) + abs(b - average_color[2])
            if diff > 100: 
                is_different = True
                for color in used_color:
                    diff = abs(r - color[0]) + abs(g - color[1]) + abs(b - color[2])
                    if diff < 80:
                        is_different = False
                        break
                if is_different: 
                    return (b, g, r)
        except:
            return (b, g, r)


def resize_and_pad(image, orgimg_size, target_size=(1024, 1024), ratio=None):
    ref_h, ref_w, _ = orgimg_size
    image = Image.fromarray(image)

    img_size = image.size 
    if max(img_size)/min(img_size)>5.8:
        total_ratio = min(target_size[0]/img_size[0], target_size[1]/img_size[1]) #特殊处理
    else:
        ratio = float(max(target_size)) / max(orgimg_size) 
        max_up_ratio = min(ref_w/img_size[0], ref_h/img_size[1]) 
        up_ratio = max(ref_w/img_size[0], ref_h/img_size[1]) 
        sub_ratio = max_up_ratio**0.5
        total_ratio = ratio * sub_ratio
    new_size = tuple([int(x * total_ratio) for x in img_size]) 

    image = image.resize(new_size, Image.Resampling.LANCZOS)

    target_size = max(new_size[0], new_size[1])
    target_size = (target_size, target_size)
    new_im = Image.new("RGB", target_size, (255, 255, 255))
    paste_position = [(target_size[0] - new_size[0]) // 2,
                        (target_size[1] - new_size[1]) // 2]
    new_im.paste(image, paste_position)

    return np.array(new_im)


def compute_iou(mask1, mask2):
    assert mask1.shape == mask2.shape
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union !=0 else 0

    return iou


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray):
    # Prompting SAM with detected boxes
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
        
    return np.array(result_masks)


def draw_bbox(image, organized_caption, random_p=0.99):
    image_pil = image
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    scale = 1
    used_color = []
    for idx in range(len(organized_caption["object_list"])):
        if "position" in organized_caption["object_list"][idx]:
            item = organized_caption["object_list"][idx]["position"]
            x_min, y_min, x_max, y_max = int(item[0] / scale), int(item[1] / scale), int(item[2] / scale), int(item[3] / scale)

            average_color = image_pil.crop((x_min, y_max-20, x_min+20, y_max)).resize((1, 1)).getpixel((0, 0)) #RGB xmin ymin xmax ymax
            color = create_color(average_color,used_color)

            used_color.append(color[::-1])
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), tuple(color), 10)
            cv2.putText(image, f'{organized_caption["object_list"][idx]["name"]}', (x_min, y_max), cv2.FONT_HERSHEY_SIMPLEX, 3, tuple(color), 8)
            
        else:
            print (f'No bounding box for {organized_caption["object_list"][idx]["name"]}')
            organized_caption["object_list"][idx]["position"] = [0,0,0,0] 

    height, width = image.shape[:2]
    scale = 1024 / max(height, width)
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
    return resized_image


def drop_index(name):
    if name.strip()[-1] in [f'{i}' for i in range(10)]:
        name = name.strip()[:-1].strip()

    return name


def ti_match(organized_caption, mobj_features, obj_name, obj_features,detections, clip, preprocess, matched_segmentation={}, threshold=0.2, weights=[0.4, 0.4, 0.2, 0], verbose=False, return_confidence=False):
    text_total_features = None
    names, names2inds = [],dict()
    for ind, obj in enumerate(organized_caption["object_list"]):
        if obj_name not in obj["name"]: 
            continue
        names2inds[obj["name"]]=ind
        names.append(obj["name"])
        text_name = f'This is a {obj["name"]}.'
        try:
            color_text = '.'.join(obj["color"]["content"]) #可能没有颜色信息
        except:
            print('no color content')
            color_text = '.'
        text_color = f'The color of this image is {color_text}'
        max_len = 290
        if len(text_color)>max_len: text_color = text_color[:max_len]
        text_desription = '.'.join(obj["description"]["content"])
        if len(text_desription)>max_len: text_desription = text_desription[:max_len]
        text = f'This is a {obj["name"]}.' + text_desription
        text_total = text + f'. The color information is {color_text}'

        if len(text_total) > max_len:
            len1 = len(f'. The color information is {color_text}')
            text_total = text[:max_len-len1] + f'. The color information is {color_text}'

        feature_name = extract_clip_feature(text_name, clip, preprocess, mode='t')
        feature_color = extract_clip_feature(text_color, clip, preprocess, mode='t')
        feature_description = extract_clip_feature(text_desription, clip, preprocess, mode='t')
        feature_total = extract_clip_feature(text_total, clip, preprocess, mode='t')
        if text_total_features is None:
            text_name_features = feature_name
            text_color_features = feature_color
            text_description_features = feature_description
            text_total_features = feature_total
        else:
            text_name_features = torch.cat([text_name_features, feature_name], dim=0)
            text_color_features = torch.cat([text_color_features, feature_color], dim=0)
            text_description_features = torch.cat([text_description_features, feature_description], dim=0)
            text_total_features = torch.cat([text_total_features, feature_total], dim=0)        

    text_features = [text_name_features, text_color_features, text_description_features, text_total_features]
    cosine_similarity_name = torch.matmul(obj_features, text_name_features.T) 
    cosine_similarity_color = torch.matmul(obj_features, text_color_features.T)
    cosine_similarity_description = torch.matmul(obj_features, text_description_features.T)
    cosine_similarity_total = torch.matmul(obj_features, text_total_features.T)
    cosine_similarity_ = weights[0] * cosine_similarity_name + weights[1] * cosine_similarity_color + weights[2] * cosine_similarity_description + weights[3] * cosine_similarity_total

    cosine_similarity_name = torch.matmul(mobj_features, text_name_features.T) 
    cosine_similarity_color = torch.matmul(mobj_features, text_color_features.T)
    cosine_similarity_description = torch.matmul(mobj_features, text_description_features.T)
    cosine_similarity_total = torch.matmul(mobj_features, text_total_features.T)
    cosine_similarity = weights[0] * cosine_similarity_name + weights[1] * cosine_similarity_color + weights[2] * cosine_similarity_description + weights[3] * cosine_similarity_total
    cosine_similarity = cosine_similarity*0.5 + cosine_similarity_*0.5
    if verbose:
        print(cosine_similarity)        
    matches = []
    for image_index, text_similarities in enumerate(cosine_similarity):
        for text_index, sim in enumerate(text_similarities[0]):
            if sim > threshold:
                matches.append((image_index, names[text_index], sim.item()))
    matches.sort(key=lambda x: x[2], reverse=True)
    if verbose:
        print ('matches', matches)
    matched_images,matched_texts = set(), set()
    final_matches = []
    for image_index, text_, sim in matches:
        if text_ not in matched_texts and image_index not in matched_images:
            final_matches.append((image_index, text_, sim))
            matched_texts.add(text_)
            matched_images.add(image_index)
    if verbose:
        print ('after', final_matches)

    for match in final_matches:
        (image_index, text_, _) = match
        organized_caption["object_list"][names2inds[text_]]["position"] = detections.xyxy[image_index].tolist()
        if return_confidence:
            organized_caption["object_list"][names2inds[text_]]["cofidence"] = detections.confidence[image_index].tolist()
        matched_segmentation[organized_caption["object_list"][names2inds[text_]]["name"].strip()] = detections.mask[image_index]
    return organized_caption, matched_segmentation


def grounding(image, organized_caption, cfg, verbose=False, return_confidence=False, return_segmentation=False):
    image_pil = image
    clip, preprocess = prepare_clip(cfg.clip_path)
    
    # grounding
    object_list = [drop_index(item["name"].strip()) for item in organized_caption["object_list"]]
    source_image = image
    w, h = source_image.width, source_image.height
    total_area = w * h
    matched_segmentation = {}
    bbox_image = None
    for obj_name in set(object_list): 
        image_source = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        CLASSES = [obj_name]
        try:
            detections = cfg.grounding_config.grounding_dino_model.predict_with_classes( # detect objects
                image=image_source,
                classes=CLASSES,
                box_threshold=cfg.grounding_config.BOX_THRESHOLD,
                text_threshold=cfg.grounding_config.TEXT_THRESHOLD,
            )   
        except:
            continue     
            
        detections.mask = segment(
            sam_predictor=cfg.grounding_config.sam_predictor,
            image=cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB), 
            xyxy=detections.xyxy
        )
        xyxy, confidence = detections.xyxy, detections.confidence
        
        h, w, _ = image_source.shape
        xyxy, confidence = detections.xyxy, detections.confidence
        llava_mask_1, llava_mask_2 = torch.zeros(len(xyxy)), torch.zeros(len(xyxy))
        obj_features_list, mobj_features_list = [], []
        # llava find error
        for i, bbox in enumerate(xyxy):
            ew, eh = (bbox[2] - bbox[0]) * 0.05, (bbox[3] - bbox[1]) * 0.05
            x1, x2 = max(int(bbox[0] - ew), 0), min(int(bbox[2] + ew), w)
            y1, y2 = max(int(bbox[1] - eh), 0), min(int(bbox[3] + eh), h)
            sub_image = image_source[y1:y2, x1:x2, :]
            sam_mask = np.expand_dims(detections.mask[i][y1:y2, x1:x2], axis=2)
            masked_sub_image = np.where(sam_mask, sub_image, 255)
                
            rgb_sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
            rgb_masked_sub_image = cv2.cvtColor(masked_sub_image, cv2.COLOR_BGR2RGB)
            pil_sub_image = Image.fromarray(rgb_sub_image)
            masked_pil_sub_image = Image.fromarray(resize_and_pad(rgb_masked_sub_image, image_source.shape))
            qs_list = [f'please tell me whether the main content in the picture is {obj_name}, you can only answer yes or no.',
                    f"This main part of this is {obj_name}, is that right? Just tell me Yes or No without more words.",] * ((cfg.repeat_times) // 2)
            out_list,out_list2 = [], []
            for qi in range(cfg.repeat_times):
                qs = qs_list[qi]
                out1 = llava_inference(qs, pil_sub_image, cfg.llava_config, cfg.llava_tokenizer, cfg.llava_model, cfg.llava_image_processor, mode='pil') 
                out2 = llava_inference(qs, masked_pil_sub_image, cfg.llava_config, cfg.llava_tokenizer, cfg.llava_model, cfg.llava_image_processor, mode='pil')
                out_list.append(out1)
                out_list2.append(out2)
                if out1 is not None:
                    if out1.startswith('Y') or out1.startswith('y'):
                        llava_mask_1[i] += 1
                if out2 is not None:
                    if out2.startswith('Y') or out2.startswith('y'):
                        llava_mask_2[i] += 1
            llava_mask_1[i] = 1 if llava_mask_1[i] > (cfg.repeat_times+1) // 2 else 0 
            llava_mask_2[i] = 1 if llava_mask_2[i] > (cfg.repeat_times+1) // 2 else 0
            llava_mask_1[i] = int(llava_mask_1[i]) | int(llava_mask_2[i])

            mobj_features_list.append(extract_clip_feature(masked_pil_sub_image, clip, preprocess)) 
            pil_sub_image = Image.fromarray(resize_and_pad(rgb_sub_image,image_source.shape))
            obj_features_list.append(extract_clip_feature(pil_sub_image, clip, preprocess))

        keep_mask = llava_mask_1 == 1 
        if len(keep_mask) == 1: keep_mask = [True] if keep_mask[0] else [False]
        detections.xyxy, detections.confidence, detections.class_id, detections.mask = xyxy[keep_mask], confidence[keep_mask], detections.class_id[keep_mask], detections.mask[keep_mask]
        if len(detections.xyxy) == 0: 
            print('llavas not agree {0} !!!'.format(obj_name))
            continue 
            
        # align
        mobj_features, obj_features = torch.stack(mobj_features_list), torch.stack(obj_features_list)
        mobj_features, obj_features = mobj_features[keep_mask], obj_features[keep_mask]
        organized_caption, matched_segmentation = ti_match(organized_caption, mobj_features, obj_name, obj_features, detections, clip, preprocess, matched_segmentation, verbose=verbose, return_confidence=return_confidence)

        bbox_image = draw_bbox(image_pil, organized_caption)
    if bbox_image is not None:
        bbox_image = Image.fromarray(bbox_image)

    if return_segmentation:
        return bbox_image, organized_caption, matched_segmentation
    else:
        return bbox_image, organized_caption


def grounding_dino(image_path, prompt, cfg, verbose=False, return_confidence=False, return_segmentation=False):
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=cfg.grounding_config.grounding_dino_model,
        image=image,
        caption=prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    bbox_image = Image.fromarray(annotated_frame_rgb)

    return bbox_image, boxes, phrases