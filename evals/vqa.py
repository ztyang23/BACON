import json
import os
import torch
import re
from tqdm import tqdm
from bacon_modules.utils.dataloader import NLVR2, OK_VQA, VQAv1, VQAv2
from bacon_modules.utils.tools import prepare_llava, llava_inference


class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


def obtain_vqa_answer_score(cfg, dataset, caption_file, question_format_dict, save_path, mode, benchmark, weight=True):
    caption_dict = {}
    with open(caption_file, "r") as file:
        for line in tqdm(file):
            data = json.loads(line)
            image_path = data["path"].split("/")[-1]
            caption = data["caption"]
            caption_dict[image_path] = caption

    total_score = 0
    total_number = 0
    if os.path.exists(save_path):
        prediction_dict = torch.load(save_path)
    else:
        prediction_dict = {}
    for qa_obj in tqdm(dataset.qa_list):
        image_path = qa_obj["image_path"]
        image_name = image_path.split("/")[-1]
        core_question = qa_obj["question"]
        answers = qa_obj["answers"]
        tp = qa_obj["type"]
        question_format = question_format_dict[tp]
        if image_name in caption_dict:
            caption = caption_dict[image_name]
            question = question_format.format(caption, core_question)
            # try:
            key = image_name + "$$" + core_question
            if os.path.exists(save_path) and key in prediction_dict:
                prediction = prediction_dict[key]
            else:
                prediction = llava_inference(question, None, cfg.llava_config, cfg.tokenizer, cfg.model, cfg.image_processor, "pil")
                prediction_dict[key] = prediction
            if benchmark == "nlvr":
                for ans in answers.keys():
                    if ans.lower() in prediction.lower():
                        total_score = total_score + answers[ans] if weight else total_score + 1
            elif benchmark == "ok_vqa":
                for ans in answers.keys():
                    if prediction.lower() == ans.lower():
                        total_score = total_score + answers[ans] if weight else total_score + 1
            elif benchmark == "vqav1":
                if tp == "yes/no":
                    ans = list(answers.keys())[0]
                    if ans.lower() in prediction.lower():
                        total_score = total_score + answers[ans] if weight else total_score + 1
                elif tp == "number":
                    for ans in answers.keys():
                        if ans.lower() in prediction.lower():
                            total_score = total_score + answers[ans] if weight else total_score + 1
                else:
                    for ans in answers.keys():
                        if prediction.lower() == ans.lower():
                            total_score = total_score + answers[ans] if weight else total_score + 1
            total_number += 1
            # except:
            #     pass

    if not os.path.exists(save_path):
        torch.save(prediction_dict, save_path)

    score = total_score / total_number
    print (f"The score of {mode} for {benchmark} is {score}")

    return score


def obtain_vqa_answer_for_upload(cfg, dataset, caption_file, question_format_dict, save_path, test_path):
    test_split = [json.loads(line) for line in open(test_path)]
    if os.path.exists(save_path):
        with open(save_path, "r") as inputfile:
            results = json.load(inputfile)
    else:
        caption_dict = {}
        with open(caption_file, "r") as file:
            for line in tqdm(file):
                try:
                    data = json.loads(line)
                except:
                    print ("line", line)
                image_path = data["path"].split("/")[-1]
                caption = data["caption"]
                caption_dict[image_path] = caption

        results = []
        for qa_obj in tqdm(dataset.qa_list):
            image_path = qa_obj["image_path"]
            core_question = qa_obj["question"]
            tp = qa_obj["type"]
            question_format = question_format_dict[tp]
            if image_path in caption_dict:
                caption = caption_dict[image_path]
                question = question_format.format(caption, core_question)
                try:
                    prediction = llava_inference(question, None, cfg.llava_config, cfg.tokenizer, cfg.model, cfg.image_processor, "pil")
                except:
                    break

                result_obj = {}
                result_obj["question_id"] = qa_obj["question_id"]
                result_obj["answer"] = prediction
                results.append(result_obj)

    results = {x['question_id']: x['answer'] for x in results}
    all_answers = []
    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    with open(save_path, 'w') as file:
        json.dump(all_answers, file)


def main(cfg):

    cfg.tokenizer, cfg.model, cfg.image_processor = prepare_llava(cfg.llava_config)

    nlvr2_dataset = NLVR2(cfg.vqa.nlvr2.question_answer_file, cfg.vqa.nlvr2.image_dir)
    nlvr2_dataset.get_qa_list()
    obtain_vqa_answer_score(cfg, nlvr2_dataset, cfg.vqa.nlvr2.bacon_result_path, cfg.vqa.nlvr2.question_format, cfg.vqa.nlvr2.bacon_answers_save_path, "bacon", "nlvr")
    obtain_vqa_answer_score(cfg, nlvr2_dataset, cfg.vqa.nlvr2.llava_result_path, cfg.vqa.nlvr2.question_format, cfg.vqa.nlvr2.llava_answers_save_path, "llava", "nlvr")
    obtain_vqa_answer_score(cfg, nlvr2_dataset, cfg.vqa.nlvr2.sharegpt4v_result_path, cfg.vqa.nlvr2.question_format, cfg.vqa.nlvr2.sharegpt4v_answers_save_path, "sharegpt4v", "nlvr")
    obtain_vqa_answer_score(cfg, nlvr2_dataset, cfg.vqa.nlvr2.qwen_result_path, cfg.vqa.nlvr2.question_format, cfg.vqa.nlvr2.qwen_answers_save_path, "qwen", "nlvr")

    ok_vqa_dataset = OK_VQA(cfg.vqa.ok_vqa.question_file, cfg.vqa.ok_vqa.answer_file, cfg.vqa.ok_vqa.image_dir, cfg.vqa.ok_vqa.image_path)
    ok_vqa_dataset.get_qa_list()
    obtain_vqa_answer_score(cfg, ok_vqa_dataset, cfg.vqa.ok_vqa.bacon_result_path, cfg.vqa.ok_vqa.question_format, cfg.vqa.ok_vqa.bacon_answers_save_path, "bacon", "ok_vqa")
    obtain_vqa_answer_score(cfg, ok_vqa_dataset, cfg.vqa.ok_vqa.llava_result_path, cfg.vqa.ok_vqa.question_format, cfg.vqa.ok_vqa.llava_answers_save_path, "llava", "ok_vqa")
    obtain_vqa_answer_score(cfg, ok_vqa_dataset, cfg.vqa.ok_vqa.sharegpt4v_result_path, cfg.vqa.ok_vqa.question_format, cfg.vqa.ok_vqa.sharegpt4v_answers_save_path, "sharegpt4v", "ok_vqa")
    obtain_vqa_answer_score(cfg, ok_vqa_dataset, cfg.vqa.ok_vqa.qwen_result_path, cfg.vqa.ok_vqa.question_format, cfg.vqa.ok_vqa.qwen_answers_save_path, "qwen", "ok_vqa")

    vqav1_dataset = VQAv1(cfg.vqa.vqav1.question_answer_file, cfg.vqa.vqav1.image_dir)
    vqav1_dataset.get_qa_list()
    obtain_vqa_answer_score(cfg, vqav1_dataset, cfg.vqa.vqav1.bacon_result_path, cfg.vqa.vqav1.question_format, cfg.vqa.vqav1.bacon_answers_save_path, "bacon", "vqav1")
    obtain_vqa_answer_score(cfg, vqav1_dataset, cfg.vqa.vqav1.llava_result_path, cfg.vqa.vqav1.question_format, cfg.vqa.vqav1.llava_answers_save_path, "llava", "vqav1")
    obtain_vqa_answer_score(cfg, vqav1_dataset, cfg.vqa.vqav1.sharegpt4v_result_path, cfg.vqa.vqav1.question_format, cfg.vqa.vqav1.sharegpt4v_answers_save_path, "sharegpt4v", "vqav1")
    obtain_vqa_answer_score(cfg, vqav1_dataset, cfg.vqa.vqav1.qwen_result_path, cfg.vqa.vqav1.question_format, cfg.vqa.vqav1.qwen_answers_save_path, "qwen", "vqav1")

    vqav2_dataset = VQAv2(cfg.vqa.vqav2.question_answer_file, cfg.vqa.vqav2.image_dir)
    vqav2_dataset.get_qa_list()
    obtain_vqa_answer_for_upload(cfg, vqav2_dataset, cfg.vqa.vqav2.bacon_result_path, cfg.vqa.vqav2.question_format, cfg.vqa.vqav2.bacon_upload_file_path, cfg.vqa.vqav2.test_split_file)
    obtain_vqa_answer_for_upload(cfg, vqav2_dataset, cfg.vqa.vqav2.llava_result_path, cfg.vqa.vqav2.question_format, cfg.vqa.vqav2.llava_upload_file_path, cfg.vqa.vqav2.test_split_file)
    obtain_vqa_answer_for_upload(cfg, vqav2_dataset, cfg.vqa.vqav2.sharegpt4v_result_path, cfg.vqa.vqav2.question_format, cfg.vqa.vqav2.sharegpt4v_upload_file_path, cfg.vqa.vqav2.test_split_file)
    obtain_vqa_answer_for_upload(cfg, vqav2_dataset, cfg.vqa.vqav2.qwen_result_path, cfg.vqa.vqav2.question_format, cfg.vqa.vqav2.qwen_upload_file_path, cfg.vqa.vqav2.test_split_file)