from easydict import EasyDict
from evals.ovdetection import main as eval_ovd
from evals.ovsgg import main as eval_ovsgg
from evals.pointingqa import main as eval_pointingqa
from evals.pointqa import main as eval_pointqa
from evals.vqa import main as eval_vqa
from evals.plan import main as eval_plan


def main(cfg):

    if cfg.task == "ovd":
        eval_ovd(cfg)

    elif cfg.task == "pointingqa":
        eval_pointingqa(cfg)

    elif cfg.task == "pointqa":
        eval_pointqa(cfg)

    elif cfg.task == "ovsgg":
        eval_ovsgg(cfg)

    elif cfg.task == "vqa":
        eval_vqa(cfg)

    elif cfg.task == "plan":
        eval_plan(cfg)


if __name__ == "__main__":
    cfg = EasyDict(__name__='Config: Eval')

    cfg.test_benchmark_path = "data/test_dataset.jsonl"
    cfg.image_dir = "data/coco2017_test"
    cfg.clip_path = "ckpt/ViT-B-32.pt"

    cfg.cache_path = "cache/cache.pt"

    cfg.task = "vqa" # [ovd, ovsgg, pointqa, pointingqa, vqa, plan]

    cfg.llava_config = EasyDict(__name__='Config: Bacon-Captioner')
    cfg.llava_config.model_path = 'ckpt/llava-v1.5-13b'
    cfg.llava_config.model_base = None
    cfg.llava_config.conv_mode = "llava_v1"
    cfg.llava_config.num_chunks = 1
    cfg.llava_config.chunk_idx = 0
    cfg.llava_config.temperature = 0.2
    cfg.llava_config.top_p = None
    cfg.llava_config.num_beams = 1
    

    cfg.ovdetection = EasyDict(__name__='Config: Evaluation of open-vocabulary object detection')
    cfg.ovdetection.ovdquo_result = "results/ov-dquo_bacon_benchmark.jsonl"
    cfg.ovdetection.devit_result = "results/devit_bacon_benchmark.jsonl"
    cfg.ovdetection.grounding_dino_result = "results/grounding_dino_bacon_benchmark.jsonl"
    cfg.ovdetection.bacon_result = "results/bacon_captioner_w_grounding_bacon_benchmark.jsonl"
    cfg.ovdetection.glamm_result = "results/glamm_bacon_benchmark.jsonl"
    cfg.ovdetection.kosmos_result = "results/kosmos_bacon_benchmark.jsonl"
    cfg.ovdetection.next_chat_result = "results/next_chat_bacon_benchmark.jsonl"
    cfg.ovdetection.clip_threshold = 0.85
    cfg.ovdetection.bbox_threshold = 0.5


    cfg.ovsgg = EasyDict(__name__='Config: Evaluation of open-vocabulary scene graph generation')
    cfg.ovsgg.image_dir = "data/visual_genome"
    cfg.ovsgg.testset_file = "data/vg150.json"
    cfg.ovsgg.question_answer_file = "data/vg_question_answers.json"
    cfg.ovsgg.scene_graphs_file = "data/vg_scene_graphs.json"
    cfg.ovsgg.bbox_file = "data/vg_attributes.json"
    cfg.ovsgg.object_file = "data/vg_object_list.txt"
    cfg.ovsgg.relationship_file = "data/vg_relationship_list.txt"
    cfg.ovsgg.vg_bacon_path = "results/bacon_captioner_w_grounding_vg.jsonl"
    cfg.ovsgg.vg_gpsnet_dir = "results/sgg/vg/gpsnet"
    cfg.ovsgg.vg_motifs_dir = "results/sgg/vg/motifs"
    cfg.ovsgg.vg_psgformer_dir = "results/sgg/vg/psgformer"
    cfg.ovsgg.vg_psgtr_dir = "results/sgg/vg/psgtr"
    cfg.ovsgg.vg_vctree_dir = "results/sgg/vg/vctree"
    cfg.ovsgg.vg_imp_dir = "results/sgg/vg/imp"
    cfg.ovsgg.bacondata_bacon_path = "results/bacon_captioner_w_grounding_bacon_benchmark.jsonl"
    cfg.ovsgg.bacondata_gpsnet_dir = "results/sgg/bacon_dataset/gpsnet"
    cfg.ovsgg.bacondata_motifs_dir = "results/sgg/bacon_dataset/motifs"
    cfg.ovsgg.bacondata_psgformer_dir = "results/sgg/bacon_dataset/psgformer"
    cfg.ovsgg.bacondata_psgtr_dir = "results/sgg/bacon_dataset/psgtr"
    cfg.ovsgg.bacondata_vctree_dir = "results/sgg/bacon_dataset/vctree"
    cfg.ovsgg.bacondata_imp_dir = "results/sgg/bacon_dataset/imp"
    cfg.ovsgg.clip_threshold = 0.85
    cfg.ovsgg.iou_threshold = 0.5


    cfg.pointqa = EasyDict(__name__='Config: Evaluation of point question answering')
    cfg.pointqa.image_dir = "data/visual_genome"
    cfg.pointqa.question_answer_file = "data/pointqa_local_test.jsonl"
    cfg.pointqa.bacon_path = "results/bacon_captioner_w_grounding_vg.jsonl"
    cfg.pointqa.glamm_path = "results/glamm_vg.jsonl"
    cfg.pointqa.next_chat_path = "results/next_chat_looktwiceqa.jsonl"
    cfg.pointqa.kosmos_path = "results/kosmos_looktwiceqa.jsonl"
    cfg.pointqa.question_format = {
        "normal": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Please note that your answer should be only one word."
    }


    cfg.pointingqa = EasyDict(__name__='Config: Evaluation of pointing question answering')
    cfg.pointingqa.image_dir = "data/visual_genome"
    cfg.pointingqa.question_answer_file = "data/v7w_pointing_test.jsonl"
    cfg.pointingqa.bacon_path = "results/bacon_captioner_w_grounding_vg.jsonl"
    cfg.pointingqa.glamm_path = "results/glamm_vg.jsonl"
    cfg.pointingqa.next_chat_path = "results/next_chat_v7w.jsonl"
    cfg.pointingqa.kosmos_path = "results/kosmos_v7w.jsonl"


    cfg.vqa = EasyDict(__name__='Config: Evaluation of visual question answering')
    cfg.vqa.nlvr2 = EasyDict(__name__='Config: VQA on NLVR2')
    cfg.vqa.nlvr2.image_dir = "data/nlvr2_test1"
    cfg.vqa.nlvr2.question_answer_file = "data/nlvr2_test1.json"
    cfg.vqa.nlvr2.bacon_result_path = "results/bacon_captioner_wo_grounding_nlvr2_test1.jsonl"
    cfg.vqa.nlvr2.llava_result_path = "results/llava_nlvr2_test1.jsonl"
    cfg.vqa.nlvr2.sharegpt4v_result_path = "results/sharegpt4v_nlvr2_test1.jsonl"
    cfg.vqa.nlvr2.qwen_result_path = "results/qwen_nlvr2_test1.jsonl"
    cfg.vqa.nlvr2.bacon_answers_save_path = "cache/bacon_answers_nlvr2_test1.pt"
    cfg.vqa.nlvr2.llava_answers_save_path = "cache/llava_answers_nlvr2_test1.pt"
    cfg.vqa.nlvr2.sharegpt4v_answers_save_path = "cache/sharegpt4v_answers_nlvr2_test1.pt"
    cfg.vqa.nlvr2.qwen_answers_save_path = "cache/qwen_answers_nlvr2_test1.pt"
    cfg.vqa.nlvr2.question_format = {
        "normal": "I will give you a description of a picture, that is {} \n" \
                    "Please judge whether this sentence {} is correct based on this description.\n" \
                    "Please note that you can only answer True or False."
    }
    cfg.vqa.ok_vqa = EasyDict(__name__='Config: VQA on QK-VQA')
    cfg.vqa.ok_vqa.image_dir = "data/coco2014_val"
    cfg.vqa.ok_vqa.image_path = "data/coco2014_val/COCO_val2014_{}.jpg"
    cfg.vqa.ok_vqa.question_file = "data/okvqa_OpenEnded_mscoco_val2014_questions.json"
    cfg.vqa.ok_vqa.answer_file = "data/okvqa_mscoco_val2014_annotations.json"
    cfg.vqa.ok_vqa.bacon_result_path = "results/bacon_captioner_wo_grounding_coco2014_val.jsonl"
    cfg.vqa.ok_vqa.llava_result_path = "results/llava_coco2014_val.jsonl"
    cfg.vqa.ok_vqa.sharegpt4v_result_path = "results/sharegpt4v_coco2014_val.jsonl"
    cfg.vqa.ok_vqa.qwen_result_path = "results/qwen_coco2014_val.jsonl"
    cfg.vqa.ok_vqa.bacon_answers_save_path = "cache/bacon_answers_okvqa.pt"
    cfg.vqa.ok_vqa.llava_answers_save_path = "cache/llava_answers_okvqa.pt"
    cfg.vqa.ok_vqa.sharegpt4v_answers_save_path = "cache/sharegpt4v_answers_okvqa.pt"
    cfg.vqa.ok_vqa.qwen_answers_save_path = "cache/qwen_answers_okvqa.pt"
    cfg.vqa.ok_vqa.question_format = {
        "normal": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Please note that your answer should be as simple as possible with only one word."
    }
    cfg.vqa.vqav1 = EasyDict(__name__='Config: VQA on VQAv1')
    cfg.vqa.vqav1.image_dir = "data/coco2014_val"
    cfg.vqa.vqav1.question_answer_file = "data/vqav1_vqa_E_val.jsonl"
    cfg.vqa.vqav1.bacon_result_path = "results/bacon_captioner_wo_grounding_coco2014_val.jsonl"
    cfg.vqa.vqav1.llava_result_path = "results/llava_coco2014_val.jsonl"
    cfg.vqa.vqav1.sharegpt4v_result_path = "results/sharegpt4v_coco2014_val.jsonl"
    cfg.vqa.vqav1.qwen_result_path = "results/qwen_coco2014_val.jsonl"
    cfg.vqa.vqav1.bacon_answers_save_path = "cache/bacon_answers_vqav1.pt"
    cfg.vqa.vqav1.llava_answers_save_path = "cache/llava_answers_vqav1.pt"
    cfg.vqa.vqav1.sharegpt4v_answers_save_path = "cache/sharegpt4v_answers_vqav1.pt"
    cfg.vqa.vqav1.qwen_answers_save_path = "cache/qwen_answers_vqav1.pt"
    cfg.vqa.vqav1.question_format = {
        "number": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Please note that you can only answer an Arabic numerals without any redundant words.",
        "yes/no": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Please note that you can only answer yes or no without any redundant words.",
        "other": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Please note that your answer should be within three words."
    }
    cfg.vqa.vqav2 = EasyDict(__name__='Config: VQA on VQAv2')
    cfg.vqa.vqav2.image_dir = "data/coco2015_test"
    cfg.vqa.vqav2.question_answer_file = "data/vqav2_OpenEnded_mscoco_test-dev2015_questions.jsonl"
    cfg.vqa.vqav2.test_split_file = "data/vqav2_OpenEnded_mscoco_test2015_questions.jsonl"
    cfg.vqa.vqav2.bacon_result_path = "results/bacon_captioner_wo_grounding_coco2015_test.jsonl"
    cfg.vqa.vqav2.llava_result_path = "results/llava_coco2015_test.jsonl"
    cfg.vqa.vqav2.sharegpt4v_result_path = "results/sharegpt4v_coco2015_test.jsonl"
    cfg.vqa.vqav2.qwen_result_path = "results/qwen_coco2015_test.jsonl"
    cfg.vqa.vqav2.bacon_upload_file_path = "cache/bacon_upload_file.json"
    cfg.vqa.vqav2.llava_upload_file_path = "cache/llava_upload_file.json"
    cfg.vqa.vqav2.sharegpt4v_upload_file_path = "cache/sharegpt4v_upload_file.json"
    cfg.vqa.vqav2.qwen_upload_file_path = "cache/qwen_upload_file.json"
    cfg.vqa.vqav2.question_format = {
        "normal": "I will give you a description of a picture, that is {} \n" \
                    "Please answer the question according to the content of the image. {}\n" \
                    "Answer the question using a single word or phrase."
    }

    cfg.plan = EasyDict(__name__='Config: Evaluation of plan')
    cfg.plan.coco_image_dir = "data/coco2017_val"
    cfg.plan.coco_imageid = "data/coco_image_ids.txt"
    cfg.plan.coco_threshold = 0.95
    cfg.plan.coco_annotations = "data/instances_val2017.json"
    cfg.plan.coco_layoutgpt_result = "results/layoutgpt_plan_coco.json"
    cfg.plan.coco_bacon_result = "results/bacon_captioner_plan_coco.json"
    cfg.plan.bacondata_image_dir = "data/coco2017_test"
    cfg.plan.bacondata_imageid = "data/bacondata_image_ids.txt"
    cfg.plan.bacondata_threshold = 0.6
    cfg.plan.bacondata_layoutgpt_result = "results/layoutgpt_plan_bacon_benchmark.json"
    cfg.plan.bacondata_bacon_result = "results/bacon_captioner_plan_bacon_benchmark.json"

    main(cfg)