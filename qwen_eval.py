import argparse
import json
from collections import defaultdict
import numpy as np
import os
import sys
# sys.path.insert(0, "/home/smslab1/PycharmProjects/AnomalyDetection/IAD_Framework/patchcore-inspection-main/src")
# import patchcore.metrics
# 获取 expert_base.py 所在的目录的绝对路径
expert_base_dir = '/home/smslab1/PycharmProjects/AnomalyDetection/Eagle/Eagle/bin/models'
# 将该目录添加到 Python 的模块搜索路径中
sys.path.insert(0, expert_base_dir)
from expert_base import ExpertModel_PatchCore, set_torch_device
from expert_base import ExpertModel_PatchCore, set_torch_device
from connector_sort import Connector
from mllm_base import MLLM_Qwen
from framework import Framework
from summary import caculate_accuracy_mmad, plot_subcategory_accuracy, caculate_accuracy_selected_dataset,caculate_accuracy_selected_dataset_v
# 设置环境变量 export HF_HOME=~/.cache/huggingface
os.environ["HF_HOME"] = "~/.cache/huggingface"
import random
import torch
from tqdm import tqdm
import sys
from evaluation.eval_set import QwenQuery

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _has_saved_patchcore(path):
    if not os.path.isdir(path):
        return False
    for fname in os.listdir(path):
        if fname.endswith(".faiss") or fname.endswith(".npy") or fname.endswith(".pth"):
            return True
    return False

import copy

def filter_text_gt_by_type(text_gt, target_type="Anomaly Detection", keep_first=True):
    """
    Return a copy of text_gt whose 'conversation' contains only entries
    with 'type' == target_type. If keep_first is True, keep only the first match.
    """
    conv = text_gt.get("conversation", [])
    matches = [c for c in conv if c.get("type") == target_type]
    if keep_first:
        matches = matches[:1]
    if not matches:
        return None
    new = copy.deepcopy(text_gt)
    new["conversation"] = matches
    return new

def filter_text_gt_by_question(text_gt, question_text="Is there any defect in the object?", keep_first=True):
    """
    Return a copy of text_gt whose 'conversation' contains only entries
    with 'Question' equal (case-insensitive) to question_text.
    """
    q_lower = question_text.strip().lower()
    conv = text_gt.get("conversation", [])
    matches = [c for c in conv if c.get("Question", "").strip().lower() == q_lower]
    if keep_first:
        matches = matches[:1]
    if not matches:
        return None
    new = copy.deepcopy(text_gt)
    new["conversation"] = matches
    return new

if __name__ == "__main__":

    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON config file.")
    parser.add_argument("--data_path_image", type=str, default=None, help="Path to the MVTec-AD dataset (can also be provided via --config).")
    parser.add_argument("--backbone_names", type=str, nargs="+", default=["wideresnet50"])
    parser.add_argument("--layers_to_extract_from", type=str, default=['layer2', 'layer3'])
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=8)
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--target_embed_dimension", type=int, default=1024)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--anomaly_scorer_num_nn", type=int, default=5)
    parser.add_argument("--percentage", type=float, default=0.1)
    parser.add_argument("--train_val_split", type=float, default=1.0)
    parser.add_argument("--threshold_method", type=str, default="real_label")
    parser.add_argument("--q", type=float, default=99)
    parser.add_argument("--init_strategy", type=str, default="center_farthest")
    parser.add_argument('--out_csv', type=str, default=None, help='Path to output CSV file')

    parser.add_argument("--model_path", type=str, default="../../InternVL/pretrained/InternVL2-1B")
    parser.add_argument("--model_name", type=str, default="internvl")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=0)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")

    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--domain_knowledge_path", type=str, default="../../../dataset/MMAD/domain_knowledge.json")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--agent_model", type=str, default="GT", choices=["GT", "PatchCore", "AnomalyCLIP", "AnomalyCLIP_mvtec","EfficientAD"])
    parser.add_argument("--agent_notation", type=str, default="bbox", choices=["bbox", "contour", "highlight", "mask", "heatmap"])
    parser.add_argument(
        "--resume_file",  # 两个减号，可选参数
        type=str,
        default=None,  # 不写时就是 None
        help="Existing answers JSON filename to resume from (under result/ directory).",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--defect_shot", type=int, default=0)
    parser.add_argument("--dataset_name", type =str, default=None, help="select dataset")
    parser.add_argument("--class_name", type=str, default=None, help="select class")
    parser.add_argument("--Prompt", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--text_prior_only", action="store_true")
    parser.add_argument("--task_type", type=str, choices=["all", "Anomaly Detection", "IsThereAnyDefect"],
                        default="Anomaly Detection",
                        help="Control which conversation entry to keep from text_gt. 'all' keeps full conversation. "
                             "'Anomaly Detection' filters by type. 'IsThereAnyDefect' filters by exact question.")

    def load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"--config not found: {path}")

        if p.suffix in [".yml", ".yaml"]:
            try:
                import yaml  # pip install pyyaml
            except ImportError as e:
                raise ImportError("Please install pyyaml: pip install pyyaml") from e
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError("Config must be .yaml/.yml or .json")

    def apply_config_defaults(parser, cfg: dict):
        valid = {a.dest for a in parser._actions}
        unknown = [k for k in cfg.keys() if k not in valid]
        if unknown:
            raise ValueError(f"Unknown keys in config: {unknown}")

        parser.set_defaults(**cfg)

    tmp_args, _ = parser.parse_known_args()

    if tmp_args.config:
        cfg = load_config(tmp_args.config)
        apply_config_defaults(parser, cfg)

    args = parser.parse_args()

    if args.dataset_name == "DS-MVTec":
        from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
    elif args.dataset_name == "VisA":
        from patchcore.datasets.Visa import VisaDataset, DatasetSplit

    torch.manual_seed(1234)
    model_path = args.model_path
    # model_name = os.path.split(model_path.rstrip('/'))[-1]
    torch.set_grad_enabled(False)

    if args.resume_file:
        # 直接指定已有文件，继续往里面写
        answers_json_path = os.path.join("result", args.resume_file)
        print("Resume from existing file:", answers_json_path)
        if os.path.exists(answers_json_path):
            with open(answers_json_path, "r") as file:
                all_answers_json = json.load(file)
        else:
            print("WARNING: resume_file does not exist, start from empty.")
            all_answers_json = []
    else:
        if args.debug:
            args.model_name += "_Debug"
        if args.defect_shot >= 1:
            args.model_name += f"_{args.defect_shot}_defect_shot"
        # Base file name parts
        base_name = f"answers_{args.few_shot_model}_shot_{args.model_name}"
        suffix_parts = []
        # Conditionally add parts to the file name
        if args.dataset_name:
            suffix_parts.append(args.dataset_name)
        if args.class_name:
            suffix_parts.append(args.class_name)
        # This assumes args.status and args.reproduce are mutually exclusive or can be combined
        if args.status:
            suffix_parts.append("status")
        if args.threshold_method :
            suffix_parts.append(args.threshold_method)
            if args.q:
                suffix_parts.append(args.q)

        if args.init_strategy :
            suffix_parts.append(args.init_strategy)
        if args.reproduce:
            # Assuming 'reproduce' implies a specific version or configuration
            suffix_parts.append("reproduce")
        # Join the parts with underscores
        if suffix_parts:
            full_file_name = f"{base_name}_{'_'.join([str(x) for x in suffix_parts])}.json"
        else:
            full_file_name = f"{base_name}.json"

        if not os.path.exists("result"):
            os.makedirs("result")
        # Construct the full path
        answers_json_path = os.path.join("result", full_file_name)
        # Only add numeric suffix if reproduce=True
        if args.reproduce and os.path.exists(answers_json_path):
            i = 1
            base, ext = os.path.splitext(answers_json_path)
            new_path = f"{base}_{i}{ext}"
            while os.path.exists(new_path):
                i += 1
                new_path = f"{base}_{i}{ext}"
            answers_json_path = new_path

        print(f"Answers will be saved at {answers_json_path}")


        # 用于存储所有答案
        if os.path.exists(answers_json_path):
            with open(answers_json_path, "r") as file:
                all_answers_json = json.load(file)
        else:
            all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "/home/smslab1/PycharmProjects/AnomalyDetection/MMAD-main/dataset/MMAD",
        "json_path": "/home/smslab1/PycharmProjects/AnomalyDetection/MMAD-main/dataset/MMAD/mmad.json",
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    if args.debug:
        # 固定随机种子
        random.seed(1)
        # random.seed(10)
        sample_keys = random.sample(list(chat_ad.keys()), 1600)
    else:
        sample_keys = chat_ad.keys()

    ######################################### add new experiment ########################################
    if args.dataset_name :
        selected_dataset_keys = [key for key in sample_keys if key.startswith(args.dataset_name +"/")]
        sample_keys = selected_dataset_keys
    else:
         sample_keys = sample_keys

    if args.class_name :
        selected_class_keys = []
        for key in sample_keys:
            if args.class_name in key:
                selected_class_keys.append(key)
                sample_keys = selected_class_keys
    else:
        sample_keys = sample_keys

    device = set_torch_device()
    print(f"Using device: {device}")
    # 1. 实例化 ExpertModel 并加载模型
    print("\n--- Step 1: Initializing and loading ExpertModel ---")
    expert = ExpertModel_PatchCore(device)
    PatchCore_list = expert.load_model(args)
    print("ExpertModel loaded successfully.")
    # 2. 实例化 Connector
    print("\n--- Step 2: Initializing Connector ---")
    connector = Connector(expert_model=expert, status=args.status, threshold_method=args.threshold_method, q=args.q)
    # 3. 实例化 MLLM 并加载模型
    print("\n--- Step 2: Initializing and loading MLLM ---")
    mllm = MLLM_Qwen(model_path=args.model_path, device=device, status=args.status,args= args)
    mllm.load_model()
    print("MLLM loaded successfully.")
    framework = Framework(expert,connector,mllm)

    defect_images = defaultdict(list)

    for image_path in sample_keys:
        dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
        dataset_name = image_path.split("/")[0].replace("MVTec-AD", "MVTec")
        object_name = image_path.split("/")[1]
        defect_name = image_path.split("/")[2]
        # 使用 (dataset_name, object_name, defect_name) 作为键
        defect_key = (dataset_name, object_name, defect_name)
        defect_images[defect_key].append(image_path)

    all_scores_by_class = {}
    all_labels_by_class = {}
    all_scores_val_by_class = {}
    all_training_scores = {}
    last_object_name = None

    for data_id, image_path in enumerate(tqdm(sample_keys)):
        if image_path in existing_images and not args.reproduce:
            continue
        text_gt = chat_ad[image_path]

        if args.task_type == "all":
            # keep full conversation
            pass
        elif args.task_type == "Anomaly Detection":
            filtered_text_gt = filter_text_gt_by_type(chat_ad[image_path], target_type="Anomaly Detection",
                                                      keep_first=True)
            if filtered_text_gt is None:
                # no matching entry, skip sample
                continue
            text_gt = filtered_text_gt
        elif args.task_type == "IsThereAnyDefect":
            filtered_text_gt = filter_text_gt_by_question(chat_ad[image_path],
                                                          question_text="Is there any defect in the object?",
                                                          keep_first=True)
            if filtered_text_gt is None:
                continue
            text_gt = filtered_text_gt
        else:
            # fallback: try to filter by a custom type string provided in args.task_type
            filtered_text_gt = filter_text_gt_by_type(chat_ad[image_path], target_type=args.task_type, keep_first=True)
            if filtered_text_gt is None:
                continue
            text_gt = filtered_text_gt

        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]

        if args.domain_knowledge:
            dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
            object_name = image_path.split("/")[1]

        else:
            domain_knowledge = None

        images_in_defect = defect_images[defect_key]
        defect_shot = random.sample([img for img in images_in_defect if img != image_path],
                                        min(args.defect_shot, len(images_in_defect) - 1))
        rel_defect_shot = [os.path.join(args.data_path, path) for path in defect_shot]
        if text_gt["mask_path"]:
            rel_mask_path = os.path.join(args.data_path, image_path.split("/")[0], image_path.split("/")[1],text_gt["mask_path"])
        else:
            rel_mask_path = None

        _DATASETS = {
                     "DS-MVTec": ["patchcore.datasets.mvtec", "MVTecDataset"],
                     "VisA": ["patchcore.datasets.Visa", "VisaDataset"],
                     "MVTec-LOCO":["patchcore.datasets.mvtec_loco", "MVTecLocoDataset"],
                     "GoodsAD": ["patchcore.datasets.GoodsAD", "GoodsADDataset"],
                     }
        dataset_info = _DATASETS[args.dataset_name]
        dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
        # 3.构建训练集 dataloader
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            args.data_path_image,
            classname= object_name,
            resize=args.resize,
            train_val_split=args.train_val_split,
            imagesize=args.imagesize,
            split=DatasetSplit.TRAIN,
            seed=0,
            augment=False,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 构建测试集 dataloader
        test_dataset = dataset_library.__dict__[dataset_info[1]](
            args.data_path_image,
            classname=object_name,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TEST,
            seed=0,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if args.train_val_split < 1:
            val_dataset = dataset_library.__dict__[dataset_info[1]](
                args.data_path_image,
                classname=object_name,
                resize=args.resize,
                train_val_split=args.train_val_split,
                imagesize=args.imagesize,
                split=dataset_library.DatasetSplit.VAL,
                seed=0,
            )

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            val_dataloader = None

        anomaly_labels = [
            x[1] != "good" for x in test_dataloader.dataset.data_to_iterate
        ]

        fixed_path = image_path.replace("DS-MVTec", "MVTec-AD").replace("/image/", "/test/")
        fixed_path = os.path.join(cfg["data_path"], fixed_path)

        processed_output, image_info,apply_attention = framework.evaluate(
            train_dataloader=train_dataloader,
            object_name =object_name,
            patchcore_list=PatchCore_list,  #change
            test_image=fixed_path,  #
            args=args,
            image_path=rel_image_path,  #
            model_name=args.model_name,
            anomaly_labels=anomaly_labels,
            force_retrain=False
        )

        qwenquery = QwenQuery(
            image_path=rel_image_path,
            text_gt=text_gt,
            processor=mllm.processor,
            model=mllm,
            few_shot=rel_few_shot,
            visualization=args.visualization,
            domain_knowledge=object_name,
            mask_path=rel_mask_path,
            CoT=args.CoT,
            Prompt=args.Prompt,
            image_info=image_info,
            defect_shot=rel_defect_shot,
            args=args,
            status=args.status,
            class_name=object_name,
            apply_attention= apply_attention
        )
        questions, answers, gpt_answers = qwenquery.generate_answer()

        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]

        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        if data_id % 10 == 0 or data_id == len(chat_ad.keys()) - 1:
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)


    caculate_accuracy_selected_dataset_v(answers_json_path, args.dataset_name)
