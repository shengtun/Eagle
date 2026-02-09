import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import os
import sys
sys.path.insert(0, '/home/smslab1/PycharmProjects/AnomalyDetection/Eagle')
from Eagle.attention_editing.atten_Edit import AttentionPatcher

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import cv2

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(int(blocks)):  # 确保 blocks 是整数
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image = Image.open(image_file).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def draw_bbox_on_image(image_path, bboxes, color=(0, 0, 255), thickness=2):
    image = cv2.imread(image_path)
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
class MLLM_Qwen:
    def __init__(self, model_path: str, device: str = "cuda", status=None,args=None):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.input_size = 448
        self.status = status
        self.processor = None
        self.args = args
    def load_model(self):

        print(f"Loading InternVL3 model from {self.model_path} on device {self.device}...")
        device = 'cuda'
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().to(device)
        min_pixels = 512 * 28 * 28
        max_pixels = 512* 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_path, min_pixels=min_pixels, max_pixels=max_pixels,trust_remote_code=True, padding_side='left', use_fast=True)

    def preprocess_image(self, image_path, image_info, max_num=12):
        """
        对图像进行预处理，使其符合 InternVL3 和 Qwen 的输入要求。

        返回两种形式：
            1. tensor_list：InternVL 的多图输入（拼接成 pixel_values）
            2. fileuri_list：Qwen 可用的 "file:///path/to" 列表
        """
        import tempfile
        from PIL import Image

        # def save_pil_to_temp_uri(pil_img):
        #     """将 PIL 转成本地文件并返回 file:/// 路径"""
        #     tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=True)
        #     pil_img.save(tmp.name)
        #     return "file://" + tmp.name

        import os, shutil, uuid
        from pathlib import Path

        TMP_DIR = Path("/tmp/eagle_tmp_images")
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        def cleanup_tmp_dir():
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            TMP_DIR.mkdir(parents=True, exist_ok=True)

        def save_pil_to_tmp_dir(pil_img):
            fname = f"{uuid.uuid4().hex}.png"
            path = TMP_DIR / fname
            pil_img.save(path)
            return "file://" + str(path)

        self.image_path = image_path
        query_pil = Image.open(self.image_path).convert('RGB')
        query_tensor = load_image(self.image_path, max_num=1).to(torch.bfloat16).cuda()

        # Qwen 需要 file:// path
        query_uri = "file://" + os.path.abspath(self.image_path)

        # 初始化列表
        tensor_list = [query_tensor]
        fileuri_list = [query_uri]

        bboxes = image_info[self.image_path]["bboxes"]

        # 绘制红框图
        image_with_bbox = draw_bbox_on_image(self.image_path, bboxes)
        bbox_image_rgb = cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
        bbox_pil = Image.fromarray(bbox_image_rgb)

        # 生成 tensor（InternVL）
        bbox_tensor = load_image(bbox_pil, max_num=1).to(torch.bfloat16).cuda()
        tensor_list.append(bbox_tensor)

        # 生成 file://（Qwen）
        cleanup_tmp_dir()
        bbox_uri = save_pil_to_tmp_dir(bbox_pil)
        fileuri_list.append(bbox_uri)
        # 返回：InternVL 用的 tensor 列表，Qwen 用的 file:// 列表

        return tensor_list, fileuri_list


    def create_prompt_qwen(self, image_info, image_uris):
        """
        为 Qwen 生成 prompt（content 列表形式），而不是带 <image> 的纯文本。

        Args:
            processed_output: 上游专家模型返回的结果字典，包含
                - processed_output["image_status"][image_path] -> "normal"/"anomalous"
                - processed_output["bbox_info"][image_path] -> list of bboxes
            image_path: 当前 query image 的路径（原始）
            image_uris: 该 query 对应的图像 URI 列表，例如：
                image_uris[0]: 原图  file:///...
                image_uris[1]: 带 red bbox 的图（如果有） file:///...

        Returns:
            incontext: Qwen 所需的 content 列表（list[dict]）
            image_info: {image_path: {"status": ..., "bboxes": ...}}
        """

        incontext = []
        if self.status and image_info:
            # status = processed_output["image_status"].get(image_path)
            status =next(iter(image_info.values()))['status']
            # ---- 正常情况 ----
            if status == "normal":
                incontext.append({
                    "type": "text",
                    "text": "Following is the query image. The query image is predicted as normal."
                })
                # 原图
                if len(image_uris) >= 1:
                    incontext.append({
                        "type": "image",
                        "image": image_uris[0]
                    })

            # ---- 异常情况 ----
            else:
                # 原始 query image
                if len(image_uris) >= 1:
                    incontext.append({
                        "type": "text",
                        "text": "Following is the query image. The query image is predicted as anomalous."
                    })
                    incontext.append({
                        "type": "image",
                        "image": image_uris[0]
                    })

                # 带红框的 query image（如果有第二张）
                if len(image_uris) >= 2:
                    incontext.append({
                        "type": "text",
                        "text": (
                            "Following is the query image with red bounding box. "
                            "The position of the red bounding box on the query image "
                            "is the predicted defect location."
                        )
                    })
                    incontext.append({
                        "type": "image",
                        "image": image_uris[1]
                    })

        else:
            # 没有 processed_output 或者 status 关闭：只给一张 query image
            incontext.append({
                "type": "text",
                "text": "Following is the query image:"
            })
            if len(image_uris) >= 1:
                incontext.append({
                    "type": "image",
                    "image": image_uris[0]
                })
            if len(image_uris) >= 2:
                incontext.append({
                    "type": "text",
                    "text": (
                        "Following is the query image with red bounding box. "
                        "The position of the red bounding box on the query image "
                        "is the predicted defect location."
                    )
                })
                incontext.append({
                    "type": "image",
                    "image": image_uris[1]
                })

        return incontext

    def processor_framework(self,instruction_P:str,conversation:str,image_path: str, image_info:dict,):
        image_tensor_list, image_url_list = self.preprocess_image(image_path, image_info)
        final_prompt = self.create_prompt_qwen(image_info, image_url_list)
        incontext = []
        incontext.extend(final_prompt)

        if self.args.text_prior_only:
            # 只有在存在 bbox 描述时才删除
            if (
                    len(incontext) >= 4
                    and incontext[-1].get("type") == "image"
                    and incontext[-2].get("type") == "text"
                    and "red bounding box" in incontext[-2].get("text", "").lower()
            ):
                incontext = incontext[:-2]

        texts = [c["text"] for c in conversation if isinstance(c, dict) and "text" in c]
        prompt_text = "\n".join(t.strip() for t in texts if t.strip())
        payload = ([
                       {"type": "text", "text": instruction_P},
                   ] + [
                       {"type": "text", "text": f"Answer with the option's letter from the given choices directly! "}]
                   + incontext + [
                       {"type": "text", "text": f"Following is the question list: " + prompt_text}
                   ])
        return payload

    def generate_response(self, content:str):
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128,do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        with torch.no_grad():
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return response

    def generate_response_attention(self, content:str):
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        patcher = AttentionPatcher(cfg=None)  # or pass a default dict if you want global cfg
        patcher.enable()
        cfg = self.attentionchange(inputs)
        patcher.set_cfg(cfg)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128,do_sample = False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        with torch.no_grad():
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        return response
    def find_nth_occurrence(self,ids_row, token_id, n):
        ids = ids_row.tolist()
        positions = [i for i, t in enumerate(ids) if t == token_id]
        return positions[n - 1] if len(positions) >= n else -1
    def attentionchange(self,inputs):
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        n_image = 1
        batch_idx = 0
        ids_main = inputs["input_ids"][batch_idx]
        start_idx_main = self.find_nth_occurrence(ids_main, vision_start_token_id, n_image)
        end_idx_main = self.find_nth_occurrence(ids_main, vision_end_token_id, n_image)

        start_idx_template = self.find_nth_occurrence(ids_main, vision_start_token_id, 1)
        end_idx_gen_template = self.find_nth_occurrence(ids_main, vision_end_token_id, 1)

        if start_idx_main == -1 or end_idx_main == -1:
            raise ValueError(f"{n_image}-th image tokens not found in `inputs`")
        pos_main = start_idx_main + 1
        pos_end_main = end_idx_main

        pos_template = start_idx_template + 1
        pos_end_template = end_idx_gen_template

        target_text = "The query image is predicted as anomalous"
        # target_text = "The query image is predicted as normal"
        def find_subsequence(haystack, needle):
            n, m = len(haystack), len(needle)
            if m == 0 or m > n:
                return -1
            for i in range(n - m + 1):
                if haystack[i:i + m] == needle:
                    return i
            return -1
        needle_ids = self.processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        full_ids = inputs["input_ids"][0].tolist()
        start = find_subsequence(full_ids, needle_ids)
        if start == -1:
            # try dropping the first token
            start = find_subsequence(full_ids, needle_ids[1:])
            if start != -1:
                needle_ids = needle_ids[1:]  # 同步更新长度

        end = start + len(needle_ids)
        print("Target span:", start, end, "len=", len(needle_ids))
        img_slice = slice(pos_main, pos_end_main)  # 第二张图
        text_slice = slice(start, end)  # "predicted anomalous" 那段 token span
        cfg = {
            "enabled": True,
            "decode_only": False,  # or True to only affect decode steps
            "layers": [9, 10, 11, 12, 13, 14],
            "heads_mask": None,
            "img_slice": img_slice,  # set these indices per-query
            "text_slice": text_slice,
            "weight_img": 1.0,
            "weight_txt": 1.0,
            "boost_img": 0.0,
            "suppr_txt": 0.0,
        }

        return  cfg
