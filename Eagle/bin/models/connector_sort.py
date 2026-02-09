import numpy as np
import os
import torch
import sys
# 导入所有必要的模块
# 确保 expert 能够被导入
sys.path.insert(0, "/home/smslab1/PycharmProjects/AnomalyDetection/Eagle/Eagle/src")
import json
import argparse
from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
import patchcore.metrics
import patchcore.utils
import numpy as np
from EVT import pick_percentile

class Connector:
    """
    用于连接 ExpertModel 与 MLLM 的类
    负责将视觉模型输出转换成MLLM可用的prompt和输入格式
    """
    def __init__(self, expert_model, status=None, image_path=None, threshold_method=None, q = None):
        self.expert_model = expert_model
        self.status = status
        # self.few_shot = few_shot if few_shot is not None else []

        self.threshold_method = threshold_method
        self.image_path = image_path
        self.q = q
    def threshold_selection(self, scores):
        scores_max = np.max(scores)
        if self.threshold_method == "real_label":
            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            optimal_image_threshold = image_metrics["threshold"][np.argmax(image_metrics["tpr"] - image_metrics["fpr"])]

        else:
            val_scores = scores
            # 方法1 q-percentile (简单但对异常值敏感)
            if self.threshold_method == "p-quantile":
                optimal_image_threshold = np.percentile(val_scores, q=self.q)
            if self.threshold_method == "mad":
            # 方法2: 中位数 + 绝对标准差 (对异常值更鲁棒) MAD
                median_val = np.median(val_scores)
                mad = np.median(np.abs(val_scores - median_val))
                optimal_image_threshold = median_val + 3 * mad
            if self.threshold_method == "k-sigma":
            # 方法3: 均值 + 3倍标准差 (适用于近似正态分布) k-sigma
                mean_val = np.mean(val_scores)
                std_val = np.std(val_scores)
                optimal_image_threshold = mean_val + 3 * std_val

            # TODO: A threshold is calculated for each cycle, it's not necessary to calculate it every time.
            if self.threshold_method == "EVT":
                from scipy import stats
                import scipy.stats as st
                shape_param, loc_param, scale_param = st.genextreme.fit(val_scores)
                x = np.linspace(val_scores.min(), val_scores.max() + 1, 100)
                pdf_values = st.genextreme.pdf(x, shape_param, loc=loc_param, scale=scale_param)
                # 为了在直方图上叠加，我们需要调整PDF的尺度
                # 计算直方图的最大高度，用于缩放PDF曲线
                hist_counts, hist_bins = np.histogram(val_scores, bins=20)
                max_hist_count = np.max(hist_counts)
                max_pdf_value = np.max(pdf_values)

                # 缩放因子：将概率密度转换为与直方图计数相匹配的尺度
                scaling_factor = max_hist_count / max_pdf_value if max_pdf_value > 0 else 1
                scaled_pdf = pdf_values * scaling_factor

                from EVT import pick_percentile
                # confidence_level = 0.98
                confidence_level = pick_percentile(shape_param, 0.3)
                optimal_image_threshold = st.genextreme.ppf(confidence_level, shape_param, loc=loc_param,
                                                                scale=scale_param)

        return optimal_image_threshold

    def process_expert_output(self, optimal_image_threshold, score_Test, anomaly_map_Test, image_path,anomaly_labels=None):
        """
        处理expert模型的输出，scores,segmentations

        """
        image_status = {}
        if score_Test >= optimal_image_threshold:
            status = "anomalous"
        else:
            status = "normal"

        image_status[image_path] = status

        bbox_info = {}
        # result = patchcore.utils.visualize_anomaly_with_mask_fast(self.image_path, anomaly_map_Test, notation="bbox")
        result = patchcore.utils.visualize_anomaly_with_mask_fast_topk_merge(image_path, anomaly_map_Test, notation="bbox")
        bbox_info.update(result)

        result = {
            "bbox_info": bbox_info,
            "image_status": image_status
        }
        return result

    def create_prompt(self, processed_output,image_path):
        """Generate prompt from processed_output when available; otherwise return a default prompt."""
        incontext = ''
        image_info = {}
        if self.status and processed_output:
            status = processed_output["image_status"].get(image_path)
            bboxes = processed_output["bbox_info"].get(image_path, [])
            image_info[image_path] = {"status": status, "bboxes": bboxes}

            if image_info and image_info.get(image_path, {}).get('status') == "normal":
                incontext += "\nFollowing is the query image, The query image is predicted as normal: \n<image> "
                # incontext += "\nFollowing is the query image: \n<image> " + "\nThe query image is predicted as normal, You are not allowed to change normal to anomalous unless there is an obvious defect."
            else:
                incontext += (
                    "Following is the query image,The query image is predicted as anomalous: \n<image> "
                    "\nFollowing is the query image with red bounding box, The position of the red bounding box on the query image is the predicted defect location.\n<image>"
                    # "\nYou are allowed to change anomalous to normal if you find no visible defect."
                )
            payload = incontext
        else:
            # No processed output or status disabled: return the system prompt (or empty) and empty image_info
            incontext = f"Following is the query image: " + '\n' + f"<image> "
            payload = incontext
            image_info = {}
        return payload, image_info

    def run_inference(self, optimal_image_threshold , score_Test, anomaly_map_Test, image_path,anomaly_labels, model_name):
        """
        Run full inference pipeline only when self.status is True.
        If self.status is False, skip processing and return a default prompt.
        """
        if model_name == "internvl":
            if self.status:
                processed_output = self.process_expert_output(optimal_image_threshold, score_Test, anomaly_map_Test,image_path, anomaly_labels)
                prompt, image_info = self.create_prompt(processed_output,image_path)
            else:
                prompt, image_info = self.create_prompt(None)
            return prompt, image_info
        if model_name  in("qwen","llava-next","llava-1.5"):
            processed_output = self.process_expert_output(optimal_image_threshold, score_Test, anomaly_map_Test,image_path, anomaly_labels)
            image_info = {}
            status = processed_output["image_status"].get(image_path)
            bboxes = processed_output["bbox_info"].get(image_path, [])
            image_info[image_path] = {"status": status, "bboxes": bboxes}
            return processed_output ,image_info