import csv
import logging
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
from torch.nn import functional as F
LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))] # 当anomaly_scores变量为None时, 创建一个新的列表，长度与image_paths相同, 列表中的每个元素都被设置为字符串"-1",_是一个惯用的占位符变量，表示这个循环变量不会被使用

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()
######################################### segmantation image have problems ######################################
        if masks_provided:
            if mask_path is not None:
                if os.path.isdir(mask_path):  # 检查是否是目录
                    # 处理目录中的多个掩码
                    mask_files = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)
                                         if f.endswith(('.png', '.jpg', '.jpeg'))])
                    if mask_files:
                        # 读取所有掩码并合并
                        masks = []
                        for mask_file in mask_files:
                            mask_img = PIL.Image.open(mask_file).convert("RGB")
                            mask_transformed = mask_transform(mask_img)
                            masks.append(mask_transformed)

                        if isinstance(masks[0], np.ndarray):
                            # 如果是numpy数组，用numpy的方式合并
                            mask = np.maximum.reduce(masks)
                        else:
                            # 如果是torch.Tensor，转换为numpy
                            masks_numpy = [m.numpy() if not isinstance(m, np.ndarray) else m for m in masks]
                            mask = np.maximum.reduce(masks_numpy)
                    else:
                        # 目录为空，创建全零掩码
                        mask = np.zeros_like(image)
                else:
                    # 处理单个掩码文件
                    mask = PIL.Image.open(mask_path).convert("RGB")
                    mask = mask_transform(mask)
                    if not isinstance(mask, np.ndarray):
                        mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
##################################################################
        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()
def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

def embedding_concat(x, y):
    """
    FOR ATTENTION MODULE
    CONCATE LAYER2 AND Attention-LAYER3
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    """
    FOR ATTENTION MODULE FETURE MAP RESHAPE
    """
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

import matplotlib.pyplot as plt

def save_heatmap(seg_map, save_path):
    plt.imsave(save_path, seg_map, cmap='jet')



import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def visualize_anomaly_with_mask_fast(
        image_path: object,
        anomaly_map: object,
        save_path: object = None,
        notation: object = "bbox",
) -> dict[Any, list[Any]]:
    """
    将 anomaly_map 叠加到原图上，支持 bbox/contour 标注，保存到 save_path
    只跳过 GrabCut 前景分割步骤以提升速度
    """
    # 读取原图
    query_image = Image.open(image_path).convert('RGB')
    original_size = query_image.size

    if anomaly_map.ndim == 3:
        # anomaly_map = anomaly_map[..., 0]
        anomaly_map = anomaly_map.squeeze()
        # 结果的形状将是: (224, 224)
    # for anomaly_map resize 到原图大小
    anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)

    # 归一化 ndarray:(900,900)
    anomaly_map = anomaly_map.astype(np.float32)
    # anomaly_map = anomaly_map / (np.max(anomaly_map) + 1e-8)


    # 直接使用原始 anomaly_map，不做前景过滤
    anomaly_map_normalized = anomaly_map / (np.max(anomaly_map) + 1e-8)

    bboxes = []  # peng :new for return bboxes information

    if notation == "contour" or notation == "bbox":
        initial_threshold = anomaly_map_normalized.max() * (1 - 1 / 3)
        _, binary_map = cv2.threshold(anomaly_map_normalized, initial_threshold, 1, cv2.THRESH_BINARY)
        if np.count_nonzero(binary_map) / (binary_map.shape[0] * binary_map.shape[1]) > 0.5:
            sorted_values = np.sort(anomaly_map_normalized.flatten())
            threshold_value = sorted_values[int(0.5 * len(sorted_values))]
            _, binary_map = cv2.threshold(anomaly_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)
        binary_map = (binary_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取图像尺寸用于轮廓过滤
        height, width = anomaly_map.shape[:2]
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0.001 * width * height]

        query_image_np = np.array(query_image.convert("RGBA"))
        if notation == "contour":
            for contour in contours:
                cv2.drawContours(query_image_np, [contour], -1, (255, 0, 0, 255), 2)
        elif notation == "bbox":
            # 简化版本：直接为每个contour创建bbox，不做复杂合并
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            for x, y, w, h in bounding_boxes:
                x1, y1, x2, y2 = x, y, x + w, y + h
                bboxes.append((x1, y1, x2, y2))
                cv2.rectangle(query_image_np, (x1, y1), (x2, y2), (255, 0, 0, 255), 2)
        combined_image = Image.fromarray(query_image_np)
    else:
        combined_image = query_image
    # combined_image.save(save_path)
    return {image_path: bboxes}


def visualize_anomaly_with_mask_fast_topk_merge(
        image_path: object,
        anomaly_map: object,
        save_path: object = None,
        notation: object = "bbox",
        top_k: int = 3,
        iou_threshold: float = 0.3,
        merge_distance: int = 20,
) -> dict[Any, list[Any]]:
    """
    Overlay anomaly_map on the original image, return up to `top_k` merged boxes ordered by anomaly score (desc).

    - `top_k`: if provided, keep at most top_k boxes after merging.
    - `iou_threshold`: merge boxes whose IoU > this threshold.
    - `merge_distance`: also merge boxes whose center distance < this (pixels).
    """
    from math import sqrt

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union = areaA + areaB - interArea
        return interArea / (union + 1e-8) if union > 0 else 0.0

    def center_dist(boxA, boxB):
        cxA = 0.5 * (boxA[0] + boxA[2])
        cyA = 0.5 * (boxA[1] + boxA[3])
        cxB = 0.5 * (boxB[0] + boxB[2])
        cyB = 0.5 * (boxB[1] + boxB[3])
        return sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)

    def union_box(boxes):
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return (int(x1), int(y1), int(x2), int(y2))

    # read image and anomaly map
    query_image = Image.open(image_path).convert('RGB')
    original_size = query_image.size  # (width, height)

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.squeeze()

    anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map_normalized = anomaly_map / (np.max(anomaly_map) + 1e-8)

    bboxes = []

    if notation in ("contour", "bbox"):
        initial_threshold = anomaly_map_normalized.max() * (1 - 1 / 3)
        _, binary_map = cv2.threshold(anomaly_map_normalized, initial_threshold, 1, cv2.THRESH_BINARY)
        if np.count_nonzero(binary_map) / (binary_map.shape[0] * binary_map.shape[1]) > 0.5:
            sorted_values = np.sort(anomaly_map_normalized.flatten())
            threshold_value = sorted_values[int(0.5 * len(sorted_values))]
            _, binary_map = cv2.threshold(anomaly_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)
        binary_map = (binary_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = anomaly_map.shape[:2]
        contours = [c for c in contours if cv2.contourArea(c) > 0.001 * width * height]

        candidate_boxes = []
        candidate_scores = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = anomaly_map_normalized[y1:y2, x1:x2]
            score = float(np.max(region)) if region.size > 0 else 0.0
            candidate_boxes.append((x1, y1, x2, y2))
            candidate_scores.append(score)

        # merge overlapping / nearby boxes by greedy grouping (high->low)
        merged_boxes = []
        merged_scores = []
        if candidate_boxes:
            order = list(np.argsort(candidate_scores)[::-1])
            used = set()
            while order:
                idx = order.pop(0)
                if idx in used:
                    continue
                base_box = candidate_boxes[idx]
                group = [base_box]
                group_scores = [candidate_scores[idx]]
                to_remove = []
                for j in order:
                    if j in used:
                        continue
                    other_box = candidate_boxes[j]
                    if iou(base_box, other_box) > iou_threshold or center_dist(base_box, other_box) < merge_distance:
                        group.append(other_box)
                        group_scores.append(candidate_scores[j])
                        to_remove.append(j)
                # remove grouped indices from order
                order = [o for o in order if o not in to_remove]
                used.update(to_remove)
                merged = union_box(group)
                merged_boxes.append(merged)
                merged_scores.append(max(group_scores))

        # sort merged by score desc and pick top_k
        if merged_boxes:
            order = np.argsort(merged_scores)[::-1]
            if top_k is not None:
                order = order[:top_k]
            selected = [merged_boxes[i] for i in order]
        else:
            selected = []

        # draw on image
        query_image_np = np.array(query_image.convert("RGBA"))
        for (x1, y1, x2, y2) in selected:
            bboxes.append((x1, y1, x2, y2))
            cv2.rectangle(query_image_np, (x1, y1), (x2, y2), (255, 0, 0, 255), 2)
        combined_image = Image.fromarray(query_image_np)
    else:
        combined_image = query_image

    if save_path is not None and 'combined_image' in locals():
        try:
            combined_image.save(save_path)
        except Exception:
            pass

    return {image_path: bboxes}



def visualize_anomaly_with_mask(
        image_path,
        anomaly_map,
        notation="bbox",
):
    """
    将 anomaly_map 叠加到原图上，支持 bbox/contour 标注，保存到 save_path
    只跳过 GrabCut 前景分割步骤以提升速度
    """
    # 读取原图
    query_image = Image.open(image_path).convert('RGB')
    original_size = query_image.size

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[..., 0]
    # for anomaly_map resize 到原图大小
    anomaly_map = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)

    # 归一化 ndarray:(900,900)
    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map = anomaly_map / (np.max(anomaly_map) + 1e-8)

    # 跳过 GrabCut 分割前景的步骤，直接使用原始 anomaly_map
    # 注释掉以下 GrabCut 相关代码：
    # query_image_np = np.array(query_image)
    # mask = np.zeros(query_image_np.shape[:2], np.uint8)
    # bgd_model = np.zeros((1, 65), np.float64)
    # fgd_model = np.zeros((1, 65), np.float64)
    # height, width = query_image_np.shape[:2]
    # rect = (int(width * 0.01), int(height * 0.01), int(width * 0.98), int(height * 0.98))
    # cv2.grabCut(query_image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    # fg_mask = mask != 2
    # foreground_area = np.sum(fg_mask)
    # min_foreground_area = 0.05 * fg_mask.size
    # if foreground_area < min_foreground_area:
    #     filtered_anomaly_map = anomaly_map
    # else:
    #     filtered_anomaly_map = np.zeros_like(anomaly_map)
    #     filtered_anomaly_map[fg_mask] = anomaly_map[fg_mask]
    #     anomaly_map = filtered_anomaly_map

    # 直接使用原始 anomaly_map，不做前景过滤
    anomaly_map_normalized = anomaly_map / (np.max(anomaly_map) + 1e-8)

    bboxes = []  # peng :new for return bboxes information

    if notation == "contour" or notation == "bbox":
        initial_threshold = anomaly_map_normalized.max() * (1 - 1 / 3)
        _, binary_map = cv2.threshold(anomaly_map_normalized, initial_threshold, 1, cv2.THRESH_BINARY)
        if np.count_nonzero(binary_map) / (binary_map.shape[0] * binary_map.shape[1]) > 0.5:
            sorted_values = np.sort(anomaly_map_normalized.flatten())
            threshold_value = sorted_values[int(0.5 * len(sorted_values))]
            _, binary_map = cv2.threshold(anomaly_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)
        binary_map = (binary_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 获取图像尺寸用于轮廓过滤
        height, width = anomaly_map.shape[:2]
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0.001 * width * height]

        query_image_np = np.array(query_image.convert("RGBA"))
        if notation == "contour":
            for contour in contours:
                cv2.drawContours(query_image_np, [contour], -1, (255, 0, 0, 255), 2)
        elif notation == "bbox":
            # 简化版本：直接为每个contour创建bbox，不做复杂合并
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            for x, y, w, h in bounding_boxes:
                x1, y1, x2, y2 = x, y, x + w, y + h
                bboxes.append((x1, y1, x2, y2))
                cv2.rectangle(query_image_np, (x1, y1), (x2, y2), (255, 0, 0, 255), 2)
        combined_image = Image.fromarray(query_image_np)
    else:
        combined_image = query_image

    return {image_path: bboxes}