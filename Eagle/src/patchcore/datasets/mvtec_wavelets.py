import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

# ==== 新增 import ====
import cv2
import numpy as np
try:
    import pywt
except ImportError:
    pywt = None
    print("⚠️ 未安装 pywt，将跳过小波增强。pip install PyWavelets")
from .GLCM import compute_glcm_contrast  # 直接用你现成的实现

# ... 保留原有 import 与代码 ...

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        # ===== 自适应增强的可选参数 =====
        enable_adaptive_enhance=True,
        tau=20,                    # GLCM 对比度阈值（建议: 训练集上用分位数估出来后传入）
        wavelet_gain=1.2,
        wavelet_name="db2",
        wavelet_level=2,
        **kwargs,
    ):
        # --- 原有初始化 ---
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.enable_adaptive_enhance = enable_adaptive_enhance
        self.tau = tau
        self.wavelet_gain = wavelet_gain
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    # ===== 新增：小波增强（仅增强亮度通道）=====
    def _wavelet_enhance_y(self, rgb: np.ndarray) -> np.ndarray:
        if pywt is None:
            return rgb  # 没有 pywt 就不处理
        # RGB -> YCrCb，拿亮度
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)

        coeffs = pywt.wavedec2(y, wavelet=self.wavelet_name, level=self.wavelet_level)
        cA, details = coeffs[0], coeffs[1:]
        new_details = [(cH * self.wavelet_gain, cV * self.wavelet_gain, cD * self.wavelet_gain)
                       for (cH, cV, cD) in details]
        y_enh = pywt.waverec2([cA] + new_details, wavelet=self.wavelet_name)
        y_enh = np.clip(y_enh, 0, 255).astype(np.uint8)

        ycrcb[:, :, 0] = y_enh
        rgb_out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return rgb_out

    # ===== 新增：是否需要增强的判定（GLCM 对比度，复用你的函数）=====
    def _maybe_adaptive_enhance(self, pil_image: PIL.Image.Image) -> PIL.Image.Image:
        if not self.enable_adaptive_enhance or self.tau is None:
            return pil_image

        rgb = np.array(pil_image)  # PIL RGB -> np.uint8 RGB
        # 亮度通道上测 GLCM 对比度（与 GLCM.py 保持 levels/angles 的一致性）
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        _, _, _, c = compute_glcm_contrast(y)  # 来自 GLCM.py【见上方引用】
        if c < float(self.tau):
            rgb = self._wavelet_enhance_y(rgb)
            pil_image = PIL.Image.fromarray(rgb)
        return pil_image

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            next_idx = (idx + 1) % len(self.data_to_iterate)
            if next_idx != idx:
                return self.__getitem__(next_idx)
            else:
                raise ValueError("No valid images found")

        # --- 先读原图 ---
        image = PIL.Image.open(image_path).convert("RGB")

        # --- 新增：自适应前处理（与 train/val/test 一致）---
        image = self._maybe_adaptive_enhance(image)

        # --- 再做你现有的几何/归一化变换 ---
        image = self.transform_img(image)

        # === 以下保持你的 mask 逻辑不变 ===
        if self.split == DatasetSplit.TEST and mask_path is not None:
            if os.path.isdir(mask_path):
                mask_files = sorted(os.listdir(mask_path))
                mask_list = [
                    self.transform_mask(PIL.Image.open(os.path.join(mask_path, f)))
                    for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))
                ]
                if mask_list:
                    mask = torch.max(torch.stack(mask_list), dim=0)[0]
            else:
                mask = self.transform_mask(PIL.Image.open(mask_path))
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            for classname in self.classnames_to_use:
                if self.split == DatasetSplit.TEST:
                    classpath = os.path.join(self.source, classname, "test")
                else:
                    classpath = os.path.join(self.source, classname, "train")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # # Unrolls the data dictionary to an easy-to-iterate list.
        # data_to_iterate = []
        # for classname in sorted(imgpaths_per_class.keys()):
        #     for anomaly in sorted(imgpaths_per_class[classname].keys()):
        #         for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
        #             data_tuple = [classname, anomaly, image_path]
        #             if self.split == DatasetSplit.TEST and anomaly != "good":
        #                 data_tuple.append(maskpaths_per_class[classname][anomaly][i])
        #             else:
        #                 data_tuple.append(None)
        #             data_to_iterate.append(data_tuple)
        # Unrolls the data dictionary to an easy-to-iterate list.
        image_extensions = ('.png', '.jpg', '.jpeg')

        data_to_iterate = []

        if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
            for classname in sorted(imgpaths_per_class.keys()):
                for anomaly in sorted(imgpaths_per_class[classname].keys()):
                    for image_path in imgpaths_per_class[classname][anomaly]:
                        if not image_path.lower().endswith(image_extensions):
                            continue
                        data_tuple = [classname, anomaly, image_path]
                        data_tuple.append(None)  # TRAIN 没有 mask
                        data_to_iterate.append(data_tuple)

        elif self.split == DatasetSplit.TEST:
            for classname in sorted(imgpaths_per_class.keys()):
                for anomaly in sorted(imgpaths_per_class[classname].keys()):
                    if anomaly == "good":
                        # TEST 中的 good 图片没有 mask
                        for image_path in imgpaths_per_class[classname][anomaly]:
                            if image_path.lower().endswith(image_extensions):
                                data_to_iterate.append([classname, anomaly, image_path, None])
                    else:
                        # TEST 中的异常图像配对 mask
                        for image_path in imgpaths_per_class[classname][anomaly]:
                            if not image_path.lower().endswith(image_extensions):
                                continue  # 跳过 .txt 等非图像文件

                            # 构造对应的 mask 路径
                            basename = os.path.splitext(os.path.basename(image_path))[0]
                            # 假设 image_path 是：.../test/<anomaly>/<filename>.png
                            # mask 路径为：.../ground_truth/<anomaly>/<filename>.png
                            test_dir = os.path.dirname(os.path.dirname(image_path))  # .../test
                            root_dir = os.path.dirname(test_dir)  # .../<classname>
                            mask_path = os.path.join(root_dir, "ground_truth", anomaly, basename + "_mask.png")

                            if not os.path.exists(mask_path):
                                print(f"⚠️  mask 不存在: {mask_path}")

                            data_to_iterate.append([classname, anomaly, image_path, mask_path])
        return imgpaths_per_class, data_to_iterate
