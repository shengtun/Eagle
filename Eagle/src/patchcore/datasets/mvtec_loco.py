import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "juice_bottle",
    "pushpins",
    "breakfast_box"
    "screw_bag"
    "splicing_connectors",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecLocoDataset(torch.utils.data.Dataset):
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
        **kwargs,
    ):
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # self.transform_img = [
        #     transforms.Resize(resize),
        #     transforms.CenterCrop(imagesize),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # ]
        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_img = transforms.Compose(self.transform_img)


        # self.transform_mask = [
        #     transforms.Resize((resize, resize)),
        #     transforms.CenterCrop(imagesize),
        #     transforms.ToTensor(),
        # ]
        self.transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ]

        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)              # imagesize:224

    def __getitem__(self, idx):
        #
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 递归处理下一个索引（需要边界检查）
            next_idx = (idx + 1) % len(self.data_to_iterate)
            if next_idx != idx:  # 避免无限循环
                return self.__getitem__(next_idx)
            else:
                raise ValueError("No valid images found")
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            if os.path.isdir(mask_path):  # 处理多掩码情况
                mask_files = sorted(os.listdir(mask_path))  # 读取文件夹内的所有掩码
                mask_list = [
                    self.transform_mask(PIL.Image.open(os.path.join(mask_path, f)))
                    for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))  # 过滤非图片文件
                ]
                if mask_list:
                    mask = torch.max(torch.stack(mask_list), dim=0)[0]  # 取多个掩码的最大值合并
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
            if self.split == DatasetSplit.TEST:
                classpath = os.path.join(self.source, classname, "test")
            else:
                classpath = os.path.join(self.source, classname, "train")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            # anomaly_types = os.listdir(classpath)
            anomaly_types = [
                d for d in os.listdir(classpath)
                if os.path.isdir(os.path.join(classpath, d))
            ]

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

        image_extensions = ('.png', '.jpg', '.jpeg')
        # Unrolls the data dictionary to an easy-to-iterate list.
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
                            mask_path = os.path.join(root_dir, "ground_truth", anomaly, basename)

                            if not os.path.exists(mask_path):
                                print(f"⚠️  mask 不存在: {mask_path}")

                            data_to_iterate.append([classname, anomaly, image_path, mask_path])
        #
        # ################## original ##################
        # for classname in sorted(imgpaths_per_class.keys()):
        #     for anomaly in sorted(imgpaths_per_class[classname].keys()):
        #         for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
        #             data_tuple = [classname, anomaly, image_path]
        #             if self.split == DatasetSplit.TEST and anomaly != "good":
        #                 data_tuple.append(maskpaths_per_class[classname][anomaly][i])
        #             else:
        #                 data_tuple.append(None)





                    # data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
