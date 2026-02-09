from abc import ABC, abstractmethod
import sys
sys.path.insert(0,"/home/smslab1/PycharmProjects/AnomalyDetection/Eagle/Eagle/src")
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.soft_patch
import patchcore.sampler
import patchcore.soft_sampler
import patchcore.utils
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pynvml

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def set_torch_device():
    # 初始化pynvml
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    max_free_memory = 0
    best_device = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i

    # 释放pynvml资源
    pynvml.nvmlShutdown()

    # 设置PyTorch设备
    device = torch.device(f'cuda:{best_device}' if torch.cuda.is_available() else 'cpu')
    return device
device = set_torch_device()

class ExpertModel_PatchCore:
    """
    视觉领域的预训练模型基类（Expert Model）
    用于提取图像特征或生成初步预测结果
    """

    def __init__(self, device: str = "cuda"):
        # self.model_path = model_path
        self.device = device
        self.model = None  # 存放加载好的模型



    def load_model(self, args):
        """加载预训练视觉模型"""
        input_shape = (3, args.imagesize, args.imagesize)
        backbone_names = list(args.backbone_names)
        if len(backbone_names) > 1:
            layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
            for layer in args.layers_to_extract_from:
                idx = int(layer.split(".")[0])
                layer = ".".join(layer.split(".")[1:])
                layers_to_extract_from_coll[idx].append(layer)
        else:
            layers_to_extract_from_coll = [args.layers_to_extract_from]

        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
                backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers)
            # sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.percentage, device)
            sampler = patchcore.soft_sampler.ApproximateGreedyCoresetSampler(args.percentage, device,
                                                                            number_of_starting_points=10,
                                                                             dimension_to_project_features_to=128,
                                                                             init_strategy= args.init_strategy)

            # patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance = patchcore.soft_patch.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device= device,
                input_shape=input_shape,
                pretrain_embed_dimension=args.pretrain_embed_dimension,
                target_embed_dimension=args.target_embed_dimension,
                patchsize= args.patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=args.anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)

        return loaded_patchcores


    def preprocess(self, image):
        """对输入图像进行预处理，返回模型可用的张量"""
        pass

    def fit(self, support_image, model):
        """对支持图像进行训练，更新模型参数"""
        Patchcore_list = model
        for Patchcore in Patchcore_list:
            Patchcore.fit(support_image)
            # the scores of normal images
            unselected_feats = Patchcore.nonselected_feats
            unselected_feats = Patchcore.nonselected_feats
            scores = Patchcore._predict_from_nonselected_features(unselected_feats)

        return scores

    def predict(self, support_image, test_image, model, args):
        self.transform_img = [
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, args.imagesize, args.imagesize)
        """对输入图像进行推理，返回预测结果（如bounding box, mask, 分数等）,image = {support image, query image}"""
        Patchcore_list = model # 假设model是PatchCore模型实例

        ###########################################################
        for Patchcore in Patchcore_list:
            if isinstance(test_image, torch.utils.data.DataLoader):
                scores, anomaly_maps, labels_gt, masks_gt = Patchcore.predict(test_image)
            else:
                test_image = Image.open(test_image).convert("RGB")
                test_image = self.transform_img(test_image)
                # test_image = test_image.resize((args.imagesize, args.imagesize))  # imagesize需和训练时一致
                # test_image_tensor = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0  # [C,H,W]
                test_image_tensor = test_image.unsqueeze(0)  # [1, C, H, W]
                scores, anomaly_maps = Patchcore.predict(test_image_tensor)

        return scores, anomaly_maps


    def postprocess(self, raw_output):
        """对模型原始输出进行后处理，转成结构化结果"""
        pass
