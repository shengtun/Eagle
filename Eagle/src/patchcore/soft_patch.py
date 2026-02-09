"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.soft_sampler
import patchcore.Attention
import patchcore.utils
LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.soft_sampler.ApproximateGreedyCoresetSampler(0.1, device=0,
                                                                            number_of_starting_points=10,
                                                                             dimension_to_project_features_to=128,
                                                                             init_strategy="center_farthest"),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.embedding_list = []


    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    feature, patch_shapes = self._embed(input_image,provide_patch_shapes=True)
                    features.append(feature)
                    self.patch_shapes = patch_shapes
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            # 将layer3 插值
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)


    def attention_step(self, features):  # save locally aware patch features
        """
        Attention module
        """
        # x, _, file_name, _ = input_data
        # features = self(features)
        embeddings = []
        for feature in features:
            # print(feature.shape)
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)
            # mvtec :layer2:28,layer3:14/ mvtec_loco:layer2:32,layer3:16
            # mvtec :layer2:28,layer3:14/ mvtec_loco:layer2:32,layer3:16
            if feature.shape[3] == 32:
                saconv_out = avep(feature)

            elif feature.shape[3] == 16:
                attention_in1 = maxp(feature[:])
                attention_in2 = maxp(feature[:])
                attention_v = maxp(feature[:])
                width = attention_in1.shape[3]

                # Flatten: [B, C, H, W] → [B, C, H*W]
                attention_in1_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_in2_flatten = torch.flatten(attention_in1[:], 2, 3)
                attention_v = torch.flatten(attention_v[:], 2, 3)

                # Reshape for attention: [B, C, HW, 1] and [B, C, 1, HW]
                attention_in1 = torch.reshape(attention_in1_flatten, (
                attention_in1_flatten.shape[0], attention_in1_flatten.shape[1], attention_in1_flatten.shape[2], 1))
                attention_in2 = torch.reshape(attention_in2_flatten, (
                attention_in2_flatten.shape[0], attention_in2_flatten.shape[1], 1, attention_in2_flatten.shape[2]))
                attention_v = torch.reshape(attention_v,
                                            (attention_v.shape[0], attention_v.shape[1], attention_v.shape[2], 1))

                attention_out = torch.matmul(attention_in1, attention_in2)
                attention_out = torch.nn.functional.softmax(attention_out, dim=-1)
                attention_out = torch.matmul(attention_out, attention_v)
                saconv_out = torch.reshape(attention_out,
                                           (attention_out.shape[0], attention_out.shape[1], width, width))

            embeddings.append(saconv_out)
        # only use attention-layer3, do not add another bench
        # embedding = patchcore.utils.embedding_concat(embeddings[0], embeddings[1])
        # self.embedding_list.extend(patchcore.utils.reshape_embedding(np.array(embedding)))

        return embeddings

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank_valid(training_data)

    def _fill_memory_bank_valid(self, input_data):
        """Build memory bank (S) and also extract non-selected patches (U) with image ids."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)  # [P, D] tensor or np.ndarray

        feats_list, img_ids_list = [], []
        img_counter = 0

        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for batch in data_iterator:
                if isinstance(batch, dict):
                    batch = batch["image"]
                if isinstance(batch, torch.Tensor) and batch.ndim > 3:
                    batchsize = batch.shape[0]
                    for i in range(batchsize):
                        image = batch[i].unsqueeze(0)
                        feats = _image_to_features(image)
                        if isinstance(feats, torch.Tensor):
                            feats = feats.detach().cpu().numpy()
                        P = len(feats)
                        feats_list.append(feats)
                        img_ids_list.extend([img_counter] * P)
                        img_counter += 1
                else:
                    feats = _image_to_features(batch)
                    if isinstance(feats, torch.Tensor):
                        feats = feats.detach().cpu().numpy()
                    P = len(feats)
                    feats_list.append(feats)
                    img_ids_list.extend([img_counter] * P)
                    img_counter += 1

        # 采样前（全体 patch）
        features_all = np.concatenate(feats_list, axis=0)  # [N_total_patches, D]
        # self.features_all = features_all  # 仅供可视化检查
        img_ids_all = np.asarray(img_ids_list, dtype=np.int32)  # [N_total_patches]
        assert features_all.shape[0] == img_ids_all.shape[0]

        # coreset 采样，得到 S 的全局索引（相对 features_all）
        features_S, sample_indices = self.featuresampler.run(features_all, return_indices=True)
        selected_img_ids = img_ids_all[sample_indices]  # 与 memory bank 顺序对齐

        N = features_all.shape[0]
        print("N =", N,
              "min(sample_idx) =", int(sample_indices.min()),
              "max(sample_idx) =", int(sample_indices.max()))
        assert 0 <= sample_indices.min() and sample_indices.max() < N

        n_images = int(img_ids_all.max()) + 1
        sel_cnt = np.bincount(img_ids_all[sample_indices], minlength=n_images)
        all_cnt = np.bincount(img_ids_all, minlength=n_images)
        uns_cnt = all_cnt - sel_cnt
        print("images with 0 selected:", np.sum(sel_cnt == 0))
        print("example (first 20):")
        for i in range(min(20, n_images)):
            print(i, "all=", int(all_cnt[i]), "S=", int(sel_cnt[i]), "U=", int(uns_cnt[i]))

        # U = 非 S 的补丁
        all_idx = np.arange(len(img_ids_all))
        unselected_sample_indices= np.setdiff1d(all_idx, sample_indices, assume_unique=False)
        unselected_feats = features_all[unselected_sample_indices]
        unselected_img_ids = img_ids_all[unselected_sample_indices]

        # —— 缓存 ——（后面预测会用）
        # Memory bank（S）
        self.memory_bank_indices = sample_indices  # S 的全局 patch 索引
        self.memory_bank_img_ids = selected_img_ids  # S 的 image 索引（与 memory bank 顺序一致）
        # 非 S（U）
        self.nonselected_indices = unselected_sample_indices
        self.nonselected_feats = unselected_feats  # U 的特征
        self.nonselected_img_ids = unselected_img_ids  # U 的 image 索引

        # 在 S 上建库（fit Faiss 索引）
        self.anomaly_scorer.fit(detection_features=[features_S])
        self.sel_cnt = sel_cnt
        self.all_cnt = all_cnt
        self.uns_cnt = uns_cnt


        return unselected_feats

        # scores = self._predict_from_nonselected_features(unselected_feats)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            # resnet
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            # vit
            # patch_scores = patch_scores.reshape(batchsize, 14, 14)

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def _predict_from_nonselected_features(self, features):
        """
        用 _fill_memory_bank_valid() 得到的非 S patch 计算各图像的 anomaly score（避免信息泄露）。
        仅计算 image-level 分数；不返回 mask（因U_i缺失S_i，无法稳妥复原栅格）。

        依赖字段：
          - self.nonselected_feats        : (|U|, D)
          - self.nonselected_img_ids      : (|U|,)
          - self.memory_bank_img_ids      : (|S|,)
          - self.anomaly_scorer.nn_method : 已在S上fit好的Faiss索引，支持 run(k, query, index=subset)

        返回：
          image_scores : (N_images,) float32，每张图的异常分数（不足 min_unselected 的为 NaN）
        """
        _ = self.forward_modules.eval()
        batchsize = 2
        min_unselected = 32

        assert hasattr(self, "nonselected_feats") and hasattr(self, "nonselected_img_ids"), \
            "Call _fill_memory_bank_valid() first to populate nonselected_*."
        assert hasattr(self, "memory_bank_img_ids"), \
            "memory_bank_img_ids missing; ensure _fill_memory_bank_valid() completed."

        import numpy as np, math

        U_feats = self.nonselected_feats  # (|U|, D) (225792,1024)
        U_img_ids = self.nonselected_img_ids  # (|U|,) (225792,)
        S_img_ids = self.memory_bank_img_ids  # (|S|,) (25088,)

        if len(U_feats) == 0:
            return np.array([], dtype=np.float32)

        n_images = int(U_img_ids.max()) + 1
        image_scores = np.full((n_images,), np.nan, dtype=np.float32)

        # U_i：每张图未选中的patch行号（在 U_feats 中的行号）
        U_sets = [np.where(U_img_ids == i)[0] for i in range(n_images)]

        # memory bank 的“局部索引”为 0..|S|-1
        ref_all_local = np.arange(len(S_img_ids), dtype=np.int64)

        with torch.no_grad():
            for i in tqdm.tqdm(range(n_images), desc = "threshold scoring"):
                U_i = U_sets[i]
                if len(U_i) < min_unselected:
                    continue
                query = U_feats[U_i] # 该图的非S补丁作为查询
                #####################################
                s_i_local = np.where(S_img_ids ==i)[0]  # 该图的S补丁在 memory bank 中的局部索引
                ref_allowed = np.setdiff1d(ref_all_local, s_i_local, assume_unique=False)
                if len(ref_allowed) == 0:
                    continue
                ######################################
                self.features_all = self.anomaly_scorer.detection_features
                index_features = self.features_all[ref_allowed]
                torch.cuda.empty_cache()
                query_distances, query_nns= self.anomaly_scorer.nn_method.run(
                    self.anomaly_scorer.n_nearest_neighbours, query, index_features
                )
                patch_scores = np.mean(query_distances, axis=-1)
                # patch_scores = self.anomaly_scorer.predict([query])[0]  # （1568，）
                image_score = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=1
                )
                image_score = image_score.reshape(*image_score.shape[:2], -1)
                image_score = self.patch_maker.score(image_score)

                image_scores[i] = image_score

        return image_scores
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        # 是在构建一个 Unfold 模块，用于把 feature map 切成 patch（滑动窗口）
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

import matplotlib.pyplot as plt

def save_heatmap(seg_map, save_path):
    plt.imsave(save_path, seg_map, cmap='jet')