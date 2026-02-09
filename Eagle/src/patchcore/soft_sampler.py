import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

class GreedyCoresetSampler_valid(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
            self,
            features: Union[torch.Tensor, np.ndarray],
            return_indices: bool = False,  # <<< 新增：可选是否返回 sample_indices
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            if return_indices:
                if isinstance(features, np.ndarray):
                    N = features.shape[0]
                    return features, np.arange(N, dtype=np.int64)
                else:
                    N = features.shape[0]
                    return features, np.arange(N, dtype=np.int64)
            return features

        self._store_type(features)

        if isinstance(features, np.ndarray):
            features_t = torch.from_numpy(features)
        else:
            features_t = features

        reduced_features = self._reduce_features(features_t)
        sample_indices_t = self._compute_greedy_coreset_indices(reduced_features)  # torch.LongTensor 或 ndarray

        # 统一成 torch.LongTensor
        if not isinstance(sample_indices_t, torch.Tensor):
            sample_indices_t = torch.as_tensor(sample_indices_t, dtype=torch.long)
        # 原始特征的采样
        sampled_features = features_t[sample_indices_t]

        sampled_features = self._restore_type(sampled_features)
        sample_indices_np = sample_indices_t.detach().cpu().numpy()

        if return_indices:
            return sampled_features, sample_indices_np
        return sampled_features
    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

class ApproximateGreedyCoresetSampler(GreedyCoresetSampler_valid):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        init_strategy: str = "center_farthest",  # Add strategy selector
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        self.init_strategy = init_strategy
        self.init_indices = None
        super().__init__(percentage, device, dimension_to_project_features_to)


    def _pick_start_points(self, features: torch.Tensor) -> list:
        N = len(features)
        k = int(np.clip(self.number_of_starting_points, 1, N))
        if self.init_indices is not None:
            return list(np.unique(self.init_indices)[:k])

        if self.init_strategy == "center_farthest":
            center = features.mean(dim=0, keepdim=True)
            d = self._compute_batchwise_differences(features, center).reshape(-1)
            c_idx = int(torch.argmin(d))
            seeds = [c_idx]
            d_all = self._compute_batchwise_differences(features, features[c_idx:c_idx+1]).reshape(-1)
            _, far_idx = torch.topk(d_all, k-1, largest=True)
            seeds += far_idx.cpu().tolist()
            return seeds

        if self.init_strategy == "pca_extremes":
            X = features - features.mean(dim=0, keepdim=True)
            u, s, vT = torch.linalg.svd(X, full_matrices=False)
            pc1 = vT[0:1, :]
            proj = (X @ pc1.T).reshape(-1)
            max_idx = int(torch.argmax(proj))
            min_idx = int(torch.argmin(proj))
            seeds = [max_idx, min_idx]
            if k > 2:
                vals, idx = torch.sort(proj)
                grid = torch.linspace(0, N-1, steps=k-2).round().long()
                extra = idx[grid].cpu().tolist()
                seeds += extra
            return list(dict.fromkeys(seeds))[:k]
        # Default: random
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        return np.random.choice(len(features), number_of_starting_points, replace=False ).tolist()


    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )

        start_points = self._pick_start_points(features)
        # start_points = np.random.choice(
        #     len(features), number_of_starting_points, replace=False
        # ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]

class WeightedGreedyCoresetSampler(ApproximateGreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)
        self.sampling_weight = None

    def set_sampling_weight(self, sampling_weight):
        self.sampling_weight = sampling_weight


# class GreedyCoresetSampler_valid(BaseSampler):
#     def __init__(
#             self,
#             percentage: float,
#             device: torch.device,
#             dimension_to_project_features_to=128,
#     ):
#         """Greedy Coreset sampling base class."""
#         super().__init__(percentage)
#
#         self.device = device
#         self.dimension_to_project_features_to = dimension_to_project_features_to
#
#     def _reduce_features(self, features):
#         if features.shape[1] == self.dimension_to_project_features_to:
#             return features
#         mapper = torch.nn.Linear(
#             features.shape[1], self.dimension_to_project_features_to, bias=False
#         )
#         _ = mapper.to(self.device)
#         features = features.to(self.device)
#         return mapper(features)
#
#     def run(
#             self,
#             features: Union[torch.Tensor, np.ndarray],
#             return_indices: bool = False,  # <<< 新增：可选是否返回 sample_indices
#     ) -> Union[torch.Tensor, np.ndarray]:
#         """Subsamples features using Greedy Coreset.
#
#         Args:
#             features: [N x D]
#         """
#         if self.percentage == 1:
#             if return_indices:
#                 if isinstance(features, np.ndarray):
#                     N = features.shape[0]
#                     return features, np.arange(N, dtype=np.int64)
#                 else:
#                     N = features.shape[0]
#                     return features, np.arange(N, dtype=np.int64)
#             return features
#
#         self._store_type(features)
#         if isinstance(features, np.ndarray):
#             features = torch.from_numpy(features)
#
#         reduced_features = self._reduce_features(features)
#         sample_indices = self._compute_greedy_coreset_indices(reduced_features)
#         features = features[sample_indices]
#         return self._restore_type(features)
#         if isinstance(features, np.ndarray):
#             features_t = torch.from_numpy(features)
#         else:
#             features_t = features
#
#         reduced_features = self._reduce_features(features_t)
#         sample_indices_t = self._compute_greedy_coreset_indices(reduced_features)  # torch.LongTensor 或 ndarray
#
#         # 统一成 torch.LongTensor
#         if not isinstance(sample_indices_t, torch.Tensor):
#             sample_indices_t = torch.as_tensor(sample_indices_t, dtype=torch.long)
#         # 原始特征的采样
#         sampled_features = features_t[sample_indices_t]
#
#         sampled_features = self._restore_type(sampled_features)
#         sample_indices_np = sample_indices_t.detach().cpu().numpy()
#
#         if return_indices:
#             return sampled_features, sample_indices_np
#         return sampled_features