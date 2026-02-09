import argparse
import numpy as np
import torch
import tqdm
import os
import math
import matplotlib.pyplot as plt


# -------------- Utils --------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)


@torch.no_grad()
def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance matrix between rows of a and b.
    a: [Na, D], b: [Nb, D] -> [Na, Nb]
    """
    a2 = (a * a).sum(dim=1, keepdim=True)       # [Na,1]
    b2 = (b * b).sum(dim=1, keepdim=True).T     # [1,Nb]
    ab = a @ b.T                                 # [Na,Nb]
    d2 = torch.clamp(a2 + b2 - 2 * ab, min=0)
    return torch.sqrt(d2 + 1e-12)


@torch.no_grad()
def jl_project(features: torch.Tensor, out_dim: int = 128, device="cpu"):
    """
    Random linear projection (Johnson–Lindenstrauss style), no bias.
    Keeps distribution roughly, speeds up distance ops.
    """
    N, D = features.shape
    if D == out_dim:
        return features.to(device)
    W = torch.randn(D, out_dim, device=device) / math.sqrt(out_dim)
    return (features.to(device) @ W).to(device)


@torch.no_grad()
def pca_project_2d(features: torch.Tensor) -> torch.Tensor:
    """
    PCA to 2D via SVD: returns [N,2]
    """
    X = features - features.mean(dim=0, keepdim=True)
    # economical SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    PC2 = Vh[:2].T                  # [D,2]
    return X @ PC2                  # [N,2]


@torch.no_grad()
def coverage_metrics(features: torch.Tensor, coreset_idx: np.ndarray):
    """
    Compute k-center style coverage metrics:
    - max_min_dist:  max over points of min distance to coreset
    - mean_min_dist: average of min distances
    - p95_min_dist:  95th percentile of min distances
    """
    cores = features[coreset_idx]          # [k,D]
    D = pairwise_dist(features, cores)     # [N,k]
    minD, _ = torch.min(D, dim=1)          # [N]
    max_min = float(minD.max().item())
    mean_min = float(minD.mean().item())
    p95_min = float(torch.quantile(minD, 0.95).item())
    return dict(max_min_dist=max_min, mean_min_dist=mean_min, p95_min_dist=p95_min)


# -------------- Init strategies --------------
@torch.no_grad()
def init_random(N: int, k: int):
    return np.random.choice(N, k, replace=False).tolist()


@torch.no_grad()
def init_center_farthest(features: torch.Tensor, k: int):
    """
    1) pick the sample closest to global mean
    2) pick top-(k-1) farthest samples from this center
    """
    N = features.shape[0]
    k = int(np.clip(k, 1, N))

    center = features.mean(dim=0, keepdim=True)            # [1,D]
    d_to_center = pairwise_dist(features, center).reshape(-1)   # [N]
    c_idx = int(torch.argmin(d_to_center).item())

    seeds = [c_idx]
    if k > 1:
        d_all = pairwise_dist(features, features[c_idx:c_idx+1]).reshape(-1)  # [N]
        vals, idx = torch.topk(d_all, k-1, largest=True)
        add = idx.cpu().tolist()
        seeds += add
        # ensure uniqueness
        seeds = list(dict.fromkeys(seeds))[:k]
        # in rare case of duplicates, pad randomly
        if len(seeds) < k:
            extra = [i for i in range(N) if i not in seeds]
            np.random.shuffle(extra)
            seeds += extra[:(k - len(seeds))]
    return seeds


@torch.no_grad()
def init_pca_extremes(features: torch.Tensor, k: int):
    """
    Use PCA first component extremes as anchors, then fill with quantiles along PC1.
    """
    N = features.shape[0]
    k = int(np.clip(k, 1, N))
    X = features - features.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    pc1 = Vh[0:1, :]                          # [1,D]
    proj = (X @ pc1.T).reshape(-1)            # [N]
    max_idx = int(torch.argmax(proj).item())
    min_idx = int(torch.argmin(proj).item())
    seeds = [max_idx, min_idx]
    if k > 2:
        vals, order = torch.sort(proj)        # ascending
        # pick (k-2) approx. evenly spaced indices
        grid = torch.linspace(0, N - 1, steps=k - 2).round().long()
        extra = order[grid].cpu().tolist()
        seeds += extra
    # ensure uniqueness & cap to k
    seeds = list(dict.fromkeys(seeds))[:k]
    if len(seeds) < k:
        # fallback: add farthest from current seeds
        left = [i for i in range(N) if i not in seeds]
        if len(seeds) > 0:
            D = pairwise_dist(features[left], features[seeds])   # [L,|seeds|]
            minD, _ = torch.min(D, dim=1)                        # [L]
            _, add_idx = torch.topk(minD, k - len(seeds), largest=True)
            seeds += [left[i] for i in add_idx.cpu().tolist()]
        else:
            np.random.shuffle(left)
            seeds += left[:(k - len(seeds))]
    return seeds


INIT_STRATS = {
    "random": init_random,
    "center_farthest": init_center_farthest,
    "pca_extremes": init_pca_extremes,
}


# -------------- Approx Greedy Coreset --------------
@torch.no_grad()
def approximate_greedy_coreset(features: torch.Tensor,
                               percentage: float = 0.01,
                               k_starts: int = 10,
                               init_strategy: str = "random",
                               jl_dim: int = 128,
                               device: str = "cpu",
                               desc: str = "Subsampling"):
    """
    Approximate greedy coreset selection, with customizable init seeds.
    - features: [N,D] torch.Tensor (will not be modified)
    """
    assert 0 < percentage <= 1.0
    N, D = features.shape
    device = torch.device(device)

    # JL projection (for speed & memory)
    feats = jl_project(features, out_dim=jl_dim, device=device).contiguous()  # [N,jl_dim]

    # pick k starting anchors (distinct)
    k_starts = int(np.clip(k_starts, 1, N))
    seeds = INIT_STRATS[init_strategy](feats, k_starts) if init_strategy != "random" \
        else INIT_STRATS[init_strategy](N, k_starts)

    # init approximate distances: mean distance to seeds
    D0 = pairwise_dist(feats, feats[seeds])               # [N,k_starts]
    approx_minD = torch.min(D0, dim=1).values.view(-1, 1)  # [N,1]

    # number of samples to pick
    num_pick = int(max(1, round(N * percentage)))
    coreset_indices = []

    for _ in tqdm.tqdm(range(num_pick), desc=desc):
        # pick the farthest point from current anchors
        sel = int(torch.argmax(approx_minD).item())
        coreset_indices.append(sel)

        # update min distances
        d_new = pairwise_dist(feats, feats[sel: sel + 1])   # [N,1]
        approx_minD = torch.minimum(approx_minD, d_new)

    return np.array(coreset_indices), seeds


# -------------- Visualization --------------
@torch.no_grad()
def visualize_2d(features: torch.Tensor,
                 result_dict: dict,
                 savefig: str = None):
    """
    result_dict: {name: {'coreset_idx': np.ndarray, 'seeds': list}}
    """
    Z = pca_project_2d(features).cpu().numpy()  # [N,2]
    N = Z.shape[0]

    ncols = len(result_dict)
    plt.figure(figsize=(5 * ncols, 5))

    for i, (name, res) in enumerate(result_dict.items(), start=1):
        coreset_idx = set(res["coreset_idx"].tolist())
        seeds_idx = set(res["seeds"])

        ax = plt.subplot(1, ncols, i)
        ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.2, label="all")
        # mark coreset points
        cs = np.array(list(coreset_idx))
        ax.scatter(Z[cs, 0], Z[cs, 1], s=12, alpha=0.9, label="coreset")
        # mark seeds
        if len(seeds_idx) > 0:
            sd = np.array(list(seeds_idx))
            ax.scatter(Z[sd, 0], Z[sd, 1], s=24, marker="x", label="seeds")

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=200)
        print(f"[Saved] {savefig}")
    else:
        plt.show()


# -------------- Main --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None, help="Path to .npy feature array [N,D].")
    parser.add_argument("--synthetic", type=int, default=0, help="Use synthetic Gaussian blobs if 1.")
    parser.add_argument("--n", type=int, default=6000, help="Synthetic N.")
    parser.add_argument("--d", type=int, default=256, help="Synthetic D.")
    parser.add_argument("--percentage", type=float, default=0.01, help="Coreset percentage (0,1].")
    parser.add_argument("--k", type=int, default=10, help="number_of_starting_points (distinct).")
    parser.add_argument("--jl_dim", type=int, default=128, help="Projection dim for JL.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--savefig", type=str, default=None, help="Path to save visualization PNG.")

    args = parser.parse_args()
    set_seed(42)

    # Load or synthesize features
    if args.synthetic or args.features is None:
        # Make 3 Gaussian blobs to mimic multi-cluster industrial features
        n_per = args.n // 3
        centers = [np.zeros(args.d), np.ones(args.d)*3.0, np.concatenate([np.ones(args.d//2)*-3.0, np.ones(args.d - args.d//2)*2.0])]
        X = []
        for c in centers:
            X.append(np.random.randn(n_per, args.d) * 0.7 + c)
        rem = args.n - n_per * 3
        if rem > 0:
            X.append(np.random.randn(rem, args.d) * 0.7 + centers[0])
        feats_np = np.vstack(X).astype(np.float32)
        print(f"[Synthetic] features shape: {feats_np.shape}")
    else:
        feats_np = np.load(args.features)
        assert feats_np.ndim == 2, "features must be [N,D]"
        feats_np = feats_np.astype(np.float32)
        print(f"[Load] features: {args.features}, shape={feats_np.shape}")

    feats = torch.from_numpy(feats_np)

    # Run each strategy
    strategies = ["random", "center_farthest", "pca_extremes"]
    results = {}

    for strat in strategies:
        print(f"\n=== Strategy: {strat} ===")
        idx, seeds = approximate_greedy_coreset(
            feats, percentage=args.percentage, k_starts=args.k,
            init_strategy=strat, jl_dim=args.jl_dim, device=args.device,
            desc=f"Subsampling ({strat})"
        )
        metrics = coverage_metrics(feats, idx)
        for k, v in metrics.items():
            print(f"{k:>15}: {v:.6f}")
        results[strat] = {"coreset_idx": idx, "seeds": seeds, "metrics": metrics}

    # Optional visualization (2D PCA)
    if args.savefig is not None:
        visualize_2d(feats, results, savefig=args.savefig)


if __name__ == "__main__":
    main()
