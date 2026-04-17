# build_flashhead_cache_qwen3.py
# 作用：
# 1) 从 Qwen3-1.7B 的 lm_head.weight 提取 token embeddings
# 2) 执行 spherical k-means（cosine-based）
# 3) 保存 FlashHead 所需的 clustering_cache.safetensors
#
# 输出内容：
#   - centroids: [num_clusters, hidden_dim]
#   - cluster_assignments: [vocab_size]
#
# 用法示例：
# python build_flashhead_cache_qwen3.py \
#   --model_path /data/users/tongf/master_thesis_tang/models/Qwen3-1.7B \
#   --output_dir /data/users/tongf/master_thesis_tang/models/Qwen3-1.7B/flash_head_cache \
#   --num_clusters 8192 \
#   --max_iters 100 \
#   --seed 42
#
# 建议：
# - 第一轮先用 4096 clusters + 50~100 iters 跑通
# - 正式实验再尝试 8192 clusters + 更高迭代数

import os
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_clusters", type=int, default=8192)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--tol_fraction_changed", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default="auto")

    # batched assignment / update
    parser.add_argument("--assign_batch_size", type=int, default=4096)
    parser.add_argument("--init_candidate_batch_size", type=int, default=8192)

    # 是否使用 tied weights 对应的 lm_head
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def batched_argmax_assign(
    X: torch.Tensor,          # [N, D], normalized
    C: torch.Tensor,          # [K, D], normalized
    batch_size: int = 4096,
):
    """
    对每个样本分配最近 centroid（cosine = dot，因为都已归一化）。
    返回：
      assignments: [N]
      max_scores: [N]
    """
    N = X.shape[0]
    device = X.device

    assignments = torch.empty(N, dtype=torch.long, device=device)
    max_scores = torch.empty(N, dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = X[start:end]                       # [B, D]
        sims = xb @ C.T                         # [B, K]
        scores, idx = sims.max(dim=1)
        assignments[start:end] = idx
        max_scores[start:end] = scores.float()

    return assignments, max_scores


@torch.no_grad()
def recompute_centroids(
    X: torch.Tensor,            # [N, D], normalized
    assignments: torch.Tensor,  # [N]
    num_clusters: int,
):
    """
    根据 assignments 重算 centroid，然后做单位化。
    若某个 cluster 空了，先留零向量，后续由调用方处理。
    """
    N, D = X.shape
    device = X.device

    centroids = torch.zeros((num_clusters, D), dtype=X.dtype, device=device)
    counts = torch.zeros((num_clusters,), dtype=torch.long, device=device)

    centroids.index_add_(0, assignments, X)
    counts.index_add_(0, assignments, torch.ones_like(assignments, dtype=torch.long))

    non_empty = counts > 0
    centroids[non_empty] = F.normalize(centroids[non_empty], dim=1)

    return centroids, counts


@torch.no_grad()
def refill_empty_clusters(
    X: torch.Tensor,
    centroids: torch.Tensor,
    counts: torch.Tensor,
    assignments: torch.Tensor,
    scores: torch.Tensor,
):
    """
    处理空 cluster：
    用“当前最差匹配”的样本来补空 cluster。
    这是工程上常见的稳定化做法。
    """
    empty = (counts == 0).nonzero(as_tuple=False).flatten()
    if empty.numel() == 0:
        return centroids, assignments

    # 找到最差匹配样本（score 越小表示离其当前 centroid 越远）
    worst_idx = torch.argsort(scores)[: empty.numel()]

    for e, wi in zip(empty.tolist(), worst_idx.tolist()):
        centroids[e] = X[wi]
        assignments[wi] = e

    centroids = F.normalize(centroids, dim=1)
    return centroids, assignments


@torch.no_grad()
def spherical_kmeans_plus_plus_init(
    X: torch.Tensor,        # [N, D], normalized
    num_clusters: int,
    candidate_batch_size: int = 8192,
    seed: int = 42,
):
    """
    一个可用的 cosine / spherical k-means++ 初始化。
    思路：
      - 第一个中心随机选
      - 后续中心偏向选择“与现有中心最大相似度较低”的点
    注意：
      这是面向工程落地的 batched 版本，不追求极致数学精致，但足够实用。
    """
    set_seed(seed)
    device = X.device
    N, D = X.shape

    first_idx = torch.randint(0, N, (1,), device=device).item()
    centers = [X[first_idx].clone()]   # list of [D]

    # 维护每个点到已选中心的最大 cosine 相似度
    best_sim = torch.full((N,), -1.0, dtype=torch.float32, device=device)

    def update_best_sim(new_center: torch.Tensor):
        nc = new_center.unsqueeze(1)  # [D,1] not used directly
        for start in range(0, N, candidate_batch_size):
            end = min(start + candidate_batch_size, N)
            xb = X[start:end]
            sim = (xb @ new_center).float()  # [B]
            best_sim[start:end] = torch.maximum(best_sim[start:end], sim)

    update_best_sim(centers[0])

    for k in range(1, num_clusters):
        # 与现有中心越不像，weight 越大
        # best_sim in [-1,1]，转成非负权重
        weights = (1.0 - best_sim).clamp_min(1e-8)
        probs = weights / weights.sum()

        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_center = X[next_idx].clone()

        centers.append(next_center)
        update_best_sim(next_center)

        if k % 500 == 0 or k == num_clusters - 1:
            print(f"[init] selected {k + 1}/{num_clusters} centers")

    C = torch.stack(centers, dim=0)  # [K, D]
    C = F.normalize(C, dim=1)
    return C


@torch.no_grad()
def spherical_kmeans(
    X: torch.Tensor,
    num_clusters: int,
    max_iters: int = 100,
    tol_fraction_changed: float = 1e-4,
    assign_batch_size: int = 4096,
    init_candidate_batch_size: int = 8192,
    seed: int = 42,
):
    """
    spherical k-means 主过程
    """
    print("Initializing centroids with spherical k-means++ ...")
    C = spherical_kmeans_plus_plus_init(
        X=X,
        num_clusters=num_clusters,
        candidate_batch_size=init_candidate_batch_size,
        seed=seed,
    )

    prev_assignments = None

    for it in range(max_iters):
        t0 = time.perf_counter()

        # Step 1: assign
        assignments, scores = batched_argmax_assign(
            X=X,
            C=C,
            batch_size=assign_batch_size,
        )

        # Step 2: recompute
        C_new, counts = recompute_centroids(
            X=X,
            assignments=assignments,
            num_clusters=num_clusters,
        )

        # Step 3: refill empties if needed
        if (counts == 0).any():
            C_new, assignments = refill_empty_clusters(
                X=X,
                centroids=C_new,
                counts=counts,
                assignments=assignments,
                scores=scores,
            )
            C_new, counts = recompute_centroids(
                X=X,
                assignments=assignments,
                num_clusters=num_clusters,
            )

        t1 = time.perf_counter()

        # convergence check
        if prev_assignments is None:
            frac_changed = 1.0
        else:
            frac_changed = (assignments != prev_assignments).float().mean().item()

        non_empty = (counts > 0).sum().item()
        min_count = counts[counts > 0].min().item() if non_empty > 0 else 0
        max_count = counts.max().item()

        print(
            f"[iter {it + 1:03d}] "
            f"time={t1 - t0:.2f}s | "
            f"frac_changed={frac_changed:.6f} | "
            f"non_empty={non_empty}/{num_clusters} | "
            f"cluster_size[min,max]=[{min_count},{max_count}]"
        )

        C = C_new
        prev_assignments = assignments.clone()

        if frac_changed < tol_fraction_changed:
            print(f"Converged at iter {it + 1} with frac_changed={frac_changed:.6f}")
            break

    return C, assignments


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    device = torch.device(args.device)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # 取 lm_head.weight
    with torch.no_grad():
        E = model.lm_head.weight.detach()

    # 如果用了 device_map="auto"，lm_head 可能已经在某张 GPU 上；统一搬到指定 device
    E = E.to(device=device, dtype=torch.float32)

    vocab_size, hidden_dim = E.shape
    print(f"lm_head.weight shape = [{vocab_size}, {hidden_dim}]")
    print(f"num_clusters         = {args.num_clusters}")
    print(f"max_iters            = {args.max_iters}")

    if args.num_clusters >= vocab_size:
        raise ValueError("num_clusters must be < vocab_size")

    # spherical k-means：先单位化
    print("Normalizing embeddings...")
    X = normalize_rows(E)  # [V, D], float32

    # 聚类
    start_time = time.perf_counter()
    centroids, cluster_assignments = spherical_kmeans(
        X=X,
        num_clusters=args.num_clusters,
        max_iters=args.max_iters,
        tol_fraction_changed=args.tol_fraction_changed,
        assign_batch_size=args.assign_batch_size,
        init_candidate_batch_size=args.init_candidate_batch_size,
        seed=args.seed,
    )
    end_time = time.perf_counter()

    # 保存
    cache_path = output_dir / "clustering_cache.safetensors"
    save_file(
        {
            "centroids": centroids.cpu().contiguous(),
            "cluster_assignments": cluster_assignments.cpu().contiguous(),
        },
        str(cache_path),
    )

    meta = {
        "model_path": args.model_path,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_clusters": args.num_clusters,
        "max_iters": args.max_iters,
        "tol_fraction_changed": args.tol_fraction_changed,
        "seed": args.seed,
        "dtype_for_model_load": args.dtype,
        "device": args.device,
        "assign_batch_size": args.assign_batch_size,
        "init_candidate_batch_size": args.init_candidate_batch_size,
        "elapsed_seconds": end_time - start_time,
        "output_file": str(cache_path),
    }

    with open(output_dir / "clustering_cache_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved cache to: {cache_path}")
    print(f"Saved metadata to: {output_dir / 'clustering_cache_meta.json'}")
    print(f"Elapsed: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()