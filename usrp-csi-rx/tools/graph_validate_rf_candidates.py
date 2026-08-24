#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def pairwise_dist(P):
    d = P[:, None, :] - P[None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("candidates_npz")
    ap.add_argument("--out", required=True)

    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sigma_xyz", type=float, default=0.15)
    ap.add_argument("--sigma_score", type=float, default=0.15)

    ap.add_argument("--lambda_graph", type=float, default=0.35)
    ap.add_argument("--iterations", type=int, default=20)

    ap.add_argument("--keep_threshold", type=float, default=0.60)
    ap.add_argument("--max_points", type=int, default=300)

    args = ap.parse_args()

    p = Path(args.candidates_npz)
    d = np.load(p)

    P = d["points_xyz"].astype(np.float32)
    score = d["values"].astype(np.float32)
    conf = d["confidence"].astype(np.float32)

    if P.ndim != 2 or P.shape[1] != 3:
        raise SystemExit(
            f"[ERROR] expected points_xyz [N,3], got {P.shape}"
        )

    N = len(P)

    if N == 0:
        raise SystemExit("[ERROR] candidate set is empty")

    # Combined initial RF reliability.
    score_n = normalize01(score)
    conf_n = normalize01(conf)

    initial = (
        0.75 * score_n +
        0.25 * conf_n
    ).astype(np.float32)

    # Pairwise geometry.
    D = pairwise_dist(P)

    np.fill_diagonal(D, np.inf)

    k = min(args.k, max(1, N - 1))

    nbr_idx = np.argpartition(
        D,
        kth=k - 1,
        axis=1
    )[:, :k]

    W = np.zeros((N, N), dtype=np.float32)

    sx = max(args.sigma_xyz, 1e-6)
    ss = max(args.sigma_score, 1e-6)

    for i in range(N):
        for j in nbr_idx[i]:

            dij = D[i, j]
            ds = initial[i] - initial[j]

            w_xyz = np.exp(
                -0.5 * (dij / sx) ** 2
            )

            w_score = np.exp(
                -0.5 * (ds / ss) ** 2
            )

            W[i, j] = float(
                w_xyz * w_score
            )

    # Symmetrize graph.
    W = np.maximum(W, W.T)

    degree = W.sum(axis=1)

    Wn = W / (
        degree[:, None] + 1e-8
    )

    refined = initial.copy()

    lam = float(
        np.clip(args.lambda_graph, 0.0, 1.0)
    )

    # Graph regularization / confidence propagation.
    for _ in range(args.iterations):

        neighbor_support = Wn @ refined

        refined = (
            (1.0 - lam) * initial
            + lam * neighbor_support
        ).astype(np.float32)

    refined = normalize01(refined)

    graph_support = normalize01(
        W @ refined
    )

    final_score = normalize01(
        0.70 * refined
        + 0.30 * graph_support
    )

    order = np.argsort(
        final_score
    )[::-1]

    keep = order[
        final_score[order] >= args.keep_threshold
    ]

    if len(keep) == 0:
        print(
            "[WARN] no points passed keep_threshold; "
            "falling back to highest-scoring points"
        )
        keep = order

    keep = keep[:args.max_points]

    P_keep = P[keep]
    score_keep = score[keep]
    conf_keep = conf[keep]
    initial_keep = initial[keep]
    refined_keep = refined[keep]
    support_keep = graph_support[keep]
    final_keep = final_score[keep]

    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        out,

        points_xyz=P_keep.astype(np.float32),

        rf_score=score_keep.astype(np.float32),
        rf_confidence=conf_keep.astype(np.float32),

        initial_score=initial_keep.astype(np.float32),
        graph_refined_score=refined_keep.astype(np.float32),
        graph_support=support_keep.astype(np.float32),
        final_score=final_keep.astype(np.float32),

        original_indices=keep.astype(np.int64),

        graph_degree=degree[keep].astype(np.float32),

        source=str(p),

        k=np.array(k),
        sigma_xyz=np.array(args.sigma_xyz),
        sigma_score=np.array(args.sigma_score),
        lambda_graph=np.array(args.lambda_graph),
        iterations=np.array(args.iterations),
        keep_threshold=np.array(args.keep_threshold),
    )

    print(f"[OK] wrote: {out}")

    print("[GRAPH] input points:", N)
    print("[GRAPH] kept points:", len(keep))
    print("[GRAPH] k:", k)

    print(
        "[GRAPH] degree mean/std:",
        float(degree.mean()),
        float(degree.std())
    )

    print(
        "[GRAPH] initial mean/std:",
        float(initial.mean()),
        float(initial.std())
    )

    print(
        "[GRAPH] refined mean/std:",
        float(refined.mean()),
        float(refined.std())
    )

    print(
        "[GRAPH] final mean/std/min/max:",
        float(final_score.mean()),
        float(final_score.std()),
        float(final_score.min()),
        float(final_score.max())
    )

    print("[GRAPH] top-20 refined candidates:")

    for rank, idx in enumerate(keep[:20], start=1):

        print(
            f"{rank:02d}: "
            f"xyz={P[idx].tolist()} "
            f"rf={float(score[idx]):.6f} "
            f"conf={float(conf[idx]):.6f} "
            f"graph={float(graph_support[idx]):.6f} "
            f"final={float(final_score[idx]):.6f}"
        )


if __name__ == "__main__":
    main()
