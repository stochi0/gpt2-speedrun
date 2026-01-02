"""
Visualize (scaled dot-product) self-attention step-by-step with plots.

This script is intentionally small and constructs
Q/K/V via a single fused projection (GPT-2 style `c_attn`), then shows:
  - Q, K, V for a chosen head
  - raw scores = Q @ K^T / sqrt(d_head)   (the transpose that makes shapes work)
  - optional causal masking
  - softmax probabilities
  - output = probs @ V                   (weighted sum over values)

Usage:
  uv run python scripts/attention_viz.py --outdir artifacts/attn_viz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _heatmap(
    ax: plt.Axes,
    data: torch.Tensor,
    title: str,
    x_label: str,
    y_label: str,
    *,
    xticks: list[str] | None = None,
    yticks: list[str] | None = None,
    cmap: str = "viridis",
    annotate: bool = False,
) -> None:
    # data expected 2D (H, W)
    img = ax.imshow(data.detach().cpu().float().numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks)

    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        h, w = data.shape
        for i in range(h):
            for j in range(w):
                ax.text(
                    j,
                    i,
                    f"{float(data[i, j]):.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if float(data[i, j]) < float(data.mean()) else "black",
                )


def _save(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def scaled_dot_product_attention(
    x: torch.Tensor,
    *,
    n_head: int,
    causal: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    """
    Build a tiny self-attention forward pass with a GPT-2 style fused qkv projection.

    Shapes:
      x:   (B, T, D)
      qkv: (B, T, 3D) -> split into q,k,v each (B, T, D)
      q,k,v -> reshape to heads:
        (B, T, n_head, d_head) -> transpose to (B, n_head, T, d_head)
      scores: (B, n_head, T, T) from q @ k^T
      probs:  (B, n_head, T, T) softmax over last dim
      out:    (B, n_head, T, d_head) from probs @ v
    """
    torch.manual_seed(seed)

    b, t, d = x.shape
    assert d % n_head == 0, "D must be divisible by n_head"
    d_head = d // n_head

    # Fused "c_attn": one projection produces [Q | K | V] along the last dimension.
    # In GPT-2 this is historically named Conv1D, but it is effectively a Linear.
    w_qkv = torch.randn(d, 3 * d) / math.sqrt(d)
    b_qkv = torch.zeros(3 * d)
    qkv = x @ w_qkv + b_qkv  # (B, T, 3D)
    q, k, v = qkv.split(d, dim=2)  # split on dim=2 because that's the feature axis

    # Reshape to multi-head: split D into (n_head, d_head)
    q = q.view(b, t, n_head, d_head).transpose(1, 2)  # (B, nh, T, dh)
    k = k.view(b, t, n_head, d_head).transpose(1, 2)  # (B, nh, T, dh)
    v = v.view(b, t, n_head, d_head).transpose(1, 2)  # (B, nh, T, dh)

    # Core attention: scores = Q K^T / sqrt(d_head)
    # The transpose is on the last two axes of K:
    #   Q: (T, dh)  and  K^T: (dh, T)  -> (T, T)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)  # (B, nh, T, T)

    if causal:
        # Mask out attending to "future" positions (upper triangle).
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)  # row-wise: for each query token, distribution over key tokens
    out = probs @ v  # (B, nh, T, dh)

    # Merge heads back: (B, T, D)
    out_merged = out.transpose(1, 2).contiguous().view(b, t, d)

    return {
        "x": x,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "probs": probs,
        "out": out,
        "out_merged": out_merged,
        "d_head": torch.tensor(d_head),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize self-attention with annotated plots.")
    parser.add_argument("--outdir", type=str, default="artifacts/attn_viz", help="Output directory for PNGs.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--head", type=int, default=0, help="Which head to visualize.")
    parser.add_argument("--causal", action="store_true", help="Apply causal mask.")
    parser.add_argument("--annotate", action="store_true", help="Write numeric values into heatmaps.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    b, t, d, nh = args.B, args.T, args.D, args.n_head
    if d % nh != 0:
        raise SystemExit(f"--D ({d}) must be divisible by --n_head ({nh})")
    if not (0 <= args.head < nh):
        raise SystemExit(f"--head must be in [0, {nh-1}]")

    torch.manual_seed(args.seed)
    # Tiny toy input: (B, T, D). Think of each row as a token embedding.
    x = torch.randn(b, t, d)

    tensors = scaled_dot_product_attention(x, n_head=nh, causal=args.causal, seed=args.seed + 1)
    dh = int(tensors["d_head"].item())

    # Choose a single (batch, head) slice for visualization.
    bh = 0
    h = args.head
    Q = tensors["q"][bh, h]  # (T, dh)
    K = tensors["k"][bh, h]  # (T, dh)
    V = tensors["v"][bh, h]  # (T, dh)
    S = tensors["scores"][bh, h]  # (T, T)
    P = tensors["probs"][bh, h]  # (T, T)
    O = tensors["out"][bh, h]  # (T, dh)

    token_labels = [f"t{i}" for i in range(t)]
    dim_labels = [f"d{i}" for i in range(dh)]

    # 1) Q/K/V heatmaps for one head
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    _heatmap(
        axes[0],
        Q,
        title=f"Q (head {h})  shape=(T, dh)=({t}, {dh})",
        x_label="head_dim",
        y_label="token (query index)",
        xticks=dim_labels,
        yticks=token_labels,
        cmap="magma",
        annotate=args.annotate,
    )
    _heatmap(
        axes[1],
        K,
        title=f"K (head {h})  shape=(T, dh)=({t}, {dh})",
        x_label="head_dim",
        y_label="token (key index)",
        xticks=dim_labels,
        yticks=token_labels,
        cmap="magma",
        annotate=args.annotate,
    )
    _heatmap(
        axes[2],
        V,
        title=f"V (head {h})  shape=(T, dh)=({t}, {dh})",
        x_label="head_dim",
        y_label="token (value index)",
        xticks=dim_labels,
        yticks=token_labels,
        cmap="magma",
        annotate=args.annotate,
    )
    _save(fig, outdir / "01_qkv.png")

    # 2) Show K^T explicitly (this is the transpose that makes Q @ K^T produce T×T)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    _heatmap(
        axes[0],
        K,
        title=f"K  shape=(T, dh)=({t}, {dh})",
        x_label="head_dim",
        y_label="token",
        xticks=dim_labels,
        yticks=token_labels,
        cmap="viridis",
        annotate=args.annotate,
    )
    _heatmap(
        axes[1],
        K.transpose(0, 1),
        title=f"K^T  shape=(dh, T)=({dh}, {t})",
        x_label="token",
        y_label="head_dim",
        xticks=token_labels,
        yticks=dim_labels,
        cmap="viridis",
        annotate=args.annotate,
    )
    fig.suptitle("Why transpose? Q:(T,dh) times K^T:(dh,T) -> scores:(T,T)", y=1.05)
    _save(fig, outdir / "02_k_transpose.png")

    # 3) Scores matrix (T×T): each row is a query token, columns are key tokens.
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _heatmap(
        ax,
        S,
        title=f"Scores = Q K^T / sqrt(dh)  shape=(T,T)=({t},{t})" + ("  [causal masked]" if args.causal else ""),
        x_label="key token index",
        y_label="query token index",
        xticks=token_labels,
        yticks=token_labels,
        cmap="coolwarm",
        annotate=args.annotate,
    )
    _save(fig, outdir / ("03_scores_causal.png" if args.causal else "03_scores.png"))

    # 4) Softmax probs (T×T): each row sums to 1.
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _heatmap(
        ax,
        P,
        title=f"Probs = softmax(scores) over keys  shape=(T,T)=({t},{t})",
        x_label="key token index",
        y_label="query token index",
        xticks=token_labels,
        yticks=token_labels,
        cmap="Blues",
        annotate=args.annotate,
    )
    _save(fig, outdir / ("04_probs_causal.png" if args.causal else "04_probs.png"))

    # 5) Output vectors for that head: (T, dh)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    _heatmap(
        ax,
        O,
        title=f"Output = probs @ V  shape=(T, dh)=({t},{dh})",
        x_label="head_dim",
        y_label="token index",
        xticks=dim_labels,
        yticks=token_labels,
        cmap="plasma",
        annotate=args.annotate,
    )
    _save(fig, outdir / "05_out.png")

    # 6) One-token “weighted sum” view (bar chart of attention weights for a chosen query token)
    query_idx = min(2, t - 1)
    weights = P[query_idx]  # (T,)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(token_labels, weights.detach().cpu().float().numpy())
    axes[0].set_title(f"Attention weights for query token t{query_idx} (row of probs)")
    axes[0].set_xlabel("key token")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Show the weighted-sum result for that token (a dh-dimensional vector)
    out_vec = O[query_idx]  # (dh,)
    axes[1].bar(dim_labels, out_vec.detach().cpu().float().numpy())
    axes[1].set_title(f"Output vector for t{query_idx} (head {h}) = Σ_j w_j * V_j")
    axes[1].set_xlabel("head_dim")
    axes[1].set_ylabel("value")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("The core idea: output token is a weighted sum of value vectors", y=1.05)
    _save(fig, outdir / "06_weighted_sum.png")

    # Write a tiny textual summary to help navigate the plots.
    summary = outdir / "README.txt"
    summary.write_text(
        "\n".join(
            [
                "Attention visualization outputs:",
                "",
                "01_qkv.png            - Q/K/V matrices for one head",
                "02_k_transpose.png    - K and K^T (the critical transpose in QK^T)",
                "03_scores*.png        - scores = QK^T/sqrt(dh) (optionally causal-masked)",
                "04_probs*.png         - softmax(scores) row-wise over keys",
                "05_out.png            - out = probs @ V for that head",
                "06_weighted_sum.png   - bar chart for one query token's weights + resulting vector",
                "",
                f"Config: B={b}, T={t}, D={d}, n_head={nh}, d_head={dh}, head={h}, causal={args.causal}",
                "",
                "Notes:",
                "- Q,K,V shown are for a single (batch, head).",
                "- scores[i,j] is 'how much query token i matches key token j'.",
                "- softmax is taken over j (keys) so each row sums to 1.",
                "- output token i is Σ_j probs[i,j] * V[j].",
                "",
            ]
        )
    )

    print(f"Wrote attention visualizations to: {outdir.resolve()}")
    print(f"Open: {summary.resolve()}")


if __name__ == "__main__":
    main()


