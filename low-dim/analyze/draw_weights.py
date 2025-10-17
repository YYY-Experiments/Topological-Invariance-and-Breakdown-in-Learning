import torch as tc
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def draw(exp_id, logged_weights, frame_prop):
    if isinstance(logged_weights, tc.Tensor):
        weights_np = logged_weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(logged_weights)

    assert weights_np.ndim == 3 and weights_np.shape[2] == 2

    num_frames, num_points, _ = weights_np.shape
    target_frame = int(num_frames * frame_prop)
    target_frame = min(max(target_frame, 0), num_frames - 1)
    data = weights_np[target_frame]  # (D, 2)

    plt.rcParams.update({"font.size": 20})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(
        data[:, 0], data[:, 1],
        s=40, c="#4a90e2", alpha=0.6,
        edgecolors="w", linewidths=0.6
    )

    ax.set_xlabel("$w_{i,1}$", fontsize=20, fontweight="bold")
    ax.set_ylabel("$a_{i}$", fontsize=20, fontweight="bold")
    ax.grid(False)

    plt.tight_layout()

    save_path = f"figures/exp_{exp_id}/epoch_{target_frame}.png"
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="20")
    parser.add_argument("--frame_prop", type=float, default=0.)
    args = parser.parse_args()

    save_path = f"./results/{args.exp_id}.pkl"
    with open(save_path, "rb") as fil:
        info = pickle.load(fil)
    logged_weights = info["logged_weights"]

    draw(
        exp_id=args.exp_id,
        logged_weights=logged_weights,
        frame_prop=args.frame_prop,
    )
