import torch as tc
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def draw(exp_id, logged_weights, frame_prop, elev=30, azim=-60, zoom=1.0, show=False):
    if isinstance(logged_weights, tc.Tensor):
        weights_np = logged_weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(logged_weights)

    assert weights_np.ndim == 3 and weights_np.shape[2] == 3

    num_frames, num_points, _ = weights_np.shape
    target_frame = int(num_frames * frame_prop)
    target_frame = min(max(target_frame, 0), num_frames - 1)
    data = weights_np[target_frame]  # (D, 3)

    plt.rcParams.update({"font.size": 20})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        s=40, c="#4a90e2", alpha=0.6,
        edgecolors="w", linewidths=0.6
    )

    ax.set_xlabel("$w_{i,1}$", fontsize=20, fontweight="bold", labelpad=12)
    ax.set_ylabel("$w_{i,2}$", fontsize=20, fontweight="bold", labelpad=12)
    ax.set_zlabel("$a_{i}$", fontsize=20, fontweight="bold", labelpad=12)

    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
    ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min], dtype=float)
    max_range = ranges.max() if np.isfinite(ranges).all() else 1.0
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0
    half = max_range * zoom / 2 if max_range > 0 else 0.5
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, color="gray")
    # ax.grid(False)


    ax.w_xaxis.line.set_color((1, 1, 1, 0))
    ax.w_yaxis.line.set_color((1, 1, 1, 0))
    ax.w_zaxis.line.set_color((1, 1, 1, 0))

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    save_path = f"figures/exp_{exp_id}/epoch_{target_frame}.png"
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True)
    if show:
        plt.show()
        return 
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="20")
    parser.add_argument("--frame_prop", type=float, default=0.) 
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--zoom", type=float, default=0.6)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    save_path = f"./results/{args.exp_id}.pkl"
    with open(save_path, "rb") as fil:
        info = pickle.load(fil)
    logged_weights = info["logged_weights"]

    draw(
        exp_id = args.exp_id, 
        logged_weights=logged_weights,
        frame_prop=args.frame_prop,
        elev=args.elev,
        azim=args.azim,
        zoom=args.zoom,
        show=args.show,
    )
