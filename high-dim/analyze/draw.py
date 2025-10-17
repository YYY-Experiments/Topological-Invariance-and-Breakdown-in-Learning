import torch as tc
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import os 
from tqdm import tqdm
from analyze.base import get_data

sns.set_theme(style="whitegrid")

def mean_and_std(seq_list):
    assert len(seq_list) > 0
    min_len = min(len(s) for s in seq_list)
    data = tc.tensor([s[:min_len] for s in seq_list], dtype=tc.float32)  # [n_exp, T]
    mean = data.mean(dim=0)
    std = data.var(dim=0, unbiased=False).sqrt() 
    return mean, std


def draw(all_b0, all_b1, all_b2, ax, small=False):
    b0_mean, b0_std = mean_and_std(all_b0)
    b1_mean, b1_std = mean_and_std(all_b1)
    b2_mean, b2_std = mean_and_std(all_b2)
    x = list(range(len(b0_mean)))

    c_b0, c_b1, c_b2 = "#ff7f0e", "#2ca02c", "#d62728"
    linewidth = 1 if small else 3
    shade_alpha = 0.1 if small else 0.18

    ax.plot(x, b0_mean.tolist(), label="$b_0$", color=c_b0, linewidth=linewidth, alpha = 0.8)
    ax.fill_between(x,
                     (b0_mean - b0_std).tolist(),
                     (b0_mean + b0_std).tolist(),
                     color=c_b0, alpha=shade_alpha, linewidth=0)

    ax.plot(x, b1_mean.tolist(), label="$b_1$", color=c_b1, linewidth=linewidth, alpha = 0.8)
    ax.fill_between(x,
                     (b1_mean - b1_std).tolist(),
                     (b1_mean + b1_std).tolist(),
                     color=c_b1, alpha=shade_alpha, linewidth=0)

    ax.plot(x, b2_mean.tolist(), label="$b_2$", color=c_b2, linewidth=linewidth, linestyle = '--', alpha=0.8)
    ax.fill_between(x,
                     (b2_mean - b2_std).tolist(),
                     (b2_mean + b2_std).tolist(),
                     color=c_b2, alpha=shade_alpha, linewidth=0)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_ids", type=str, default="20")
    parser.add_argument("--inset_exp_ids", type=str, default="20")
    parser.add_argument("--inset_pos", type=str, default="top", choices=["top", "mid"])
    args = parser.parse_args()

    exp_ids = args.exp_ids.split(",")

    all_b0, all_b1, all_b2 = [], [], []
    for exp_id in tqdm(exp_ids, desc="Loading experiments"):
        all_b0.append(get_data(exp_id, "b0"))
        all_b1.append(get_data(exp_id, "b1"))
        all_b2.append(get_data(exp_id, "b2"))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    draw(all_b0, all_b1, all_b2, ax1)


    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Betti numbers", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(False)

    if args.inset_pos == "top":
        axins = ax1.inset_axes([0.08, 0.50, 0.38, 0.38])
    elif args.inset_pos == "mid":
        axins = ax1.inset_axes([0.08, 0.38, 0.38, 0.30])
    else:
        raise ValueError(f"Invalid inset position: {args.inset_pos}")

    inset_exp_ids = args.inset_exp_ids.split(",")
    ins_all_b0, ins_all_b1, ins_all_b2 = [], [], []
    for exp_id in tqdm(inset_exp_ids, desc="Loading inset experiments"):

        ins_all_b0.append(get_data(exp_id, "b0"))
        ins_all_b1.append(get_data(exp_id, "b1"))
        ins_all_b2.append(get_data(exp_id, "b2"))

    draw(ins_all_b0, ins_all_b1, ins_all_b2, axins)

    axins.tick_params(axis="y", labelsize=12)
    axins.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axins.grid(False)
    axins.set_title("Small $\eta$", fontsize=14)
    axins.set_xticklabels([])

    ax1.legend(fontsize=20, loc="upper right", frameon=True)

    os.makedirs("figures/betti", exist_ok=True)
    plt.tight_layout()
    save_name = f"[{exp_ids[0]}][{inset_exp_ids[0]}].png"
    plt.savefig(f"figures/betti/{save_name}", dpi=300)