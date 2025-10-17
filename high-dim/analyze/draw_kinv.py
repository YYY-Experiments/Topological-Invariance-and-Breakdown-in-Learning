import torch as tc
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from tqdm import tqdm
import os
from analyze.base import get_data

sns.set_theme(style="whitegrid")

def mean_and_std(seq_list):
    assert len(seq_list) > 0
    min_len = min(len(s) for s in seq_list)
    data = tc.tensor([s[:min_len] for s in seq_list], dtype=tc.float32)  # [n_exp, T]
    mean = data.mean(dim=0)
    std = data.var(dim=0, unbiased=False).sqrt()  # 标准差；阴影显示均值±1σ
    return mean, std

def draw(Kinvs, label , small=False):
    Kinvs_mean, Kinvs_std = mean_and_std(Kinvs)
    x = list(range(len(Kinvs_mean)))
    c_acc = "#1f77b4" if not small else "#17becf"
    linewidth = 1 if small else 2
    shade_alpha = 0.1 if small else 0.22

    plt.plot(x, Kinvs_mean.tolist(), label=label, color=c_acc, linewidth=linewidth)
    plt.fill_between(x,
                    (Kinvs_mean - Kinvs_std).tolist(),
                    (Kinvs_mean + Kinvs_std).tolist(),
                    color=c_acc, alpha=shade_alpha, linewidth=0)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_ids", type=str, default="20")
    args = parser.parse_args()


    exp_ids = args.exp_ids.split(",")

    all_Kinvs = []
    for exp_id in tqdm(exp_ids, desc="Loading experiments"):
        all_Kinvs.append(get_data(exp_id, "Kinv"))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    draw(all_Kinvs, label="$1/K$")

    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Sharpness Inversion", fontsize=20)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(False)

    ax1.legend(fontsize=20, frameon=True)

    os.makedirs("figures/betti", exist_ok=True)
    plt.tight_layout()
    save_name = f"[{exp_ids[0]}].png"
    plt.savefig(f"figures/betti/Kinvs_{save_name}", dpi=300)