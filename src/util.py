import os
import json
import numpy as np
import matplotlib.pyplot as plt


def save_json(key, data, path="figures/results.json"):
    db = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                db = json.load(f)
        except Exception:
            db = {}

    db[key] = data
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def plot_baseline(key="baseline", json_path="figures/results.json", save_path="figures/baseline.png"):
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        db = json.load(f)

    if key not in db:
        return

    data = db[key]
    r_data, f_data = data.get("resampled", {}), data.get("fixed", {})

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

    all_means = []

    if r_data:
        M_r = sorted([int(k) for k in r_data.keys()])
        mu_r = [np.mean([x["eval_stat"] if isinstance(x, dict) else x for x in r_data[str(m)]]) for m in M_r]
        sd_r = [np.std([x["eval_stat"] if isinstance(x, dict) else x for x in r_data[str(m)]]) for m in M_r]
        all_means.extend(mu_r)
        ax.plot(M_r, mu_r, marker="o", color="#2ca02c", linewidth=2.5, label="Resampled")
        ax.fill_between(M_r, [m - s for m, s in zip(mu_r, sd_r)], [m + s for m, s in zip(mu_r, sd_r)], color="#2ca02c", alpha=0.15)

    if f_data:
        M_f = sorted([int(k) for k in f_data.keys()])
        mu_f = [np.mean([x["eval_stat"] if isinstance(x, dict) else x for x in f_data[str(m)]]) for m in M_f]
        sd_f = [np.std([x["eval_stat"] if isinstance(x, dict) else x for x in f_data[str(m)]]) for m in M_f]
        all_means.extend(mu_f)
        ax.plot(M_f, mu_f, marker="s", color="#1f77b4", linewidth=2.5, label="Fixed")
        ax.fill_between(M_f, [m - s for m, s in zip(mu_f, sd_f)], [m + s for m, s in zip(mu_f, sd_f)], color="#1f77b4", alpha=0.15)
    ax.set_xlabel("M (number of directions)", fontsize=11)
    ax.set_ylabel(r"$\mathbb{E}_a[T(\{a^\top f_\theta(x_n)\}_{n=1}^N)]$", fontsize=11)

    ax.margins(y=0.1)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
