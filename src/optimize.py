import random
import numpy as np
import torch
from lejepa.univariate import EppsPulley
from slicing import sigreg, eval_D
from util import save_json, plot_baseline


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_emb(N=1000, K=512, M=16, steps=10000, lr=0.01, seed=42, t_max=5.0, n_points=17, resample=True, device="cuda"):
    set_seeds(seed)
    Z = torch.randn(N, K, device=device, dtype=torch.float32)
    Z[:, 1] = Z[:, 0].sign() * torch.randn(N, device=device, dtype=torch.float32).abs()
    Z = torch.nn.Parameter(Z)

    if resample:
        loss_fn = sigreg(M, t_max=t_max, n_points=n_points).to(device)
    else:
        ep = EppsPulley(t_max=t_max, n_points=n_points).to(device)
        g = torch.Generator(device=device).manual_seed(seed)
        A = torch.randn(K, M, device=device, dtype=torch.float32, generator=g)
        A = A / A.norm(p=2, dim=0)

    optimizer = torch.optim.Adam([Z], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        if resample:
            loss = loss_fn(Z)
        else:
            proj = Z @ A  # (N, M)
            loss = ep(proj).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Z_d = Z.detach()
        eval_stat = eval_D(Z_d, M_eval=2048, t_max=t_max, n_points=n_points, seed=seed + 10000)
        L = torch.linalg.eigvalsh(torch.cov(Z_d.T))

    return {"eval_stat": eval_stat, "cond": (L[-1] / L[0]).item(), "trace": L.sum().item()}


def run_sweep(
    N=1000,
    K=512,
    M_values=(4, 8, 16, 32, 64, 128, 256, 512, 1024),
    steps=10000,
    lr=0.01,
    n_seeds=3,
    t_max=5.0,
    n_points=17,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_resample = {}
    results_fixed = {}

    for M in M_values:
        resample_stats = [
            train_emb(N, K, M, steps, lr, 42 + s, t_max, n_points, True, device)
            for s in range(n_seeds)
        ]
        fixed_stats = [
            train_emb(N, K, M, steps, lr, 42 + s, t_max, n_points, False, device)
            for s in range(n_seeds)
        ]
        results_resample[str(M)] = resample_stats
        results_fixed[str(M)] = fixed_stats

    data = {
        "resampled": results_resample,
        "fixed": results_fixed,
    }

    save_json("baseline", data, "figures/results.json")
    plot_baseline("baseline", "figures/results.json", "figures/baseline.png")

    return data


if __name__ == "__main__":
    run_sweep()
