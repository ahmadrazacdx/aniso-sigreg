import torch
from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest


def sigreg(M=1024, t_max=5.0, n_points=17):
    ep = EppsPulley(t_max=t_max, n_points=n_points)
    return SlicingUnivariateTest(
        univariate_test=ep,
        num_slices=M,
        reduction="mean",
    )


def eval_D(Z, M_eval=2048, t_max=5.0, n_points=17, seed=0):
    K = Z.shape[1]
    gen = torch.Generator(device=Z.device)
    gen.manual_seed(seed)

    A = torch.randn(K, M_eval, device=Z.device, dtype=Z.dtype, generator=gen)
    A = A / A.norm(p=2, dim=0)

    ep = EppsPulley(t_max=t_max, n_points=n_points).to(Z.device)
    proj = Z @ A  # (N, M_eval)
    stats = ep(proj)  # (M_eval,)
    return stats.mean().item()
