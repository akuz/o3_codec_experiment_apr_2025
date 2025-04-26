import torch


def bilinear_deposit(lattice: torch.Tensor,
                     f_hat: torch.Tensor,
                     n_hat: torch.Tensor,
                     val: torch.Tensor) -> None:
    """
    Differentiable scatter-add into `lattice` using bilinear weights.

    Parameters
    ----------
    lattice : (F,N) complex tensor – modified in place
    f_hat, n_hat, val : broadcastable tensors of same shape
    """
    f1 = torch.floor(f_hat).long()
    n1 = torch.floor(n_hat).long()
    wf = (f_hat - f1).clamp(0, 1)
    wn = (n_hat - n1).clamp(0, 1)
    f2 = f1 + 1
    n2 = n1 + 1

    for fi, wf_part in [(f1, 1 - wf), (f2, wf)]:
        mask_f = (fi >= 0) & (fi < lattice.size(0))
        for nj, wn_part in [(n1, 1 - wn), (n2, wn)]:
            mask = mask_f & (nj >= 0) & (nj < lattice.size(1))
            if not mask.any():
                continue
            lattice.index_put_((fi[mask], nj[mask]),
                               (wf_part * wn_part * val)[mask],
                               accumulate=True)
