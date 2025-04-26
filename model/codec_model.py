import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pattern_bank import PatternBank
from .occurrences import OccurrenceParameters
from .synthesiser import bilinear_deposit


class CodecModel(nn.Module):
    """
    Full differentiable codec model (patterns + occurrences + lattice build).
    """

    def __init__(self, F: int, N: int, K: int, M: int):
        super().__init__()
        self.F, self.N = F, N
        self.bank = PatternBank(K)
        self.occ  = OccurrenceParameters(M, K)

    # --------------------------------------------------------------------- #
    def forward(self, tau: float) -> torch.Tensor:
        # unpack both gates
        P, dF, dT, g_soft, g_hard = self.bank()
        sel   = self.occ.select_patterns(tau)          # (M,K)
        k_idx = sel.argmax(-1)                         # (M,)

        P_sel  = P[k_idx]                              # (M,9)
        dF_sel = dF[k_idx]
        dT_sel = dT[k_idx]
        g_sel  = g_soft[k_idx]                         # (M,9)  weights 0–1

        f_c = (self.F / (2 * math.pi)) * self.occ.zeta_f
        n_c = (self.N / (2 * math.pi)) * self.occ.zeta_t
        amp = torch.exp(self.occ.log_rho) * torch.exp(1j * self.occ.theta)

        A = torch.zeros(self.F, self.N,
                        dtype=torch.cfloat, device=P.device)

        for c in range(9):
            w_c = g_sel[:, c]                          # soft weights
            if w_c.max() < 1e-6:                       # skip if truly zero
                continue
            f_hat = f_c + dF_sel[:, c]
            n_hat = n_c + dT_sel[:, c]
            val   = amp * P_sel[:, c] * w_c            # weight by gate
            bilinear_deposit(A, f_hat, n_hat, val)
        return A

    # --------------------------------------------------------------------- #
    def loss(self,
             A_pred: torch.Tensor,
             A_target: torch.Tensor,
             tau: float) -> torch.Tensor:
        """Combined reconstruction + regularisers."""
        mse = (A_pred - A_target).abs().pow(2).mean()

        sparsity_g   = self.bank.logit_gate.sigmoid().mean()
        sparsity_occ = F.softplus(self.occ.log_rho).mean()

        pattern_sel  = self.occ.select_patterns(tau)
        entropy = -(pattern_sel.mean(0) *
                    torch.log_softmax(self.occ.alpha, -1)).sum()

        return mse + 1e-3 * sparsity_g + 1e-3 * sparsity_occ + 1e-4 * entropy
