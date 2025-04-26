import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternBank(nn.Module):
    """
    Dictionary of K patterns.
    Each pattern owns ≤ 9 active components inside a 5×5 stencil.
    Continuous offsets (no rounding) and straight-through gates.
    """

    def __init__(self, K: int):
        super().__init__()
        self.K = K

        # parameters ---------------------------------------------------------
        self.logit_gate = nn.Parameter(torch.randn(K, 9))         # gating logits
        self.m_raw      = nn.Parameter(0.01 * torch.randn(K, 9))  # raw magnitudes
        self.phi_int    = nn.Parameter(torch.randn(K, 9))         # intrinsic φ
        self.phi_f      = nn.Parameter(torch.randn(K, 9))         # rel-freq tag
        self.phi_t      = nn.Parameter(torch.randn(K, 9))         # rel-time tag
        # --------------------------------------------------------------------

    # ------------------------------------------------------------------ #
    def forward(self):
        """
        Returns
        -------
        P        : (K,9)  complex   – component values  m·e^{jφ}
        δf, δt   : (K,9)  real      – continuous offsets (no rounding)
        g_soft   : (K,9)  float     – sigmoid gates in [0,1]  (for weighting)
        g_hard   : (K,9)  0/1 float – hard gates after threshold (for display)
        """
        g_soft = self.logit_gate.sigmoid()          # (K,9)
        g_hard = (g_soft > 0.5).float()             # hard view

        # ---------- use **soft** gate for magnitudes so grads flow ----------
        m = torch.nn.functional.softplus(self.m_raw) * g_soft
        m = m / (m.pow(2).sum(-1, keepdim=True).clamp_min(1e-8).sqrt())

        P = m * torch.exp(1j * self.phi_int)        # complex (K,9)
        δf = (5.0 / (2 * math.pi)) * self.phi_f     # continuous offsets
        δt = (5.0 / (2 * math.pi)) * self.phi_t
        return P, δf, δt, g_soft, g_hard
