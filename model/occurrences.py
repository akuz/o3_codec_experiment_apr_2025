import torch
import torch.nn as nn
import torch.nn.functional as F


class OccurrenceParameters(nn.Module):
    """
    Track-specific set of at most M occurrences.
    Uses Gumbel-Softmax to choose pattern ids in a differentiable way.
    """

    def __init__(self, M: int, K: int):
        super().__init__()
        self.M, self.K = M, K
        # logits over patterns
        self.alpha   = nn.Parameter(torch.randn(M, K))
        # absolute position phases (periods F, N handled in codec model)
        self.zeta_f  = nn.Parameter(torch.rand(M) * 2 * torch.pi - torch.pi)
        self.zeta_t  = nn.Parameter(torch.rand(M) * 2 * torch.pi - torch.pi)
        # amplitude
        self.log_rho = nn.Parameter(torch.zeros(M))
        self.theta   = nn.Parameter(torch.rand(M) * 2 * torch.pi - torch.pi)

    # -------- utilities -----------------------------------------------------

    def select_patterns(self, tau: float):
        """
        Soft pattern selection with Gumbel-Softmax (temperature `tau`).
        Straight-through: hard one-hot in forward; gradients via soft probs.
        Returns
        -------
        sel : (M,K) float tensor  – one-hot rows in forward
        """

        # NOTE (AK): this also works, since we are not using "soft" for now:
        # return F.gumbel_softmax(self.alpha, tau=tau, hard=True, dim=-1)

        soft = F.gumbel_softmax(self.alpha, tau=tau, hard=False, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1,
                                               soft.argmax(dim=-1, keepdim=True),
                                               1.0)
        sel = hard.detach() + soft - soft.detach()
        return sel
    
