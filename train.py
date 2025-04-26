"""
Minimal demo training script — learns on synthetic random spectrogram.
Replace `target` with your real (F,N) complex lattice for actual use.
"""
import math
import torch
from torch.optim import AdamW

from model.codec_model import CodecModel
from model.inspect import pretty_print, show_lattices

def main():
    
    # ------------------------------------------------ synthetic demo data
    F, N = 64, 256
    rng = torch.Generator().manual_seed(0)

    # target = (torch.randn(F, N, generator=rng) +
    #           1j * torch.randn(F, N, generator=rng)) * 0.1
    
    # 2-D sinusoid:  period N/3 along time, F/2 along freq
    f_idx = torch.arange(F).float().unsqueeze(1)          # (F,1)
    t_idx = torch.arange(N).float().unsqueeze(0)          # (1,N)

    phase_real = 2*math.pi * (t_idx / (N/3) + f_idx / (F/2))
    phase_imag = 2*math.pi * (t_idx / (N/3) - f_idx / (F/2))   # opposite tilt

    target = torch.sin(phase_real) + 1j * torch.sin(phase_imag)   # (F,N) complex

    # ------------------------------------------------ hyper-params
    # K, M = 64, 500            # patterns, occurrences
    K, M = 4, 5000            # patterns, occurrences
    epochs = 5000

    model = CodecModel(F, N, K, M)
    opt   = AdamW(model.parameters(), lr=1e-3)

    tau_sched = torch.linspace(1.0, 0.1, epochs)

    for epoch, tau in enumerate(tau_sched):
        opt.zero_grad()
        A_pred = model(float(tau))
        loss   = model.loss(A_pred, target, float(tau))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 20 == 0:
            print(f"epoch {epoch:03d}  loss={loss.item():.5f}")

    print("Done — trained on random spectrogram.")

    pretty_print(model, top=10)

    show_lattices(model, target, tau=0.05, rows=24, cols=80)

if __name__ == "__main__":
    main()

