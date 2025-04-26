# ────────── project/model/inspect.py  (replace with this block) ──────────
import math
import torch
from .codec_model import CodecModel

# ── helper to build 5×5 ASCII heat-map ───────────────────────────────────
def _ascii_grid(mags, g_soft, dF, dT):
    shades = " .░▒▓█"       # 0-4 (lighter to darker)
    grid = [[" " for _ in range(5)] for _ in range(5)]

    if (g_soft > 0).any():
        m_max = mags[g_soft > 0].max().item()
    else:
        m_max = 1.0

    for c in range(9):
        if g_soft[c] < 1e-4:
            continue
        r = int(round(dF[c].item())) + 2
        c_ = int(round(dT[c].item())) + 2
        if 0 <= r < 5 and 0 <= c_ < 5:
            level = int((mags[c] / (m_max + 1e-8)) * (len(shades) - 1))
            grid[r][c_] = shades[level]
    return "\n".join("".join(row) for row in grid)

# ── pretty printer ───────────────────────────────────────────────────────
def pretty_print(model: CodecModel, top: int = 3):
    P, dF, dT, g_soft, g_hard = model.bank()

    print(f"\n=== PATTERN DICTIONARY ===")
    for k in range(P.size(0)):
        mags = P[k].abs()
        print(f"\nPattern {k:02d}")
        # numeric table
        print("idx |  δf   δt |   |m|      ∠φ  | gate_soft")
        print("-----------------------------------------------")
        for c in range(9):
            print(f"{c:2d} | {dF[k,c]:+5.2f} {dT[k,c]:+5.2f} | "
                  f"{mags[c]:8.6f} {torch.angle(P[k,c]):+5.2f} | "
                  f"{g_soft[k,c]:7.3f}")
        # ASCII thumbnail
        print("\n", _ascii_grid(mags, g_soft[k], dF[k], dT[k]), "\n")

    # ---------- occurrences ----------
    sel   = model.occ.select_patterns(tau=0.1)
    k_idx = sel.argmax(-1)
    f_c = (model.F / (2 * math.pi)) * model.occ.zeta_f
    n_c = (model.N / (2 * math.pi)) * model.occ.zeta_t
    rho = torch.exp(model.occ.log_rho)
    theta = model.occ.theta

    order = torch.argsort(rho, descending=True)
    print(f"\n=== OCCURRENCES (top {top}) ===")
    print("idx | pat |   row    col  |  |A|     ∠A")
    print("----------------------------------------------")
    for j in range(min(top, len(order))):
        idx = order[j]
        print(f"{idx:3d} | {int(k_idx[idx]):3d} | "
              f"{f_c[idx]:7.2f} {n_c[idx]:7.2f} | "
              f"{rho[idx]:7.4f} {theta[idx]:+6.2f}")

# ── ASCII visualisation of lattices ──────────────────────────────────────
def _ascii_lattice(lat: torch.Tensor,
                   vmax: float,
                   rows: int = 24,
                   cols: int = 80) -> str:
    """
    Down-sample `lat` (complex, shape F×N) to rows×cols and map magnitudes
    to ASCII shades.  Uses a shared vmax for consistent brightness.
    """
    # shades = " .:-=+*#%@"
    shades = " .░▒▓█"
    F, N   = lat.shape
    lat_abs = lat.abs()

    # average-pool to (rows,cols)
    pool_F = max(1, F // rows)
    pool_N = max(1, N // cols)
    small = lat_abs.unfold(0, pool_F, pool_F).unfold(1, pool_N, pool_N)
    small = small.contiguous().mean(-1).mean(-1)      # (rows,cols)

    # scale to 0-1
    small = (small / (vmax + 1e-8)).clamp(0, 1)

    # map to ASCII
    out_lines = []
    for r in range(small.size(0)):
        line = "".join(shades[int(v.item() * (len(shades) - 1))]
                       for v in small[r])
        out_lines.append(line)
    return "\n".join(out_lines)


def show_lattices(model: CodecModel,
                  target: torch.Tensor,
                  tau: float = 0.05,
                  rows: int = 24,
                  cols: int = 80):
    """
    Print ASCII maps of TARGET, RECONSTRUCTION, and ABS DIFFERENCE.
    """
    with torch.no_grad():
        recon = model(tau)

    diff = (target - recon).abs()
    vmax = torch.stack([target.abs(), recon.abs(), diff]).max().item()

    print("\n=== TARGET (magnitude) ===")
    print(_ascii_lattice(target, vmax, rows, cols))

    print("\n=== RECONSTRUCTION ===")
    print(_ascii_lattice(recon, vmax, rows, cols))

    print("\n=== ABS DIFFERENCE ===")
    print(_ascii_lattice(diff, vmax, rows, cols))
