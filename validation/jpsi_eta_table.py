"""Production radiative correction in W x t' bins, from a Born and a rad sample.

    python3 validation/jpsi_eta_table.py <born run dir> <rad run dir> [theta_lo theta_hi]

eta(bin) = [n_rad(bin)/N_rad * sigma_rad] / [n_born(bin)/N_born * sigma_born],
with Q2, W, t reconstructed from the scattered electron and the J/psi of the
LUND record (the generator's observed kinematics), t' = |t| - |t|_min(W, Q2).

sigma of the WRITTEN sample: sigma_nb in config_*.json is counted before the
hadron/lepton acceptance cut, so each part's sigma_nb is scaled by
written/accepted (accept2 with EXACT_ACCEPT, else accepted), then averaged
over parts.  The optional theta window (degrees) on the scattered electron is
applied to both samples offline -- e.g. 2.5 4.5 for the Forward Tagger.
Both samples should use the same idecay, so the decay radiation cancels.
"""
import glob
import json
import os
import sys

import numpy as np

G = np.array([1, -1, -1, -1.0])
MP, MJ = 0.93827208816, 3.0969


def load(d):
    rows, sig = [], []
    for f in sorted(glob.glob(os.path.join(d, "jpsi", "*", "lund", "*.lund"))):
        part = f.rsplit("_", 1)[-1].split(".")[0]
        cfg = glob.glob(os.path.join(d, "jpsi", "*", "inputs", f"config_{part}.json"))
        r = json.load(open(cfg[0]))["report"]
        L = [l for l in open(f) if l.strip() and not l.startswith("#")]
        i = n = 0
        while i < len(L):
            h = L[i].split()
            k, E = int(h[0]), float(h[6])
            P = [l.split() for l in L[i + 1:i + 1 + k]]
            i += 1 + k
            n += 1
            v4 = lambda p: np.array([float(p[9]), float(p[6]), float(p[7]), float(p[8])])  # noqa: E731
            e = next(v4(p) for p in P if p[3] == "11" and p[4] == "0")
            v = next(v4(p) for p in P if p[3] == "443")
            th = np.degrees(np.arccos(e[3] / np.linalg.norm(e[1:])))
            q = np.array([E, 0, 0, E]) - e
            Q2 = -(q @ (G * q))
            W2 = MP * MP + 2 * MP * q[0] - Q2
            t = (q - v) @ (G * (q - v))
            rows.append((Q2, W2, t, th))
        acc = r["accept2"] if r.get("accept2", 0) > 0 else r["accepted"]
        sig.append(r["sigma_nb"] * n / acc)
    Q2, W2, t, th = np.array(rows).T
    W = np.sqrt(W2)
    pi = np.sqrt(((W2 + Q2 + MP ** 2) / (2 * W)) ** 2 - MP ** 2)
    ev = (W2 + MJ ** 2 - MP ** 2) / (2 * W)
    pf = np.sqrt(np.maximum(ev * ev - MJ ** 2, 0))
    tmin = (pi - pf) ** 2 - ((Q2 + MJ ** 2) / (2 * W)) ** 2
    return dict(Q2=Q2, W=W, tp=-t - tmin, th=th, sigma=float(np.mean(sig)), N=len(W))


def main(db, dr, thlo=0.0, thhi=180.0):
    b, r = load(db), load(dr)
    sb = (b["th"] >= thlo) & (b["th"] <= thhi)
    sr = (r["th"] >= thlo) & (r["th"] <= thhi)

    def eta(mb, mr):
        nb, nr = mb.sum(), mr.sum()
        if nb < 100 or nr < 30:
            return None
        x = (nr / r["N"] * r["sigma"]) / (nb / b["N"] * b["sigma"])
        return x, x * np.sqrt(1 / nr + 1 / nb)

    print(f"written: Born {b['N']}, rad {r['N']};  theta_e {thlo}-{thhi} deg keeps "
          f"Born {sb.sum()} ({100*sb.mean():.1f} %), rad {sr.sum()} ({100*sr.mean():.1f} %)")
    q = b["Q2"][sb]
    print(f"selected Born: Q2 median {np.median(q):.3f}, 10-90 % {np.percentile(q,10):.3f}-"
          f"{np.percentile(q,90):.3f} GeV2;  W quartiles {np.round(np.percentile(b['W'][sb],[25,50,75]),3)};"
          f"  t' quartiles {np.round(np.percentile(b['tp'][sb],[25,50,75]),2)}")
    x = eta(np.ones(b["N"], bool), np.ones(r["N"], bool))
    print(f"all written events:   sigma_rad/sigma_born = {x[0]:.4f} +- {x[1]:.4f}")
    x = eta(sb, sr)
    print(f"theta window total:   eta = {x[0]:.4f} +- {x[1]:.4f}\n")
    We = [4.04, 4.16, 4.28, 4.40, 4.56]
    Te = [0, 0.3, 0.8, 1.5, 3, 7]
    print("W \\ t'     " + "".join(f"{Te[j]:>6.1f}-{Te[j+1]:<6.1f}  " for j in range(5)) + "   all t'     Born share")
    for i in range(4):
        wb = sb & (b["W"] >= We[i]) & (b["W"] < We[i + 1])
        wr = sr & (r["W"] >= We[i]) & (r["W"] < We[i + 1])
        row = f"{We[i]:.2f}-{We[i+1]:.2f}  "
        for j in range(5):
            x = eta(wb & (b["tp"] >= Te[j]) & (b["tp"] < Te[j + 1]),
                    wr & (r["tp"] >= Te[j]) & (r["tp"] < Te[j + 1]))
            row += f"{'-':>14}" if x is None else f"{x[0]:6.3f}+-{x[1]:.3f}  "
        x = eta(wb, wr)
        row += f"  {x[0]:.3f}+-{x[1]:.3f}   {100*wb.sum()/sb.sum():5.1f} %"
        print(row)
    row = "all W      "
    for j in range(5):
        x = eta(sb & (b["tp"] >= Te[j]) & (b["tp"] < Te[j + 1]),
                sr & (r["tp"] >= Te[j]) & (r["tp"] < Te[j + 1]))
        row += f"{'-':>14}" if x is None else f"{x[0]:6.3f}+-{x[1]:.3f}  "
    print(row)
    print("Born share per t' column: " + "  ".join(
        f"{100*(sb & (b['tp'] >= Te[j]) & (b['tp'] < Te[j+1])).sum()/sb.sum():.1f} %" for j in range(5)))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(float(x) for x in a[2:4]))
