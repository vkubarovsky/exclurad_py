"""How many hard-photon rad events survive the cut chain: true hadronic W' = M(p' g g)
and generated v = MM2(e'p) - m_pi^2 of the accepted rad events (September amp2609 sample).

The uncut rad sample has 20 % of its events at W' < 1.8 (outside the model's fitted range).
They are harmless only if the exclusivity cuts reject them, so the surviving fraction is
quoted per cut chain."""
import numpy as np
import rc_method_pi0 as R
from rc_method_pi0_combos import note_cuts


def wtrue(a):
    p = [a[:, 1 + 3 * k:4 + 3 * k] for k in range(1, 4)]
    E = np.sqrt((p[0] ** 2).sum(1) + R.MP ** 2) + np.linalg.norm(p[1], axis=1) + np.linalg.norm(p[2], axis=1)
    P = p[0] + p[1] + p[2]
    return np.sqrt(np.maximum(E ** 2 - (P ** 2).sum(1), 0))


def main():
    rng = np.random.default_rng(20261021)
    rows = {}
    for pol in ("inb", "outb"):
        S = {a: R.load(a, pol, rng) for a in ("rad", "born")}
        cal = R.calibrate(S["born"]["rec"])
        S = S["rad"]
        a = np.fromfile(f"{R.DATA}/ntuple/rad_{pol}.f32", "<f4").reshape(-1, R.NF).astype(np.float64)
        Wt = wtrue(a)
        v = S["gen"]["mm2ep"] - R.MPI ** 2
        k = S["rec"]
        dis = (k["Q2"] > 1) & (k["W"] > 2) & (k["pe"] > 2)
        chains = {
            "generated (no reco)": np.ones(len(a), bool),
            "reco topology": S["ok"].copy(),
        }
        for name, m in (("+ DIS", dis),
                        ("+ DIS + |ME|<0.2", dis & (np.abs(k["me"]) < 0.2)),
                        ("+ DIS + |MM2tot|<0.004", dis & (np.abs(k["mm2tot"]) < 0.004)),
                        ("full note chain", note_cuts(k, cal))):
            full = np.zeros(len(a), bool); full[S["ok"]] = m
            chains[name] = full
        for name, m in chains.items():
            r = rows.setdefault(name, [])
            r.append((Wt[m], v[m]))
    print(f"{'selection':28s} {'N':>8s} {'W<1.8':>7s} {'W<1.4':>7s} {'v>0.2':>7s} {'v>0.5':>7s}")
    for name, r in rows.items():
        W = np.concatenate([x[0] for x in r]); v = np.concatenate([x[1] for x in r])
        print(f"{name:28s} {len(W):8d} {np.mean(W < 1.8):7.4f} {np.mean(W < 1.4):7.4f} "
              f"{np.mean(v > 0.2):7.4f} {np.mean(v > 0.5):7.4f}")


if __name__ == "__main__":
    main()
