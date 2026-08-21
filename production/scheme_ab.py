"""Систематика экспоненцирования: delta со схемой A и без неё.

Это свойство ФОРМАЛИЗМА, а не выборки: rc_factor(exponentiate=) считает обе
версии в одной точке.  Гонять ради этого генератор бессмысленно вдвойне --
он читает EXPO_TAIL (events.py:203), а квадратура всегда экспоненцирует, так
что «прогон схемы B» сравнивал бы одно с другим.
"""
import os, sys
import numpy as np
sys.path.insert(0, "/Users/vpk/exclurad_py/production"); sys.path.insert(0, "/Users/vpk/exclurad_py")
os.environ["CARD_PREFIX"] = "g1"
from validate_channel import parse_card, get_model
from exclurad_py.core.rc import rc_factor


def main():
    rng = np.random.default_rng(20260821)
    for ch in ("pi0", "eta", "rho", "phi", "jpsi"):
        c = parse_card(ch); sf = get_model(c, ch)
        mx2 = c["vcut"] + sf.mU ** 2 if c["vcut"] > 0 else None
        d = []
        for _ in range(160):
            Q2 = rng.uniform(*c["Q2"]); W2 = rng.uniform(*c["W2"])
            t = rng.uniform(c["t"][0], c["t"][1]); ph = rng.uniform(0, 360)
            if hasattr(sf, "t_nucl"):
                sf.t_nucl = float(t)
            kw = dict(Ebeam=c["ebeam"], Q2=Q2, W2=W2, t=t, phi=ph,
                      sfmodel=sf, h=+1, mx2_cut=mx2)
            a = rc_factor(**kw, exponentiate=True)
            b = rc_factor(**kw, exponentiate=False)
            if a["sigma_born"] > 0 and np.isfinite(a["sigma_obs"]) \
               and np.isfinite(b["sigma_obs"]):
                d.append((a["sigma_obs"] / a["sigma_born"],
                          b["sigma_obs"] / b["sigma_born"]))
        d = np.array(d)
        r = d[:, 1] / d[:, 0] - 1.0
        print(f"  {ch:5}: delta(A) {d[:,0].mean():.4f}   "
              f"B/A - 1: среднее {100*r.mean():+6.3f}%, "
              f"размах [{100*r.min():+6.3f}, {100*r.max():+6.3f}]%  "
              f"({len(d)} точек)", flush=True)


if __name__ == "__main__":
    main()
