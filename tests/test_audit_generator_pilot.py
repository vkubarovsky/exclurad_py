"""Focused checks for the production pilot gate."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.audit_generator_pilot import audit


def make_fixture():
    root = Path(tempfile.mkdtemp())
    (root / "tables").mkdir()
    (root / "rad" / "prov").mkdir(parents=True)
    (root / "rad" / "lund_osg").mkdir()

    M = np.array([[[[2.0]]], [[[1.0]]]])
    B = np.array([[[[1.5]]], [[[2.0]]]])
    edges = {
        "Q2": np.array([1.0, 2.0, 4.0]),
        "W2": np.array([4.0, 5.0]),
        "t": np.array([-1.0, 0.0]),
        "phi": np.array([0.0, 1.0]),
    }
    volume = np.array([1.0, 2.0]).reshape(2, 1, 1, 1)
    mass = (M * B * volume).ravel()
    cdf = np.cumsum(mass) / mass.sum()
    np.savez_compressed(
        root / "tables" / "majorant_rad.npz",
        M=M, B=B, cdf=cdf, shape=np.array(M.shape),
        b_shape=np.array(B.shape), B_global=np.array(2.0),
        diag_margin=np.array(2.0), diag_probes_per_cell=np.array(1600),
        diag_workers=np.array(32), diag_max_cells=np.array(400),
        diag_min_usable=np.array(20),
        **{"e_" + key: value for key, value in edges.items()},
    )
    normalization = float(mass.sum())
    for index in (1, 2):
        trials, events = 100, 10
        record = {
            "card": {"nev": events},
            "stamp": {"git_commit": "abcdef1"},
            "report": {
                "trials": trials, "accept2": events,
                "accepted": 2000,
                "hard": 6, "soft": 4, "b_viol": 0,
                "max_b_viol": 1.0, "overweight": 0,
                "sigma_nb": events / trials * normalization,
            },
            "health": [],
            "gen_s": 12.0,
        }
        (root / "rad" / "prov" / f"rad_{index:05d}.config.json").write_text(
            json.dumps(record))
        (root / "rad" / "lund_osg" / f"sample_{index:05d}.lund").touch()
    return root


def main():
    root = make_fixture()
    result = audit(root, 2, 10, "abcdef1234567890")
    assert result["passed"], result["failures"]
    assert result["bound"]["margin"] == 2.0
    assert result["bound"]["probes_per_cell"] == 1600

    path = root / "rad" / "prov" / "rad_00002.config.json"
    record = json.loads(path.read_text())
    record["report"]["b_viol"] = 1
    record["report"]["max_b_viol"] = 1.2
    path.write_text(json.dumps(record))
    tolerated = audit(root, 2, 10, "abcdef1234567890")
    assert tolerated["passed"], tolerated["failures"]

    record["report"]["b_viol"] = 3
    path.write_text(json.dumps(record))
    failed = audit(root, 2, 10, "abcdef1234567890")
    assert not failed["passed"]
    assert any("bound violations=3" in item for item in failed["failures"])
    print("OK: pilot audit tolerates rare b_viol and rejects an excessive rate")


if __name__ == "__main__":
    main()
