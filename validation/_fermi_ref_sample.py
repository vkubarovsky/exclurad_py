#!/usr/bin/env python3
"""Generate the reference sample from whatever checkout is passed in.

Used by fermi_identity.py to run the PRODUCTION code, so it must not import
anything that only exists on this branch.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from exclurad_py import models                                   # noqa: E402
from exclurad_py.generator.events import Box, Generator          # noqa: E402
from exclurad_py.generator.lund import write_lund                # noqa: E402

nev = int(sys.argv[2])
sf = models.get("pi0.amp2609", t_nucl=-0.30)
sf.Ebeam = 10.6
g = Generator(sf, Ebeam=10.6, box=Box(Q2=(1.0, 5.0), W2=(4.5, 12.0),
                                      t=(-2.0, -0.10)),
              mx2_cut=sf.mU ** 2 + 0.20, helicity=0, seed=20260925,
              born_only=True, majorant_shape=(4, 4, 4, 6))
g.setup()
ev = g.generate(nev)
out = Path(tempfile.mkdtemp()) / "o.lund"
write_lund(str(out), ev, sf.channel, topology=sf.topology)
print(hashlib.sha256(out.read_bytes()).hexdigest())
