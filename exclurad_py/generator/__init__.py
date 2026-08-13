"""Unweighted event generation with exact radiative corrections.

    from exclurad_py import models
    from exclurad_py.generator import Generator, Box, write_lund

    sf  = models.get("eta.tableB1", t_nucl=-0.2)
    gen = Generator(sf, Ebeam=10.6,
                    box=Box(Q2=(1.0, 2.0), W2=(4.0, 6.0), t=(-0.4, -0.1)),
                    mx2_cut=0.5)
    evts = gen.generate(10_000)
    write_lund("eta.lund", evts, channel="eta")

Events are UNWEIGHTED -- see generator.lund for why that is not optional.
"""

from .tail import TailSampler, check_vmin_independence, choose_vmin, photon_at
from .grid import CellGrid, ChannelGeometry
from .frames import LabTransform, assemble_event
from .events import Generator, Box
from .tables import RCTable
from .lund import write_lund, event_lines, PID
from .decay import decay_two_body, in_acceptance, DECAY, ACCEPTANCE, p_theta

__all__ = ["TailSampler", "check_vmin_independence", "choose_vmin",
           "CellGrid", "ChannelGeometry", "LabTransform", "assemble_event",
           "Generator", "Box", "RCTable", "write_lund", "event_lines", "PID",
           "decay_two_body", "in_acceptance", "DECAY", "ACCEPTANCE", "p_theta"]
