"""
Unweighted event generation with exact radiative corrections.

Structure ("measured-first", as in exclugen and diffrad_vm v2)
--------------------------------------------------------------
1. Sample the OBSERVED kinematics (Q2, W2, t, phi) from sigma_obs -- the
   radiatively corrected cross section, not the Born one -- by accept/reject
   against a piecewise-constant majorant.  Every surviving event is equally
   probable, so the output is UNWEIGHTED.
2. Split sigma_obs into the unresolved (soft) and resolved (hard) branches
   at v_min and throw for the branch.
3. If hard, sample the photon from the positive-definite tail density and
   rebuild the detected hadron at that inelasticity.
4. Close four-momentum conservation on the recoil and write LUND.

Sampling the observed variables from sigma_obs -- rather than sampling Born
and applying a correction afterwards -- is what makes the generator inherit
the exact RC instead of approximating it.

Cost, and how it is managed
---------------------------
`rc_factor` costs ~0.1 s per point, so it cannot be called per event, and a
prescan that calls it thousands of times cannot be afforded either.  Two
devices fix this:

  * sigma_obs = sigma_Born * eta, with sigma_Born exact per event
    (microseconds) and eta interpolated from a precomputed RCTable.  The
    table's interpolation error is measured, not assumed.
  * the majorant is built from sigma_Born alone (cheap, fine grid) times a
    scalar bound on eta taken from the table.

The photon branch has two implementations.  The CELL-BASED one reuses a
TailSampler per prescan cell (~0.4 s each to build) and holds the photon
density constant across it -- an approximation, and the reason for the
warning that used to stand here.  The PER-EVENT one builds rc_factor and
TailSampler at the event's own kinematics and inverts the exact sigma_obs
ladder for v, with geometric bisection in the deep segment where the
log-log form degenerates.

Production runs the per-event path and has since scheme A was adopted:
EXPO_TAIL and EXACT_ACCEPT each force exact_per_event (see __init__), and
every production card sets both.  Measured on the 2026-08-19 pass, all five
channels: tail_cells_built = 0 against 4000-5000 hard events per channel,
i.e. the cell path was not entered once.  So the photon spectrum is exact
per event, and the standing "APPROXIMATION FLAGGED" note here was stale --
it described a path we stopped using on 2026-08-15 and misled a reader
(me) into reporting it as the last open physics approximation.

The cost of that exactness is the ~0.5 s rc_factor plus ~0.4 s TailSampler
per ACCEPTED event, which is affordable only because accepted events number
~1e4 against ~1e7 trials.
"""

import json
import os

import numpy as np

from ..core.born import born_xsec
from ..core.kinematics import EventKin
from ..core.rc import rc_factor
from .breit_wigner import DELTA_WIDTH as BW_DELTA, sample_mass, width as bw_width
from .tail import TailSampler, choose_vmin, photon_at
from .frames import assemble_event
from .fermi import (CONFIG, FermiSampler, flux_factor, flux_max,
                    split_deuteron, working_frame, M_D as _M_D)
from .tables import RCTable


def _cell_cdf(M, edges):
    """Cumulative cell probability, weighted by CELL VOLUME.

    With linear edges every cell has the same volume and cumsum(M) is
    equivalent, which is why the original code could omit it.  Geometric
    edges break that: the narrow low-Q2 cells would be chosen far too
    often, biasing the sample.  The volume belongs here for the same reason
    it already appears in sigma_total().
    """
    vol = 1.0
    for k, ax in (("Q2", 0), ("W2", 1), ("t", 2), ("phi", 3)):
        d = np.abs(np.diff(edges[k]))
        vol = vol * d.reshape([-1 if i == ax else 1 for i in range(4)])
    w = (M * vol).ravel()
    tot = w.sum()
    return np.cumsum(w) / tot if tot > 0 else None


class _SoftOnly(Exception):
    """Internal: event's radiative tail is below resolution -- emit soft."""


class Box:
    """Sampling box in the observed variables."""

    def __init__(self, Q2, W2, t, phi=(0.0, 2.0 * np.pi)):
        self.Q2, self.W2, self.t, self.phi = Q2, W2, t, phi

    def volume(self):
        return ((self.Q2[1] - self.Q2[0]) * (self.W2[1] - self.W2[0])
                * (self.t[1] - self.t[0]) * (self.phi[1] - self.phi[0]))

    def as_dict(self):
        return dict(Q2=list(self.Q2), W2=list(self.W2),
                    t=list(self.t), phi=list(self.phi))


class Generator:
    """Unweighted generator for one channel / model / beam energy."""

    def __init__(self, sfmodel, Ebeam, box, mx2_cut=None, helicity=0, *, ptarg=0.0,
                 seed=1234, majorant_shape=(8, 8, 8, 12),
                 table_shape=(5, 5, 5, 8), tail_shape=(2, 2, 2, 2),
                 safety=1.6, cache=None, born_only=False, cuts=None,
                 v_min=None, maj_cache=None, itarg=0, vz=(0.0, 0.0),
                 fermi="none", fermi_m_star_min=0.30):
        """born_only=True: no radiative corrections at all.

        sigma_obs == sigma_Born, eta == 1, no photon is ever emitted, and
        neither the RC table nor the tail samplers are built.  This is the
        discriminating test when a radiative sample disagrees with another
        generator: if the BORN distributions already differ, the problem is
        the cross section, the flux convention or the sampling -- not the
        radiative tail.
        """
        self.sf = sfmodel
        self.Ebeam = float(Ebeam)
        # Энергия ТЕКУЩЕГО события.  Равна номинальной всегда, кроме случая
        # включённого до-вершинного радиатора: там электрон теряет часть
        # энергии в мишени ДО взаимодействия, и вся кинематика события, его
        # сечение и его радиационный хвост считаются при пониженной энергии.
        # Таблица eta при этом остаётся построенной при номинальной -- она
        # только форма предложения, а вторая ступень пересчитывает точно.
        self._E = float(Ebeam)
        # --- до-вершинный внешний радиатор (страгглинг в мишени) ---------
        # Электрон проходит часть мишени ДО взаимодействия и успевает
        # излучить.  GEMC этого дать не может: он стартует с вершины, а
        # энергию пучка задаёт генератор, поэтому входную ногу может
        # обеспечить только он.  Выходную ногу здесь считать НЕЛЬЗЯ -- её
        # посчитает Geant4, и получится двойной счёт (exclugen ставит на
        # этот счёт запретительный комментарий, exclugen.f90:28).
        self.itarg = int(itarg)
        self.vzmin, self.vzmax = float(vz[0]), float(vz[1])
        self.X0_LH2 = 890.4      # радиационная длина жидкого водорода, см
        self.BRAD = 4.0 / 3.0
        self._vz = 0.0           # вершина текущего события
        self.n_targ_retry = 0    # катастрофические потери, перерозыгрыш

        # --- масса мезона как свойство события (Брейт-Вигнер) ------------
        # У rho ширина 150 МэВ на массе 775: фиксированная масса кладёт все
        # пи+пи- пары в одну точку там, где в эксперименте широкий резонанс.
        # Масса входит не только в форму пика -- через неё идут порог по W,
        # границы по t и все четыре-векторы, поэтому она разыгрывается ДО
        # взвешивания события, как и энергия пучка.
        #
        # Только для топологии meson_detected: у pi0/eta регистрируется
        # протон (mV = масса протона), а сам мезон узок.
        # BW=0 выключает розыгрыш: тогда ни одного обращения к генератору
        # случайных чисел не происходит и выборка побайтово совпадает с той,
        # что давал код до Брейта-Вигнера.  Это и есть проверка, что правка
        # ничего не сдвинула помимо самой массы.
        self._mV = float(self.sf.mV)
        # Initial-state nucleon at rest.  Proton everywhere except the
        # free-neutron channels; taken from the model so the whole chain
        # (kinematics, v_max, RC, electron acceptance) uses ONE target mass.
        from ..core.constants import M_P as _MP_T
        # TWO target masses, and the distinction is the whole Fermi design.
        #   _mT     -- the ON-SHELL nucleon mass of the model.  Static.  It is
        #              what the cross section and the radiative corrections
        #              are evaluated at, exactly as before (VPK 25.09: M_N in
        #              the RC), so the four rc_factor call sites are untouched.
        #   _mT_kin -- the mass of the target in the frame the event is BUILT
        #              in.  With a free nucleon the two are the same.  With a
        #              deuteron the struck nucleon is off shell, and
        #              frames.assemble_event can only map its working frame
        #              onto the laboratory if the working-frame target mass
        #              equals the lab nucleon's invariant mass m*.  Using _mT
        #              there would break beam + d = e' + meson + N' + N_s by
        #              ~20 MeV.  See FERMI_NOTE.md.
        self._mT = float(getattr(self.sf, "mT", _MP_T))
        self._mT_kin = self._mT

        # ---------------- deuteron target (Fermi motion) ------------------
        # OFF unless the card asks.  When off, _draw_fermi() returns before
        # touching the RNG, which is what makes the output byte-identical to
        # a free-nucleon run for the same seed (acceptance criterion 1).
        self.fermi = str(fermi or "none")
        self.fermi_m_star_min = float(fermi_m_star_min)
        self._fermi = None
        self._E_lab = self.Ebeam
        self._spectator = None
        self._to_lab = None
        if self.fermi != "none":
            if abs(float(ptarg)) > 0:
                raise ValueError(
                    "fermi != none with ptarg != 0: phase 1 is an UNPOLARISED "
                    "deuteron, and ptarg is a longitudinal NUCLEON "
                    "polarisation, which is not what a polarised deuteron "
                    "means (tensor polarisation is out of scope). Set one of "
                    "them to zero.")
            self._fermi = FermiSampler(self.fermi)
            # Moller flux factor ceiling, for the rejection below.  Computed
            # once from the m* floor, since that is what bounds it.
            self._flux_max = flux_max(self.fermi_m_star_min, struck=str(
                getattr(getattr(self.sf, "topology", None), "target", "p")))
            # Which nucleon is struck is already in the model: the pi0/eta
            # topologies declare target 'p', the pi0n/etan ones 'n'.  The
            # spectator is then the other one -- the deuteron has one of each.
            self._struck = str(getattr(getattr(self.sf, "topology", None),
                                       "target", "p"))
        self.n_fermi_reject = 0
        self.n_fermi_unphys = 0
        self.n_fermi_flux = 0     # draws thinned by the flux factor
        self.sum_flux = 0.0       # running <F> over draws that passed the guards
        self.n_flux_drawn = 0
        self.bw_channel = None
        if os.environ.get("BW", "1") != "0" and \
                getattr(getattr(self.sf, "topology", None), "name", "") == \
                "meson_detected":
            ch = str(getattr(self.sf, "channel", "") or "")
            if bw_width(ch) >= BW_DELTA:
                self.bw_channel = ch
        self.n_bw_thresh = 0     # sqrt(W2) < m_recoil + m: порог не пройден
        self.n_bw_tkin = 0       # (t, m) вне кинематических границ

        # Откуда взялась мажоранта и была ли она откалибрована -- НЕ
        # предположение вызывающего, а факт, записанный тем, кто её строил.
        # Прежде run.py в режиме части утверждал в отчёте прогона, что
        # загружена общая откалиброванная мажоранта; в боевой раскладке
        # majorant.npz не создаётся никогда, так что каждая выборка несла
        # про себя неправду.
        self.maj_from_cache = False
        self.maj_calibrated = False
        self.box = box
        self.mx2_cut = mx2_cut
        self.helicity = helicity
        # Longitudinal target polarisation.  Zero is the old behaviour in
        # every respect: the polarised terms of the hadronic tensor are gated
        # on it, so nothing is computed and nothing changes.
        self.ptarg = float(ptarg)
        self.rng = np.random.default_rng(seed)
        # MAJ_SHAPE / MAJ_SAFETY override the defaults, because one setting
        # does not fit every channel.  J/psi sits against a kinematic
        # boundary that moves fast with W2, so its majorant cells straddle
        # the edge and the cell maximum found by probing misses the peak:
        # with the defaults the Born run broke the majorant 40305 times in
        # 6.6e6 trials, the worst by a factor 861.  Refining alone does not
        # help (0.655% at shape 8 against 0.719% at 14 -- the structure is
        # inside a cell), but shape 20 with safety 4.0 takes it to 1 in
        # 158456 at a factor 1.1, for a 2.7x cost in Born-only wall time
        # that the radiative mode does not even notice.  Measured 2026-08-18.
        _ms = os.environ.get("MAJ_SHAPE", "")
        if _ms:
            majorant_shape = tuple(int(x) for x in _ms.split(","))
        self.majorant_shape = tuple(majorant_shape)
        # TABLE_SHAPE mirrors MAJ_SHAPE.  The eta table is a proposal, so a
        # coarse one costs speed -- but its interpolation rms is also the
        # soft/hard split, and on the wide RG-A box (5,5,5,8) gives rms 1.19
        # against the 0.02 the split needs.  So this one is correctness as
        # well as cost, and it has to be reachable from the environment for a
        # scan and for production to pin it.
        # Stage-2 bound grid and probe budget.  B only has to majorise the
        # exact/proposal ratio, so both are pure sampling efficiency -- the
        # sample is exactly unweighted for any B above the true maximum, and
        # violations are counted and reported either way.  They were tied to
        # the majorant grid at 300 probes over 6144 cells: 5% of cells got
        # their own bound and the other 95% used the global maximum set by the
        # single worst cell, which is why the mean ratio was 0.16 against
        # B = 1.57 and ten exact evaluations were needed per accepted event.
        _bs = os.environ.get("B_SHAPE", "")
        self.b_shape = tuple(int(x) for x in _bs.split(",")) if _bs else None
        self.n_probe_b = int(os.environ.get("B_PROBES", "300"))
        self.b_from_cache = False
        _ts = os.environ.get("TABLE_SHAPE", "")
        if _ts:
            table_shape = tuple(int(x) for x in _ts.split(","))
        self.table_shape = tuple(table_shape)
        self.tail_shape = tuple(tail_shape)
        self.safety = float(os.environ.get("MAJ_SAFETY", "") or safety)
        self.cache = cache
        # The majorant prescan (~19 s) is identical for every worker of a
        # parallel run.  Rebuilding it per worker is why --jobs made things
        # SLOWER than one core: 12 x 19 s of duplicated setup against 40 s
        # of actual generation each.  Cached to disk and shared instead.
        self.maj_cache = maj_cache
        self.born_only = bool(born_only)
        # Cuts the REFERENCE generators apply at generation time.  Without
        # these the two samples are different populations and any rate
        # comparison is meaningless -- see decay.py.
        self.cuts = dict(cuts or {})
        # Explicit soft/hard split, as exclugen takes it (vmin=1e-4).  None
        # -> choose_vmin picks one.  Exposing it matters: the radiative
        # FRACTION is set by this number, so two generators cannot be
        # compared on it unless both are told the same value.
        self.v_min_fixed = v_min
        from ..core.constants import M_P as _MP
        globals().setdefault("M_P_", _MP)
        globals().setdefault("M_P_SQ", _MP * _MP)
        _iacc = (cuts or {}).get("iacc")
        self._acc_electron = None
        if _iacc:
            from .decay import ACCEPTANCE
            _w = ACCEPTANCE.get(_iacc)
            if _w and "electron" in _w:
                self._acc_electron = list(_w["electron"])
        # An explicit theta window from the card INTERSECTS the preset rather
        # than replacing it: the two answer different questions -- the preset
        # is a channel's detector window, this is where the drift chambers
        # stop finding the track at all.  Measured on the 2026-09-25 born
        # million, inbending: a negative track matched to the generated
        # electron exists in 0.3 % of events at theta 5-6 deg and 3.5 % at
        # 6-7, against 44 % at 9-10 and 74 % at 15-20.  Cutting at 6 deg
        # removes 22 % of the generation and loses ZERO reconstructed events.
        _the = (cuts or {}).get("theta_e")
        if _the and (float(_the[0]) > 0.0 or float(_the[1]) < 180.0):
            if self._acc_electron is None:
                self._acc_electron = [0.0, 1.0e9, 0.0, 180.0]
            self._acc_electron[2] = max(self._acc_electron[2], float(_the[0]))
            self._acc_electron[3] = min(self._acc_electron[3], float(_the[1]))
        self.n_cut_eacc = 0
        self.n_cut_yxb = 0
        self.n_cut_acc = 0
        self.n_vmin_fallback = 0

        self.table = None
        self.maj = None
        self._tail_cache = {}

        self.n_trials = 0
        self.n_accept = 0
        self.n_hard = 0
        self.n_soft = 0
        # INSTRUMENTATION: where hard events are lost between "branch chosen"
        # and "photon written".  The tail sampler is built at the tail-CELL
        # CENTRE but its draw is tested against THIS event's v_max, and every
        # failure silently falls through to the soft branch.
        # ETA_PROPOSAL_CAP caps the TABLE eta used in the accept weight.  The
        # unclipped table is honest but its 1e3-1e5 corners crater the
        # unweighting efficiency (~100x).  With a cap the PROPOSAL is biased,
        # so an unweighted sample is only valid after reweighting each event
        # by eta_exact/min(eta_table, cap) -- production/weighted_val.py.
        # Interim measure until the two-stage exact-eta accept is built in.
        _cap = float(os.environ.get("ETA_PROPOSAL_CAP", "0"))
        self.eta_proposal_cap = _cap if _cap > 0 else None
        # exact per-event soft/hard split and photon density (see generate())
        self.exact_per_event = os.environ.get("EXACT_PER_EVENT", "") == "1"
        # stage-2 exact accept: unweighted output follows sigma_obs_exact even
        # with a capped proposal.  Implies exact_per_event (needs rc_e anyway).
        self.exact_accept = os.environ.get("EXACT_ACCEPT", "") == "1"
        if self.exact_accept:
            self.exact_per_event = True
        # VPK 2026-08-15: production scheme is EXPONENTIATED (variant A)
        self.expo_tail = os.environ.get("EXPO_TAIL", "") == "1"
        if self.expo_tail:
            self.exact_per_event = True
        # Разыгранная масса запрещает путь по ячейкам.  Тот путь строит
        # TailSampler один раз на ячейку и кладёт в кэш с ключом (ячейка,
        # спиральность) -- массы в ключе нет и быть не может, она
        # непрерывна.  Значит первое событие, зашедшее в ячейку, вморозило
        # бы свою массу, и все следующие получили бы плотность фотона,
        # посчитанную для чужой кинематики.  Продакшн и так идёт поточечно
        # (EXACT_ACCEPT/EXPO_TAIL), но полагаться на переменную окружения
        # в вопросе правильности нельзя.
        if self.bw_channel is not None and not born_only \
                and not self.exact_per_event:
            self.exact_per_event = True
            print(f"[{self.bw_channel}] масса разыгрывается по Брейту-Вигнеру: "
                  f"включён поточечный путь (кэш по ячейкам с ней несовместим)")
        self._B = None            # per-majorant-cell bound for stage 2
        self._B_global = 3.0
        self.n_accept2 = 0
        self.n_b_viol = 0
        self.max_b_viol = 1.0
        self.n_hard_chosen = 0     # rng said "hard"
        self.n_drop_no_tail = 0    # no tail sampler for the cell at all
        self.n_drop_v_ge_vmax = 0  # drawn v >= this event's v_max
        self.n_drop_photon_at = 0  # photon_at() could not build the photon
        self.n_drop_except = 0     # ValueError/ZeroDivisionError in the draw
        self.n_closure_fail = 0
        self.n_overweight = 0
        self.max_overweight = 1.0

    # ------------------------------------------------------------------ #
    @property
    def _t_kind(self):
        """Which invariant the box's t axis means -- from the topology.

        rc_factor takes this from the model's topology; the generator did
        NOT, so it built every event in the default "photon" convention
        (t = (q-p')^2) while pi0/eta topologies mean "target"
        (t = (p-p')^2). The sampled t and the t you recompute from the
        written proton then disagree -- 0.42 vs 3.33 -- and the |t| spectrum
        came out 8x too hard against exclugen.
        """
        return getattr(self.sf.topology, "t_kind", "photon")

    def _kin(self, Q2, W2, t, phi):
        return EventKin(self._E, Q2, W2, t, phi, self._mV,
                        phi_offset=float(getattr(getattr(self.sf, 'topology', None),
                                                 'phi_offset', 0.0)),
                        v=0.0, mU=self.sf.mU, M=self._mT_kin,
                        t_kind=self._t_kind)

    def _vmax(self, Q2, W2, t):
        vk = EventKin.vmax_kin(self._E, Q2, W2, t, self._mV,
                               mU=self.sf.mU, M=self._mT_kin,
                               t_kind=self._t_kind)
        if self.mx2_cut is None:
            return vk
        return min(vk, self.mx2_cut - self.sf.mU ** 2)

    def _set_event_t(self, t):
        """Point a fixed-t_N model at THIS event's observed transfer.

        pi0/eta models are functions of the nucleon transfer supplied at
        construction (see models.base.ProtonDetectedModel). In generation t
        changes event to event, so it has to be updated -- while the model
        still ignores the shifted t inside the radiative integral, because
        t does not shift.
        """
        if hasattr(self.sf, "t_nucl"):
            self.sf.t_nucl = float(t)

    def _kin_xsec(self, Q2, W2, t, phi):
        """EventKin for the CROSS SECTION, always at the ON-SHELL mass.

        Identical to _kin for a free nucleon.  For a deuteron the two differ,
        and the difference is not cosmetic: born_H feeds kin.M into the Hand
        flux factor kappa = 8 (W2 - M^2)^2 / alpha, which is part of the
        DEFINITION of the structure functions the model returns.  Those were
        fitted with M = M_N, so evaluating kappa at m* would silently change
        the cross section -- measured 0.3% at the median Fermi momentum and
        2.4% at p_s = 300 MeV/c.  VPK's decision of 25.09 is M_N in the cross
        section, so the cross section is the free-nucleon one at this event's
        invariants, and m* appears only where the event is BUILT.
        """
        return EventKin(self._E, Q2, W2, t, phi, self._mV,
                        phi_offset=float(getattr(getattr(self.sf, 'topology', None),
                                                 'phi_offset', 0.0)),
                        v=0.0, mU=self.sf.mU, M=self._mT,
                        t_kind=self._t_kind)

    def _born(self, Q2, W2, t, phi, h):
        self._set_event_t(t)
        try:
            return born_xsec(self._kin_xsec(Q2, W2, t, phi), self.sf, h,
                             self.ptarg)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None

    # ------------------------------------------------------------------ #
    def setup(self, verbose=False):
        """Build the eta table and the Born-based majorant."""
        h = self.helicity if self.helicity != 0 else +1

        if self.born_only:
            self.table = None
            eta_hi = 1.0
        else:
            eta_hi = None
        self.table = None
        # A polarised run needs stage 2.  The eta table is built at ONE
        # helicity, so the proposal it feeds is helicity-blind; with
        # sigma_LT' nonzero the true eta differs between the two signs and
        # an unweighted sample drawn against that proposal is biased in the
        # beam-spin asymmetry -- silently, since every other distribution
        # looks right.  EXACT_ACCEPT=1 removes the bias, because stage 2
        # re-accepts against sigma_obs evaluated at the EVENT's helicity.
        # Probe the model rather than trusting a flag: fits gain a fifth
        # structure function without anyone editing this file.
        if not self.born_only and not self.exact_accept:
            try:
                _p = self.sf(np.array([sum(self.box.W2) / 2.0]),
                             np.array([sum(self.box.Q2) / 2.0]),
                             np.array([sum(self.box.t) / 2.0]))
                if np.any(np.abs(_p["sLTp"]) > 0.0):
                    print(" *** WARNING: sigma_LT' is nonzero and "
                          "EXACT_ACCEPT is off.  The eta table is built at a "
                          "single helicity, so the beam-spin asymmetry of "
                          "this sample will be biased.  Set EXACT_ACCEPT=1. "
                          "***", flush=True)
            except Exception:
                pass

        if self.born_only:
            pass
        elif self.cache:
            self.table = RCTable.load(self.cache, self.sf, self.Ebeam,
                                      self.box, self.mx2_cut, h,
                                      self.table_shape, ptarg=self.ptarg)
        if self.table is None and not self.born_only:
            if verbose:
                print("building eta table "
                      f"({int(np.prod(self.table_shape))} RC points)...")
            self.table = RCTable(self.sf, self.Ebeam, self.box, self.mx2_cut,
                                 h, self.table_shape,
                                 ptarg=self.ptarg).build(verbose=verbose)
            if self.cache:
                self.table.save(self.cache)

        eta_hi = 1.0 if self.born_only else float(np.nanmax(self.table.eta))
        # the majorant bound must match the (possibly capped) proposal weight,
        # or the acceptance ratio min(eta,cap)/eta_max craters to ~1e-5
        if not self.born_only and self.eta_proposal_cap:
            eta_hi = min(eta_hi, self.eta_proposal_cap)

        if self.maj_cache and os.path.exists(self.maj_cache):
            d = np.load(self.maj_cache, allow_pickle=True)
            key = json.loads(str(d["key"]))
            if key == self._maj_key(eta_hi):
                self.maj = dict(edges={k: d["e_" + k] for k in ("Q2", "W2", "t", "phi")},
                                M=d["M"], shape=tuple(d["shape"]),
                                cdf=d["cdf"], eta_hi=float(d["eta_hi"]))
                self.maj_from_cache = True
                if "B" in d.files and not self.born_only:
                    self._B = d["B"]
                    self._B_global = float(d["B_global"])
                    self.b_shape = tuple(int(x) for x in d["b_shape"])
                    self.b_from_cache = True
                if verbose:
                    print(f" majorant loaded from {os.path.basename(self.maj_cache)}"
                          + (" (with stage-2 bound)" if self.b_from_cache else ""))
                return self

        nQ, nW, nt, nph = self.majorant_shape
        # Q2 and t get GEOMETRIC edges, W2 and phi stay linear.  The Born
        # cross section is a power law in both: measured d ln sigma / d ln Q2
        # is -1.54 (pi0), -1.15 (eta), -3.72 (rho), -2.65 (phi), -2.56
        # (J/psi), and the t dependence is exponential-into-power-law.  On a
        # linear grid the first cell then has to cover most of the dynamic
        # range on its own -- across J/psi's Q2 box that is a factor 1.4e6,
        # and its majorant duly broke, 40305 violations in 6.6e6 trials with
        # the worst at x861.  Geometric edges make the cells equal in
        # log-space, which is where the function is straight.
        def _geo(lo, hi, n):
            a, b = abs(lo), abs(hi)
            sgn = -1.0 if (lo < 0 or hi < 0) else 1.0
            e = np.geomspace(max(min(a, b), 1e-9), max(a, b), n + 1)
            return sgn * (e[::-1] if lo < hi and sgn < 0 else e)

        edges = dict(Q2=_geo(*self.box.Q2, nQ),
                     W2=np.linspace(*self.box.W2, nW + 1),
                     t=_geo(*self.box.t, nt),
                     phi=np.linspace(*self.box.phi, nph + 1))
        M = np.zeros(self.majorant_shape)
        # Stratified random sub-sampling inside each cell. A full 3^4 corner
        # scan costs 81 Born evaluations per cell (~140k calls for a 6x6x6x8
        # grid, minutes); n_probe random points plus the cell corners in the
        # steepest variable is enough, and the safety factor plus the
        # overweight counter catch any cell that was underestimated.
        rng = np.random.default_rng(20260811)
        n_probe = 24

        # Клетки, целиком лежащие за катами, обязаны получить НОЛЬ.
        # Мажоранта строится из голого борновского сечения, а каты по y, xB
        # и приёмке электрона применяются в цикле розыгрыша ПОСЛЕ выбора
        # клетки и до принятия -- то есть проба, попавшая в закатанную
        # клетку, засчитывается в знаменатель и гибнет.  У J/psi сечение при
        # Q2 = 0.001 в миллион раз больше, чем при 0.25, и электрон оттуда
        # уходит в трубу пучка: нижняя клетка по Q2 держала 66 % интеграла
        # мажоранты при 3 % сечения, перерасход в 876 раз.
        #
        # Проверка идёт по ГУСТОЙ сетке (Q2, W2) -- каты зависят только от
        # этих двух и не требуют вычисления сечения, так что это почти
        # даром.  Густота здесь не роскошь: пропустить узкую проходящую
        # полоску значит выбросить кусок фазового объёма, а это смещение, а
        # не медленный счёт.  Двадцати четырёх случайных проб для такого
        # решения мало.
        NC = 7
        cut_ok = np.ones((nQ, nW), dtype=bool)
        for i in range(nQ):
            qs = np.linspace(edges["Q2"][i], edges["Q2"][i + 1], NC)
            for j in range(nW):
                ws = np.linspace(edges["W2"][j], edges["W2"][j + 1], NC)
                ok = False
                for Q2 in qs:
                    for W2 in ws:
                        if self._pass_yxb(Q2, W2) and \
                                self._pass_electron_acc(Q2, W2):
                            ok = True
                            break
                    if ok:
                        break
                cut_ok[i, j] = ok
        n_dead = int((~cut_ok).sum())
        if verbose and n_dead:
            print(f" majorant: {n_dead}/{nQ * nW} (Q2,W2) cells fully outside "
                  f"the cuts -- zeroed")

        for i in range(nQ):
            for j in range(nW):
                if not cut_ok[i, j]:
                    continue                      # клетка мертва: M остаётся 0
                for k in range(nt):
                    for l in range(nph):
                        u = rng.random((n_probe, 4))
                        u[0] = [0.02, 0.02, 0.02, 0.02]
                        u[1] = [0.98, 0.98, 0.98, 0.98]
                        u[2] = [0.5, 0.5, 0.5, 0.5]
                        Q2s = edges["Q2"][i] + u[:, 0] * np.diff(edges["Q2"])[i]
                        W2s = edges["W2"][j] + u[:, 1] * np.diff(edges["W2"])[j]
                        ts_ = edges["t"][k] + u[:, 2] * np.diff(edges["t"])[k]
                        phs = edges["phi"][l] + u[:, 3] * np.diff(edges["phi"])[l]
                        best = 0.0
                        for Q2, W2, t, ph in zip(Q2s, W2s, ts_, phs):
                            b = self._born(Q2, W2, t, ph, h)
                            if b is not None and np.isfinite(b):
                                best = max(best, abs(b))
                        M[i, j, k, l] = best
        M *= eta_hi * self.safety
        tot = M.sum()
        if not np.isfinite(tot) or tot <= 0:
            raise RuntimeError(
                "мажоранта вышла нулевой: коробка целиком за катами?")
        self.maj = dict(edges=edges, M=M, shape=self.majorant_shape,
                        cdf=_cell_cdf(M, edges),
                        eta_hi=eta_hi)
        return self

    def _maj_key(self, eta_hi):
        return dict(model=self.sf.name, Ebeam=self.Ebeam, box=self.box.as_dict(),
                    mx2_cut=self.mx2_cut, helicity=self.helicity, ptarg=self.ptarg,
                    shape=list(self.majorant_shape), safety=self.safety,
                    born_only=self.born_only, eta_hi=round(float(eta_hi), 9),
                    # A deuteron majorant is not a free-nucleon majorant: the
                    # per-event boost widens r inside every cell, so B_cell is
                    # calibrated against a different distribution.  Nothing
                    # else in this key notices -- model, box, shape and safety
                    # are all identical -- so a free-nucleon majorant would
                    # load silently against a Fermi run.
                    fermi=self.fermi,
                    # The cell EDGES are part of what a cached majorant is,
                    # and nothing else in this key notices them: model, box,
                    # shape and safety are all identical between a linear and
                    # a geometric grid.  A stale linear majorant would load
                    # silently against geometric sampling, and the cell
                    # probabilities would be wrong -- the cdf is built from
                    # M * cell volume, and the volumes would not match the
                    # edges being sampled.  Same gap the eta table had twice
                    # (phi convention, then the t axis); bump the tag with
                    # any change to how edges are laid out.
                    edges="geom-q2t-cutmask-v2",
                    cuts={k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in sorted(self.cuts.items())})
        # NOTE: t_nucl is deliberately NOT part of the key. Since the
        # per-event-t fix it is mutable state -- the generator rewrites it
        # for every event -- so including it recorded whatever t the last
        # calibration event happened to have (-0.432 instead of the card's
        # -0.30). The key never matched, every worker silently rebuilt the
        # majorant, and --jobs looked far slower than it is.

    def save_majorant(self, path):
        """Persist the CALIBRATED majorant so parallel workers can share it."""
        s = self.maj
        extra = {}
        if self._B is not None:
            # The stage-2 bound belongs in here.  It was rebuilt by every job
            # from 300 fresh probes, so 200 jobs ran 200 independent lotteries
            # for a quantity that is a property of the box, not of the job.
            extra = dict(B=self._B, B_global=self._B_global,
                         b_shape=np.array(self.b_shape))
        np.savez_compressed(path, M=s["M"], cdf=s["cdf"],
                            shape=np.array(s["shape"]), eta_hi=s["eta_hi"],
                            key=json.dumps(self._maj_key(s["eta_hi"])),
                            **extra,
                            **{"e_" + k: v for k, v in s["edges"].items()})
        return path

    def _pass_electron_acc(self, Q2, W2):
        """Scattered-electron acceptance, testable BEFORE any expensive work.

        The generator samples OBSERVED kinematics, so |p| and theta of the
        scattered electron follow from (Q2, W2) and the beam alone -- the
        radiated photon does not move them, and neither do t or phi.  So
        this is the identical cut the assembled event would face later, only
        applied for free.

        It is worth applying: of J/psi events generated without it, 6% pass
        the electron window, so 94% of the rc_factor calls in that channel
        were being spent on events destined to be thrown away.  Measured
        cost with the card's cuts, 1942 -> 36957 ms per written event; this
        is where the factor of 19 came from.

        NOTE this moves the acceptance in front of the n_accept counter, so
        sigma_total() now reports the cross section WITHIN the electron
        acceptance rather than over the whole box.  For a card that declares
        iacc that is the more useful number, but it is a change of meaning.
        """
        acc = self._acc_electron
        if acc is None:
            return True
        nu = (W2 + Q2 - self._mT_kin ** 2) / (2.0 * self._mT_kin)
        Ep = self._E - nu
        if Ep <= 0.0:
            return False
        c = 1.0 - Q2 / (2.0 * self._E * Ep)
        if abs(c) > 1.0:
            return False
        th = np.degrees(np.arccos(c))
        return (acc[0] <= Ep <= acc[1]) and (acc[2] <= th <= acc[3])

    def _pass_yxb(self, Q2, W2):
        ylo, yhi = self.cuts.get("y", (0.0, 1.0))
        xlo, xhi = self.cuts.get("xB", (0.0, 1.0))
        if (ylo, yhi, xlo, xhi) == (0.0, 1.0, 0.0, 1.0):
            return True
        M = self.sf.topology.m_detected if False else 0.93827208816
        nu = (W2 + Q2 - M * M) / (2.0 * M)
        if nu <= 0:
            return False
        y = nu / self._E
        xB = Q2 / (W2 + Q2 - M * M)
        return (ylo <= y <= yhi) and (xlo <= xB <= xhi)

    # ------------------------------------------------------------------ #
    def _draw_vertex(self):
        """Вершина по длине мишени и потеря энергии до неё.

        Разыгрывается на КАЖДОЙ пробе, а не на принятом событии: энергия
        входит в сечение, поэтому она должна быть известна до того, как
        событие взвешивается.

        Закон страгглинга: доля уцелевшей энергии распределена как
        P(loss < y) = y^(bt), откуда loss = u^(1/bt) при равномерном u.
        Тот же вид, что в exclugen (exclugen.f90:247) и в aao_rad
        (aao_rad.F:675).  Потеря больше 0.9 -- перерозыгрыш: там страгглинг
        уже не описывает процесс, а событие всё равно не выживет.
        """
        if not self.itarg or self.vzmax <= self.vzmin:
            self._vz = 0.0
            self._E = self.Ebeam
            return True
        self._vz = self.vzmin + (self.vzmax - self.vzmin) * self.rng.random()
        bt = self.BRAD * (self._vz - self.vzmin) / self.X0_LH2
        if bt <= 1e-12:
            self._E = self.Ebeam
            return True
        u = max(self.rng.random(), 1e-30)
        vloss = np.exp(np.log(u) / bt)
        if vloss > 0.9:
            self.n_targ_retry += 1
            return False
        self._E = self.Ebeam * (1.0 - vloss)
        return True

    def _draw_fermi(self):
        """Fermi momentum of the struck nucleon, and the frame it implies.

        Drawn on EVERY TRIAL, beside _draw_vertex and for the same reason: it
        changes the energy that enters the cross section, so it has to be
        known before the event is weighted.

        When the target is a free nucleon this returns immediately WITHOUT
        drawing, so the random stream is bit-for-bit the one the free-nucleon
        run had.  That is acceptance criterion 1 and it is what keeps this
        branch from disturbing the running campaign.

        What it sets:
            _E        beam energy in the struck nucleon's rest frame.  The
                      generator only knows how to work with a target at rest,
                      so that frame is where the event is built; self._E
                      already carries a per-event energy for the straggling
                      radiator, and this composes with it.
            _mT_kin   the OFF-SHELL mass m*.  The working frame must have the
                      target at this mass or the map back to the laboratory is
                      not a Lorentz transformation.
            _to_lab   that map.
            _spectator  the on-shell spectator, appended to the final state.

        Returns False -- the whole trial dies, as in _draw_mass -- when the
        draw cannot produce an event.  It is never RE-drawn: re-drawing would
        quietly truncate n(k) and the sample would no longer follow the wave
        function.  Two reasons to reject:

          * s below the box.  A nucleon running away from the beam gives a
            near-co-moving collision; at p_s = 0.7 GeV/c sqrt(s) falls to
            0.3 GeV and no final state exists at all.  No proposal from the
            majorant could be satisfied, so the trial is dead either way.
          * m* below fermi_m_star_min (default 0.30 GeV).  m* -> 0 at
            p_s = 0.7025 GeV/c, where the nucleon becomes lightlike and its
            "rest frame" needs an unbounded boost -- 1151 GeV at the ceiling.
            PWIA with an on-shell cross section has no meaning there.  Costs
            0.14% of the distribution and caps the working-frame energy at
            about 45 GeV.
        """
        if self._fermi is None:
            return True
        self._E_lab = self._E
        p_spec = self._fermi.draw(self.rng)
        struck4, spec4, m_star = split_deuteron(p_spec, self._struck)
        if m_star < self.fermi_m_star_min:
            self.n_fermi_reject += 1
            return False
        k_lab = np.array([self._E, 0.0, 0.0, self._E])
        tot = struck4 + k_lab
        s_inv = tot[0] ** 2 - float(np.dot(tot[1:], tot[1:]))
        if s_inv <= self.box.W2[0]:
            self.n_fermi_reject += 1
            return False
        # ---- Moller flux factor, by rejection ---------------------------
        # The rate per beam particle per nucleon is sigma * F, not sigma, with
        # F = 1 - beta_z.  Applying it as a WEIGHT would force the majorant to
        # bound sigma*F, i.e. to be raised by F_max = 1.91 everywhere -- and a
        # 1.91x looser bound costs 1.91x more FULL trials, each of which has
        # already paid for _propose, the cuts and _born.  Applying it here, as
        # a rejection on the draw itself, costs 1.91x more draws of three
        # uniforms and nothing else, and leaves the majorant untouched: the
        # accepted spectator distribution is n(k)*F rather than n(k), which is
        # the physical one.
        #
        # <F> = 1 exactly over an isotropic distribution, so this reweights the
        # shape without changing the normalisation; the guards above correlate
        # weakly with F and leave <F> = 1.0002, which is tracked in sum_flux and
        # reported by health() rather than assumed.
        flux = float(flux_factor(struck4))
        self.n_flux_drawn += 1
        self.sum_flux += flux
        if self.rng.random() * self._flux_max > flux:
            self.n_fermi_flux += 1
            return False

        self._E, self._to_lab = working_frame(self._E, struck4)
        self._mT_kin = m_star
        self._spectator = spec4
        self._flux = flux
        return True

    def _boost_to_lab(self, ev):
        """Take one assembled event from the working frame to the laboratory.

        A no-op for a free nucleon.  For a deuteron every four-vector goes
        through the SAME linear map, so the conservation that
        assemble_event closed on the recoil survives exactly; adding the
        spectator then completes beam + d = e' + meson + N' + N_s (+ gamma),
        because struck + spectator = (M_d, 0, 0, 0) by construction.
        """
        if self._to_lab is None:
            return ev
        L = self._to_lab
        for key in ("beam", "target", "scattered", "meson", "photon", "recoil"):
            v = ev.get(key)
            if v is not None:
                ev[key] = L @ np.asarray(v, dtype=float)
        ev["spectator"] = self._spectator
        ev["spectator_pid"] = CONFIG[self._struck][2]
        ev["struck"] = ev["target"]
        ev["p_fermi"] = float(np.linalg.norm(self._spectator[1:]))
        ev["m_star"] = float(self._mT_kin)
        return ev

    def _pass_fermi_frame(self, Q2, W2, t, phi):
        """Is the proposed point physical in THIS event's frame?

        The majorant box is fixed, but with Fermi motion the frame the event
        is built in is not: a nucleon moving away from the beam sees less
        energy, and a (Q2, W2, t) that is perfectly physical against a nucleon
        at rest can be past the kinematic limit against that one.  vmax_kin
        then raises, and before this guard it raised AFTER n_accept had already
        been incremented, which would have biased sigma_total as well as
        crashing the run.

        Rejected here, beside the other cuts and before the Born evaluation:
        the trial counts in the denominator with zero weight, exactly as a
        trial whose mass fell outside the t limits does.  A no-op for a free
        nucleon, so the byte-identity test is unaffected.
        """
        if self._fermi is None:
            return True
        try:
            # BOTH have to be asked.  vmax_kin and EventKin fail on different
            # conditions -- "Born point itself unphysical at this t" and
            # "unphysical lepton kinematics (|cos| > 1)" -- and a trial that
            # clears the first can still die in the second, which is how the
            # first 100k deuteron run ended after calibration.
            if self._vmax(Q2, W2, t) <= 0:
                self.n_fermi_unphys += 1
                return False
            self._kin(Q2, W2, t, phi)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            self.n_fermi_unphys += 1
            return False
        return True

    def _draw_mass(self, Q2, W2, t):
        """Масса мезона для этой пробы, порог по W и границы по t.

        Порядок как в diffrad_vm.f90:1447: сначала масса, потом проверка
        sqrt(W2) >= m_recoil + m.  Потолок розыгрыша -- тот же
        sqrt(W2) - m_recoil, поэтому порог срабатывает редко и только когда
        места нет вовсе.

        Возвращает False -- проба отбрасывается целиком, ровно как goto 100
        в фортране.  Отбрасывание СОБЫТИЯ, а не перерозыгрыш одной массы:
        иначе у порога вес события молча перестал бы соответствовать
        разыгранной плотности.
        """
        if self.bw_channel is None:
            return True
        W = np.sqrt(max(W2, 0.0))
        m_max = W - self.sf.mU
        if m_max <= 0.0:
            self.n_bw_thresh += 1
            return False
        self._mV = sample_mass(self.bw_channel, float(self.sf.mV), m_max,
                               self.rng)
        if W < self.sf.mU + self._mV:
            self.n_bw_thresh += 1
            return False
        # Границы по t зависят от массы: у широкого rho они гуляют от
        # события к событию, и уже разыгранное t может из них выпасть.
        # Без явной проверки проба всё равно умерла бы -- born вернул бы
        # None на |cos theta_V| > 1 -- но молча, вперемешку с настоящими
        # аномалиями.  Здесь у неё своя причина и свой счётчик.
        lim = EventKin.t_limits(Q2, W2, self._mV, self.sf.mU,
                                t_kind=self._t_kind)
        if lim is None or not (lim[0] <= t <= lim[1]):
            self.n_bw_tkin += 1
            return False
        return True

    def _propose(self):
        s = self.maj
        flat = int(np.searchsorted(s["cdf"], self.rng.random()))
        flat = min(flat, int(np.prod(s["shape"])) - 1)
        i, j, k, l = np.unravel_index(flat, s["shape"])
        e, u = s["edges"], self.rng.random(4)
        return (e["Q2"][i] + u[0] * np.diff(e["Q2"])[i],
                e["W2"][j] + u[1] * np.diff(e["W2"])[j],
                e["t"][k] + u[2] * np.diff(e["t"])[k],
                e["phi"][l] + u[3] * np.diff(e["phi"])[l],
                s["M"][i, j, k, l], (i, j, k, l))

    def _tail_cell(self, cell):
        """Map a fine majorant cell onto the coarser tail-sampler grid.

        One TailSampler costs ~0.4 s to build, so one per majorant cell
        (hundreds) is unaffordable. The tail grid is deliberately coarse:
        within a tail cell the photon density is treated as constant in the
        observed kinematics. APPROXIMATION -- see REPORT.md gap 3.
        """
        return tuple(int(c * ts // ms) for c, ms, ts
                     in zip(cell, self.majorant_shape, self.tail_shape))

    def _tail_for_cell(self, cell, h, sigma_obs):
        key = (self._tail_cell(cell), h)
        if key in self._tail_cache:
            return self._tail_cache[key]
        e = self.maj["edges"]
        tc = key[0]
        # centre of the TAIL cell, in each observed variable
        def centre(name, idx, n_tail):
            lo, hi = e[name][0], e[name][-1]
            w = (hi - lo) / n_tail
            return lo + (idx + 0.5) * w
        cQ2 = centre("Q2", tc[0], self.tail_shape[0])
        cW2 = centre("W2", tc[1], self.tail_shape[1])
        ct = centre("t", tc[2], self.tail_shape[2])
        cph = centre("phi", tc[3], self.tail_shape[3])
        try:
            kin = self._kin(cQ2, cW2, ct, cph)
            vm = self._vmax(cQ2, cW2, ct)
            if vm <= 0:
                self._tail_cache[key] = None
                return None
            # sigma_obs MUST be evaluated at the same point as sigma_hard.
            #
            # sigma_hard below is built at the tail-CELL CENTRE, but the caller
            # passes this EVENT's sigma_obs (Born x interpolated eta).  With
            # tail_shape=(2,2,2,2) a tail cell spans e.g. Q2 0.2-5.1, across
            # which sigma_born moves by orders of magnitude -- so
            # frac = sigma_hard(centre)/sigma_obs(event) is not a probability
            # at all.  It collapsed to a median of 6e-4 where the correct value
            # at fixed kinematics is 0.59-0.62, which is what emptied the hard
            # branch and left every RAD sample effectively Born.
            #
            # One exact rc_factor per tail cell (there are only
            # prod(tail_shape) of them) puts numerator and denominator back at
            # the same kinematics and takes the eta table out of the split.
            self._set_event_t(ct)
            rc_c = rc_factor(M=self._mT,
Ebeam=self._E, Q2=cQ2, W2=cW2, t=ct, phi=cph,
                             sfmodel=self.sf, h=h, lam=self.ptarg, mx2_cut=self.mx2_cut,
                             t_kind=self._t_kind, mV=self._mV)
            sigma_obs_c = float(rc_c["sigma_obs"])
            if not np.isfinite(sigma_obs_c) or sigma_obs_c <= 0:
                self._tail_cache[key] = None
                return None

            if self.v_min_fixed:
                v_min = min(self.v_min_fixed, 0.5 * vm)
                from .tail import TailSampler as _TS
                ts_probe = _TS(kin, self.sf, h, vm, v_min=v_min)
                soft, hard, hf = ts_probe.split(sigma_obs_c)
                soft_frac = 1.0 - hf
                if soft <= 0:
                    v_min, soft_frac = choose_vmin(kin, self.sf, h, vm,
                                                   sigma_obs_c)
                    self.n_vmin_fallback += 1
            else:
                v_min, soft_frac = choose_vmin(kin, self.sf, h, vm, sigma_obs_c)
            if v_min is None:
                self._tail_cache[key] = None
                return None
            self._tail_cache[key] = dict(
                ts=TailSampler(kin, self.sf, h, vm, v_min=v_min),
                soft_frac=soft_frac, buf=None, ibuf=0)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            self._tail_cache[key] = None
        return self._tail_cache[key]

    def _resolve_b_shape(self, shape, per_cell=20):
        """A bound grid the probe budget can actually measure.

        Unset, `b_shape` fell back to the MAJORANT grid -- 6144 cells against
        300 probes, so under 5 % of cells were ever measured and the rest took
        the global fallback.  That is how the 2026-09-25 production came out
        with the bound exceeded in 198 of 200 chunks, worst x3.35: the bound
        was a lottery of 300 draws, redrawn independently by every job.

        Coarsen every axis by a common factor until the grid fits the budget
        at `per_cell` probes each.  Keeping the majorant's aspect matters: the
        large-r region sits at high Q2 and high W, and a grid that collapses
        those axes puts a bound sized for that corner onto the whole box.
        """
        if self.b_shape is not None:
            return tuple(self.b_shape)
        budget = max(1, self.n_probe_b // per_cell)
        best = tuple(1 for _ in shape)
        for f in range(1, max(shape) + 1):
            cand = tuple(max(1, -(-m // f)) for m in shape)
            if int(np.prod(cand)) <= budget:
                best = cand
                break
        return best

    def _b_grid_cell(self, cell):
        """Map a fine majorant cell onto the (coarser) stage-2 bound grid.

        Same integer mapping as _tail_cell.  Separating the two grids is the
        point: the majorant wants to be fine, the bound grid wants to be
        coarse enough that 300-odd probes actually reach every cell.
        """
        if self.b_shape is None:
            return cell
        return tuple(int(c * bs // ms) for c, ms, bs
                     in zip(cell, self.majorant_shape, self.b_shape))

    def _b_cell(self, cell):
        """Stage-2 bound for this majorant cell (calibrated; global fallback)."""
        if self._B is None:
            return self._B_global
        v = float(self._B[self._b_grid_cell(cell)])
        return v if v > 0 else self._B_global

    def _b_full(self, shape):
        """The stage-2 bound on the MAJORANT grid, cell by cell.

        The same integer mapping as _b_grid_cell, vectorised over each axis:
        the proposal (_cell_cdf) and the acceptance (_b_cell) have to read the
        identical bound for every cell, so both go through this one mapping.
        """
        if self._B is None:
            return np.full(shape, self._B_global)
        if self.b_shape is None:
            return self._B
        idx = [np.minimum((np.arange(ms) * bs) // ms, bs - 1)
               for ms, bs in zip(shape, self.b_shape)]
        return self._B[np.ix_(*idx)]

    def _draw_photon(self, entry, block=4096):
        """One (v, nhat) from a cell's sampler, refilling in blocks.

        Drawing photons one at a time wastes the vectorization: sample(1)
        still proposes a whole candidate block. Buffering cuts the
        per-event cost by an order of magnitude.
        """
        buf, i = entry["buf"], entry["ibuf"]
        if buf is None or i >= len(buf["v"]):
            buf = entry["ts"].sample(self.rng, block)
            entry["buf"], entry["ibuf"] = buf, 0
            i = 0
        entry["ibuf"] = i + 1
        return dict(v=buf["v"][i], nhat=buf["nhat"][i])

    # ------------------------------------------------------------------ #
    def majorant_provenance(self):
        """Одной строкой: откуда мажоранта и калибровалась ли она."""
        src = ("loaded from %s" % os.path.basename(self.maj_cache)
               if self.maj_from_cache else "built in this process")
        return "majorant %s; %s" % (
            src, "calibrated" if self.maj_calibrated else "NOT calibrated")

    def calibrate(self, n_probe=20000, margin=1.35, verbose=False):
        """Raise the majorant where a trial run exceeds it, then reset.

        A prescan samples each cell at a finite number of points and can
        miss its true maximum; the wider the box, the more often.  Any
        exceedance biases an unweighted sample -- those events are
        under-represented -- so it must be fixed BEFORE generating, not
        reported afterwards.

        This makes a throwaway pass, records the worst sigma_obs/M per
        cell, inflates those cells by that ratio times `margin`, and
        discards the events.  The generated sample is then produced against
        the corrected majorant.
        """
        s = self.maj
        shape = s["shape"]
        worst = np.ones(shape)
        h0 = self.helicity if self.helicity != 0 else +1
        for _ in range(n_probe):
            if not self._draw_vertex():
                continue
            if not self._draw_fermi():
                continue
            Q2, W2, t, phi, M, cell = self._propose()
            if M <= 0 or not self._pass_yxb(Q2, W2):
                continue
            if not self._draw_mass(Q2, W2, t):
                continue
            h = h0 if self.helicity != 0 else (+1 if self.rng.random() < 0.5 else -1)
            b = self._born(Q2, W2, t, phi, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = 1.0 if self.born_only else float(self.table(Q2, W2, t, phi)[0])
            if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                eta = self.eta_proposal_cap
            r = (b * eta) / M
            if r > worst[cell]:
                worst[cell] = r
        # stage-2 bound calibration: exact/proposal ratio per cell, from up to
        # 300 probe points (rc_factor is ~0.1-1 s, the full prescan would be
        # unaffordable).  Cells never probed fall back to the global max.
        # born_only has no eta table (there is no radiative proposal to
        # correct), so stage 2 is both meaningless and a crash: self.table
        # is None.  The two other table reads, lines ~467 and ~544, already
        # guard with `1.0 if self.born_only else ...`; this one did not.
        # Never hit before because run_production_100k.sh only ever exports
        # EXACT_ACCEPT=1 for the rad cards -- iborn=1 with the production
        # environment first happened on 2026-08-17.
        if self.exact_accept and not self.born_only and self.b_from_cache:
            if verbose:
                print(f" stage-2 bound from cache: shape {self.b_shape}, "
                      f"B {self._B.min():.2f}-{self._B.max():.2f}, "
                      f"global {self._B_global:.2f}")
        elif self.exact_accept and not self.born_only:
            self.b_shape = self._resolve_b_shape(shape)
            bshape = self.b_shape
            bmax = np.zeros(bshape)
            n_b = 0
            for _ in range(max(n_probe, self.n_probe_b)):
                if n_b >= self.n_probe_b:
                    break
                if not self._draw_vertex():
                    continue
                if not self._draw_fermi():
                    continue
                Q2, W2, t, phi, Mv, cell = self._propose()
                if Mv <= 0 or not self._pass_yxb(Q2, W2):
                    continue
                if not self._draw_mass(Q2, W2, t):
                    continue
                # Draw the helicity the way generate() does, not h0.  The
                # bound this loop calibrates is r = sigma_obs(h) / (born(h) *
                # eta_table), and eta_table is built at h0, so r = eta(h) /
                # eta(h0) -- identically 1 at h0 and NOT 1 at the other sign
                # once sigma_LT' is nonzero.  Probing only h0 would calibrate
                # the bound on the one helicity where it is trivially tight.
                # Today this is a no-op: sigma_LT' = 0 in every fit we carry,
                # so nothing depends on h at all.  With a 5% asymmetry the
                # miss would be ~0.06% (eta 0.88255 vs 0.88305 at v_cut 0.1)
                # against a 1.3 safety factor -- small, but there is no
                # reason to leave it wrong.
                h_p = h0 if self.helicity != 0 else \
                    (+1 if self.rng.random() < 0.5 else -1)
                b = self._born(Q2, W2, t, phi, h_p)
                if b is None or not np.isfinite(b) or b <= 0:
                    continue
                eta = float(self.table(Q2, W2, t, phi)[0])
                if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                    eta = self.eta_proposal_cap
                try:
                    self._set_event_t(t)
                    so = float(rc_factor(M=self._mT,
Ebeam=self._E, Q2=Q2, W2=W2, t=t, phi=phi,
                        sfmodel=self.sf, h=h_p, lam=self.ptarg, mx2_cut=self.mx2_cut,
                        t_kind=self._t_kind, mV=self._mV)["sigma_obs"])
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    continue
                if not np.isfinite(so) or so <= 0:
                    continue
                n_b += 1
                r = so / (b * eta)
                bc = self._b_grid_cell(cell)
                if r > bmax[bc]:
                    bmax[bc] = r
            self._B_global = float(max(bmax.max() * 1.3, 1.05))
            # The global fallback for never-probed cells is BAKED IN here
            # rather than applied later in _b_cell: the proposal now carries
            # this same array (see _b_full), and the two must read the
            # identical number for every cell or the composition breaks again.
            self._B = np.where(bmax > 0, bmax * 1.3, self._B_global)
            if verbose:
                print(f" stage-2 bounds: {n_b} probes, global B = "
                      f"{self._B_global:.2f}")

        n_bad = int((worst > 1.0).sum())
        s["M"] = s["M"] * np.where(worst > 1.0, worst * margin, 1.0)
        # THE PROPOSAL MUST CARRY B_cell (fixed 2026-09-25).
        #
        # Stage 1 accepts with s_obs/M, stage 2 with sigma_exact/(s_obs*B_cell),
        # so the accepted density is (proposal) * s_obs/M * so/(s_obs*B_cell).
        # With the cell drawn ~ M that leaves so/B_cell -- the SHAPE of the
        # sample divided by a piecewise-constant function of the kinematics.
        # Measured on 400k events each of the 2026-09-24 born and rad samples:
        # the rad/born ratio stepped by -21.6% (22.5 sigma) across the W2 cell
        # edge at W = 2.20 and by +21.8% at W = 3.80, with nothing at the
        # eta-table edges -- which is what identified this grid as the cause.
        # Drawing the cell ~ M*B_cell cancels the 1/B_cell exactly and keeps
        # the per-cell tightness that made a per-cell bound worth having; a
        # single global B would also be unbiased but throws that away.
        s["Mtot"] = (s["M"] * self._b_full(shape)
                     if (self.exact_accept and not self.born_only) else s["M"])
        s["cdf"] = _cell_cdf(s["Mtot"], s["edges"])
        self.n_trials = self.n_accept = 0
        self.n_overweight = 0
        self.max_overweight = 1.0
        self.n_cut_yxb = self.n_cut_acc = 0
        self.maj_calibrated = True
        if verbose:
            print(f" calibrated majorant: {n_bad} of {int(np.prod(shape))} cells "
                  f"raised (worst factor {worst.max():.2f})")
        return dict(cells_raised=n_bad, worst=float(worst.max()))

    # ------------------------------------------------------------------ #
    def generate(self, n_events, progress=None):
        if self.maj is None:
            self.setup()
        out = []
        while len(out) < n_events:
            # энергия события -- до всего остального: от неё зависят и каты,
            # и сечение, и хвост
            if not self._draw_vertex():
                continue
            if not self._draw_fermi():
                continue
            Q2, W2, t, phi, M, cell = self._propose()
            h = self.helicity if self.helicity != 0 else \
                (+1 if self.rng.random() < 0.5 else -1)
            self.n_trials += 1

            # y / xB cuts first: they are arithmetic and reject before the
            # (much more expensive) Born evaluation.
            if not self._pass_yxb(Q2, W2):
                self.n_cut_yxb += 1
                continue
            if not self._pass_electron_acc(Q2, W2):
                self.n_cut_eacc += 1
                continue

            # Масса -- ПОСЛЕ n_trials, вместе с катами, а не до счётчика.
            # sigma_total считается как (accept/trials) * интеграл мажоранты,
            # поэтому проба, у которой предложение из мажоранты уже взято, а
            # масса выпала из границ по t, обязана войти в знаменатель с
            # нулевым весом.  У rho такие пробы -- 37%, и не считай их,
            # сечение выросло бы в полтора раза на пустом месте.
            if not self._draw_mass(Q2, W2, t):
                continue
            if not self._pass_fermi_frame(Q2, W2, t, phi):
                continue

            b = self._born(Q2, W2, t, phi, h)
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            eta = 1.0 if self.born_only else float(self.table(Q2, W2, t, phi)[0])
            if self.eta_proposal_cap and eta > self.eta_proposal_cap:
                eta = self.eta_proposal_cap
            s_obs = b * eta
            if s_obs <= 0:
                continue
            if s_obs > M:
                self.n_overweight += 1
                self.max_overweight = max(self.max_overweight, s_obs / M)
            if self.rng.random() * M > s_obs:
                continue
            self.n_accept += 1

            v_max = self._vmax(Q2, W2, t)
            if v_max <= 0:
                continue
            self._set_event_t(t)
            kin = self._kin(Q2, W2, t, phi)

            photon4 = meson4 = None
            if self.exact_per_event and not self.born_only:
                # EXACT PER-EVENT RADIATION (env EXACT_PER_EVENT=1).
                # The cell-based path holds the soft/hard split and the photon
                # density constant across a tail cell (16 cells over the whole
                # box), which biased frac(v > 0.3) by ~2x against the exact
                # quadrature even after the population reweighting.  Here both
                # come from THIS event's kinematics: one rc_factor (~0.1 s)
                # and one TailSampler (~0.4 s) per ACCEPTED event -- affordable
                # because accepted events are ~1e4, not the 1e7 trials.
                try:
                    rc_e = rc_factor(M=self._mT,
Ebeam=self._E, Q2=Q2, W2=W2, t=t,
                                     phi=phi, sfmodel=self.sf, h=h,
                                     lam=self.ptarg, mx2_cut=self.mx2_cut,
                                     t_kind=self._t_kind, mV=self._mV)
                    so_e = float(rc_e["sigma_obs"])

                    # STAGE 2 (optional, EXACT_ACCEPT=1): re-accept on
                    # sigma_obs_exact / (proposal * B_cell).  Removes the
                    # capped-proposal bias from the UNWEIGHTED sample itself;
                    # B_cell is calibrated in calibrate(), violations counted.
                    if self.exact_accept:
                        prop = float(s_obs)
                        B = self._b_cell(cell)
                        r = so_e / (prop * B) if prop > 0 else 0.0
                        if r > 1.0:
                            self.n_b_viol += 1
                            self.max_b_viol = max(self.max_b_viol, r)
                        if self.rng.random() > min(r, 1.0):
                            continue          # rejected by the exact stage
                        self.n_accept2 += 1

                    vmn = min(self.v_min_fixed or 1.0e-4, 0.5 * v_max)
                    # degenerate tail: photons with v < ~10*vmn are below any
                    # missing-mass resolution; treating the event as soft is
                    # physically free and avoids the near-threshold sampler
                    # stall that cost 9 h on the J/psi box (vmax ~ 1e-3).
                    if v_max < 10.0 * vmn:
                        raise _SoftOnly

                    if self.expo_tail:
                        # EXPONENTIATED scheme (VPK decision 2026-08-15):
                        # the observable frac(v > x) must equal
                        # [sigma_obs(vmax)-sigma_obs(x)]/sigma_obs(vmax) with
                        # the YFS-exponentiated soft factor -- what the
                        # published eta(v_cut) tables mean.  Hard fraction and
                        # the v-draw both come from the sigma_obs ladder; the
                        # bare tail supplies only the photon ANGLES.
                        mU2 = self.sf.mU ** 2

                        def sobs(x):
                            return float(rc_factor(M=self._mT,
Ebeam=self._E, Q2=Q2, W2=W2, t=t, phi=phi,
                                sfmodel=self.sf, h=h, lam=self.ptarg,
                                mx2_cut=x + mU2,
                                t_kind=self._t_kind, mV=self._mV)["sigma_obs"])

                        N0 = max(so_e - sobs(vmn), 0.0)
                        if self.rng.random() < N0 / so_e:
                            self.n_hard_chosen += 1
                            nodes = np.geomspace(vmn, v_max, 8)
                            N = [N0] + [max(so_e - sobs(x), 0.0)
                                        for x in nodes[1:-1]] + [0.0]
                            u = self.rng.random() * N0
                            j = next(k for k in range(7) if N[k + 1] <= u)
                            lo, hi = nodes[j], nodes[j + 1]
                            nl, nh = N[j], N[j + 1]
                            if nh > 0 and nl > nh and u > 0:
                                # log-log interpolation: N(v) ~ power law
                                fr = np.log(u / nl) / np.log(nh / nl)
                                v_star = float(
                                    lo * (hi / lo) ** np.clip(fr, 0.0, 1.0))
                            else:
                                # LAST segment: N(v_max) == 0 makes log-log
                                # degenerate and piled every deep photon onto
                                # the segment's left edge (pi0 scheme-A came
                                # out with EXACTLY zero events above v=0.3).
                                # Invert N(v)=u by geometric bisection with
                                # exact sigma_obs -- ~5 rc calls on the ~10%
                                # deepest hard events only.
                                lo_v, hi_v = lo, hi
                                for _ in range(5):
                                    mid = float(np.sqrt(lo_v * hi_v))
                                    Nm = max(so_e - sobs(mid), 0.0)
                                    if Nm > u:
                                        lo_v = mid
                                    else:
                                        hi_v = mid
                                v_star = float(np.sqrt(lo_v * hi_v))
                            ts_e = TailSampler(kin, self.sf, h, v_max,
                                               v_min=vmn)
                            s = ts_e.sample(self.rng, 1)
                            got2 = photon_at(kin, v_star, s["nhat"][0])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                    else:
                        ts_e = TailSampler(kin, self.sf, h, v_max, v_min=vmn)
                        _, _, hf = ts_e.split(so_e)
                        if np.isfinite(hf) and \
                                self.rng.random() < max(hf, 0.0):
                            self.n_hard_chosen += 1
                            s = ts_e.sample(self.rng, 1)
                            got2 = photon_at(kin, float(s["v"][0]),
                                             s["nhat"][0])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                except _SoftOnly:
                    photon4 = meson4 = None
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    photon4 = meson4 = None
                    self.n_drop_except += 1
                got = None
            elif self.born_only:
                got = None
            else:
                got = self._tail_for_cell(cell, h, s_obs)
            if got is None:
                if not self.born_only and not self.exact_per_event:
                    self.n_drop_no_tail += 1
            else:
                if self.rng.random() > got["soft_frac"]:
                    self.n_hard_chosen += 1
                    try:
                        s = self._draw_photon(got)
                        v = float(s["v"])
                        if v < v_max:
                            # rebuild at THIS event's kinematics: omega
                            # depends on z, which the cell centre does not
                            # share (see tail.photon_at)
                            got2 = photon_at(kin, v, s["nhat"])
                            if got2 is not None:
                                photon4, meson4 = got2
                            else:
                                self.n_drop_photon_at += 1
                        else:
                            self.n_drop_v_ge_vmax += 1
                    except (ValueError, ZeroDivisionError):
                        photon4 = meson4 = None
                        self.n_drop_except += 1

            if photon4 is not None:
                self.n_hard += 1
            else:
                self.n_soft += 1

            ev = assemble_event(kin, meson4=meson4, photon4=photon4,
                                azimuth=self.rng.uniform(0, 2 * np.pi))
            if not ev["ok"]:
                self.n_closure_fail += 1
                continue
            # LAB vectors from here on: the acceptance cut and the vector-meson
            # decay below both work on ev["scattered"] / ["recoil"] / ["meson"],
            # so the boost has to come first or a deuteron run would be cut on
            # working-frame momenta.
            ev = self._boost_to_lab(ev)
            ev.update(weight=1.0, Q2=Q2, W2=W2, t=t, phi=phi, helicity=h,
                      eta=eta, radiative=photon4 is not None,
                      # вершина события: при выключенном радиаторе ноль, как
                      # и было; при включённом -- точка внутри мишени, где
                      # электрон провзаимодействовал.  GEMC ждёт её здесь.
                      vz=self._vz, Ebeam_eff=self._E)

            acc = self.cuts.get("iacc", "none")
            if acc != "none":
                from .decay import in_acceptance, DECAY, decay_two_body
                dau = None
                ch = self.sf.channel
                if ch in DECAY:
                    a, bq = decay_two_body(ev["meson"], DECAY[ch][2], self.rng)
                    if a is None:
                        self.n_cut_acc += 1
                        continue
                    ev["daughters"] = (a, bq)
                    dau = (a, bq)
                if not in_acceptance(acc, ev["scattered"], ev["recoil"], dau)[0]:
                    self.n_cut_acc += 1
                    continue
            out.append(ev)
            if progress and len(out) % progress == 0:
                print(f"  {len(out)}/{n_events}")
        return out

    # ------------------------------------------------------------------ #
    def sigma_total(self):
        """Absolute cross section of THIS run, nb (accept-reject estimator).

        sigma = (n_accept / n_trials) * integral of the majorant over the box.
        The generator was printing none of this, so a delta = sigma_obs/sigma_born
        test had no normalisation and could only check the SHAPE (2026-08-17).
        """
        if self.maj is None or self.n_trials == 0:
            return float("nan")
        e = self.maj["edges"]
        dQ2 = np.diff(e["Q2"])[:, None, None, None]
        dW2 = np.diff(e["W2"])[None, :, None, None]
        dt = np.diff(e["t"])[None, None, :, None]
        dph = np.diff(e["phi"])[None, None, None, :]
        vol = dQ2 * dW2 * dt * dph
        # With stage 2 on, the sample IS the exact one, so its normalisation
        # has to come from the same estimator: a trial survives both stages
        # with (s_obs/M) * (so/(s_obs*B)) = so/(M*B), and the cell is drawn
        # ~ M*B, hence (n_accept2/n_trials) * integral(M*B) = integral(sigma
        # exact).  Reporting stage 1 here would give integral(b*eta_capped)
        # for a sample distributed as sigma_exact -- shape and normalisation
        # from two different densities.
        if self.exact_accept and not self.born_only and self.n_accept2 > 0:
            Mt = self.maj.get("Mtot", self.maj["M"])
            return float((self.n_accept2 / self.n_trials)
                         * float((Mt * vol).sum()))
        return float((self.n_accept / self.n_trials)
                     * float((self.maj["M"] * vol).sum()))

    def report(self):
        eff = self.n_accept / self.n_trials if self.n_trials else float("nan")
        n = self.n_hard + self.n_soft
        return dict(trials=self.n_trials, accepted=self.n_accept,
                    sigma_nb=self.sigma_total(),
                    efficiency=eff, hard=self.n_hard, soft=self.n_soft,
                    radiative_fraction=self.n_hard / n if n else float("nan"),
                    closure_failures=self.n_closure_fail,
                    overweight=self.n_overweight,
                    max_overweight=self.max_overweight,
                    cut_yxb=self.n_cut_yxb, cut_acceptance=self.n_cut_acc,
                    cut_eacceptance=self.n_cut_eacc,
                    vmin_fallback=self.n_vmin_fallback,
                    tail_cells_built=len(self._tail_cache),
                    hard_chosen=self.n_hard_chosen,
                    accept2=self.n_accept2, b_viol=self.n_b_viol,
                    max_b_viol=self.max_b_viol,
                    drop_no_tail=self.n_drop_no_tail,
                    drop_v_ge_vmax=self.n_drop_v_ge_vmax,
                    drop_photon_at=self.n_drop_photon_at,
                    drop_except=self.n_drop_except,
                    bw_channel=self.bw_channel,
                    bw_thresh=self.n_bw_thresh, bw_tkin=self.n_bw_tkin)

    def health(self):
        r = self.report()
        msgs = []
        if r["overweight"] > 1e-3 * max(r["trials"], 1):
            msgs.append(f"majorant exceeded in {r['overweight']}/{r['trials']} "
                        f"trials (max x{r['max_overweight']:.2f}) -- raise "
                        f"safety or refine majorant_shape")
        if r["closure_failures"]:
            msgs.append(f"{r['closure_failures']} events failed four-momentum "
                        f"closure and were discarded")
        # <F> = 1 exactly over an isotropic distribution, so the flux rejection
        # reweights the shape and leaves the normalisation alone.  The guards
        # correlate weakly with F, so report the measured value rather than
        # assert the identity: sigma_total is low by this factor.
        if self.n_flux_drawn:
            mf = self.sum_flux / self.n_flux_drawn
            if abs(mf - 1.0) > 0.005:
                msgs.append(f"<flux factor> = {mf:.4f}, not 1 -- sigma_total is "
                            f"low by that factor; the guards are biting F")
        if r["efficiency"] < 0.01:
            msgs.append(f"unweighting efficiency {r['efficiency']:.4f} is low "
                        f"-- refine majorant_shape")
        # Stage-2 bound violations were NOT reported here, and that is the one
        # number that says whether the exact accept actually held.  A run that
        # broke the bound 491 times out of 24678 accepts, by up to a factor
        # 45, still printed "health: clean" (2026-08-18).  Those events are
        # accepted with probability 1 where they deserved more, so they are
        # UNDER-represented: the pi0 delta profile came out 17-29% low at high
        # W, which normalisation then mirrored into a 20% "excess" at low W --
        # 9 sigma, and the only axis still failing after the phi fix.
        nb, tot = r.get("b_viol", 0), max(r.get("accepted", 0), 1)
        if nb > 1e-3 * tot:
            msgs.append(f"stage-2 bound exceeded in {nb}/{tot} accepts (max "
                        f"x{r.get('max_b_viol', 0):.2f}) -- the sample is NOT "
                        f"exactly unweighted; refine table_shape")
        return msgs
