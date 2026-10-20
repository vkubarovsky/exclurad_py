"""Reading a vector-meson campaign's f32 ntuples, and turning them into barns.

The record is the one vm_to_ntuple.groovy writes: 73 little-endian float32 per
event.  The layout is repeated here because a reader that guesses it produces
plots that look right and are wrong -- on 2026-10-11 a column read as pz where
E belonged gave a whole table of t_N that was not t_N.  Every momentum taken
out of here is checked against its particle's mass before it is used.

    [0]       MC::Event weight
    [1..18]   generated px,py,pz for slot k = 0..5, at 1+3k
    [19..66]  reconstructed block of 8 at 19+8k:
              found(0/1) pid charge status quality px py pz
    [67][68]  n_rec(negative daughter pid), n_rec(positive daughter pid)
    [69][70]  n_hadron, n_gamma      [71] n_rec rows
    [72]      bit mask of generated particles present (1<<k)

Slots, by role and not by pid, so phi and rho read the same way:
    0 e'   1 p   2 h+   3 h-   4 gamma(decay)   5 gamma(production)
"""
import glob
import json
import os

import numpy as np

# Two record layouts, because two groovy scripts write them.  The vector one
# (vm_to_ntuple.groovy) carries six slots and 73 floats; the pseudoscalar one
# (hipo_to_ntuple.groovy) carries four and 50.  Reading one with the other's
# offsets does not fail -- it returns numbers -- which is how a column of pz
# was once read where E belonged and a whole table of t_N was not t_N.  So
# the layout is named, not assumed, and every momentum is checked against its
# particle's mass before it is used.
#
#   vector      [0] weight  [1+3k] gen px,py,pz   [19+8k] rec block of 8
#               slots  0 e'  1 p  2 h+  3 h-  4 gamma(decay)  5 gamma(prod)
#   pseudoscalar[0] weight  [1+3k] gen px,py,pz   [13+8k] rec block of 8
#               slots  0 e'  1 N  2 g1  3 g2
#               NOTE: no radiated-photon slot at all.  hipo_to_ntuple keeps
#               only the two photons whose MC::Lund parent is the pi0 row, so
#               the production photon never reaches the ntuple and the
#               "where the radiation goes" page cannot be built for pi0/eta.
LAYOUT = {
    "vector": dict(nf=73, rec0=19, slots=("escat", "prot", "hplus", "hminus",
                                          "gdec", "gprod")),
    "pseudo": dict(nf=50, rec0=13, slots=("escat", "prot", "hplus", "hminus")),
}
MP = 0.938272
MN = 0.939565
ME = 0.000511
# hplus/hminus are the two decay products whatever they are: K+/K- for phi,
# e+/e- for J/psi, and the two photons for pi0 and eta.  "meson" is the state
# they come from.  `baryon` is what the topology detects -- the PROTON on a
# proton target and the struck NEUTRON on a deuteron one, where picking the
# spectator proton instead is a silent error that costs the whole sample.
# `names` are what the slots are CALLED on a plot.  The generic hplus/hminus
# are the right internal names -- they are the positive and negative decay
# product -- but printing them on a pi0 page says "p_h+" for a photon, which
# is simply wrong, and VPK caught it on the first book.  For pi0 and eta the
# two slots are the two decay photons IN MC::Lund ORDER, not sorted by
# energy: measured on the rad million, slot 2 is the harder one in 50.4 % of
# events and the two medians agree to 1 %.  So they are gamma_1 and gamma_2
# with no implied ordering, and the page says so.
CHANNELS = {
    "jpsi": dict(layout="vector", hplus=ME, hminus=ME, meson=3.096900, baryon=MP,
                 names=("e^+", "e^-"), meson_name=r"J/\psi", baryon_name="p'", ordered=True,
                 kinds=("charged",) * 4),
    "phi":  dict(layout="vector", hplus=0.493677, hminus=0.493677, meson=1.019461, baryon=MP,
                 names=("K^+", "K^-"), meson_name=r"\phi", baryon_name="p'", ordered=True,
                 kinds=("charged",) * 4),
    "rho":  dict(layout="vector", hplus=0.139570, hminus=0.139570, meson=0.775260, baryon=MP,
                 names=(r"\pi^+", r"\pi^-"), meson_name=r"\rho", baryon_name="p'", ordered=True,
                 kinds=("charged",) * 4),
    "pi0":  dict(layout="pseudo", hplus=0.0, hminus=0.0, meson=0.1349768, baryon=MP,
                 names=(r"\gamma_1", r"\gamma_2"), meson_name=r"\pi^0", baryon_name="p'", ordered=False,
                 kinds=("charged", "charged", "photon", "photon")),
    "eta":  dict(layout="pseudo", hplus=0.0, hminus=0.0, meson=0.547862, baryon=MP,
                 names=(r"\gamma_1", r"\gamma_2"), meson_name=r"\eta", baryon_name="p'", ordered=False,
                 kinds=("charged", "charged", "photon", "photon")),
    "pi0n": dict(layout="pseudo", hplus=0.0, hminus=0.0, meson=0.1349768, baryon=MN,
                 names=(r"\gamma_1", r"\gamma_2"), meson_name=r"\pi^0", baryon_name="n'", ordered=False,
                 kinds=("charged", "neutron", "photon", "photon")),
    "etan": dict(layout="pseudo", hplus=0.0, hminus=0.0, meson=0.547862, baryon=MN,
                 names=(r"\gamma_1", r"\gamma_2"), meson_name=r"\eta", baryon_name="n'", ordered=False,
                 kinds=("charged", "neutron", "photon", "photon")),
}
MASS = CHANNELS            # kept: older callers index MASS[channel]["meson"]


def _p3(a, k, rec, rec0):
    b = rec0 + 8 * k + 5 if rec else 1 + 3 * k
    return a[:, b:b + 3]


def _pmag_theta_phi(p):
    m = np.linalg.norm(p, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = np.degrees(np.arccos(np.clip(p[:, 2] / np.where(m > 0, m, 1), -1, 1)))
    ph = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return m, th, ph


def sigma_nb(prov_dir, mode):
    """Mean sigma_nb over a set's chunks, and its spread.

    Every chunk of a set ran the same box with a different seed, so the chunk
    sigmas are repeated measurements of one number: the mean is the sample's
    normalisation and the spread over sqrt(n) is its error.  A chunk whose
    sigma sits far from the others is not averaged in silently -- it is
    returned, because that is a symptom (a stale majorant cache gave sigmas
    low by <B> ~ 2x for a week in September and nothing said so).
    """
    vals = []
    for f in sorted(glob.glob(os.path.join(prov_dir, f"{mode}_*.config.json"))):
        try:
            with open(f) as fh:
                vals.append(json.load(fh)["report"]["sigma_nb"])
        except (KeyError, ValueError, OSError):
            pass
    v = np.array(vals, dtype=float)
    if v.size == 0:
        raise RuntimeError(f"no sigma_nb found in {prov_dir} for {mode}")
    return dict(mean=v.mean(), err=v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0,
                n=v.size, min=v.min(), max=v.max(), spread_rel=(v.max() - v.min()) / v.mean())


# The two topologies Mariana Tenorio's analysis uses, as slot tuples.  The
# exclusive one is the four-track sample; the other drops the proton, which
# costs the t measurement and buys roughly four times the statistics -- in
# F18 her exclusive J/psi peak holds 73 events and the three-track one 471,
# so a shape comparison that needs the proton and one that does not are two
# different conversations.
# How well the reconstructed momentum must agree with the generated one for
# the match to count.  Chosen 2026-10-20 from the Delta p / p distributions,
# which separate into a resolution core at zero and a shelf running to -1
# where MC::GenMatch paired a generated particle with a different
# reconstructed one.
#   charged 0.20  sits in the valley on every charged panel, and is 3-10x the
#                 core's own width, so it cuts the shelf and not the core.
#   photon  0.40  the calorimeter core runs -0.24 .. +0.12 and there is NO
#                 valley -- it merges into the shelf -- so 0.20 would cut the
#                 core.  0.40 keeps it; the residue is better removed with a
#                 window on the reconstructed M(gamma gamma), which is a
#                 physics cut and not a matching criterion.
#   neutron None  there is no core to speak of: q16..q84 is -0.78 .. +0.01.
#                 The neutron's momentum is not measured in these data, so a
#                 threshold would be arbitrary and none is applied.  Say so
#                 wherever a pi0n number is quoted.
DP_MAX = {"charged": 0.20, "photon": 0.40, "neutron": None}

TOPOLOGY = {"ee+e-p": (0, 1, 2, 3), "ee+e-": (0, 2, 3),
            "full": (0, 1, 2, 3), "no_baryon": (0, 2, 3)}


def load(pattern, channel, ebeam=10.6, rec=False, require=(0, 1, 2, 3),
         max_files=None, dp_cut=True):
    """Read every f32 matching `pattern` into one dict of arrays.

    rec=False gives the generated event; rec=True the reconstructed one, and
    then `require` is the tuple of slots whose REC track must be matched
    (0 e', 1 p, 2 h+, 3 h-) -- pass TOPOLOGY["ee+e-"] to drop the proton.
    Kinematics are built from the particles, never from a stored column, so
    there is nothing to get out of step with the generator.
    """
    ch = CHANNELS[channel]
    lay = LAYOUT[ch["layout"]]
    NF, REC0, SLOTS = lay["nf"], lay["rec0"], lay["slots"]
    files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"no files match {pattern}")
    chunks = []
    for f in files:
        a = np.fromfile(f, dtype="<f4")
        # A record size that does not divide the file is the loudest symptom
        # of reading one layout with the other's, so it is an error and not
        # a truncation.
        if a.size % NF:
            raise RuntimeError(f"{f}: {a.size} floats is not a multiple of {NF} "
                               f"-- wrong layout for channel {channel}?")
        chunks.append(a.reshape(-1, NF))
    a = np.concatenate(chunks).astype(np.float64)
    n_read = len(a)

    if rec and require:
        # FOUND IS NOT USABLE.  MC::GenMatch happily matches a generated
        # particle to a REC::Particle row with pid 0, 22 or 2112 and a
        # momentum of exactly zero -- a neutral calorimeter cluster with no
        # direction assigned.  The found flag is set, and the four-vector is
        # not there.  Measured 2026-10-20 on the rad samples, as a fraction
        # of the events where all four were "found":
        #     J/psi  8.99 %   (the FT electron, 8.03 % on its own)
        #     phi    8.08 %
        #     pi0 p  13.61 %
        #     pi0 n  58.38 %  -- the NEUTRON alone is 51.88 %
        # Used as a four-vector such a row gives theta = 90 deg, nu = E_beam
        # and W = W_max, which piles up at the edge of every panel and looks
        # like a plotting artefact rather than a selection error.  So a slot
        # counts as reconstructed only when it is found AND carries momentum.
        keep = np.ones(len(a), dtype=bool)
        nzero, nmis = {}, {}
        for k in require:
            if k >= len(SLOTS):
                continue
            found = a[:, REC0 + 8 * k] > 0.5
            pk = np.linalg.norm(a[:, REC0 + 8 * k + 5: REC0 + 8 * k + 8], axis=1)
            ok = found & (pk > 0.0)
            nzero[SLOTS[k]] = int((found & (pk <= 0.0)).sum())
            thr = DP_MAX[ch["kinds"][k]] if dp_cut else None
            if thr is not None:
                pg = np.linalg.norm(a[:, 1 + 3 * k: 4 + 3 * k], axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = np.abs(pk - pg) / np.where(pg > 0, pg, 1.0)
                nmis[SLOTS[k]] = int((ok & ~(rel < thr)).sum())
                ok &= rel < thr
            keep &= ok
        a = a[keep]

    m = ch
    MB = ch["baryon"]
    out = dict(n_files=len(files), n_read=n_read, n_kept=len(a), weight=a[:, 0],
               channel=channel, layout=ch["layout"],
               n_zero_momentum=(nzero if rec and require else {}),
               n_dp_rejected=(nmis if rec and require else {}),
               dp_cut=bool(dp_cut) and bool(rec and require))
    for k, name in enumerate(SLOTS):
        p = _p3(a, k, rec, REC0)
        out[name + "_p"], out[name + "_theta"], out[name + "_phi"] = _pmag_theta_phi(p)
        out[name + "_p3"] = p
    for miss in ("gdec", "gprod"):                 # absent in the pseudoscalar record
        if miss + "_p" not in out:
            z = np.zeros(len(a))
            out[miss + "_p"], out[miss + "_theta"], out[miss + "_phi"] = z, z, z
            out[miss + "_p3"] = np.zeros((len(a), 3))
            out["has_" + miss] = False

    # --- the scattered electron: Q2, nu, W ---------------------------------
    pe = out["escat_p3"]
    Ee = np.sqrt((pe ** 2).sum(1) + ME ** 2)
    cth = np.where(out["escat_p"] > 0, pe[:, 2] / np.where(out["escat_p"] > 0, out["escat_p"], 1), 1)
    out["Q2"] = 2 * ebeam * Ee * (1 - cth)
    out["nu"] = ebeam - Ee                     # = E_gamma for quasi-real photoproduction
    out["W2"] = MB ** 2 + 2 * MB * out["nu"] - out["Q2"]
    out["W"] = np.sqrt(np.maximum(out["W2"], 0))
    out["y"] = out["nu"] / ebeam

    # --- the recoil proton: t, straight from its energy --------------------
    # t = (p - p')^2 = 2M^2 - 2M E_p'.  The proton is the MEASURED particle in
    # this topology, so this is the t the box was drawn in, not a derived one.
    pp = out["prot_p3"]
    Ep = np.sqrt((pp ** 2).sum(1) + MB ** 2)
    out["t"] = 2 * MB ** 2 - 2 * MB * Ep
    out["mt"] = -out["t"]

    # --- the decay pair: the meson's invariant mass ------------------------
    ph, pm = out["hplus_p3"], out["hminus_p3"]
    Eh = np.sqrt((ph ** 2).sum(1) + m["hplus"] ** 2) + np.sqrt((pm ** 2).sum(1) + m["hminus"] ** 2)
    pv = ph + pm
    out["m_pair"] = np.sqrt(np.maximum(Eh ** 2 - (pv ** 2).sum(1), 0))
    out["meson_p"], out["meson_theta"], out["meson_phi"] = _pmag_theta_phi(pv)

    # --- t', the distance from the physical forward edge --------------------
    # |t_min| runs over three orders of magnitude across a wide box, so |t|
    # and t' are not the same picture at all: a fixed cut in |t| removes a
    # different slice of t' in every (W2, Q2) cell.  Both are plotted.
    mV = m["meson"]
    W = np.sqrt(np.maximum(out["W2"], 1e-9))
    Eg = (out["W2"] - out["Q2"] - MB ** 2) / (2 * W)
    pg = np.sqrt(np.maximum(Eg ** 2 + out["Q2"], 0))
    EV = (out["W2"] + mV ** 2 - MB ** 2) / (2 * W)
    pV_cm = np.sqrt(np.maximum(EV ** 2 - mV ** 2, 0))
    t0 = -out["Q2"] + mV ** 2 - 2 * (Eg * EV - pg * pV_cm)
    out["t_min"] = t0
    out["tprime"] = np.abs(t0) - out["mt"]      # >= 0 inside the physical region
    out["mtprime"] = out["mt"] - np.abs(t0)     # = -t' , the usual plotting variable
    return out
