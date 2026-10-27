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
import re
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
# nf is the record before the vertices were appended on 2026-10-22, nf_vz the
# record with them.  Both are accepted: a file written by the old reader still
# reads, it simply has no vertex and the vertex pages say so.  The vertices
# went on the END precisely so that every existing offset stayed where it was.
LAYOUT = {
    "vector": dict(nf=73, nf_vz=80, vz_gen=73, vz_rec0=74, rec0=19,
                   slots=("escat", "prot", "hplus", "hminus", "gdec", "gprod")),
    "pseudo": dict(nf=50, nf_vz=55, vz_gen=50, vz_rec0=51, rec0=13,
                   slots=("escat", "prot", "hplus", "hminus")),
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


def chunks_present(root, mode):
    """The chunk indices that actually have an ntuple, as a sorted list.

    The normalisation has to come from the chunks you LOADED, not from every
    chunk the provenance directory happens to hold.  On 2026-10-23 a second
    million was generated as chunks 201-400 while the ntuples were still only
    1-200: sigma_nb then averaged 400 sidecars for a 200-chunk sample, and the
    plot book printed 0.94679 for a ratio that is 0.94658.  The peer session
    found it in the book's own header.
    """
    idx = set()
    for od in sorted(glob.glob(os.path.join(root, mode, "out_*"))):
        for f in glob.glob(os.path.join(od, "*.f32")):
            m = re.search(r"_(\d+)\.f32$", os.path.basename(f))
            if m:
                idx.add(int(m.group(1)))
    return sorted(idx)


def events_per_chunk(root, mode, channel):
    """Records in one ntuple file, from its size -- not a remembered 5000."""
    lay = LAYOUT["vector" if channel in ("jpsi", "phi", "rho") else "pseudo"]
    for od in sorted(glob.glob(os.path.join(root, mode, "out_*"))):
        fs = glob.glob(os.path.join(od, "*.f32"))
        if not fs:
            continue
        sz = os.path.getsize(fs[0])
        for nf in (lay["nf_vz"], lay["nf"]):
            if sz % (nf * 4) == 0:
                return sz // (nf * 4)
    raise RuntimeError(f"cannot size an ntuple under {root}/{mode}")


def _preset_factor(rep, path, nopreset, unknown):
    """written / counted, for the chunk whose report dict this is.

    sigma_total() normalises with n_accept2 (rad, exact-accept) or n_accept
    (Born), and both counters are incremented INSIDE the sampling loop --
    events.py:1434 and :1469.  The iacc daughter preset (|p| > 1 GeV,
    theta 5-40 deg for the jpsi preset) is applied LATER, in the outer loop at
    events.py:1645, where a failing event increments n_cut_acc and is dropped.
    So a dropped event is absent from the LUND file and present in sigma.

    Spreading sigma_nb over the written events as sigma_nb/n_events therefore
    overweights each one by counted/written, and -- this is the part that
    matters -- by a DIFFERENT amount in the two arms, because a hard decay or
    production photon leaves a lepton below the 1 GeV threshold.  Measured over
    the whole 2026-10-09 J/psi campaign, 2000 chunks per arm:

        rad   written/counted = 0.61665 +- 0.00012
        born  written/counted = 0.62499 +- 0.00012
        ratio                 = 0.98666 +- 0.00027

    so every rad/Born ratio built from an uncorrected sigma_nb is 1.3 % high.
    sigma_rad/sigma_Born on this campaign is 0.9331, not 0.9457.  Found by the
    rc_method_jpsi study on 2026-10-27; the 0.9457 is still right as the
    PRE-acceptance ratio, it is simply not the normalisation of the events that
    were written.

    The correction is applied here, at the consumer, and not in sigma_total().
    sigma_total() reporting the cross section within the ELECTRON acceptance is
    deliberate and documented (events.py:758-775); changing what it means would
    make every existing sidecar incomparable with every future one, silently.

    iacc = none leaves cut_acceptance at 0 and this returns exactly 1.0 --
    measured on the phi and pi0 campaigns, whose cards declare none.
    """
    a2 = rep.get("accept2", 0) or 0
    a1 = rep.get("accepted", 0) or 0
    counted = a2 if a2 > 0 else a1          # exactly what sigma_total() divides by
    if counted <= 0:
        # Pre-counter sidecar: the factor cannot be established either way, and
        # assuming 1.0 is how a 1.3 % bias stayed invisible for three weeks.
        raise RuntimeError(
            f"{path}: report carries neither accept2 nor accepted, so the iacc "
            f"preset normalisation cannot be checked.  Regenerate the "
            f"provenance, or read this set with a reader that does not claim "
            f"an absolute sigma.")
    if "cut_acceptance" not in rep:
        unknown.append(path)
        return 1.0
    cut = rep["cut_acceptance"] or 0
    if cut == 0:
        nopreset.append(path)
        return 1.0
    written = counted - cut
    if written <= 0:
        raise RuntimeError(
            f"{path}: counted {counted} - cut_acceptance {cut} = {written}, "
            f"which is not a number of written events.")
    return written / counted


def sigma_nb(prov_dir, mode, chunks=None):
    """Mean sigma_nb over a set's chunks, and its spread.

    `chunks` restricts the average to those chunk indices -- pass
    chunks_present() so the normalisation matches the sample.  Without it
    every sidecar in the directory is averaged, which is right only when the
    ntuples cover every chunk.

    Every chunk of a set ran the same box with a different seed, so the chunk
    sigmas are repeated measurements of one number: the mean is the sample's
    normalisation and the spread over sqrt(n) is its error.  A chunk whose
    sigma sits far from the others is not averaged in silently -- it is
    returned, because that is a symptom (a stale majorant cache gave sigmas
    low by <B> ~ 2x for a week in September and nothing said so).
    """
    want = set(chunks) if chunks is not None else None
    vals, missing = [], []
    # Which chunks needed no correction (iacc = none) and which could not be
    # checked, reported rather than swallowed: a caller that sees
    # preset_unknown > 0 is reading a sigma whose normalisation is unverified.
    nopreset, unknown = [], []
    for f in sorted(glob.glob(os.path.join(prov_dir, f"{mode}_*.config.json"))):
        m = re.search(r"_(\d+)\.config\.json$", os.path.basename(f))
        i = int(m.group(1)) if m else None
        if want is not None and i not in want:
            continue
        try:
            with open(f) as fh:
                rep = json.load(fh)["report"]
            vals.append(rep["sigma_nb"] * _preset_factor(rep, f, nopreset, unknown))
        except (KeyError, ValueError, OSError):
            missing.append(i)
    v = np.array(vals, dtype=float)
    if v.size == 0:
        raise RuntimeError(f"no sigma_nb found in {prov_dir} for {mode}")
    if want is not None and v.size != len(want):
        # Loudly, not silently: a sample whose chunks do not all have a
        # sidecar is normalised by a subset, and that is a fact about the
        # number, not a detail of the loop.
        raise RuntimeError(
            f"{prov_dir}: {len(want)} chunks loaded but {v.size} sidecars "
            f"readable{' (unreadable: %s)' % missing[:5] if missing else ''} -- "
            f"generate the missing provenance or restrict the sample")
    return dict(mean=v.mean(), err=v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0,
                n=v.size, min=v.min(), max=v.max(), spread_rel=(v.max() - v.min()) / v.mean(),
                preset_corrected=v.size - len(nopreset) - len(unknown),
                preset_none=len(nopreset), preset_unknown=len(unknown))


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
#                 core.  0.40 keeps it.  This is the one threshold here that
#                 is a judgement and not a reading: with no valley there is
#                 nothing to aim at, and 0.40 only means "keeps the core,
#                 drops most of the shelf".  VPK reviewed it 2026-10-21 and
#                 kept it.
#                 STILL TO DO, agreed the same day: remove the residual shelf
#                 with a window on the RECONSTRUCTED M(gamma gamma) instead of
#                 leaning harder on this threshold.  That is a physics cut on
#                 the pair, not a matching criterion on one slot, and it is
#                 the right tool for what 0.40 leaves behind.
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
    REC0, SLOTS = lay["rec0"], lay["slots"]
    files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"no files match {pattern}")
    # Which record is this?  The vertices were appended on 2026-10-22, so a
    # file divides by nf_vz if it has them and by nf if it does not.  Deciding
    # per file rather than per campaign means a half-reprocessed directory
    # still reads, and mixing the two is refused rather than silently
    # reinterpreted.
    sizes = set()
    raw = []
    for f in files:
        a = np.fromfile(f, dtype="<f4")
        if a.size % lay["nf_vz"] == 0 and a.size % lay["nf"] != 0:
            nf = lay["nf_vz"]
        elif a.size % lay["nf"] == 0 and a.size % lay["nf_vz"] != 0:
            nf = lay["nf"]
        elif a.size % lay["nf_vz"] == 0:
            nf = lay["nf_vz"]          # divides by both: the longer wins
        else:
            raise RuntimeError(f"{f}: {a.size} floats divides neither {lay['nf']} "
                               f"nor {lay['nf_vz']} -- wrong layout for {channel}?")
        sizes.add(nf)
        raw.append(a.reshape(-1, nf))
    if len(sizes) > 1:
        raise RuntimeError(f"{pattern}: records of {sorted(sizes)} floats in one "
                           f"set -- reprocess them all or read them separately")
    NF = sizes.pop()
    has_vz = NF == lay["nf_vz"]
    chunks = raw
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
               has_vz=has_vz,
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
    # W where W2 < 0 is not 0, it does not exist.  Same argument as t_min
    # below: a NaN is dropped by the histogram, a 0 pulls the axis to zero.
    with np.errstate(invalid="ignore"):
        out["W"] = np.where(out["W2"] > 0, np.sqrt(np.abs(out["W2"])), np.nan)
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

    # --- the vertex, when the record carries it ----------------------------
    if has_vz:
        out["vz_gen"] = a[:, lay["vz_gen"]]
        for k, name in enumerate(SLOTS):
            out[name + "_vz"] = a[:, lay["vz_rec0"] + k]

    # --- t', the distance from the physical forward edge --------------------
    # |t| and t' are not the same picture, but NOT for the reason this comment
    # used to give.  It said "|t_min| runs over three orders of magnitude
    # across a wide box", which is true of phi (measured 213.6 here) and false
    # of J/psi (4.0), whose W2 box is narrow -- 16.29 to 20.80, with |t_min|
    # from 0.4700 to 1.8816 and the 1st to 75th percentile inside 0.470-0.526.
    # Writing one channel's number into shared code as if it were universal is
    # the same mistake as quoting one channel's CARD for another, and it was
    # made here by the same author on the same day.
    #
    # What drives the W dependence of a fixed |t| cut is where the cut sits
    # relative to t_min, and the dependence is NOT monotonic in that ratio --
    # it peaks in the shoulder of the t distribution and dies at both ends.
    # Swept on the generated J/psi sample, median |t_min| = 0.4909, lowest
    # against highest W2 decile:
    #
    #   |t| cut   cut/t_min   kept all   low W2   high W2   spread
    #     0.55        1.1       95.2 %   100.0 %   92.1 %   + 7.9 pt
    #     0.70        1.4       82.6 %    97.1 %   78.5 %   +18.6 pt   <- peak
    #     1.00        2.0       61.4 %    76.1 %   58.1 %   +18.0 pt   <- our cut
    #     1.50        3.1       38.0 %    47.6 %   35.9 %   +11.7 pt
    #     2.00        4.1       24.0 %    29.3 %   23.0 %   + 6.3 pt
    #     3.00        6.1        9.5 %    11.0 %    9.1 %   + 1.9 pt
    #     4.00        8.1        3.1 %     3.3 %    3.0 %   + 0.3 pt
    #
    # Far out, the exponential fall dominates and a shift in t_min is a small
    # relative displacement, so the cut takes the same fraction everywhere --
    # which is why phi, whose |t| > 1.0 sits at 34 x t_min, shows only 3.6
    # points despite a 214-fold spread in t_min.  Right AT t_min it dies for
    # the opposite reason: the cut keeps essentially everything in every cell
    # (100.0 % against 92.1 %) and there is nothing for t_min to move across.
    #
    # J/psi's cut sits at 2.0 x t_min, within noise of the maximum, so this
    # channel is the WORST case for the effect rather than a typical one.
    # Moving the lowest bin edge out to 2-3 GeV^2 would reduce the spread from
    # 18 to 2-6 points at the cost of 60-75 % of the sample; stated as a trade,
    # not a recommendation.  Both |t| and t' are plotted; which a binning
    # should use has to be measured per channel at the cut it actually uses.
    # Swept by the analysis session, 2026-10-23.
    mV = m["meson"]

    # t_min DOES NOT EXIST BELOW THRESHOLD, and the honest value is NaN.
    #
    # This line used to read  W = np.sqrt(np.maximum(out["W2"], 1e-9)),  and
    # the floor was there to stop a NaN.  Stopping it was the mistake.  A
    # RECONSTRUCTED W2 can come out negative -- 10 of 14393 on the Born
    # inbending set -- and the floor then gives W = 3.2e-5 and t_min = 1.8e11,
    # a number eleven orders of magnitude wrong instead of no number at all.
    # That is strictly worse than a NaN: np.histogram DROPS a NaN and the axis
    # is then set by the real entries, while 1.8e11 silently sets the axis for
    # the whole panel.  It is why -t' ran negative and the W axis started at
    # 2.5 on the reconstructed pages -- not a plotting choice, a derived
    # quantity computed where it is not defined.  Found by the analysis
    # session, 2026-10-23, chasing an artefact spike that t' had hidden.
    #
    # Measured on born inbending 45 nA: 274 of 14393 events (1.90 %) sit below
    # (M_B + m_V)^2 in reconstructed W2, and FIVE of them carried the axis.
    #
    # Mariana's data has the same disease from an independent code: her handoff
    # warns that t_prime reaches -3.6e8 because t_min goes as 1/E_gamma^2 and
    # the reconstructed E_photon passes through zero.  Two codes, one undefined
    # quantity computed anyway.  Check it in any t' this group produces.
    thr = (MB + mV) ** 2
    below = out["W2"] < thr
    out["n_below_threshold"] = int(below.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        W = np.where(out["W2"] > 0, np.sqrt(np.abs(out["W2"])), np.nan)
        Eg = (out["W2"] - out["Q2"] - MB ** 2) / (2 * W)
        pg = np.sqrt(np.maximum(Eg ** 2 + out["Q2"], 0))
        EV = (out["W2"] + mV ** 2 - MB ** 2) / (2 * W)
        pV_cm = np.sqrt(np.maximum(EV ** 2 - mV ** 2, 0))
        t0 = -out["Q2"] + mV ** 2 - 2 * (Eg * EV - pg * pV_cm)
    t0 = np.where(below, np.nan, t0)
    out["t_min"] = t0
    out["tprime"] = np.abs(t0) - out["mt"]      # >= 0 inside the physical region
    out["mtprime"] = out["mt"] - np.abs(t0)     # = -t' , the usual plotting variable
    return out


# --------------------------------------------------------------- pooling ---
def pool_weights(t_gen, samples, var="|t|"):
    """Per-event weights for several samples whose boxes OVERLAP.

    The case this exists for: a production sample over the whole range, plus a
    later sample covering only the tail, generated because the tail ran out of
    events.  They cannot simply be concatenated -- in the overlap the events
    came from two sources and would be counted twice.

    Every sample draws proportional to sigma(x) inside its own box, so the
    generation density at x is the sum over the samples whose box contains x:

        g(x) = SUM_X  N_X sigma(x) / sigma_X

    and the weight that makes the sum of weights reproduce the cross section
    is sigma(x)/g(x).  **sigma(x) cancels**, which is what makes this cheap:

        w(x) = 1 / SUM_{X : x in box_X} ( N_X / sigma_X )

    -- piecewise constant, one value per distinct overlap pattern, not a
    function of the kinematics.  It is the standard minimum-variance
    multi-channel estimator, not a local invention.

    THE BOX TEST USES THE GENERATED VALUE, never the reconstructed one.  An
    event born at |t| = 1.6 and measured at 1.4 keeps its overlap weight; test
    it on the measured value instead and the two samples double-count along
    the boundary.

    Checked 2026-10-20 at 6.535 GeV, pi0.amp2614 Born, A = (|t| < 2.5, 50k,
    1.58647 nb) pooled with B = (1.5 < |t| < 2.5, 50k, 0.09350 nb): the pooled
    weights reproduce sigma_A to 0.020 %, agree bin by bin below the overlap
    exactly and inside it to within the statistics of A alone, and multiply the
    events above |t| = 1.5 by 17.9.

    t_gen    array of the GENERATED |t| of every event, in sample order
    samples  list of dicts: {"lo": , "hi": , "sigma": , "n": } -- |t| bounds,
             the sample's cross section in nb, and its generated event count.
             `n` must be the number GENERATED, not the number that survived a
             later cut, or the density is wrong.
    """
    t = np.abs(np.asarray(t_gen, dtype=float))
    if sum(s["n"] for s in samples) != t.size:
        raise ValueError(f"t_gen has {t.size} entries but the samples declare "
                         f"{sum(s['n'] for s in samples)} generated events")
    inv = np.zeros_like(t)
    for s in samples:
        if s["sigma"] <= 0 or s["n"] <= 0:
            raise ValueError(f"sample with sigma={s['sigma']} n={s['n']}")
        inv += np.where((t >= s["lo"]) & (t < s["hi"]), s["n"] / s["sigma"], 0.0)
    if np.any(inv <= 0):
        raise ValueError(f"{int((inv <= 0).sum())} events lie in no sample's box "
                         f"-- {var} outside every (lo, hi)")
    return 1.0 / inv


# ------------------------------------------------------------------ vertex ---
def lund_vz(pattern, max_files=None, max_events=400000):
    """Vertex z of the generated events, read from the LUND files.

    It is NOT in the f32 record -- the ntuple carries momenta and nothing
    else -- so a vertex plot has to come from lund_osg/.  That is worth the
    inconvenience: GEMC keeps the LUND vertex exactly (BEAM_V and SPREAD_V in
    the gcard are not applied to LUND input, checked on 40k events per arm),
    so column 14 is where the event happens in the detector.

    This exists because on 2026-10-21 every Born event came out at z = 0,
    outside the -5.5 .. -0.5 cell, and nothing plotted it.  One flag, itarg,
    governed both the external radiator and the vertex draw; switching the
    radiator off for the Born arm switched the target length off with it.
    VPK asked for the distribution along the target to be plotted after every
    generator run, and this is the reader for it.
    """
    files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"no LUND files match {pattern}")
    vz = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                a = line.split()
                if len(a) >= 14:
                    vz.append(float(a[13]))
        if len(vz) >= max_events:
            break
    return np.asarray(vz[:max_events], dtype=float)
