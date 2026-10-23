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
    # What actually drives the W dependence of a fixed |t| cut is how close the
    # cut sits to t_min, not how much t_min varies.  Measured on the generated
    # samples, a cut |t| > 1.0 GeV^2:
    #
    #            median |t_min|   cut/t_min   kept, low W2 decile -> high
    #   J/psi        0.491            2.0          76.1 % -> 58.1 %   18 points
    #   phi          0.029           34.5          31.2 % -> 27.6 %    3.6
    #
    # So the channel with the 214-fold spread in t_min has the WEAKER W
    # dependence, because its cut is far out on the tail, while J/psi's cut
    # sits at twice t_min and bites.  Both |t| and t' are plotted; which one a
    # binning should use depends on where the cut is, and that has to be
    # measured per channel rather than inferred from the spread.
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
