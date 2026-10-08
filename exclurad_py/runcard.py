"""
Run card in the exclugen style: plain text, one value per line, `!` comment.

Positional, like `exclugen/gen_input.txt` -- the order below IS the format.
Blank lines and whole-line comments are skipped; anything after `!` on a
value line is ignored.  Fortran-style exponents (1.0d-4) are accepted.

    eta.tableB1        ! model  <channel>.<vintage>
    10.6               ! ebeam (GeV)
    100000             ! nev
    0                  ! iborn   1 = Born only, 0 = full RC
    4.5 12.0           ! W2 range (GeV^2)
    1.0 5.0            ! Q2 range (GeV^2)
    -2.0 -0.10         ! t range (GeV^2, both negative)
    0.60               ! vcut    MX^2 - m_miss^2 (GeV^2); 0 = no cut
    -0.30              ! t_nucl  (GeV^2), pi0/eta only; ignored for vector
    0                  ! ihel    +1, -1, or 0 = helicity-averaged
    20260813           ! seed
    1                  ! idecay  1 = decay vector meson h+h-, 0 = leave it
    /Volumes/wd_14tb/mc/exclurad_py   ! output root
    0.0 1.0            ! y range      (optional)
    0.0 1.0            ! xB range     (optional)
    none               ! iacc         (optional)
    1.0d-4             ! vmin         (optional)
    1                  ! itarg   1 = pre-vertex external radiator (optional)
    -7.5 -2.5          ! vertex z range, cm -- the 5 cm LH2 cell (optional)

The one departure from exclugen: line 1 is a MODEL NAME, not an `ivec`
integer.  pi0 and eta each have several vintages that give visibly
different RC (2013 / 2021 / Table B.1), and an integer channel code cannot
say which one you ran.  The model name goes into the LUND provenance
header, so a sample can always be traced back to the cross section that
made it.  `python3 -m exclurad_py.runcard --models` lists them.
"""

import os
import sys

FIELDS = [
    ("model", str, "model  <channel>.<vintage>"),
    ("ebeam", float, "ebeam (GeV)"),
    ("nev", int, "nev"),
    ("iborn", int, "iborn   1 = Born only, 0 = full RC"),
    ("W2", "pair", "W2 range (GeV^2)"),
    ("Q2", "pair", "Q2 range (GeV^2)"),
    ("t", "pair", "t range (GeV^2, both negative)"),
    ("vcut", float, "vcut    MX^2 - m_miss^2 (GeV^2); 0 = no cut"),
    ("t_nucl", float, "t_nucl  (GeV^2), pi0/eta only"),
    ("ihel", int, "ihel    +1, -1, or 0 = helicity-averaged"),
    ("seed", int, "seed"),
    ("idecay", int, "idecay  1 = decay vector meson h+h-, 2 = same + radiation of the decay leptons (J/psi), 0 = leave it"),
    ("outroot", str, "output root"),
    # --- optional trailing fields (cards without them still parse) ---
    ("y", "pair", "y range (inelasticity nu/E)"),
    ("xB", "pair", "xB range"),
    ("iacc", str, "acceptance preset: none | jpsi"),
    ("vmin", float, "soft/hard split (GeV^2); 0 = choose automatically"),
    ("itarg", int, "1 = pre-vertex external radiator (target straggling)"),
    ("vz", "pair", "vertex z range (cm) -- the target cell"),
    ("ptarg", float, "ptarg   longitudinal target polarisation, -1..+1; 0 = unpolarised"),
    ("fermi", str, "fermi   deuteron target: none | av18"),
    ("theta_e", "pair", "scattered-electron polar angle window, degrees"),
    ("fermi_mstar_min", float, "fermi_mstar_min  floor on the struck nucleon's off-shell mass m* (GeV); 0.30 for pi0/eta, 0.10 for near-threshold J/psi"),
]
N_REQUIRED = 13          # fields up to and including outroot
# itarg defaults OFF and vz to a point target: a card written before the
# radiator existed keeps producing exactly what it produced before.
# `fermi` defaults to "none", so a card written before the deuteron target
# existed produces exactly what it produced before -- byte for byte, because
# the Fermi momentum is then never drawn and the RNG stream is untouched.
# A string, not an integer: the wave function is a physics input that belongs
# in the card and in the LUND header in a form a reader understands, as with
# `iacc none`.
# vmin DEFAULTS TO A FIXED SPLIT.  It used to default to 0, which does not
# mean zero -- the soft/hard integral diverges there -- it means "choose one
# per event".  A card that simply omitted the field therefore got a boundary
# that moved from event to event, and nothing said so.  0 still selects the
# automatic mode, but now only when it is asked for explicitly.
# theta_e defaults to the whole range, so a card written before this field
# existed keeps producing exactly what it produced before.  It is applied in
# _pass_electron_acc, BEFORE the Born cross section and the radiative
# correction, because theta of the scattered electron follows from Q2 and W2
# alone -- so a rejected event costs three arithmetic operations, not a
# radiative integral.
DEFAULT_OPT = dict(y=(0.0, 1.0), xB=(0.0, 1.0), iacc="none", vmin=1.0e-4,
                   itarg=0, vz=(0.0, 0.0), ptarg=0.0, fermi="none",
                   theta_e=(0.0, 180.0), fermi_mstar_min=0.30)


def _num(tok):
    return float(tok.replace("d", "e").replace("D", "E"))


def parse(path):
    """Read a run card -> dict.  Raises ValueError with the line number."""
    vals = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("!")[0].strip()
            if line:
                vals.append((lineno, line))

    if len(vals) < N_REQUIRED:
        missing = [f[0] for f in FIELDS[len(vals):N_REQUIRED]]
        raise ValueError(f"{path}: only {len(vals)} value lines, need at least "
                         f"{N_REQUIRED}. Missing: {', '.join(missing)}")
    if len(vals) > len(FIELDS):
        raise ValueError(f"{path}: {len(vals)} value lines, expected at most "
                         f"{len(FIELDS)} (extra at line {vals[len(FIELDS)][0]})")

    cfg = dict(DEFAULT_OPT)
    for (name, kind, desc), (lineno, tok) in zip(FIELDS, vals):
        try:
            if kind == "pair":
                a, b = tok.split()[:2]
                cfg[name] = (_num(a), _num(b))
            elif kind is float:
                cfg[name] = _num(tok.split()[0])
            elif kind is int:
                cfg[name] = int(_num(tok.split()[0]))
            else:
                cfg[name] = tok.split()[0] if name != "outroot" else tok
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: cannot read {name} "
                             f"({desc}) from {tok!r}: {exc}") from exc

    _validate(cfg, path)
    return cfg


def _validate(cfg, path):
    from .models import names
    if cfg["model"] not in names():
        chan = cfg["model"].split(".")[0]
        near = [n for n in names() if n.startswith(chan + ".")]
        hint = f"  did you mean one of {near}?" if near else ""
        raise ValueError(f"{path}: unknown model {cfg['model']!r}.{hint}\n"
                         f"available: {names()}")
    for k in ("W2", "Q2"):
        if cfg[k][0] >= cfg[k][1]:
            raise ValueError(f"{path}: {k} range {cfg[k]} is not increasing")
    if not (cfg["t"][0] < cfg["t"][1] < 0):
        raise ValueError(f"{path}: t range {cfg['t']} must be increasing and "
                         f"both negative (e.g. -2.0 -0.10)")
    if cfg["ihel"] not in (-1, 0, 1):
        raise ValueError(f"{path}: ihel must be +1, -1 or 0")
    if not -1.0 <= cfg.get("ptarg", 0.0) <= 1.0:
        raise ValueError(f"{path}: ptarg must lie in [-1, +1]")
    if cfg.get("ptarg", 0.0) and cfg["ihel"] == 0:
        # ihel = 0 does NOT throw the double-spin part away.  The helicity is
        # drawn per event, the Born and the RC are both evaluated at THAT h
        # (events.py: _born(..., h) and rc_factor(..., h=h)), and since the
        # 2026-09-23 fix the drawn value is written to LUND header column 5.
        # So A_LL is recovered by splitting the one sample on column 5 -- the
        # h*Lambda term shows up as a yield difference between the two subsets
        # (measured: A = +0.21 +- 0.05 at Lambda = +1, and 0 at Lambda = 0).
        # This warning used to say the terms "average to zero"; that was true
        # only while column 5 was always 0.000 and h was therefore unrecorded.
        print("runcard: ihel = 0 with ptarg != 0 -- the helicity is drawn per "
              "event and written to LUND column 5; split on it to get A_LL. "
              "Requires EXACT_ACCEPT=1, or the RC is built at one helicity.")
    if cfg["idecay"] not in (0, 1, 2):
        raise ValueError(f"{path}: idecay must be 0, 1 or 2")
    if cfg["idecay"] == 2:
        from .generator.decay import LEPTONIC
        if cfg["model"].split(".")[0] not in LEPTONIC:
            raise ValueError(f"{path}: idecay 2 (radiation of the decay "
                             f"leptons) is for {LEPTONIC} only")
    if cfg["iborn"] not in (0, 1):
        raise ValueError(f"{path}: iborn must be 0 or 1")
    if cfg["nev"] <= 0:
        raise ValueError(f"{path}: nev must be positive")
    for k in ("y", "xB"):
        if cfg[k][0] >= cfg[k][1]:
            raise ValueError(f"{path}: {k} range {cfg[k]} is not increasing")
    from .generator.decay import ACCEPTANCE
    if cfg["iacc"] not in ("none",) + tuple(ACCEPTANCE):
        raise ValueError(f"{path}: unknown acceptance preset {cfg['iacc']!r}; "
                         f"use 'none' or one of {sorted(ACCEPTANCE)}")
    # Caught here rather than deep in the generator: a typo in the wave
    # function silently falling back to a free nucleon would produce a sample
    # that looks fine and is not the one that was asked for.
    if not 0.0 < cfg.get("fermi_mstar_min", 0.30) < 0.9:
        raise ValueError(f"{path}: fermi_mstar_min must lie in (0, 0.9) GeV")
    from .generator.fermi import WAVEFUNCTIONS
    if cfg["fermi"] not in ("none",) + tuple(WAVEFUNCTIONS):
        raise ValueError(f"{path}: unknown deuteron wave function "
                         f"{cfg['fermi']!r}; use 'none' or one of "
                         f"{sorted(WAVEFUNCTIONS)}")


def echo(cfg, fh=sys.stdout):
    """Print the parsed card back, exclugen-style, so a log records the run."""
    w = max(len(f[0]) for f in FIELDS)
    print(" exclurad_py run card", file=fh)
    print(" " + "-" * 48, file=fh)
    for name, kind, desc in FIELDS:
        v = cfg[name]
        v = f"{v[0]:g} {v[1]:g}" if isinstance(v, tuple) else v
        print(f"   {name:<{w}} = {v}", file=fh)
    print(" " + "-" * 48, file=fh)


TEMPLATE = """! exclurad_py run card -- plain text, one value per line, ! starts a comment.
! Positional (exclugen convention): the ORDER of the lines is the format.
{model:<34} ! model  <channel>.<vintage>   (--models to list)
{ebeam:<34} ! ebeam (GeV)
{nev:<34} ! nev
{iborn:<34} ! iborn   1 = Born only, 0 = full RC
{W2:<34} ! W2 range (GeV^2)
{Q2:<34} ! Q2 range (GeV^2)
{t:<34} ! t range (GeV^2, both negative)
{vcut:<34} ! vcut    MX^2 - m_miss^2 (GeV^2);  0 = no cut (kinematic limit)
{t_nucl:<34} ! t_nucl  (GeV^2) for pi0/eta models; ignored for vector mesons
{ihel:<34} ! ihel    +1, -1, or 0 = helicity-averaged
{seed:<34} ! seed
{idecay:<34} ! idecay  1 = decay vector meson to h+h- isotropically, 2 = + decay-lepton radiation (J/psi), 0 = no decay
{outroot:<34} ! output root (<channel>/<date>_<E>_<mode>_v1/ created under it)
{y:<34} ! y range  (nu/E)   -- match the reference generator's cut
{xB:<34} ! xB range          -- match the reference generator's cut
{iacc:<34} ! iacc    acceptance preset: none | jpsi  (|p|,theta on e',p',daughters)
{vmin:<34} ! vmin    soft/hard split (GeV^2); 0 = auto.  exclugen uses 1.0d-4
{itarg:<34} ! itarg   1 = pre-vertex external radiator (target straggling)
{vz:<34} ! vz      vertex z range (cm) -- the target cell
{ptarg:<34} ! ptarg   longitudinal target polarisation, -1..+1; 0 = unpolarised
{fermi:<34} ! fermi   deuteron target: none | av18  (Fermi motion + spectator)
{theta_e:<34} ! theta_e  scattered-electron polar angle window, deg
{fermi_mstar_min:<34} ! fermi_mstar_min  m* floor (GeV)
"""


def write_template(path, **kw):
    d = dict(model="eta.tableB1", ebeam="10.6", nev="100000", iborn="0",
             W2="4.5 12.0", Q2="1.0 5.0", t="-2.0 -0.10", vcut="0.20",
             t_nucl="-0.30", ihel="0", seed="20260813", idecay="1",
             outroot="/Volumes/wd_14tb/mc/exclurad_py",
             y="0.0 1.0", xB="0.0 1.0", iacc="none", vmin="1.0e-4",
             itarg="0", vz="0.0 0.0", ptarg="0.0", fermi="none",
             theta_e="0.0 180.0", fermi_mstar_min="0.30")
    d.update({k: str(v) for k, v in kw.items()})
    with open(path, "w") as fh:
        fh.write(TEMPLATE.format(**d))
    return path


if __name__ == "__main__":
    if "--models" in sys.argv:
        from .models import list_models
        print(list_models())
    elif "--template" in sys.argv:
        i = sys.argv.index("--template")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "gen_input.txt"
        print("wrote", write_template(out))
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        echo(parse(sys.argv[1]))
    else:
        print(__doc__)
