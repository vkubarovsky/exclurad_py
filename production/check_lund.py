"""Integrity of the LUND output -- the checks a plot cannot make.

    python3 production/check_lund.py <channel> [file-or-glob ...]
    python3 production/check_lund.py pi0 "/path/*_rad_v1/lund/*.lund"

validate_lund.py draws twelve distributions and is useful for spotting a
shape that looks wrong.  It cannot see a violated conservation law: a decay
that does not close, a photon written with the wrong parent, a meson whose
written mass and four-vector disagree all leave the distributions looking
entirely reasonable.  Nothing in the suite tested those until 2026-08-21.

Per event, against the format in generator/lund.py:

  1. the header's particle count matches the lines that follow;
  2. four-momentum: the STATUS 1 particles sum to beam + target.  A decayed
     meson carries status 0 (documentation) and its daughters status 1, so
     summing status 1 counts each final state exactly once;
  3. mass shell: E^2 - p^2 equals the mass in column 11, and that mass
     equals the PDG value for the particle's PID;
  4. decay closure: a status 0 meson equals the sum of its daughters;
  5. charge: the final state sums to zero (e- p in, e- p X out);
  6. the radiated photon is massless and has positive energy;
  7. header column 5 (beam polarisation) is -1, 0 or +1.

Tolerances are set by the file itself, not by taste: momenta are written as
%18.10e, so a component near 10 GeV round-trips to about 1e-9.  Summing five
of them gives a few times that, and ATOL = 1e-6
# Насколько остаток может отклоняться от «светоподобный, вдоль пучка».
# Печать в LUND идёт с шестью знаками, отсюда и порядок.
PREVERTEX_TOL = 1e-5 GeV sits three decades above
the noise and far below anything physical.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exclurad_py.generator.breit_wigner import (            # noqa: E402
    DELTA_WIDTH, M_DAUGHTER, WIDTH)

M_P = 0.93827208816
ATOL = 1e-6
# Насколько остаток может отклоняться от «светоподобный, вдоль пучка».
# Печать в LUND идёт с шестью знаками, отсюда и порядок.
PREVERTEX_TOL = 1e-5

PDG_MASS = {11: 0.51099895e-3, -11: 0.51099895e-3, 22: 0.0,
            111: 0.1349768, 221: 0.547862, 113: 0.77526, 333: 1.019461,
            443: 3.096900, 211: 0.13957039, -211: 0.13957039,
            321: 0.493677, -321: 0.493677, 2212: M_P, 2112: 0.93956542}
CHARGE = {11: -1, -11: +1, 22: 0, 111: 0, 221: 0, 113: 0, 333: 0, 443: 0,
          211: +1, -211: -1, 321: +1, -321: -1, 2212: +1, 2112: 0}
CHANNEL_PID = {"pi0": 111, "eta": 221, "rho": 113, "phi": 333, "jpsi": 443}

# phi is WRITTEN as 1.019412: the diffrad_vm transcription value
# (models/vector.py:290), a deliberate 49 keV convention rather than an
# error.  read_events already tolerates it; do the same here and say so.
MASS_TOL = {333: 6e-5}

# Каналы, у которых масса РАЗЫГРЫВАЕТСЯ.  Сравнивать её с полюсом нельзя: у
# phi так падали 1962 события из 2000, и падали правильно с точки зрения
# проверки и неправильно по сути -- масса и не должна быть полюсной.
# Согласие записанной массы с четырёх-вектором проверяется отдельно
# (массовая оболочка) и остаётся в силе; здесь проверяется, что масса лежит
# в том окне, из которого её вообще могли разыграть.  Границы берутся из
# таблицы генератора, а не переписываются сюда.
BW_PID = {113: "rho", 333: "phi", 223: "omega"}


def check_file(path, ch, nmax=0):
    """Counts of each failure mode in one file, plus the worst residuals."""
    mpid = CHANNEL_PID[ch]
    bad = dict(events=0, count=0, momentum=0, shell=0, pdg_mass=0,
               decay=0, charge=0, photon=0, polarisation=0, malformed=0)
    worst = dict(momentum=0.0, shell=0.0, decay=0.0, prevertex=0.0)
    n_rad_photon = 0
    n_prevertex = 0          # событий с до-вершинной потерей
    max_prevertex = 0.0      # самая большая потеря, ГэВ
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if line.startswith("#") or not line.strip():
                continue
            h = line.split()
            if len(h) != 10:
                bad["malformed"] += 1
                continue
            try:
                npart = int(h[0])
                ebeam = float(h[6])
                pol = float(h[4])
            except ValueError:
                bad["malformed"] += 1
                continue
            bad["events"] += 1

            rows = []
            for _ in range(npart):
                q = fh.readline().split()
                if len(q) != 14:
                    break
                rows.append(q)
            if len(rows) != npart:
                bad["count"] += 1
                continue

            if pol not in (-1.0, 0.0, 1.0):
                bad["polarisation"] += 1

            tot = np.zeros(4)                       # (px, py, pz, E)
            charge = 0
            doc = {}                                # index -> (p4, daughter)
            daughters = {}                          # parent index -> sum p4
            saw_photon = False
            shell_bad = pdg_bad = False
            for q in rows:
                idx, status, pid = int(q[0]), int(q[2]), int(q[3])
                parent = int(q[4])
                p4 = np.array([float(q[6]), float(q[7]), float(q[8]),
                               float(q[9])])
                m_written = float(q[10])

                # Test m^2, not m.  m^2 = E^2 - p^2 is a difference of two
                # nearly equal numbers, so its absolute error scales with
                # E^2 and NOT with the mass; converting to a tolerance on m
                # by dividing by 2m diverges exactly where the cancellation
                # is worst.  Two attempts at this on 2026-08-21 failed 10083
                # and then 19421 events out of 28000 on pure round-off, and
                # the offender both times was the radiated photon: written
                # mass 0, reconstructed 2.8e-5 at E = 1.4 GeV, which is
                # sqrt(4e-10) * E and nothing else.
                #
                # Measured residuals of |m^2 - m_written^2| / E^2 in a
                # production file: proton 2.3e-9, photon 4e-10, electron
                # 2.9e-11.  K = 1e-7 sits a factor 40 above the worst of
                # those and orders of magnitude below any real error -- a
                # misassigned mass moves m^2 by its own square.
                m2 = p4[3] ** 2 - p4[:3] @ p4[:3]
                d_shell = abs(m2 - m_written ** 2)
                tol_shell = 1e-7 * max(p4[3] ** 2, 1.0)
                worst["shell"] = max(worst["shell"], d_shell / tol_shell)
                if d_shell > tol_shell:
                    shell_bad = True
                bwc = BW_PID.get(pid)
                if bwc is not None and WIDTH.get(bwc, 0.0) >= DELTA_WIDTH:
                    lo = 2.0 * M_DAUGHTER[bwc] - 1e-6
                    hi = PDG_MASS[pid] + 5.0 * WIDTH[bwc] + 1e-6
                    if not (lo <= m_written <= hi):
                        pdg_bad = True
                elif pid in PDG_MASS:
                    if abs(m_written - PDG_MASS[pid]) > MASS_TOL.get(pid, 1e-6):
                        pdg_bad = True

                if status == 1:
                    tot += p4
                    charge += CHARGE.get(pid, 0)
                    if pid == 22 and parent == 0:
                        saw_photon = True
                elif status == 0:
                    doc[idx] = p4
                if parent:
                    daughters[parent] = daughters.get(parent, np.zeros(4)) + p4

            if shell_bad:
                bad["shell"] += 1
            if pdg_bad:
                bad["pdg_mass"] += 1
            if saw_photon:
                n_rad_photon += 1

            # Четырёхимпульс.  С включённым до-вершинным радиатором (itarg=1)
            # событие построено при ПОНИЖЕННОЙ энергии пучка, а сам излучённый
            # фотон в файл не пишется -- он поглощён в эту пониженную энергию.
            # Сравнение с номинальным пучком тогда даёт невязку, равную потере,
            # и до 24.08 эта проверка молча падала на КАЖДОЙ боевой выборке:
            # 5.8 % событий пи0, худшая невязка 8.46 ГэВ.  Докладывалась она
            # при этом как пройденная -- по прогону без радиатора.
            #
            # Правильная проверка не слабее прежней, а строже: остаток обязан
            # быть светоподобным вектором ВДОЛЬ ПУЧКА с ПОЛОЖИТЕЛЬНОЙ энергией
            # -- ровно то, чем может быть до-вершинный фотон и ничто иное.
            # Потерянная частица, перепутанный знак или незакрытый распад дадут
            # остаток непродольный, или не светоподобный, или отрицательный.
            # Измерено на боевом пи0: |px|,|py| <= 9e-11, |E-pz| <= 6.1e-8,
            # отрицательных нет.
            want = np.array([0.0, 0.0, ebeam, ebeam + M_P])
            dv = want - tot                      # (px, py, pz, E)
            d_mom = float(np.max(np.abs(dv)))
            if d_mom > ATOL:
                # остаток есть -- он обязан быть до-вершинным фотоном
                dE = float(dv[3])
                off = max(abs(float(dv[0])), abs(float(dv[1])),
                          abs(dE - float(dv[2])))
                if dE < 0.0 or off > PREVERTEX_TOL:
                    bad["momentum"] += 1
                    worst["momentum"] = max(worst["momentum"], d_mom)
                else:
                    n_prevertex += 1
                    max_prevertex = max(max_prevertex, dE)
                    worst["prevertex"] = max(worst["prevertex"], off)
            else:
                worst["momentum"] = max(worst["momentum"], d_mom)

            for idx, p4 in doc.items():
                s = daughters.get(idx)
                if s is None:
                    bad["decay"] += 1
                    continue
                d = float(np.max(np.abs(p4 - s)))
                worst["decay"] = max(worst["decay"], d)
                if d > ATOL:
                    bad["decay"] += 1

            if charge != 0:
                bad["charge"] += 1
            if nmax and bad["events"] >= nmax:
                break
    bad["prevertex"] = n_prevertex
    worst["prevertex_dE"] = max_prevertex
    return bad, worst, n_rad_photon


def main(ch, patterns, nmax=0):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.expanduser(p)))
    files = sorted(set(files))
    if not files:
        sys.exit("нет файлов")
    tot = None
    worst = dict(momentum=0.0, shell=0.0, decay=0.0, prevertex=0.0,
                 prevertex_dE=0.0)
    nph = 0
    for f in files:
        b, w, k = check_file(f, ch, nmax)
        tot = b if tot is None else {x: tot[x] + b[x] for x in tot}
        for x in worst:
            worst[x] = max(worst[x], w[x])
        nph += k
    n = tot["events"]
    print(f"[{ch}] файлов {len(files)}, событий {n}, "
          f"с радиационным фотоном {100*nph/max(n,1):.1f}%")
    print(f"  худшая невязка: импульс {worst['momentum']:.2e} ГэВ "
          f"(допуск {ATOL:.0e}), распад {worst['decay']:.2e}, "
          f"массовая оболочка {worst['shell']:.2f} от своего допуска")
    npv = tot.get("prevertex", 0)
    if npv:
        print(f"  до-вершинная потеря: {npv} событий ({100*npv/max(n,1):.2f} %), "
              f"максимум {worst['prevertex_dE']:.4f} ГэВ; отклонение остатка от "
              f"«светоподобный вдоль пучка» не больше {worst['prevertex']:.1e}")
    # prevertex здесь НЕТ намеренно: это не отказ, а измеренное свойство
    # выборки с включённым радиатором.
    order = ("malformed", "count", "momentum", "shell", "pdg_mass", "decay",
             "charge", "photon", "polarisation")
    names = dict(malformed="битые заголовки", count="счётчик частиц",
                 momentum="четырёхимпульс", shell="массовая оболочка",
                 pdg_mass="масса против PDG / окна Брейта-Вигнера", decay="замыкание распада",
                 charge="заряд", photon="радиационный фотон",
                 polarisation="поляризация в заголовке")
    fails = 0
    for k in order:
        if tot[k]:
            print(f"  ОШИБКА {names[k]:<26} {tot[k]} из {n}")
            fails += tot[k]
    print(f"[{ch}] {'ПРОШЛО' if fails == 0 else 'ПРОВАЛ'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    sys.exit(main(a[0], a[1:] or ["*.lund"],
                  int(os.environ.get("NMAX", "0"))))
