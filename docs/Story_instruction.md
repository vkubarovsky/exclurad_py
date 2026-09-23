# Story: from a run card to a simulated CLAS12 sample

Everything you need is in this repository (`exclurad_py`, branch `amp2026s`). The commands
below were all run in September 2026 and worked as written. Read `README.md` for what the
generator is; this page is only how to use it for a GEMC sample.

## 0. Setup, once

You need: a JLab account with `ifarm`, `/cache`, `/volatile` and slurm; python3 (3.9 or newer)
with numpy on the machine where you generate (any laptop is fine); a directory of your own
under `/volatile/clas12/users/<you>/` on ifarm.

```bash
git clone -b amp2026s git@github.com:vkubarovsky/exclurad_py.git
cd exclurad_py
python3 -c "import numpy; import exclurad_py; print('ok')"
```

## 1. Pick the run card

Cards are plain text, one value per line, the order is the format (`exclurad_py/runcard.py`).
The cards for the CLAS12 pi0 program, fixed with Valery on 2026-09-23:

| card | data set | beam | Q2 | y | vertex z |
|---|---|---|---|---|---|
| `cards/rad_pi0_rga.txt`, `cards/born_pi0_rga.txt` | RG-A fall 2018 and spring 2019, inbending and outbending | 10.6 | 0.8 - 10 | 0 - 0.90 | -5.5 to -0.5 cm |
| `cards/rad_pi0_rgk6535.txt`, `cards/born_pi0_rgk6535.txt` | RG-K fall 2018, 6.535 GeV | 6.535 | 0.5 - 10 | 0 - 0.85 | same |
| `cards/rad_pi0_rgk7546.txt`, `cards/born_pi0_rgk7546.txt` | RG-K fall 2018, 7.546 GeV | 7.546 | 0.5 - 10 | 0 - 0.85 | same |

Common to all: W2 3.24 - 16 (W 1.8 - 4.0 GeV), t -5 to -0.001 GeV2, no x_B cut, vcut 0 (the
missing-mass cut is an analysis cut), itarg 1 (pre-vertex external radiator). `rad_*` is the
full radiative sample, `born_*` the Born twin, identical except the `iborn` line. The reasons
for every number are in the repository `aao_norad_kin` (README, "Generator box from GEMC
acceptance"); you do not need them to run.

Before a run, edit two lines of your copy of the card: `seed` (unique per run; two runs with
the same seed are identical) and the output root (Valery's disk is in there). Check the card:

```bash
python3 -c "from exclurad_py.runcard import parse; print(parse('cards/born_pi0_rga.txt'))"
```

## 2. Generate

```bash
PYTHONPATH=. python3 -u -m exclurad_py.run cards/born_pi0_rga.txt --jobs 8
```

Output: `<outroot>/pi0/<date>_10.6_born_v1/lund/*.lund`, one file per job, plus `inputs/`
(the card and a JSON report). Born runs at 3 ms per event per core: one million events on
eight cores in about 15 minutes; `rad` is several times slower. Events are unweighted.

The LUND header is `nPart A Z polTarget polBeam beamPID beamE nucleonPID processID weight`.
Column 5 carries the beam helicity per event when the card has `ihel = 0`. Lines starting
with `#` are provenance (code version, model, cuts); keep the files as they are for the
record, strip the `#` lines only in the copy that goes to GEMC.

Start with 2000 events (`nev` line) and look at the LUND before doing a million.

## 3. Prepare the LUND for GEMC

GEMC exits silently, with zero events and no error, if a LUND file contains `#` lines.

```bash
grep -hv '^#' <outroot>/pi0/<date>_10.6_born_v1/lund/*.lund > all.lund
python3 production/farm/split_lund.py all.lund 2500 chunks/
```

2500 events per chunk keeps one farm job under three hours. Copy `chunks/` to your
directory on `/volatile`.

## 4. Simulate and reconstruct

Two routes. Configuration names are those of clas12-config on cvmfs (gemc 5.11,
coatjava 11.1.0):

| data set | gcard and yaml | torus | target centre |
|---|---|---|---|
| RG-A fall 2018 inbending | `rga_fall2018` | -1 | -3 cm |
| RG-A fall 2018 outbending | `rga_fall2018` | +1 | -3 cm |
| RG-A spring 2019 inbending | `rga_spring2019` | -1 | -3 cm |
| RG-K fall 2018, both energies | `rgk_fall2018_FTOn` | +1 | -3 cm |

**A. OSG, the official CLAS12 MC portal (gemc.jlab.org).** Upload the chunks as user LUND
files, choose the configuration and torus sign, background merging on. This produces a
sample equivalent to the official productions on `/cache` and is the route for a physics
sample. Not tested by us yet: if the portal complains about the files, the first thing to
check is that the `#` lines are gone.

**B. JLab farm with slurm**, no background merging; good for acceptance, resolution and
yield studies, and what all the numbers in the aao_norad_kin study were made with.

1. Test with one job of 200 events: copy `production/farm/gemc_test.sbatch`, replace
   `USER` in the paths, set gcard, yaml and torus sign for your data set, put one chunk as
   `lund.dat` next to it, `sbatch gemc_test.sbatch`. Expect about 8 minutes; check that
   `gemc.hipo` is a few MB, not 5 kB (5 kB means zero events).
2. Production: `production/farm/gemc_array.sbatch`, same edits plus the array size
   (`--array=1-80` for 200 000 events in chunks of 2500). Each job runs gemc (1.4 - 1.9 s per
   event), `recon-util` with the configuration's yaml (0.8 - 1.0 s per event), writes a
   filtered DST to `dst/` and a flat ntuple to `out/`.

Three rules, already in the scripts, learnt the hard way: `export TMPDIR=/tmp` before any
`module` command (the per-job TMPDIR breaks the module system silently); never put
`module load` inside a pipe; `module purge` before running the groovy tools (java 21 from
the coatjava module breaks groovy 4.0.3). Never run these on the ifarm login node: the
processes die when your last ssh session closes.

## 5. Look at the result

The flat ntuple (50 float32 per event: generated four-vectors, and for e', p and the two
photons the reconstructed particle that `MC::GenMatch` assigns, with pid, charge, status,
quality and momentum) is written by each farm job; for any other hipo file:

```bash
export PATH=/group/clas12/packages/groovy/4.0.3/bin:$PATH
/group/clas12/packages/coatjava/9.0.2/bin/run-groovy production/gemc_check/hipo_to_ntuple.groovy recon.hipo 10000000 out.f32
```

Concatenate the per-job files (`cat out/*.f32 > sample.f32`), copy to your machine and plot:

```bash
python3 production/gemc_check/plot_kinematics.py sample.f32 plots/mysample
python3 production/gemc_check/boundaries.py plots/mysample mysample=sample.f32@10.6
```

You get generated and reconstructed spectra (Q2, W, x_B, -t, momenta and angles of e', p,
pi0, M(gamma gamma), missing mass), efficiencies, a yield table and acceptance maps. Reference
numbers from the official RG-A aao_norad MC at 10.6 GeV: identified e'p in 5.6 % of generated
events inbending and 41 % outbending; e'p gamma gamma 0.9 % and 3.7 %. At 6.535 GeV outbending
(our own GEMC run): 39 % and 15 %.

## 6. Things that will confuse you

- `MC::Event.weight` in the hipo is not a weight. For exclurad_py samples it is 1.
- `MC::GenMatch` is the truth matching to use, one row per generated particle. Its index refers
  to `MC::Particle`, which does not contain the status-0 pi0 row of the LUND. `MC::RecMatch` is
  the reverse map with many ghost rows; do not pick "the best quality" from it.
- In outbending, 8.5 % of identified electrons come back with 50 - 90 % of their momentum
  missing (2 % at 6.5 GeV, 0.2 % inbending). They fill y > 0.8 and x_B near 0.
- Hadrons get pid 0 whenever no trigger electron was identified in the event.
- Photons are reconstructed only in the forward calorimeters (theta below 35 deg) and the
  forward tagger (2.5 - 4.5 deg); the pi0 needs both photons there.

## 7. Where to ask

Valery Kubarovsky. The study behind the cards and the tools: repository `aao_norad_kin`
(GitHub vkubarovsky), its README and `plots/`.
