# Story: from a run card to a simulated CLAS12 sample

Everything you need is in this repository (`exclurad_py`, branch `production`). The commands
below were all run in September 2026 and worked as written. Read `README.md` for what the
generator is; this page is only how to use it for a GEMC sample.

## 0. Setup, once

You need: a JLab account with `ifarm`, `/cache`, `/volatile` and slurm; python3 (3.9 or newer)
with numpy on the machine where you generate (any laptop is fine); a directory of your own
under `/volatile/clas12/users/<you>/` on ifarm.

```bash
git clone -b production git@github.com:vkubarovsky/exclurad_py.git
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
| `cards/{rad,born}_{pi0,eta}_rga_sp18.txt` | RG-A spring 2018 | 10.594 | 0.8 - 10 | 0 - 0.90 | -4.44 to +0.56 cm (target at -1.94 cm) |
| `cards/{rad,born}_eta_{rga,rgk6535,rgk7546}.txt` | the same six for eta (model `eta.amp2609`, otherwise identical) | | | | |

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
eight cores in about 15 minutes. Events are unweighted.

**Radiative mode is different.** It must be run with the production environment, otherwise
the beam-spin asymmetry of the sample is wrong (the generator prints a warning about
`EXACT_ACCEPT`):

```bash
export TABLE_SERIAL=1 ETA_PROPOSAL_CAP=5 EXACT_ACCEPT=1 EXPO_TAIL=1 MAJ_SAFETY=4.0
PYTHONPATH=. python3 -u -m exclurad_py.run cards/rad_pi0_rga.txt --jobs 8
```

(this is what `production/run_prod_v1.sh` sets).  It costs about 0.8 s per event per core on a
Mac and about 2 s on a JLab farm core (the hard-photon quadrature, retuned on 2026-09-24), against
2.2 s per event for GEMC plus reconstruction; so a radiative million events is 550 farm
core-hours and belongs on the farm, see `production/farm/gen_array.sbatch`.  Always read the
`health:` block at the end of a run: it reports the unweighting efficiency and any majorant
violation.

The LUND header is `nPart A Z polTarget polBeam beamPID beamE nucleonPID processID weight`.
Column 5 carries the beam helicity per event when the card has `ihel = 0`. The LUND files
contain event lines only, no comment lines, so they go to GEMC or to the OSG portal as they
are. The provenance of a run (code version, model, box, seed) and its cross section
(`sigma_nb`) are in the run's `inputs/config.json` next to the `lund/` directory; keep that
directory with the files.

Start with 2000 events (`nev` line) and look at the LUND before doing a million.

## 3. Prepare the LUND for GEMC

Nothing to edit in the files. For OSG, put the LUND files of one submission into one flat
directory and nothing else in it; that directory is what the portal picks up. For the farm,
split the sample into chunks of 2500 events, which keeps one job under three hours:

```bash
python3 production/farm/split_lund.py all.lund 2500 chunks/
```

(`all.lund` being the concatenation of the run's files; the splitter never cuts an event in two.)

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
sample. Not tested by us yet with exclurad_py files; the first submission will tell.

**B. JLab farm with slurm.**  Two arrays, symmetric for Born and rad, in `production/farm/`:

1. `gen_array.sbatch <born|rad>`: task $i$ generates one chunk (seed = base + i) into the
   directory that GEMC reads and that OSG would also be handed, and keeps the chunk's
   `config.json` (the cross section `sigma_nb` lives there) and card next to it.
2. `sim_array.sbatch <born|rad>`: gemc 5.13 + coatjava 10.0.7 (the pass-2 pair of setup.json for
   `rga_fall2018`), then `bg-merger` with the 50 nA background file and the eleven detectors the
   official OSG production merges, then `recon-util`, the DST and the ntuple.  Both Born and rad
   get the background, so that their acceptance ratio is not biased.

Submission lines are in the header of each script.  The older one-stage `gemc_test.sbatch`
(200-event check) and `gemc_array.sbatch` (no background) remain for tests; the 200-event test
first is still the rule: check that `gemc.hipo` is a few MB, not 5 kB (5 kB means gemc read
zero events, which happens if a LUND file is malformed or contains comment lines).

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
