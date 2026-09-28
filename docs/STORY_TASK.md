# Inbending and outbending: generated against reconstructed, and the cuts

Task for Story, 2026-09-28. Everything referred to here is on `production`.

## Why regenerate

Two things changed under you since 2026-09-26, and both touch what you have.

**The photon selection was wrong for radiative samples.** A radiative LUND
carries THREE photons — the radiated one is written before the pi0 pair — and
`hipo_to_ntuple.groovy` took the first two in order. So "g1" was the radiated
photon and one decay photon was dropped. Measured on our rad million: 56.7 %
of events have three photons and the pair was wrong in 26.8 % of the sample,
which read as a g1 efficiency of 39 % against born's 70 %. Fixed in `2a1c904`
by taking the two photons whose `MC::Lund` parent is the pi0 row. **Any rad
ntuple made before that commit is wrong in its photon half.** Born is
unaffected — it never has three photons.

**The stage-2 sampling bound was a lottery.** It was rebuilt from 300 probes
inside every job, on a 6144-cell grid, so each job drew its own bound anywhere
between 2.01 and 8.67 against a true maximum of 8.67. Fixed in `30a10c7` and
`da664cc`. The effect on a finished sample is small but real and it sits at
high W: our production million is 14 % low in its last W bin (3.95–4.00,
5.2 sigma) and unaffected below W = 3.9.

`git pull` gets both, plus the deuteron work merged from the `fermi` branch.

## Rebuild the tables first — the box changed

On 2026-09-28 the `t` box went to **−2.50 … −t_min** (`d581c5c`). The eta
table, the majorant and the stage-2 bound are all properties of the box, so
**everything in `tables/` from before that date is for a different box.**

There is one script now, and it did not exist when you started:

```bash
export PROD_DIR=/volatile/clas12/vpk/prod_pi0_rga_fa18_inb_<your date>
mkdir -p $PROD_DIR/cards
cp ~/exclurad_py/cards/born_pi0_rga.txt $PROD_DIR/cards/rga_born.txt
cp ~/exclurad_py/cards/rad_pi0_rga.txt  $PROD_DIR/cards/rga_rad.txt
bash ~/exclurad_py/production/farm/build_tables.sh
```

Minutes on an interactive ifarm node, not hours. It writes into
`$PROD_DIR/tables/`:

| file | what | needed by |
|---|---|---|
| `etatable.npz` | eta on a (5,5,5,8) grid | rad |
| `majorant_rad.npz` | majorant **+ stratified stage-2 bound** | rad |
| `majorant_born.npz` | majorant only — born has no stage 2 | born |
| `build.log` | what was built, and the bound's spread | you |

Then submit `gen_array.sbatch` with the same `PROD_DIR` exported. It reads
`$R/tables/majorant_$MODE.npz`, so the two modes no longer collide.

**Read two numbers out of `build.log` before you submit 200 jobs.** On the
new box, with the defaults (`NW=32`, `PER=400`):

```
bound grid (4, 4, 4, 6) = 384 cells x 400 probes
B: min 0.85  median 1.70  ...  max 12.72
cells with no usable probe: <should be a handful, not 150>
```

A **median far above ~1.7** or **many unprobed cells** means the probes did
not reach; raise `PER`. A grid of `(1,1,1,2)` means you are back to a global
bound wearing a grid's clothes — that is the shape the old code produced, and
it is what the fixed script exists to avoid. Note the maximum is **12.72** on
the −2.5 box against 8.67 on the old −2.0 one: widening `t` added a harder
corner, which is exactly why the tables cannot be carried over.

### Why you cannot skip this, and why nothing will warn you

The two caches behave differently on a stale file:

- **`etatable.npz` raises.** `RCTable.load` compares a key that carries the
  model, version, beam energy, cut, shape, box, `t_nucl`, and the phi
  convention. A mismatch is a `ValueError` and the job dies. Good.
- **`majorant*.npz` does not.** On a key mismatch it falls through and
  rebuilds inside the job — silently, 200 times. It costs 25 s per born job
  and tells you nothing.

So the rule is by hand, not by error message. **Rebuild when any of these
changes:** model, `ebeam`, `W2`/`Q2`/`t` box, `vcut`, `ihel`, `ptarg`,
`t_nucl`, `fermi`. **Do not rebuild for:** `nev`, `seed`, `outroot`,
`theta_e`, `y`, `xB`, `iacc`, `vz`, `itarg` — those are cuts and bookkeeping,
and the tables do not depend on them.

## What to produce

Four samples: **born and rad, inbending and outbending.** The generator card is
the same for both field directions — the torus sign lives in the GEMC step, not
in the card — so it is one generation per mode and two simulations:

```
production/farm/gen_array.sbatch     stage A, generation
production/farm/sim_array.sbatch     stage B, GEMC + 50 nA background + recon
production/farm/post_array.sbatch    catch-up if stage B dies after GEMC
```

Outbending needs `-SCALE_FIELD="binary_torus, +1.00"` and the matching gcard
and yaml from `clas12-config`; everything else is identical.

## What to measure

```
python3 production/gemc_check/gen_vs_rec.py <ntuple.f32> <tag> <outdir> [--log]
```

Sixteen panels: `|p|` and `theta` for e', p and both photons separately, the
pi0 built from the pair, and Q2, W, -t, MM2(e'p). Generated and reconstructed
overlaid, in counts — the gap between them IS the efficiency, which is why they
are not normalised. Run it with and without `--log`.

**Reconstructed means found AND identified**, not merely matched. On this
sample the difference is a factor five, and the reason is worth knowing before
you read any efficiency: the event start time comes from the trigger electron,
and when there is none the reconstruction takes the fastest particle for a pion
and sets the start time from it. Time-of-flight PID then collapses — the proton
is identified in 5.6 % of events without an electron against 68.3 % with one,
while its **track** is found in 78 % and 85 %. So condition everything on a
reconstructed electron, and drop the events without one.

That conditioning is for ELECTROPRODUCTION only. Photoproduction has no
scattered electron and needs a different anchor.

## What we already know for inbending — do not redo it

Measured on the born million, inbending, torus −1.00. A negative track matched
to the generated electron exists in:

| theta | track | identified | identified given a track |
|---|---:|---:|---:|
| 5–6° | 0.3 % | 0.0 % | 3 % |
| 6–7° | 3.5 % | 0.3 % | 8 % |
| 9–10° | 44 % | 30 % | 67 % |
| 15–20° | 74 % | 68 % | 91 % |

Below 6 degrees **there is no track** — this is the drift chambers, not PID.
Identification behaves normally once a track exists and reaches 91 %.

A cut at `theta_e > 6` drops 22 % of the generation and loses **zero**
reconstructed events (38530 before, 38530 after); 6.5 is also free and drops
38 %. The card field is `theta_e` (a degree window, field 22). It is currently
set to **5.0 — open — on purpose**, so the cuts can be chosen from data rather
than assumed; decide it at the 100 M stage.

**A cut on Q2 does not substitute for it.** `theta_e > 6` keeps Q2 down to 0.80
and 39 % of the surviving events sit below 1.2, because at fixed theta a low E'
gives a low Q2. Cutting Q2 > 1.2 would throw that physics away for the same
gain.

**None of this is measured for outbending**, and it will differ: outbending
bends the electron away from the beamline and should accept smaller angles.
That measurement is the point of the task.

## Practical, from `docs/gemc_howto.pdf`

- `--time=10:00:00` for the full chain, `04:00:00` for the catch-up. gemc alone
  varies by a factor two across nodes, 3.3 to 6.8 hours for 5000 events.
- `--mem=4G`. Nothing in this chain has exceeded 2.7 GB over 1232 measured
  steps; the note used to say 6G and that came from RG-K.
- `/tmp` is `noexec` on the compute nodes, so sqlite-jdbc cannot load its
  native library and both `bg-merger` and `recon-util` die. The scripts set
  `JDK_JAVA_OPTIONS` to put the JVM temp directory on `/volatile`. It does
  **not** reproduce on ifarm2402 — test with a short sbatch job, not on the
  login node.
