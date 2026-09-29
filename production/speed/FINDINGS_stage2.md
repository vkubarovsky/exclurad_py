# Where the generator's time actually goes — measured, 2026-09-29

Branch `speed-table`.  This started as "refine the eta table, worth 1.6-1.8x".
Both levers I was handed turned out to be dead, and the measurement points
somewhere neither of them did.  Nothing here is adopted; it is all evidence.

## 1. The eta table is NOT the lever.  Measured.

Six configurations, production box, 200 events, self-built tables:

| config | ms/ev | stage-2 | RC/event | b_viol | build | interp rms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 563.7 | 20.2 % | 4.96 | 1 (x1.02) | 24 s | 1.332 |
| TABLE_SHAPE 7,7,7,11 | 557.0 | 22.3 % | 4.49 | 1 (x1.34) | 47 s | 0.944 |
| TABLE_SHAPE 10,10,10,16 | 545.8 | 22.0 % | 4.54 | 1 (x1.34) | 144 s | **0.668** |
| B_PROBES=3000 | 447.0 | 37.5 % | 2.67 | 2 (x1.33) | 24 s | 1.332 |
| B_SHAPE 6,6,6,8 | 486.2 | 42.6 % | 2.35 | 1 (x1.14) | 24 s | 1.332 |
| both | 419.5 | 47.4 % | 2.11 | 2 (x1.32) | 143 s | 0.667 |

Refining the table **halved** its interpolation error and bought **3 %**, for
144 s of build.  The rms(rel) 0.292 is a real defect in the interpolant and
it is close to irrelevant to speed.  The premise -- "a poor table spreads the
exact/proposal ratio r, so tightening r raises stage-2 acceptance" -- is
wrong, because B was never tracking r.

## 2. The bound lever is real but ALREADY BANKED.  My baseline was wrong.

The 26 % that B_PROBES appears to buy above is an artefact of my own
baseline, and the table is only honest as an internal comparison.

`B_PROBES` is read **only** when the bound is not cached
(`events.py:1140`): with a majorant npz that carries `B`/`B_global`/
`b_shape`, `b_from_cache` is True and the variable does nothing.  My scan
built its own majorants, which have no stored B, so every row above
re-derived the in-job 300-probe bound:

    my scan   "stage-2 bounds: 300 probes, global B = 4.91"
    farm/prod "stage-2 bound from cache: shape (4,4,4,6), B 0.92-10.11"

and the acceptances say the same: my baseline 20.2 %, production 45.0 %.
Production is already 2.2x better than the baseline I measured against,
because `production/farm/build_tables.sh` builds a **stratified** bound,
384 cells probed uniformly at 400 probes each.  My best row, 47.4 %, lands
where production already sits.  So this scan rediscovered, from a worse
starting point, the fix that went in on 2026-09-27.  The lever is spent.

Caught by the campaign session; verified here in the code and in the logs
before accepting it.

## 3. The cost model both of us were using is wrong

"The exact RC is the whole cost of the generator" is not true.  Cutting
exact evaluations 4.96 -> 2.11 per event (2.35x) produced 1.34x overall.

Profiling **only** `Generator.generate` -- setup excluded, which is the trap
in the naive profile, since `calibrate` alone is 15 s -- on 50 events with
the production tables and environment:

| | share of generate | calls |
|---|---:|---:|
| `TailSampler._build_grid` | **47 %** | 32 (368 ms each) |
| `sobs` (stage-2 exact RC) | 33 % | 292 |
| everything else | 20 % | |

`_build_grid` -> `grid.build` -> `_weighted_density` -> `_density_batch`
-> `rc_factor` -> `sigma_F`: 403 `sigma_F` calls in the generate phase, of
which ~12.6 per tail-sampler build.

**The largest single item in generation is building the photon tail
sampler, and neither the brief nor the campaign session mentions it.**  It
is where I would look next.

## 4. A fact worth reusing: the model call is ~116 us of FIXED overhead

`sf(W2, Q2, t)` on the production model, by batch size:

| N | total us | us/point |
|---:|---:|---:|
| 1 | 116.6 | 116.62 |
| 4 | 115.2 | 28.81 |
| 16 | 117.8 | 7.37 |
| 64 | 126.1 | 1.97 |
| 1024 | 227.9 | 0.22 |

One point and sixty-four points cost the same.  Anywhere the model is
called pointwise inside a loop, batching is nearly free speed -- this is the
same disease the merged `speed` work cured inside `sigma_F`, and the profile
above says it is still present elsewhere.  A single-point `born_xsec` costs
221.6 us, 57 % of it in the model layer.

## 5. What I did NOT do

* **The quadrature is untouched.** It is the only lever that changes the
  physics, so it is VPK's decision, and it must not share a measurement
  with anything else.  `quad_cost.py` here will produce the eta comparison
  over the box when he wants it; it adopts nothing.
* Nothing in this branch is proposed for merge.  `speed` (merged, 1.657x,
  byte-identical) is unaffected by any of it.
