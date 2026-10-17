#!/usr/bin/env python3
"""Certify a small radiative generator pilot before campaign submission.

The audit checks both sampling correctness and the cached-majorant
normalisation.  A passing pilot has the requested number of complete chunks,
zero ordinary majorant overweights, a stage-2 violation fraction within the
generator's documented production tolerance, and a quoted ``sigma_nb`` equal
to the accept/reject estimator built from the same ``M * B`` density as the
cached cell CDF.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


AXES = ("Q2", "W2", "t", "phi")


def bound_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "M", "cdf", "shape", "B", "b_shape", "B_global",
            "diag_margin", "diag_probes_per_cell",
            "diag_workers", "diag_max_cells", "diag_min_usable",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"bound cache lacks required fields: {missing}")
        M = np.asarray(data["M"], dtype=float)
        B = np.asarray(data["B"], dtype=float)
        bshape = tuple(int(x) for x in np.asarray(data["b_shape"]).ravel())
        shape = tuple(int(x) for x in np.asarray(data["shape"]).ravel())
        cdf = np.asarray(data["cdf"], dtype=float)
        edges = {axis: np.asarray(data["e_" + axis], dtype=float)
                 for axis in AXES}
        metadata = {
            "margin": float(data["diag_margin"]),
            "probes_per_cell": int(data["diag_probes_per_cell"]),
            "workers": int(data["diag_workers"]),
            "max_cells": int(data["diag_max_cells"]),
            "min_usable": int(data["diag_min_usable"]),
            "B_global": float(data["B_global"]),
        }

    if M.shape != shape or B.shape != bshape:
        raise ValueError(
            f"cache shape mismatch: M {M.shape}/{shape}, B {B.shape}/{bshape}")
    idx = np.ix_(*[
        np.minimum((np.arange(shape[a]) * bshape[a]) // shape[a],
                   bshape[a] - 1)
        for a in range(4)
    ])
    B_full = B[idx]
    Mtot = M * B_full
    volume = 1.0
    for axis, name in enumerate(AXES):
        width = np.abs(np.diff(edges[name]))
        volume = volume * width.reshape(
            [-1 if position == axis else 1 for position in range(4)])
    mass = (Mtot * volume).ravel()
    total = float(mass.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("cached M*B integral is not finite and positive")
    expected_cdf = np.cumsum(mass) / total
    if cdf.shape != expected_cdf.shape or not np.allclose(
            cdf, expected_cdf, rtol=2e-14, atol=2e-15):
        raise ValueError("cached cdf was not built from the stored M*B density")
    metadata["normalization_integral_nb"] = total
    metadata["proposal_weighted_B"] = (
        float((Mtot * volume).sum()) / float((M * volume).sum()))
    return total, metadata


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def audit(root, expected_runs, expected_events, expected_revision=None,
          max_b_viol_fraction=1.0e-3):
    table = root / "tables" / "majorant_rad.npz"
    configs = sorted((root / "rad" / "prov").glob("rad_*.config.json"))
    lunds = sorted((root / "rad" / "lund_osg").glob("*.lund"))
    failures = []
    runs = []

    if len(configs) != expected_runs:
        failures.append(
            f"expected {expected_runs} config records, found {len(configs)}")
    if len(lunds) != expected_runs:
        failures.append(f"expected {expected_runs} LUND files, found {len(lunds)}")

    try:
        normalization, bound = bound_arrays(table)
    except Exception as exc:
        failures.append(f"bound cache: {exc}")
        normalization, bound = float("nan"), {}

    revisions = set()
    for config in configs:
        try:
            record = json.loads(config.read_text())
            report = record["report"]
            stamp = record["stamp"]
            card = record["card"]
        except Exception as exc:
            failures.append(f"{config.name}: cannot read record: {exc}")
            continue

        revision = str(stamp.get("git_commit", ""))
        revisions.add(revision)
        trials = int(report.get("trials", 0))
        accepted = int(report.get("accepted", 0))
        accept2 = int(report.get("accept2", 0))
        hard = int(report.get("hard", 0))
        soft = int(report.get("soft", 0))
        b_viol = int(report.get("b_viol", -1))
        overweight = int(report.get("overweight", -1))
        sigma = float(report.get("sigma_nb", float("nan")))
        quoted_events = int(card.get("nev", 0))
        expected_sigma = (accept2 / trials) * normalization if trials else float("nan")

        if quoted_events != expected_events:
            failures.append(
                f"{config.name}: card nev={quoted_events}, expected {expected_events}")
        if hard + soft != expected_events or accept2 != expected_events:
            failures.append(
                f"{config.name}: incomplete events hard+soft={hard + soft}, "
                f"accept2={accept2}, expected={expected_events}")
        b_viol_fraction = b_viol / max(accepted, 1)
        if b_viol < 0 or b_viol_fraction > max_b_viol_fraction:
            failures.append(
                f"{config.name}: stage-2 bound violations={b_viol}/"
                f"{accepted} accepts ({b_viol_fraction:.6%}), limit="
                f"{max_b_viol_fraction:.6%}, max "
                f"x{float(report.get('max_b_viol', 0.0)):.6g}")
        if overweight != 0:
            failures.append(
                f"{config.name}: ordinary majorant overweights={overweight}")
        if not math.isfinite(sigma) or sigma <= 0.0:
            failures.append(f"{config.name}: invalid sigma_nb={sigma}")
        elif not math.isclose(sigma, expected_sigma, rel_tol=2e-12, abs_tol=1e-14):
            failures.append(
                f"{config.name}: sigma_nb={sigma:.12g}, cached-MB estimator="
                f"{expected_sigma:.12g}")
        health = list(record.get("health") or [])
        unexpected_health = [
            message for message in health
            if not message.startswith("note: stage-2 bound exceeded")
        ]
        if unexpected_health:
            failures.append(f"{config.name}: health={unexpected_health}")

        runs.append({
            "config": config.name,
            "revision": revision,
            "events": hard + soft,
            "trials": trials,
            "accepted": accepted,
            "accept2": accept2,
            "b_viol": b_viol,
            "b_viol_fraction": b_viol_fraction,
            "max_b_viol": float(report.get("max_b_viol", 0.0)),
            "overweight": overweight,
            "sigma_nb": json_number(sigma),
            "expected_sigma_nb": json_number(expected_sigma),
            "generation_seconds": json_number(
                record.get("gen_s", float("nan"))),
        })

    if len(revisions) != 1:
        failures.append(f"pilot records contain revisions {sorted(revisions)}")
    if expected_revision is not None:
        bad = sorted(revision for revision in revisions
                     if not revision or not expected_revision.startswith(revision))
        if bad:
            failures.append(
                f"pilot revisions {bad} do not identify expected "
                f"revision {expected_revision}")

    return {
        "schema": "exclurad-pilot-audit-v1",
        "passed": not failures,
        "root": str(root),
        "expected_runs": expected_runs,
        "expected_events_per_run": expected_events,
        "expected_revision": expected_revision,
        "max_b_viol_fraction": max_b_viol_fraction,
        "generator_revisions": sorted(revisions),
        "bound": bound,
        "runs": runs,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-runs", type=int, default=2)
    parser.add_argument("--expected-events", type=int, default=10000)
    parser.add_argument("--expected-revision")
    parser.add_argument("--max-b-viol-fraction", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.expected_runs, args.expected_events,
                   args.expected_revision, args.max_b_viol_fraction)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
