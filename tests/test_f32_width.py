"""A .f32 reader must take the record width from the file, not from a constant.

The width went 50 -> 55 for pi0/eta (and 73 -> 80 for the vector channels) on
2026-10-22, when the vertices were appended to the END of the record.  Every
earlier offset stayed put, so a file read at the wrong width does not fail --
it returns numbers, shifted.  Four readers carried the width as a literal, two
at 50 and one at 55, against a writer that now emits 55.

The header closes this when it is there.  When it is not -- and no file on
either machine had one as of 2026-10-31 -- the weight field does: it is exactly
1.0 in every record, so a candidate width that does not put 1.0 in every
field 0 is not the width.  This test pins both paths and the refusal.
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "production", "analysis"))
import f32_format


def write_bare(d, name, nf, nrec, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nrec, nf)).astype("<f4")
    a[:, 0] = 1.0                      # the weight, as every writer sets it
    p = os.path.join(d, name)
    a.tofile(p)
    return p, a


def main():
    d = tempfile.mkdtemp()
    fail = 0

    # 1. legacy, 55 fields: inferred from the weight column
    p, a = write_bare(d, "new.f32", 55, 300)
    got, nf = f32_format.load(p, (50, 55))
    ok = nf == 55 and got.shape == (300, 55) and np.allclose(got, a)
    print(f"  legacy 55 fields -> {nf}, shape {got.shape}   {'ok' if ok else 'FAIL'}")
    fail += not ok

    # 2. legacy, 50 fields, a size that 55 does not divide
    p, a = write_bare(d, "old.f32", 50, 301)
    got, nf = f32_format.load(p, (50, 55))
    ok = nf == 50 and got.shape == (301, 50)
    print(f"  legacy 50 fields -> {nf}, shape {got.shape}   {'ok' if ok else 'FAIL'}")
    fail += not ok

    # 3. headered: the header is believed and its bytes skipped
    rng = np.random.default_rng(7)
    b = rng.normal(size=(120, 50)).astype("<f4"); b[:, 0] = 1.0
    p = os.path.join(d, "hdr.f32")
    with open(p, "wb") as fh:
        fh.write(f32_format.pack(50, 120, "pi0", "deadbee"))
        b.tofile(fh)
    got, nf = f32_format.load(p, (50, 55))
    ok = nf == 50 and got.shape == (120, 50) and np.allclose(got, b)
    print(f"  headered 50 fields -> {nf}, shape {got.shape}   {'ok' if ok else 'FAIL'}")
    fail += not ok

    # 4. THE ONE THAT MATTERS: a 50-field file whose record count also makes it
    #    divisible by 55 must be REFUSED, not silently read at 55.  550 records
    #    of 50 is 27500 floats = 500 records of 55.
    p, a = write_bare(d, "ambig.f32", 50, 550, seed=3)
    raw = np.fromfile(p, dtype="<f4")
    assert raw.size % 50 == 0 and raw.size % 55 == 0, "the trap needs both to divide"
    try:
        got, nf = f32_format.load(p, (50, 55))
    except ValueError as e:
        ok = "cannot be inferred" in str(e) or "leaves" in str(e)
        print(f"  ambiguous size -> refused   {'ok' if ok else 'FAIL: wrong error'}")
        fail += not ok
    else:
        # it is allowed to resolve, but ONLY to the true width
        ok = nf == 50
        print(f"  ambiguous size -> resolved to {nf}   "
              f"{'ok (weight column broke the tie)' if ok else 'FAIL: read at the wrong width'}")
        fail += not ok

    # 5. a file that is neither width is a fault, not a default
    p, _ = write_bare(d, "junk.f32", 37, 100)
    try:
        f32_format.load(p, (50, 55))
    except ValueError:
        print("  neither width -> refused   ok")
    else:
        print("  neither width -> ACCEPTED   FAIL")
        fail += 1

    print(f"FAILURES: {fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
