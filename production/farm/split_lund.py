"""Split a concatenated, comment-free LUND file into fixed-size chunks for a slurm array.
    python3 split_lund.py all.lund 2500 chunks/
GEMC exits silently with zero events if the LUND contains '#' lines: strip them first
(grep -v '^#' *.lund > all.lund)."""
import os, sys
src, per, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
lines = open(src).readlines()
n = chunk = 0; out = None; i = 0
while i < len(lines):
    f = lines[i].split()
    if len(f) == 10:
        if n % per == 0:
            if out: out.close()
            chunk += 1; out = open(os.path.join(outdir, "lund_%03d.dat" % chunk), "w")
        npart = int(f[0]); out.write("".join(lines[i:i + 1 + npart])); i += 1 + npart; n += 1
    else:
        i += 1
out.close(); print("events", n, "chunks", chunk)
