import sys, statistics as st, collections
base, spd, host = {}, {}, {}
guards = 0; bytes_ok = bytes_bad = 0
for line in sys.stdin:
    f = line.split()
    if not f: continue
    if f[0] == "GUARD": guards += 1
    elif f[0] == "RESULT" and len(f) >= 5 and f[2] not in ("NA",):
        try: v = float(f[3])
        except ValueError: continue
        (base if f[2] == "base" else spd)[f[1]] = v
        host[f[1]] = f[-1]
    elif f[0] == "BYTES":
        if f[2] == "identical": bytes_ok += 1
        else: bytes_bad += 1
pairs = sorted(set(base) & set(spd))
ratios = [base[k]/spd[k] for k in pairs]
def q(x, p): 
    x = sorted(x); i = (len(x)-1)*p; lo=int(i); hi=min(lo+1,len(x)-1)
    return x[lo] + (x[hi]-x[lo])*(i-lo)
print(f"paired tasks            : {len(pairs)}   (guard failures: {guards})")
print(f"LUND byte-comparison    : {bytes_ok} identical, {bytes_bad} differ")
if not pairs: sys.exit()
b = [base[k] for k in pairs]; s = [spd[k] for k in pairs]
print()
print(f"{'':10s} {'median':>9s} {'min':>9s} {'max':>9s}   ms/event")
print(f"{'base':10s} {st.median(b):9.1f} {min(b):9.1f} {max(b):9.1f}")
print(f"{'speed':10s} {st.median(s):9.1f} {min(s):9.1f} {max(s):9.1f}")
print()
print(f"per-task RATIO: median {st.median(ratios):.3f}   "
      f"10-90% {q(ratios,0.1):.3f}-{q(ratios,0.9):.3f}   "
      f"min {min(ratios):.3f} max {max(ratios):.3f}")
print(f"distinct nodes: {len(set(host.values()))}")
print()
# DO NOT divide the campaign's 2586 ms/event by this ratio unless the
# "base" arm IS the code the campaign ran.  2586 is the PRE-speed number;
# a numba A/B whose base arm is the already-merged numpy work measures a
# ratio relative to THAT, and chaining it onto 2586 silently skips a
# factor.  Quoting 2586/2.324 = 1113 ms/event was exactly this mistake.
print(f"ratio is relative to the 'base' arm of THIS run -- chain it onto the")
print(f"baseline that arm represents, not onto an unrelated campaign number.")
