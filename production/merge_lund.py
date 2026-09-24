"""Merge exclurad_py LUND files into fixed-size chunks with an honest header.

    merge(sources, out_dir, prefix, chunk=5000) -> list of manifest rows

`sources` is an ORDERED list of (path, host).  Events are streamed in that
order, copied byte for byte, and cut into files of exactly `chunk` events;
the final partial chunk is discarded (its size is reported).  Only the
comment header is rewritten:

  * lines that describe the sample (model, box, spins, E, mode, validity,
    topology) must agree across every source -- a mismatch is an error, not
    a warning, because the whole point of a merged sample is that it is one
    population;
  * lines that were per-source (run.runcard, run.seed, run.n_events, the
    sf_usage counters, git_commit/version_tag/python) are dropped from the
    common block and summarised in merge.* lines instead;
  * merge.sources lists, for this chunk, every source file with the event
    range taken from it, so any event traces back to its file, seed and host.

Verification is built in: a sha256 over the event text is kept for the
source stream (minus the discarded tail) and for the written files; they
must be equal.  Nothing here reads the physics.
"""
import hashlib
import os
import re
import time

PER_FILE = ("run.runcard", "run.seed", "run.n_events", "git_commit", "version_tag",
            "python", "sf_usage.calls", "sf_usage.points", "sf_usage.out_of_range",
            "sf_usage.out_of_range_frac", "git_branch", "run.eta_interp",
            # sf.t_nucl in the header is RUNTIME STATE, not the card value:
            # Generator._set_event_t rewrites it per event and stamp() records
            # whatever the last event left there.  Per file, meaningless.
            "sf.t_nucl")
IDENTITY = ("sf.model", "sf.channel", "sf.version", "run.mode", "run.box.Q2",
            "run.box.W2", "run.box.t", "run.vcut", "run.mx2_cut", "run.Ebeam",
            "run.ihel", "run.ptarg", "sf.target", "sf.topology")


def _header(path):
    """Return (dict of '# key: value' lines in order, list of raw non-kv lines)."""
    kv, raw = {}, []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([\w.]+):\s*(.*)$", line.rstrip("\n"))
            if m:
                kv[m.group(1)] = m.group(2)
            else:
                raw.append(line.rstrip("\n"))
    return kv, raw


def _events(path):
    """Yield each event as one string (header line + particle lines)."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            t = line.split()
            if len(t) != 10:
                raise ValueError(f"{path}: expected an event header, got: {line[:60]!r}")
            n = int(t[0])
            buf = [line]
            for _ in range(n):
                buf.append(next(fh))
            yield "".join(buf)


def merge(sources, out_dir, prefix, chunk=5000, verbose=True, extra=(), start=0):
    """`start`: number the first output file start+1, to append to an
    existing sequence rather than overwrite it.

    `extra`: ('key', 'value') pairs added to the common header, e.g. the spins
    for samples whose headers predate the spin stamp (taken from the card)."""
    os.makedirs(out_dir, exist_ok=True)
    # --- common header from the first source, identity checked on all
    kv0, raw0 = _header(sources[0][0])
    seen = {k: set() for k in PER_FILE}
    for p, _ in sources:
        kv, _ = _header(p)
        for k in IDENTITY:
            if kv.get(k) != kv0.get(k):
                raise ValueError(f"identity mismatch on {k}: {p!r} has {kv.get(k)!r}, "
                                 f"{sources[0][0]!r} has {kv0.get(k)!r}")
        for k in PER_FILE:
            if k in kv:
                seen[k].add(kv[k])
    common = [(k, v) for k, v in kv0.items() if k not in PER_FILE]
    common += [(k, v) for k, v in extra if k not in kv0]
    hosts = sorted({h for _, h in sources})
    root = os.path.commonpath([s for s, _ in sources])
    rel = lambda p: os.path.relpath(p, start=root)

    rows = []
    h_src, h_out = hashlib.sha256(), hashlib.sha256()
    n_in = n_out = 0
    buf, buf_src = [], []           # events of the chunk being built; (path, host, first, last)
    cur = None                      # (path, host, first_idx_in_file)
    ifile = start

    def flush():
        nonlocal buf, buf_src, ifile, n_out
        ifile += 1
        name = f"{prefix}_{ifile:06d}.lund"
        path = os.path.join(out_dir, name)
        hdr = [f"# {k}: {v}" for k, v in common]
        hdr += [f"# run.n_events: {chunk}",
                f"# merge.chunk: {ifile}",
                f"# merge.root: {root}",
                f"# merge.sources: " + " ; ".join(f"{rel(p)}[{h}]:{a}-{b}" for p, h, a, b in buf_src),
                f"# merge.hosts: " + " ; ".join(hosts),
                f"# merge.git_commit: " + " ; ".join(sorted(seen["git_commit"])),
                f"# merge.version_tag: " + " ; ".join(sorted(seen["version_tag"])),
                f"# merge.python: " + " ; ".join(sorted(seen["python"])),
                f"# merge.date: {time.strftime('%Y-%m-%d')}",
                "# merge.note: events copied byte for byte from the sources listed above; "
                "the source seeds are in the manifest"]
        hdr += raw0
        body = "".join(buf)
        with open(path, "w") as fh:
            fh.write("\n".join(hdr) + "\n")
            fh.write(body)
        # verify what is ON DISK, not what was in memory: re-read, drop the
        # header, hash the event text
        data = open(path, "rb").read()
        md5 = hashlib.md5(data).hexdigest()
        text = data.decode()
        i = 0
        while text.startswith("#", i):
            i = text.index("\n", i) + 1
        h_out.update(text[i:].encode())
        rows.append(dict(file=name, events=chunk,
                         sources=" ; ".join(f"{rel(p)}:{a}-{b}" for p, _, a, b in buf_src),
                         hosts=" ; ".join(sorted({h for _, h, _, _ in buf_src})),
                         seeds=" ; ".join(str(_header(p)[0].get("run.seed", "?")) for p, _, _, _ in buf_src),
                         md5=md5))
        n_out += chunk
        buf, buf_src = [], []

    for p, host in sources:
        first = None
        idx = 0
        for ev in _events(p):
            idx += 1
            if first is None:
                first = idx
            buf.append(ev)
            n_in += 1
            if len(buf) == chunk:
                buf_src.append((p, host, first, idx))
                # hash the source stream only for events that are kept
                for e in buf:
                    h_src.update(e.encode())
                flush()
                first = None
        if first is not None:
            buf_src.append((p, host, first, idx))
    tail = len(buf)
    if verbose:
        print(f"  {prefix}: {n_in} events in from {len(sources)} files -> "
              f"{ifile} files x {chunk} = {n_out}; discarded tail {tail}")
    if h_src.hexdigest() != h_out.hexdigest():
        raise RuntimeError(f"{prefix}: event stream hash mismatch after writing -- STOP")
    return rows, dict(n_in=n_in, n_out=n_out, files=ifile, tail=tail,
                      sha256=h_out.hexdigest(), hosts=hosts,
                      commits=sorted(seen["git_commit"]), tags=sorted(seen["version_tag"]))
