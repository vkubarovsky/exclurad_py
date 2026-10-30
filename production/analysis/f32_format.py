#!/usr/bin/env python3
"""The .f32 ntuple header: what it is, and why it exists.

THE PROBLEM IT SOLVES.  A .f32 was a bare stream of little-endian float32 with
no header at all, so the record length had to be INFERRED from the file size.
On 2026-10-23 a second analysis session loaded 1600 superseded files -- named
exactly like the current ones, 73 floats where the current record is 80 -- with
a library pinned to the old length, and ran a complete analysis on the wrong
production without one error being raised.  Nothing in the bytes said which
vintage they were.  Inference cannot fail loudly; a header can.

THE GUARANTEE.  The header is 64 bytes = 16 float32, and 16 is divisible by
none of the record lengths in use (80, 73, 55, 50).  So a headered file's
total float count, 16 + nrec*nf, is NEVER a multiple of its own nf -- for any
number of records.  An old reader doing size/nf therefore cannot read a new
file as a valid file of the same channel: it gets a remainder and raises.
That is the property the size was chosen for, not a coincidence.

    vector  5000 records   1 600 000 bytes bare   1 600 064 with header
    pseudo  5000 records   1 100 000              1 100 064
    neither headered size divides 80, 73, 55 or 50

The weaker half, stated honestly and NOT hypothetically: 16 + nrec*80 can be
a multiple of 73 whenever nrec = 29 mod 73, and the first test written for
this code hit that case by accident -- a 2000-record vector file is
16 + 160000 = 160016 floats and 160016 = 73 * 2192 exactly, so an old reader
would have read it as 2192 legacy records of 73 fields without complaint.
2000 = 29 mod 73.

No header size avoids this.  gcd(80, 73) = 1, so for any fixed header length
there are record counts that make the total divide the other channel's length;
choosing a different size moves which counts are bad, it does not remove them.

Production files are 5000 records (5000 = 36 mod 73) and are safe, but the
guarantee is therefore asymmetric and worth stating precisely:

    ABSOLUTE   a headered file is never readable as a file of its OWN length
    NOT        it may, for particular record counts, be readable as a file of
               the other length in its channel's layout

The magic word closes the second half for any reader that checks it, which is
why the check is first and not a fallback.

THE LAYOUT, 64 bytes, little-endian throughout:

    offset  size  type      content
      0      4    char[4]   magic 'EXF3'
      4      4    uint32    format version, currently 1
      8      4    uint32    nfields -- the record length, STATED not inferred
     12      4    uint32    nrecords, or 0 when the writer did not know
     16      4    uint32    header length in bytes, 64 -- so a later version
                            may grow it and this reader still finds the data
     20     16    char[16]  channel, NUL-padded ('jpsi', 'pi0', ...)
     36     16    char[16]  git commit of the code that wrote it, NUL-padded
     52     12    --        reserved, zero

nfields is in the file, so nothing downstream has to guess.  header_bytes is
there so version 2 can be longer without breaking version 1 readers.

BACKWARD COMPATIBILITY.  A file without the magic is a legacy file and is read
by size inference exactly as before, so every ntuple written before this
change keeps working.  Forward, a new file in an old reader raises.  Both
halves are wanted: old data must stay readable, old code must not read new
data silently.
"""
import os
import struct

MAGIC = b"EXF3"
VERSION = 1
HEADER_BYTES = 64
_RECORD_LENGTHS = (80, 73, 55, 50)

assert all(HEADER_BYTES // 4 % n for n in _RECORD_LENGTHS), \
    "the header must not be a whole number of records of any length in use"


def pack(nfields, nrecords=0, channel="", commit=""):
    """The 64 bytes that go in front of the records."""
    h = bytearray(HEADER_BYTES)
    h[0:4] = MAGIC
    struct.pack_into("<III I", h, 4, VERSION, int(nfields), int(nrecords), HEADER_BYTES)
    h[20:36] = channel.encode()[:16].ljust(16, b"\0")
    h[36:52] = commit.encode()[:16].ljust(16, b"\0")
    return bytes(h)


def read(path):
    """The header of `path`, or None when the file is a legacy bare stream.

    Raises on a file that starts with the magic but is otherwise unusable --
    a truncated or future-version header is a fault, not a legacy file.
    """
    size = os.path.getsize(path)
    if size < 4:
        return None
    with open(path, "rb") as fh:
        head = fh.read(HEADER_BYTES)
    if head[:4] != MAGIC:
        return None
    if len(head) < 20:
        raise ValueError(f"{path}: starts with {MAGIC!r} but is only {size} bytes")
    version, nfields, nrecords, hbytes = struct.unpack_from("<IIII", head, 4)
    if version > VERSION:
        raise ValueError(
            f"{path}: header version {version}, this code understands {VERSION} -- "
            f"update the reader rather than guessing at the layout")
    if nfields not in _RECORD_LENGTHS:
        raise ValueError(f"{path}: header states {nfields} fields, which is not a "
                         f"record length this code knows {_RECORD_LENGTHS}")
    if hbytes < 20 or hbytes > size:
        raise ValueError(f"{path}: header claims {hbytes} bytes in a {size}-byte file")
    body = size - hbytes
    if body % (nfields * 4):
        raise ValueError(
            f"{path}: {body} bytes of records is not a whole number of "
            f"{nfields}-field records -- the file is truncated")
    got = body // (nfields * 4)
    if nrecords and nrecords != got:
        raise ValueError(f"{path}: header says {nrecords} records, the file holds {got}")
    return dict(version=version, nfields=nfields, nrecords=got,
                header_bytes=hbytes,
                channel=head[20:36].rstrip(b"\0").decode("ascii", "replace"),
                commit=head[36:52].rstrip(b"\0").decode("ascii", "replace"))


def describe(path):
    h = read(path)
    if h is None:
        return "%s: legacy, no header" % os.path.basename(path)
    return ("%s: v%d, %d fields x %d records, channel %r, commit %r"
            % (os.path.basename(path), h["version"], h["nfields"], h["nrecords"],
               h["channel"], h["commit"]))


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    # --fields prints "<nfields> <nrecords>" and --records prints the count,
    # both for shell callers (rentuple_array.sbatch), and both exit non-zero on
    # a legacy file so the caller can fall back rather than read a blank.
    if args and args[0] in ("--fields", "--records"):
        mode, paths = args[0], args[1:]
        for p in paths:
            h = read(p)
            if h is None:
                sys.exit(1)
            print(h["nfields"], h["nrecords"]) if mode == "--fields" else print(h["nrecords"])
        sys.exit(0)
    for p in args:
        print(describe(p))
