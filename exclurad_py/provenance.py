"""
Provenance stamping.

Every table, plot, LUND file and report this package writes carries a
stamp saying which code and which cross-section model produced it.  The
point is narrow and practical: pi0 and eta each have several vintages
(2014, 2021, 2026 refit) that give visibly different RC, and a file with
no stamp cannot be traced back to the model that made it.

    from exclurad_py import stamp
    hdr = stamp(sfmodel=sf, extra=dict(v_cut=0.20))
    json.dump(hdr, open("run.json", "w"), indent=2)

`stamp()` never raises: if git is unavailable the hash is recorded as
"unknown" rather than failing the run.
"""

import os
import subprocess

# Читать версию -- значит ТОЛЬКО читать.  `git status` иначе обновляет индекс
# и берёт .git/index.lock; двести частей на одном репозитории делают это
# разом, а брошенный замок потом не даёт сделать checkout -- так и случилось
# 23.08, замок пришлось снимать руками.
_GIT_ENV = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
import sys

VERSION = "0.1.0"

_HERE = os.path.dirname(os.path.abspath(__file__))


# Версия, переданная через окружение тем, кто запускал.  Ставится один раз
# на машине, где git заведомо работает, и переживает любой отказ git внутри
# задачи.  25.08: девять slurm-частей из 551 записались с version_tag
# "unknown" при живом git на узле -- git_branch у них отработал, а git_hash и
# git_tag нет.  Причину воспроизвести не удалось (сорок одновременных вызовов
# на ферме прошли без отказов), поэтому лечится не причина, а путь: три
# независимых способа узнать версию вместо одного.
_ENV_KEYS = dict(commit="EXCLURAD_COMMIT", tag="EXCLURAD_TAG",
                 branch="EXCLURAD_BRANCH")


def _env_version(key):
    v = os.environ.get(_ENV_KEYS.get(key, ""), "").strip()
    return v or None


def _deployed_version(key):
    """Version written at deploy time, for trees shipped without .git.

    A working copy sent to a remote machine by rsync --exclude .git has no
    repository, so git_hash() returns "unknown" and the output cannot be
    traced to the code that made it.  Both 100k production samples of
    2026-08-22 are stamped that way -- the provenance header exists exactly
    to prevent that, and it failed silently.

    production/deploy.sh writes exclurad_py/DEPLOYED_VERSION next to the
    package; this reads it as the fallback.
    """
    v = _env_version(key)
    if v:
        return v
    f = os.path.join(_HERE, "DEPLOYED_VERSION")
    try:
        for line in open(f):
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def git_hash(short=True):
    """Current commit of the exclurad_py working tree, or 'unknown'."""
    try:
        args = ["git", "-C", _HERE, "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        out = subprocess.run(args, capture_output=True, text=True, timeout=20,
                             env=_GIT_ENV)
        if out.returncode != 0:
            return _deployed_version("commit")
        h = out.stdout.strip()
        if not h:
            return _deployed_version("commit")
        # Проверка на грязное дерево -- отдельно.  Раньше её отказ выбрасывал
        # уже полученный хеш и превращал известную версию в "unknown".
        try:
            dirty = subprocess.run(["git", "-C", _HERE, "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=20,
                                   env=_GIT_ENV)
            if dirty.returncode == 0 and dirty.stdout.strip():
                h += "-dirty"
        except Exception:
            h += "-?"          # чисто оно или нет -- неизвестно, но хеш есть
        return h
    except Exception:
        return _deployed_version("commit")


def git_tag():
    """Human-readable version: the nearest tag, e.g. prod-pi0-v1.

    A commit hash identifies the code but says nothing about intent.  A run
    that will be handed to an analysis needs a name a person can quote in a
    talk, and it has to travel in the file itself.
    """
    try:
        out = subprocess.run(["git", "-C", _HERE, "describe", "--tags",
                              "--always", "--dirty"],
                             capture_output=True, text=True, timeout=20,
                             env=_GIT_ENV)
        return (out.stdout.strip() if out.returncode == 0
                else _deployed_version("tag")) or _deployed_version("tag")
    except Exception:
        return _deployed_version("tag")


def git_branch():
    try:
        out = subprocess.run(["git", "-C", _HERE, "rev-parse",
                              "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=20,
                             env=_GIT_ENV)
        return (out.stdout.strip() if out.returncode == 0
                else _deployed_version("branch"))
    except Exception:
        return _deployed_version("branch")


def stamp(sfmodel=None, extra=None, timestamp=None):
    """Provenance dict for an output file.

    Parameters
    ----------
    sfmodel   : a StructureFunctions instance (its .stamp() is merged in)
    extra     : dict of run-specific settings (cuts, grids, beam energy)
    timestamp : ISO string; pass one in if you want it recorded (the
                package does not call the clock itself so that runs are
                reproducible byte-for-byte).
    """
    from .core import CORE_VERSION

    d = {
        "code": "exclurad_py",
        "version": VERSION,
        "core_version": CORE_VERSION,
        "git_commit": git_hash(),
        "git_branch": git_branch(),
        "version_tag": git_tag(),
        "python": sys.version.split()[0],
    }
    if timestamp is not None:
        d["timestamp"] = timestamp
    if sfmodel is not None:
        d["sf"] = sfmodel.stamp()
        d["sf_usage"] = sfmodel.usage()
    if extra:
        d["run"] = dict(extra)
    return d


def header_lines(stamp_dict, comment="# "):
    """Flatten a stamp into comment lines for a CSV / LUND / text header."""
    out = []

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}{k}." if prefix else f"{k}.", v)
        else:
            out.append(f"{comment}{prefix.rstrip('.')}: {obj}")

    walk("", stamp_dict)
    return out
