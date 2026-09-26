"""
Model registry: name -> class, with provenance.

The naming convention is  <channel>.<vintage>  :

    pi0.vpk2014     eta.vpk2014      jpsi.diffrad        rho.demo
    pi0.vpk2021     eta.tableB1      jpsi.roberts2026    phi.demo
                                     jpsin.roberts2026
                    eta.vpk2021      jpsi.diffrad_vm

Usage
-----
    from exclurad_py.models import get, list_models
    sf = get("eta.tableB1", t_nucl=-0.2)      # ctor args passed through
    list_models("eta")                        # -> printable table

Why a registry at all: pi0 and eta each have several vintages (2014, 2021,
2026-refit) that are easy to confuse, and the confusion is invisible in the
output numbers.  Every model stamps its name and version into the result
files, so a table can always be traced back to the cross section that made
it.
"""

import numpy as np

_REGISTRY = {}


def register(cls):
    """Class decorator: add a model to the registry."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} has no .name")
    if cls.name in _REGISTRY:
        raise ValueError(f"duplicate model name {cls.name!r} "
                         f"({cls.__name__} vs {_REGISTRY[cls.name].__name__})")
    for attr in ("channel", "version", "topology"):
        if getattr(cls, attr, None) is None:
            raise ValueError(f"{cls.name}: missing required attribute .{attr}")
    if not cls.provenance:
        raise ValueError(f"{cls.name}: provenance is mandatory")
    _REGISTRY[cls.name] = cls
    return cls


def get(name, **kwargs):
    """Instantiate a registered model. Extra kwargs go to the constructor."""
    if name not in _REGISTRY:
        close = [k for k in _REGISTRY if k.split(".")[0] == name.split(".")[0]]
        hint = f"  did you mean one of {sorted(close)}?" if close else ""
        raise KeyError(f"unknown model {name!r}.{hint}\n"
                       f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def get_class(name):
    return _REGISTRY[name]


def names(channel=None):
    if channel is None:
        return sorted(_REGISTRY)
    return sorted(k for k, v in _REGISTRY.items() if v.channel == channel)


def channels():
    return sorted({v.channel for v in _REGISTRY.values()})


def list_models(channel=None, verbose=False):
    """Return a printable table of registered models."""
    rows = []
    for k in names(channel):
        c = _REGISTRY[k]
        rows.append((k, c.version, c.topology.name, c.provenance))
    if not rows:
        return f"(no models registered for channel {channel!r})"
    w0 = max(len(r[0]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    out = []
    for k, ver, topo, prov in rows:
        if verbose:
            out.append(f"{k:<{w0}}  {ver:<6}  {topo:<{w2}}\n"
                       f"{'':<{w0}}  {prov}")
        else:
            first = prov.split("\n")[0]
            if len(first) > 78:
                first = first[:75] + "..."
            out.append(f"{k:<{w0}}  {ver:<6}  {topo:<{w2}}  {first}")
    return "\n".join(out)


def stamp_all():
    """Provenance of every registered model (for the report)."""
    out = {}
    for k, c in _REGISTRY.items():
        out[k] = dict(channel=c.channel, version=c.version,
                      provenance=c.provenance, reference=c.reference,
                      topology=c.topology.name,
                      validity={kk: list(vv)
                                for kk, vv in (c.validity or {}).items()})
    return out
