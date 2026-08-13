"""Structure-function models, organized as channel x vintage.

    from exclurad_py import models
    print(models.list_models())            # everything
    print(models.list_models("eta"))       # one channel
    sf = models.get("eta.tableB1", t_nucl=-0.2)

Adding a model: subclass StructureFunctions (or ProtonDetectedModel for the
EXCLURAD pi0/eta topology), fill in name/channel/version/provenance/
topology/validity, and decorate with @register. The registry refuses a
model with no provenance -- that is deliberate.
"""

from .base import (StructureFunctions, ProtonDetectedModel, TwoSF,
                   OutOfRange, SF_KEYS)
from .registry import (register, get, get_class, names, channels,
                       list_models, stamp_all)

# importing these modules populates the registry
from . import pseudoscalar   # noqa: F401
from . import vector         # noqa: F401

__all__ = ["StructureFunctions", "ProtonDetectedModel", "TwoSF", "OutOfRange",
           "SF_KEYS", "register", "get", "get_class", "names", "channels",
           "list_models", "stamp_all"]
