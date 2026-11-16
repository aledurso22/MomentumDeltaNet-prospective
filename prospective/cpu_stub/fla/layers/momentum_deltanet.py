"""Executes the REAL official layer source from the checkout, unmodified."""
import os

_SRC = os.path.join(os.path.dirname(__file__), *[os.pardir] * 4,
                    "flash-linear-attention", "fla", "layers",
                    "momentum_deltanet.py")
with open(os.path.normpath(_SRC)) as _fh:
    exec(compile(_fh.read(), os.path.normpath(_SRC), "exec"))
