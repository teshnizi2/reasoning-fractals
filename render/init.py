"""EqR's native initial latents and a stratified puzzle sampler for scans."""
from __future__ import annotations

import math

import numpy as np


def comp_std(std: float, lower: float = -2.0, upper: float = 2.0) -> float:
    """The std compensation EqR's ``trunc_normal_init_`` applies so a +-2 sigma truncated normal has std ``std``."""
    sqrt2 = math.sqrt(2.0)
    z = (math.erf(upper / sqrt2) - math.erf(lower / sqrt2)) / 2.0
    c = (2.0 * math.pi) ** -0.5
    pdf_u, pdf_l = c * math.exp(-0.5 * lower ** 2), c * math.exp(-0.5 * upper ** 2)
    return std / math.sqrt(1.0 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)


def trunc_normal(shape, std: float, rng: np.random.Generator) -> np.ndarray:
    """Truncated normal, EqR style (``models.common.trunc_normal_init_``): what EqR resets z_H and z_L to."""
    from scipy.special import erfinv

    sqrt2 = math.sqrt(2.0)
    a, b = math.erf(-2.0 / sqrt2), math.erf(2.0 / sqrt2)
    cs = comp_std(std)
    x = erfinv(rng.uniform(a, b, size=shape)) * sqrt2 * cs
    return np.clip(x, -2.0 * cs, 2.0 * cs).astype(np.float32)


def stratified(rows: list, edges: tuple, per_bin: int, seed: int = 0) -> list:
    """Pick ``per_bin`` distinct (source, question, answer, rating) rows from every rating bin [edge_i, edge_i+1)."""
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [r for r in rows if lo <= r[3] < hi]
        idx = rng.choice(len(pool), size=min(per_bin, len(pool)), replace=False)
        out += [pool[i] for i in sorted(idx)]
    return out
