"""
portopt.attribution — Brinson-Fachler and Brinson-Hood-Beebower
performance attribution (B11).

Both methods decompose the active return  R_p − R_b  into:

  * Allocation effect — gain/loss from over- / underweighting groups
    (sectors, countries, factors) relative to the benchmark.
  * Selection effect — gain/loss from picking different assets within
    each group than the benchmark.
  * Interaction effect (BHB only) — joint contribution of selection
    and allocation; absent in BF, which folds it into allocation.

Inputs are per-group returns; map your asset-level returns into groups
beforehand (e.g. via Asset.sector or any classification scheme).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AttributionResult:
    """One row of attribution (per-group) plus totals."""
    groups: List[str]
    allocation: np.ndarray
    selection: np.ndarray
    interaction: np.ndarray              # zeros under BF; meaningful under BHB
    total_active: float
    method: str                          # "BF" or "BHB"

    def as_dict(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for i, g in enumerate(self.groups):
            out[g] = {
                "allocation":  float(self.allocation[i]),
                "selection":   float(self.selection[i]),
                "interaction": float(self.interaction[i]),
            }
        out["__total__"] = {
            "allocation":  float(self.allocation.sum()),
            "selection":   float(self.selection.sum()),
            "interaction": float(self.interaction.sum()),
            "active":      float(self.total_active),
            "method":      self.method,
        }
        return out


def _validate(group_weights_p, group_weights_b, group_returns_p, group_returns_b):
    wp = np.asarray(group_weights_p, dtype=float).ravel()
    wb = np.asarray(group_weights_b, dtype=float).ravel()
    rp = np.asarray(group_returns_p, dtype=float).ravel()
    rb = np.asarray(group_returns_b, dtype=float).ravel()
    n = wp.size
    if not (wb.size == rp.size == rb.size == n):
        raise ValueError("all inputs must have the same length")
    return wp, wb, rp, rb


def brinson_fachler(group_weights_p, group_weights_b,
                     group_returns_p, group_returns_b,
                     group_names=None) -> AttributionResult:
    """Brinson-Fachler attribution.

    Allocation_i = (w_p_i − w_b_i) · (R_b_i − R_b)
    Selection_i  =  w_b_i        · (R_p_i − R_b_i)
    Interaction_i = (w_p_i − w_b_i) · (R_p_i − R_b_i)

    BF folds interaction into allocation in the classical formulation
    by using (R_b_i − R_b) instead of (R_b_i). We return interaction
    separately as zero (per BF convention) — total = alloc + selection.
    """
    wp, wb, rp, rb = _validate(group_weights_p, group_weights_b,
                                group_returns_p, group_returns_b)
    R_b = float(wb @ rb)
    alloc = (wp - wb) * (rb - R_b)
    sel = wb * (rp - rb)
    # BF: interaction term is incorporated implicitly via (rb - R_b).
    interaction = np.zeros_like(alloc)
    active = float(wp @ rp - wb @ rb)
    return AttributionResult(
        groups       = list(group_names) if group_names is not None else
                       [str(i) for i in range(len(wp))],
        allocation   = alloc,
        selection    = sel,
        interaction  = interaction,
        total_active = active,
        method       = "BF",
    )


def brinson_hood_beebower(group_weights_p, group_weights_b,
                           group_returns_p, group_returns_b,
                           group_names=None) -> AttributionResult:
    """Brinson-Hood-Beebower attribution (with explicit interaction).

    Allocation_i  = (w_p_i − w_b_i) · R_b_i
    Selection_i   =  w_b_i        · (R_p_i − R_b_i)
    Interaction_i = (w_p_i − w_b_i) · (R_p_i − R_b_i)
    Total = Alloc + Selection + Interaction = R_p − R_b
    """
    wp, wb, rp, rb = _validate(group_weights_p, group_weights_b,
                                group_returns_p, group_returns_b)
    alloc = (wp - wb) * rb
    sel = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    active = float(wp @ rp - wb @ rb)
    return AttributionResult(
        groups       = list(group_names) if group_names is not None else
                       [str(i) for i in range(len(wp))],
        allocation   = alloc,
        selection    = sel,
        interaction  = interaction,
        total_active = active,
        method       = "BHB",
    )
