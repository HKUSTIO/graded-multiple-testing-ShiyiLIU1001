from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    p = np.asarray(p_values)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    m = p.shape[0]
    thresh = float(alpha) / float(m)
    return p <= thresh


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    p = np.asarray(p_values)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    m = p.shape[0]
    # sort p-values and keep indices
    order = np.argsort(p)
    p_sorted = p[order]

    reject_sorted = np.zeros_like(p_sorted, dtype=bool)
    # step-down: check from smallest to largest; stop when a check fails
    for k in range(1, m + 1):
        thresh = float(alpha) / float(m - k + 1)
        if p_sorted[k - 1] <= thresh:
            reject_sorted[k - 1] = True
            # continue to next k
        else:
            # once a rank fails, all larger ranks are non-rejections
            break

    # map back to original order
    rejects = np.zeros_like(reject_sorted, dtype=bool)
    rejects[order] = reject_sorted
    return rejects


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    p = np.asarray(p_values)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    m = p.shape[0]
    order = np.argsort(p)
    p_sorted = p[order]

    # find largest k such that p_(k) <= (k/m)*alpha
    k_max = 0
    for k in range(1, m + 1):
        thresh = (float(k) / float(m)) * float(alpha)
        if p_sorted[k - 1] <= thresh:
            k_max = k

    reject_sorted = np.zeros_like(p_sorted, dtype=bool)
    if k_max > 0:
        reject_sorted[:k_max] = True

    rejects = np.zeros_like(reject_sorted, dtype=bool)
    rejects[order] = reject_sorted
    return rejects


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    p = np.asarray(p_values)
    if p.ndim != 1:
        raise ValueError("p_values must be a 1-dimensional array")
    m = p.shape[0]
    # harmonic number H_m = sum_{j=1}^m 1/j
    h_m = float(np.sum(1.0 / np.arange(1, m + 1)))

    order = np.argsort(p)
    p_sorted = p[order]

    k_max = 0
    for k in range(1, m + 1):
        thresh = (float(k) / float(m)) * (float(alpha) / h_m)
        if p_sorted[k - 1] <= thresh:
            k_max = k

    reject_sorted = np.zeros_like(p_sorted, dtype=bool)
    if k_max > 0:
        reject_sorted[:k_max] = True

    rejects = np.zeros_like(reject_sorted, dtype=bool)
    rejects[order] = reject_sorted
    return rejects


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    arr = np.asarray(rejections_null, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("rejections_null must be a 2-dimensional array of shape [L, M]")
    # fraction of rows with at least one True
    has_any = np.any(arr, axis=1)
    return float(np.mean(has_any))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    rej = np.asarray(rejections, dtype=bool)
    truth = np.asarray(is_true_null, dtype=bool)
    if rej.ndim != 1 or truth.ndim != 1:
        raise ValueError("rejections and is_true_null must be 1-dimensional arrays")
    if rej.shape[0] != truth.shape[0]:
        raise ValueError("rejections and is_true_null must have the same length")
    total_rej = int(np.sum(rej))
    if total_rej == 0:
        return 0.0
    false_disc = int(np.sum(rej & truth))
    return float(false_disc) / float(total_rej)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    rej = np.asarray(rejections, dtype=bool)
    truth = np.asarray(is_true_null, dtype=bool)
    if rej.ndim != 1 or truth.ndim != 1:
        raise ValueError("rejections and is_true_null must be 1-dimensional arrays")
    if rej.shape[0] != truth.shape[0]:
        raise ValueError("rejections and is_true_null must have the same length")
    false_nulls = ~truth
    denom = int(np.sum(false_nulls))
    if denom == 0:
        return 0.0
    true_rej = int(np.sum(rej & false_nulls))
    return float(true_rej) / float(denom)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    # null_pvalues: columns sim_id, hypothesis_id, p_value
    # mixed_pvalues: columns sim_id, hypothesis_id, p_value, is_true_null
    # Ensure proper types
    null_df = null_pvalues.copy()
    mixed_df = mixed_pvalues.copy()

    # Determine number of simulations and hypotheses from data
    # Group by sim_id and sort by hypothesis_id to create consistent vectors
    null_groups = list(
        null_df.groupby("sim_id", sort=True, as_index=False)
    )
    L_null = len(null_groups)
    if L_null == 0:
        raise ValueError("null_pvalues contains no simulations")

    # Build rejection matrix for null (uncorrected, bonferroni, holm)
    rejections_uncorr = []
    rejections_bonf = []
    rejections_holm = []
    for sim_id, group in null_groups:
        grp = group.sort_values("hypothesis_id")
        pvals = grp["p_value"].to_numpy()
        rejections_uncorr.append(pvals <= float(alpha))
        rejections_bonf.append(bonferroni_rejections(pvals, float(alpha)))
        rejections_holm.append(holm_rejections(pvals, float(alpha)))

    rejections_uncorr = np.vstack(rejections_uncorr)
    rejections_bonf = np.vstack(rejections_bonf)
    rejections_holm = np.vstack(rejections_holm)

    fwer_uncorrected = compute_fwer(rejections_uncorr)
    fwer_bonferroni = compute_fwer(rejections_bonf)
    fwer_holm = compute_fwer(rejections_holm)

    # Mixed simulations: compute FDR and power per simulation then average
    mixed_groups = list(mixed_df.groupby("sim_id", sort=True, as_index=False))
    fdr_unc_list = []
    fdr_bh_list = []
    fdr_by_list = []
    power_unc_list = []
    power_bh_list = []
    power_by_list = []

    for sim_id, group in mixed_groups:
        grp = group.sort_values("hypothesis_id")
        pvals = grp["p_value"].to_numpy()
        is_true = grp["is_true_null"].to_numpy(dtype=bool)

        # uncorrected
        rej_unc = pvals <= float(alpha)
        # BH and BY
        rej_bh = benjamini_hochberg_rejections(pvals, float(alpha))
        rej_by = benjamini_yekutieli_rejections(pvals, float(alpha))

        fdr_unc_list.append(compute_fdr(rej_unc, is_true))
        fdr_bh_list.append(compute_fdr(rej_bh, is_true))
        fdr_by_list.append(compute_fdr(rej_by, is_true))

        power_unc_list.append(compute_power(rej_unc, is_true))
        power_bh_list.append(compute_power(rej_bh, is_true))
        power_by_list.append(compute_power(rej_by, is_true))

    # Average across simulations
    fdr_uncorrected = float(np.mean(fdr_unc_list)) if len(fdr_unc_list) > 0 else 0.0
    fdr_bh = float(np.mean(fdr_bh_list)) if len(fdr_bh_list) > 0 else 0.0
    fdr_by = float(np.mean(fdr_by_list)) if len(fdr_by_list) > 0 else 0.0

    power_uncorrected = float(np.mean(power_unc_list)) if len(power_unc_list) > 0 else 0.0
    power_bh = float(np.mean(power_bh_list)) if len(power_bh_list) > 0 else 0.0
    power_by = float(np.mean(power_by_list)) if len(power_by_list) > 0 else 0.0

    return {
        "fwer_uncorrected": fwer_uncorrected,
        "fwer_bonferroni": fwer_bonferroni,
        "fwer_holm": fwer_holm,
        "fdr_uncorrected": fdr_uncorrected,
        "fdr_bh": fdr_bh,
        "fdr_by": fdr_by,
        "power_uncorrected": power_uncorrected,
        "power_bh": power_bh,
        "power_by": power_by,
    }
