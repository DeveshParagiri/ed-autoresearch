"""Shared ILAMB-style component scores used by emissions assembly diagnostics."""
from __future__ import annotations

import numpy as np


def four_scores(pred_monthly, obs_monthly, w2):
    """Return Bias, RMSE, Seasonal, Spatial scores in [0, 1]. Higher is better.

    Pred and obs are (n_months, lat, lon) monthly fields. NaN-safe.
    """
    n = pred_monthly.shape[0]
    pred_monthly = np.nan_to_num(pred_monthly, nan=0.0, posinf=0.0, neginf=0.0)
    obs_monthly = np.nan_to_num(obs_monthly, nan=0.0, posinf=0.0, neginf=0.0)

    p_tm = pred_monthly.mean(axis=0)
    o_tm = obs_monthly.mean(axis=0)
    w_sum = float(w2.sum()) + 1e-12
    p_mean = float((p_tm * w2).sum() / w_sum)
    o_mean = float((o_tm * w2).sum() / w_sum)
    bias_score = float(np.exp(-abs(p_mean - o_mean) / (abs(o_mean) + 1e-9)))
    if not np.isfinite(bias_score):
        bias_score = 0.0

    global_rmse = float(
        np.sqrt((((pred_monthly - obs_monthly) ** 2) * w2[None, :, :]).sum() / (n * w_sum))
    )
    global_std = float(
        np.sqrt(
            ((obs_monthly - obs_monthly.mean(axis=0)) ** 2 * w2[None, :, :]).sum()
            / (n * w_sum)
        )
        + 1e-9
    )
    rmse_score = float(np.exp(-global_rmse / global_std))
    if not np.isfinite(rmse_score):
        rmse_score = 0.0

    n_yr = n // 12
    pred_clim = pred_monthly.reshape(n_yr, 12, *pred_monthly.shape[1:]).mean(axis=0)
    obs_clim = obs_monthly.reshape(n_yr, 12, *obs_monthly.shape[1:]).mean(axis=0)
    pa = pred_clim - pred_clim.mean(axis=0, keepdims=True)
    oa = obs_clim - obs_clim.mean(axis=0, keepdims=True)
    num = (pa * oa).sum(axis=0)
    den = np.sqrt((pa**2).sum(axis=0) * (oa**2).sum(axis=0)) + 1e-30
    corr = np.clip(num / den, -1, 1)
    valid = (den > 1e-20) & (w2 > 0)
    if valid.sum() == 0 or w2[valid].sum() < 1e-12:
        seasonal_score = 0.0
    else:
        seasonal_score = float((((1 + corr) / 2 * w2)[valid]).sum() / w2[valid].sum())
    if not np.isfinite(seasonal_score):
        seasonal_score = 0.0

    pred_tm = pred_monthly.mean(axis=0)
    obs_tm = obs_monthly.mean(axis=0)
    m = w2 > 0
    p_v, o_v, w_v = pred_tm[m], obs_tm[m], w2[m]
    pm = (p_v * w_v).sum() / w_v.sum()
    om = (o_v * w_v).sum() / w_v.sum()
    cov = (w_v * (p_v - pm) * (o_v - om)).sum() / w_v.sum()
    vp = (w_v * (p_v - pm) ** 2).sum() / w_v.sum()
    vo = (w_v * (o_v - om) ** 2).sum() / w_v.sum()
    r = float(cov / (np.sqrt(vp * vo) + 1e-30))
    sigma_p = float(np.sqrt(vp))
    sigma_o = float(np.sqrt(vo))
    sigma_ratio = (sigma_p + 1e-12) / (sigma_o + 1e-12)
    std_penalty = float(np.exp(-abs(np.log(max(sigma_ratio, 1e-9)))))
    spatial_score = float(((1 + r) / 2) * std_penalty)
    if not np.isfinite(spatial_score):
        spatial_score = 0.0
    return bias_score, rmse_score, seasonal_score, spatial_score
