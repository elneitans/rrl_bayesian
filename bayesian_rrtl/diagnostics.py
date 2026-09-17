"""Rank/folded split R-hat and Geyer ESS with NumPy/SciPy only.

Methods: Vehtari et al. (2021), Stan posterior analysis documentation.
ESS is conservatively capped at the number of draws (including antithetic data).
Constant/stuck chains are flagged; they never count as evidence of convergence.
"""

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm, rankdata


def _split(values):
    half = values.shape[1] // 2
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)


def _rank_normalize(values):
    ranks = rankdata(values, method='average').reshape(values.shape)
    # Blom normal scores; ties receive average ranks.
    return norm.ppf((ranks - 0.375) / (values.size + 0.25))


def _variances(values):
    n = values.shape[1]
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    var_plus = (n - 1) / n * within + float(np.var(values.mean(axis=1), ddof=1))
    return within, var_plus


def _rhat(values):
    within, var_plus = _variances(values)
    if within == 0:
        return np.inf if var_plus > 0 else np.nan
    return float(np.sqrt(var_plus / within))


def _ess(values):
    """Multi-chain autocorrelation with initial positive/monotone paired sums."""
    m, n = values.shape
    within, var_plus = _variances(values)
    if var_plus == 0:
        # A constant quantile indicator carries no sampling error. Raw constant
        # observables are separately flagged by chain_diagnostics before here.
        return float(m * n)
    if within == 0:
        return 0.0
    centered = values - values.mean(axis=1, keepdims=True)
    length = next_fast_len(2 * n)
    spectrum = rfft(centered, n=length, axis=1)
    covariance = irfft(spectrum * np.conjugate(spectrum), n=length, axis=1)[:, :n] / n
    rho = 1 - (within - covariance.mean(axis=0)) / var_plus
    rho[0] = 1
    paired = []
    for lag in range(0, n - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair <= 0:
            break
        paired.append(min(pair, paired[-1]) if paired else pair)
    tau = max(1.0, -1 + 2 * sum(paired))
    return float(min(m * n, m * n / tau))


def chain_diagnostics(values):
    """One scalar observable, shape [chain, retained draw]; at least 2 chains."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 4 or not np.isfinite(values).all():
        raise ValueError('Diagnostics require finite [>=2 chains, >=4 draws]')
    split = _split(values)
    n = split.size
    if np.ptp(values) == 0:
        return {'status': 'constant', 'rhat': None, 'ess_bulk': None, 'ess_tail': None, 'draws_used': n}
    bulk = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    candidates = [_rhat(bulk), _rhat(folded)]
    rhat = max(r for r in candidates if not np.isnan(r))
    if not np.isfinite(rhat):
        return {'status': 'stuck', 'rhat': None, 'ess_bulk': 0.0, 'ess_tail': 0.0, 'draws_used': n}
    quantiles = np.quantile(values, [0.05, 0.95])
    tail = min(_ess(_split((values <= q).astype(float))) for q in quantiles)
    return {'status': 'ok', 'rhat': rhat, 'ess_bulk': _ess(bulk), 'ess_tail': tail, 'draws_used': n}


def posterior_diagnostics(posteriors, reference_batch):
    """Check identifiable ensemble observables, never individual tree indices.

    All chains must fit the same X/y, model/prior/scale and number of draws.
    Timing, seeds and burn-in may differ, but only retained sweeps are compared.
    """
    if len(posteriors) < 2:
        raise ValueError('At least two independent posterior chains are required')
    first = posteriors[0]
    for posterior in posteriors[1:]:
        for name in ('data_hash', 'catalog_hash', 'm', 'scale', 'tree_prior', 'noise_prior', 'k', 'fixed_sigma2'):
            if getattr(posterior, name) != getattr(first, name):
                raise ValueError(f'Incompatible posterior chains: {name}')
        if len(posterior.draws) != len(first.draws):
            raise ValueError('Posterior chains must have equal retained lengths')
    predictions = np.stack([p.predict_latent_draws(reference_batch) for p in posteriors])
    report = {f'prediction_{i}': chain_diagnostics(predictions[:, :, i]) for i in range(len(reference_batch))}
    report['sigma2'] = chain_diagnostics(np.array([p.sigma2 for p in posteriors]))
    for field in ('total_leaves', 'max_depth'):
        report[field] = chain_diagnostics(np.array([[getattr(step, field) for step in p.trace[p.burn_in:]]
                                                    for p in posteriors]))
    return report
