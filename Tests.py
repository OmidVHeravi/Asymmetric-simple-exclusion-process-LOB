# “””
ASEP-LOB Empirical Testing Suite (Corrected)

Seven tests for validating the ASEP framework against LOB data.

Test 1a: Mid-price fluctuation exponent (H = 2/3, superdiffusive)
Test 1b: Cumulative trade flow at fixed price (H = 1/3, subdiffusive)
Test 2:  Trade count variance scaling (gamma = 2/3) + Tracy-Widom
Test 3:  LOB shape fit (ASEP density profile)
Test 4:  Spread distribution (geometric tail, predicted xi)
Test 5:  Shock propagation speed (Rankine-Hugoniot)
Test 6:  Phase identification (LD/HD/MC/Shock)

CORRECTIONS from original version:

- Test 1 now correctly targets H=2/3 for mid-price (superdiffusive,
  from second-class particle / Airy_2 process), NOT H=1/3.
- Separate Test 1b for cumulative flow at fixed price (H=1/3).
- Tracy-Widom right tail exponent corrected: exp(-4/3 * s^{3/2}),
  NOT exp(-2/3 * s^{3/2}).
- TW moment targets flagged as GUE (F_2); stationary LOB uses
  Baik-Rains (F_0) which has different moments.
- Spread xi formula corrected: xi = 1/log(1 + omega_A/omega_D),
  NOT 1/log(1 + omega_D/omega_A).  [omega_A and omega_D were SWAPPED]
  “””

import numpy as np
from scipy.stats import linregress, kstest, pearsonr, norm
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings

# =====================================================================

# TEST 1a: Mid-Price Fluctuation Exponent (H = 2/3)

# =====================================================================

def estimate_hurst_variogram(
series: np.ndarray,
dt: float,
lags: Optional[np.ndarray] = None,
min_samples: int = 30
) -> Dict:
“””
Estimate Hurst exponent from variogram scaling.
Var[X(t+D)-X(t)] ~ D^{2H}, so slope of log-Var vs log-D is 2H.

```
For mid-prices:
    ASEP prediction: H = 2/3  (Var ~ dt^{4/3}, superdiffusive)
    Random walk:     H = 1/2
For cumulative trade flow at a fixed price level:
    ASEP prediction: H = 1/3  (Var ~ dt^{2/3}, subdiffusive)
    Poisson:         H = 1/2

Parameters
----------
series : 1D array of uniformly-sampled observations
dt : sampling interval in seconds
lags : integer lags to evaluate (auto if None)
min_samples : minimum increments per lag
"""
if lags is None:
    max_lag = len(series) // 4
    lags = np.unique(np.logspace(
        0, np.log10(max(max_lag, 2)), 50).astype(int))

log_lags, log_vars = [], []

for lag in lags:
    inc = series[lag:] - series[:-lag]
    if len(inc) < min_samples:
        continue
    v = np.var(inc)
    if v > 0:
        log_lags.append(np.log(lag * dt))
        log_vars.append(np.log(v))

log_lags = np.array(log_lags)
log_vars = np.array(log_vars)

if len(log_lags) < 3:
    return {'H': np.nan, 'se': np.nan,
            'log_lags': log_lags, 'log_vars': log_vars}

slope, intercept, r, p, se = linregress(log_lags, log_vars)
H = slope / 2.0

return {'H': H, 'se': se / 2.0, 'slope': slope, 'R2': r**2,
        'log_lags': log_lags, 'log_vars': log_vars}
```

def detrended_fluctuation_analysis(
series: np.ndarray,
scales: Optional[np.ndarray] = None
) -> Dict:
“””
DFA on the increment (return) series.

```
For mid-price returns:
    ASEP prediction: DFA exponent = H = 2/3
For trade count increments:
    ASEP prediction: DFA exponent = H = 1/3
"""
increments = np.diff(series)
N = len(increments)

if scales is None:
    scales = np.unique(np.logspace(
        1, np.log10(max(N // 4, 12)), 40).astype(int))

fluctuations, valid_scales = [], []

for n in scales:
    n_seg = N // n
    if n_seg < 2:
        continue
    F2 = 0.0
    for s in range(n_seg):
        seg = increments[s * n:(s + 1) * n]
        profile = np.cumsum(seg - np.mean(seg))
        x = np.arange(n)
        coeffs = np.polyfit(x, profile, 1)
        trend = np.polyval(coeffs, x)
        F2 += np.mean((profile - trend) ** 2)
    fluctuations.append(np.sqrt(F2 / n_seg))
    valid_scales.append(n)

if len(valid_scales) < 3:
    return {'H': np.nan, 'se': np.nan}

log_s = np.log(np.array(valid_scales, dtype=float))
log_F = np.log(np.array(fluctuations))

slope, intercept, r, p, se = linregress(log_s, log_F)

return {'H': slope, 'se': se, 'R2': r**2,
        'log_scales': log_s, 'log_F': log_F}
```

def test_midprice_hurst(midprices: np.ndarray, dt: float,
min_lag_sec: float = 0.1,
max_lag_sec: float = 600.0) -> Dict:
“””
Test 1a: Mid-price Hurst exponent.

```
ASEP prediction: H = 2/3 (superdiffusive).
The mid-price is the interface position (second-class particle),
whose variance scales as t^{4/3} by the Balazs-Cator-Seppalainan
(2006) theorem and the Airy_2 process.
"""
min_lag = max(1, int(min_lag_sec / dt))
max_lag = min(len(midprices) // 4, int(max_lag_sec / dt))
if max_lag <= min_lag:
    max_lag = len(midprices) // 4
lags = np.unique(np.logspace(
    np.log10(min_lag),
    np.log10(max(max_lag, min_lag + 1)), 60).astype(int))

vario = estimate_hurst_variogram(midprices, dt, lags)
dfa = detrended_fluctuation_analysis(
    midprices, scales=lags[lags >= 10])

H_target = 2.0 / 3.0  # CORRECTED: was 1/3

return {
    'H_variogram': vario['H'],
    'H_DFA': dfa['H'],
    'se_variogram': vario['se'],
    'se_DFA': dfa['se'],
    'H_target': H_target,
    'asep_consistent': (
        abs(vario['H'] - H_target) < 2 * vario['se']
        if not np.isnan(vario.get('se', np.nan)) else None
    ),
}
```

def test_tradeflow_hurst(cumulative_flow: np.ndarray, dt: float,
min_lag_sec: float = 0.1,
max_lag_sec: float = 600.0) -> Dict:
“””
Test 1b: Cumulative trade flow at a fixed price level.

```
ASEP prediction: H = 1/3 (subdiffusive).
This is the integrated current J(i_0, t) at a fixed bond,
whose variance scales as t^{2/3} by KPZ universality.

Parameters
----------
cumulative_flow : cumulative count of trades (or order events)
                  at a single price level over time
dt : sampling interval
"""
min_lag = max(1, int(min_lag_sec / dt))
max_lag = min(len(cumulative_flow) // 4, int(max_lag_sec / dt))
if max_lag <= min_lag:
    max_lag = len(cumulative_flow) // 4
lags = np.unique(np.logspace(
    np.log10(min_lag),
    np.log10(max(max_lag, min_lag + 1)), 60).astype(int))

vario = estimate_hurst_variogram(cumulative_flow, dt, lags)

H_target = 1.0 / 3.0

return {
    'H_variogram': vario['H'],
    'se_variogram': vario['se'],
    'H_target': H_target,
    'asep_consistent': (
        abs(vario['H'] - H_target) < 2 * vario['se']
        if not np.isnan(vario.get('se', np.nan)) else None
    ),
}
```

# =====================================================================

# TEST 2: Trade Count Fluctuations + Tracy-Widom

# =====================================================================

def trade_count_scaling(
trade_times: np.ndarray,
windows: Optional[np.ndarray] = None
) -> Dict:
“””
Test Var[N(t)] ~ t^gamma.

```
ASEP (KPZ):  gamma = 2/3  (from t^{1/3} fluctuations)
Poisson:     gamma = 1
Hawkes:      gamma = 1 (for large t)
"""
T_total = trade_times[-1] - trade_times[0]

if windows is None:
    windows = np.logspace(-1, np.log10(T_total / 10), 50)

log_w, log_var = [], []

for w in windows:
    t0 = trade_times[0]
    n_win = int(T_total / w)
    if n_win < 20:
        continue
    counts = np.zeros(n_win)
    for k in range(n_win):
        t_start = t0 + k * w
        t_end = t_start + w
        counts[k] = (np.searchsorted(trade_times, t_end)
                     - np.searchsorted(trade_times, t_start))
    v = np.var(counts)
    if v > 0:
        log_w.append(np.log(w))
        log_var.append(np.log(v))

log_w = np.array(log_w)
log_var = np.array(log_var)

if len(log_w) < 3:
    return {'gamma': np.nan, 'se': np.nan}

slope, intercept, r, p, se = linregress(log_w, log_var)

return {'gamma': slope, 'se': se, 'R2': r**2,
        'log_windows': log_w, 'log_vars': log_var}
```

def tracy_widom_cdf_approx(s: float) -> float:
“””
Rough approximation to the TW-GUE CDF.

```
Right tail: 1 - F(s) ~ exp(-4/3 * s^{3/2})   [CORRECTED: was 2/3]
Left tail:  F(s) ~ exp(-|s|^3 / 12)
"""
if s < -5:
    return max(1e-15, np.exp(-(1.0 / 12) * abs(s) ** 3))
elif s > 5:
    # CORRECTED: exponent is 4/3, not 2/3
    # Because TW CDF involves q(s)^2 where q ~ Ai(s) ~ exp(-2/3 s^{3/2})
    # Squaring: q^2 ~ exp(-4/3 s^{3/2})
    return 1.0 - np.exp(-(4.0 / 3.0) * s ** 1.5)
else:
    # Shifted/scaled Gaussian approx (GUE: mean ~ -1.2065, std ~ 0.9018)
    return norm.cdf(s, loc=-1.2065, scale=0.9018)
```

def test_tracy_widom(trade_times: np.ndarray, window_size: float) -> Dict:
“””
Test whether standardized trade counts follow Tracy-Widom.

```
Key discriminators vs Gaussian:
  TW is left-skewed (negative skewness)
  TW right tail: exp(-4/3 * s^{3/2})  [CORRECTED from 2/3]
  TW left tail:  exp(-1/12 * |s|^3)   (heavier than Gaussian)

NOTE on TW variant:
  Step IC -> GUE F_2:  skew = -0.2935, var = 0.8132
  Flat IC -> GOE F_1:  skew = -0.2935, var = 0.6390
  Stationary IC -> Baik-Rains F_0: different moments
  For the LOB (stationary book), F_0 is the relevant distribution.
  The GUE values below are reference points; the F_0 moments
  are not available in simple closed form but share the key
  feature of negative skewness.
"""
T_total = trade_times[-1] - trade_times[0]
t0 = trade_times[0]
n_win = int(T_total / window_size)
if n_win < 30:
    warnings.warn(
        f"Only {n_win} windows; need >= 30 for reliable test.")

counts = np.zeros(n_win)
for k in range(n_win):
    t_start = t0 + k * window_size
    t_end = t_start + window_size
    counts[k] = (np.searchsorted(trade_times, t_end)
                 - np.searchsorted(trade_times, t_start))

mean_N = np.mean(counts)
std_N = np.std(counts)
if std_N == 0:
    return {'skewness': 0, 'excess_kurtosis': 0}
standardized = (counts - mean_N) / std_N

skew = float(np.mean(standardized ** 3))
kurt = float(np.mean(standardized ** 4) - 3.0)

ks_gauss, p_gauss = kstest(standardized, 'norm')

return {
    'skewness': skew,
    'excess_kurtosis': kurt,
    # GUE F_2 reference values (step IC):
    'tw_skew_ref_GUE': -0.2935,
    'tw_kurt_ref_GUE': 0.1653,
    # Key qualitative test (all TW variants share this):
    'skewness_is_negative': skew < 0,
    'ks_gauss': ks_gauss,
    'p_gauss': p_gauss,
    'n_windows': n_win,
}
```

# =====================================================================

# TEST 3: LOB Shape Fit

# =====================================================================

def catalan(k: int) -> int:
“”“k-th Catalan number.”””
from math import comb
return comb(2 * k, k) // (k + 1)

def asep_density_LD(ell: np.ndarray, N: float, alpha: float,
n_terms: int = 20) -> np.ndarray:
“”“ASEP density profile in the low-density phase.”””
x = ell / N
xi = -np.log(max(4 * alpha * (1 - alpha), 1e-12))
rho = alpha * np.ones_like(x, dtype=float)
prefactor = alpha * (1 - 2 * alpha) / max(1 - alpha, 1e-12)
base = 4 * alpha * (1 - alpha)
if base <= 0 or base >= 1:
return rho
for k in range(1, n_terms + 1):
Ck = catalan(k)
rho += prefactor * Ck / base ** k * np.exp(-k * xi * x)
return np.clip(rho, 0, 1)

def asep_density_MC(ell: np.ndarray) -> np.ndarray:
“”“ASEP density in maximal current phase.”””
safe_ell = np.maximum(ell, 0.5)
return 0.5 + ((-1.0) ** ell) / (2 * np.sqrt(np.pi * safe_ell))

def empirical_lob_shape(
snapshots: List[Dict], n_levels: int = 20,
side: str = ‘ask’
) -> Tuple[np.ndarray, np.ndarray]:
“”“Compute average LOB shape from order book snapshots.”””
all_vols = []
for snap in snapshots:
levels = snap.get(‘asks’ if side == ‘ask’ else ‘bids’, [])
if side == ‘bid’:
levels = sorted(levels, key=lambda x: -x[0])
else:
levels = sorted(levels, key=lambda x: x[0])
vol = np.zeros(n_levels)
for i in range(min(n_levels, len(levels))):
vol[i] = levels[i][1]
all_vols.append(vol)

```
avg = np.mean(all_vols, axis=0)
se = np.std(all_vols, axis=0) / np.sqrt(max(len(all_vols), 1))
return avg, se
```

def fit_asep_to_lob(avg_volume: np.ndarray) -> Dict:
“”“Fit LD, MC, and simple exponential models; compare by AIC.”””
n_levels = len(avg_volume)
ells = np.arange(1, n_levels + 1, dtype=float)

```
results = {}

# LD fit
def loss_LD(params):
    alpha, N_eff, Q = params
    if alpha <= 0.01 or alpha >= 0.49 or N_eff < 3 or Q <= 0:
        return 1e12
    rho = asep_density_LD(ells, N_eff, alpha)
    return float(np.sum((avg_volume - Q * rho) ** 2))

res = minimize(loss_LD, x0=[0.2, 50, np.max(avg_volume) * 2],
               method='Nelder-Mead', options={'maxiter': 5000})
k_LD = 3
results['LD'] = {
    'params': res.x, 'loss': res.fun,
    'aic': 2 * k_LD + n_levels * np.log(
        max(res.fun / n_levels, 1e-30))}

# MC fit
def loss_MC(params):
    Q, = params
    if Q <= 0:
        return 1e12
    rho = asep_density_MC(ells)
    return float(np.sum((avg_volume - Q * rho) ** 2))

res = minimize(loss_MC, x0=[np.max(avg_volume) * 2],
               method='Nelder-Mead')
results['MC'] = {
    'params': res.x, 'loss': res.fun,
    'aic': 2 * 1 + n_levels * np.log(
        max(res.fun / n_levels, 1e-30))}

# Exponential baseline
def loss_exp(params):
    A, lam = params
    if A <= 0 or lam <= 0:
        return 1e12
    pred = A * (1 - np.exp(-lam * ells))
    return float(np.sum((avg_volume - pred) ** 2))

res = minimize(loss_exp, x0=[np.max(avg_volume), 0.1],
               method='Nelder-Mead')
results['exponential'] = {
    'params': res.x, 'loss': res.fun,
    'aic': 2 * 2 + n_levels * np.log(
        max(res.fun / n_levels, 1e-30))}

best = min(results, key=lambda k: results[k]['aic'])
results['best_model'] = best
return results
```

# =====================================================================

# TEST 4: Spread Distribution

# =====================================================================

def test_spread_distribution(
spreads_ticks: np.ndarray,
omega_A: float,
omega_D: float
) -> Dict:
“””
Test geometric tail and predicted correlation length xi.

```
P(s >= n) = (omega_D / (omega_A + omega_D))^n = exp(-n/xi)
xi = 1 / log(1 + omega_A / omega_D)

Physical check: larger omega_A -> tighter spread -> smaller xi.
"""
spreads = np.array(spreads_ticks, dtype=float)
max_s = int(np.max(spreads))

n_vals = np.arange(1, max_s + 1)
survival = np.array([np.mean(spreads >= n) for n in n_vals])

mask = survival > 0
if np.sum(mask) < 3:
    return {'xi_empirical': np.nan, 'xi_predicted': np.nan}

n_v = n_vals[mask]
log_s = np.log(survival[mask])

slope, intercept, r, p, se = linregress(n_v, log_s)
xi_emp = -1.0 / slope if slope < 0 else float('inf')

# CORRECTED: was omega_D/omega_A, should be omega_A/omega_D
# Derivation: P(s>=n) = (omega_D/(omega_A+omega_D))^n
#   = exp(-n * log((omega_A+omega_D)/omega_D))
#   = exp(-n * log(1 + omega_A/omega_D))
# So xi = 1/log(1 + omega_A/omega_D)
if omega_D > 0:
    xi_pred = 1.0 / np.log(1 + omega_A / omega_D)
else:
    xi_pred = 0.0  # infinite placement rate -> zero spread

rel_err = (abs(xi_emp - xi_pred) / max(xi_emp, 1e-12)
           if np.isfinite(xi_emp) else np.nan)

return {
    'xi_empirical': xi_emp,
    'xi_predicted': xi_pred,
    'R2_geometric': r ** 2,
    'relative_error': rel_err,
}
```

def estimate_langmuir_rates(events: List[Dict]) -> Tuple[float, float]:
“””
Estimate omega_A (spread-narrowing placements) and
omega_D (best-quote-depleting cancellations).
“””
narrowing = 0
widening = 0
T = (events[-1][‘time’] - events[0][‘time’]
if len(events) > 1 else 1.0)

```
for ev in events:
    if ev.get('narrows_spread', False):
        narrowing += 1
    if ev.get('widens_spread', False):
        widening += 1

return narrowing / T, widening / T
```

# =====================================================================

# TEST 5: Shock Propagation Speed

# =====================================================================

def measure_refill_wavefront(
snapshots_after: List[List[Tuple[float, float]]],
pre_snapshot: List[Tuple[float, float]],
n_levels: int = 10,
dt_snap: float = 0.1,
threshold_frac: float = 0.5
) -> np.ndarray:
“”“Measure refill times for each level after a large trade.”””
pre_depth = np.array([
pre_snapshot[i][1] if i < len(pre_snapshot) else 0
for i in range(n_levels)])

```
refill_times = np.full(n_levels, np.nan)

for t_idx, snap in enumerate(snapshots_after):
    cur_depth = np.array([
        snap[i][1] if i < len(snap) else 0
        for i in range(n_levels)])
    for lvl in range(n_levels):
        if (np.isnan(refill_times[lvl])
                and pre_depth[lvl] > 0
                and cur_depth[lvl] >= threshold_frac * pre_depth[lvl]):
            refill_times[lvl] = t_idx * dt_snap

return refill_times
```

def test_shock_speed(
refill_times: np.ndarray,
tick_size: float,
rho_L: float,
rho_R: float,
p_minus_q: float
) -> Dict:
“””
Test Rankine-Hugoniot prediction for refill wavefront.
v_s = (p-q)(1 - rho_L - rho_R)

```
For concave flux j(rho) = rho(1-rho), entropy shocks
require rho_L < rho_R.
"""
levels = np.arange(len(refill_times), dtype=float)
valid = ~np.isnan(refill_times)

if np.sum(valid) < 3:
    return {'v_empirical': np.nan, 'v_predicted': np.nan}

slope, intercept, r, p, se = linregress(
    levels[valid], refill_times[valid])
v_emp = tick_size / slope if slope > 0 else float('inf')
v_pred = p_minus_q * (1 - rho_L - rho_R)

return {
    'v_empirical': v_emp,
    'v_predicted': v_pred,
    'R2_linear': r ** 2,
    'relative_error': abs(v_emp - v_pred) / max(abs(v_pred), 1e-12),
}
```

# =====================================================================

# TEST 6: Phase Identification

# =====================================================================

def classify_phase(alpha: float, beta: float) -> str:
“”“Classify into LD/HD/MC/Shock.”””
if alpha >= 0.5 and beta >= 0.5:
return ‘MC’
elif alpha < 0.5 and beta < 0.5:
if abs(alpha - beta) < 0.05:
return ‘Shock’
elif alpha < beta:
return ‘LD’
else:
return ‘HD’
elif alpha >= 0.5:
return ‘HD’
else:
return ‘LD’

def classify_phase_series(
alpha_ts: np.ndarray, beta_ts: np.ndarray
) -> List[str]:
“”“Classify a time series of windows.”””
return [classify_phase(a, b) for a, b in zip(alpha_ts, beta_ts)]

def test_phase_predictions(
alpha_ts: np.ndarray,
beta_ts: np.ndarray,
J_ts: np.ndarray,
rho_ts: np.ndarray,
spread_ts: np.ndarray
) -> Dict:
“””
For each identified phase, check whether observed
(J, rho, spread) match ASEP predictions.
“””
phases = classify_phase_series(alpha_ts, beta_ts)
results = {}

```
for phase in ['LD', 'HD', 'MC', 'Shock']:
    mask = np.array([p == phase for p in phases])
    n = int(np.sum(mask))
    if n < 5:
        results[phase] = {
            'n_windows': n, 'status': 'insufficient data'}
        continue

    a, b = alpha_ts[mask], beta_ts[mask]
    J, rho, sp = J_ts[mask], rho_ts[mask], spread_ts[mask]
    J_norm = J / max(np.max(J_ts), 1e-12)

    info = {'n_windows': n, 'mean_spread': float(np.mean(sp))}

    if phase == 'LD':
        J_pred = a * (1 - a)
        if np.std(J_pred) > 0 and np.std(J_norm) > 0:
            corr, p_val = pearsonr(J_pred, J_norm)
            info['J_corr'] = corr
            info['J_corr_p'] = p_val

    elif phase == 'HD':
        J_pred = b * (1 - b)
        if np.std(J_pred) > 0 and np.std(J_norm) > 0:
            corr, p_val = pearsonr(J_pred, J_norm)
            info['J_corr'] = corr
            info['J_corr_p'] = p_val

    elif phase == 'MC':
        info['J_mean'] = float(np.mean(J))
        info['J_cv'] = float(
            np.std(J) / max(np.mean(J), 1e-12))

    elif phase == 'Shock':
        info['spread_std'] = float(np.std(sp))

    results[phase] = info

return results
```

# =====================================================================

# MASTER RUNNER

# =====================================================================

def run_all_tests(
midprices: np.ndarray,
dt: float,
trade_times: np.ndarray,
book_snapshots: List[Dict],
spreads_ticks: np.ndarray,
events: List[Dict],
omega_A: float,
omega_D: float,
verbose: bool = True
) -> Dict:
“”“Run all ASEP-LOB tests and return a summary.”””
results = {}

```
# Test 1a: Mid-price Hurst
if verbose:
    print("=" * 60)
    print("TEST 1a: Mid-price fluctuation exponent (H = 2/3)")
    print("=" * 60)
r1a = test_midprice_hurst(midprices, dt)
results['test1a_midprice_hurst'] = r1a
if verbose:
    print(f"  H (variogram) = {r1a['H_variogram']:.4f}"
          f" +/- {r1a['se_variogram']:.4f}")
    if not np.isnan(r1a.get('H_DFA', np.nan)):
        print(f"  H (DFA)       = {r1a['H_DFA']:.4f}"
              f" +/- {r1a['se_DFA']:.4f}")
    print(f"  ASEP target:    {r1a['H_target']:.4f}"
          f"  (superdiffusive)")
    print(f"  Consistent:     {r1a['asep_consistent']}")

# Test 2: Trade count scaling
if verbose:
    print("\n" + "=" * 60)
    print("TEST 2: Trade count variance scaling (gamma = 2/3)")
    print("=" * 60)
r2a = trade_count_scaling(trade_times)
results['test2_scaling'] = r2a
if verbose:
    print(f"  gamma = {r2a['gamma']:.4f} +/- {r2a['se']:.4f}")
    print(f"  ASEP target: 0.6667")

# Test 2b: Tracy-Widom shape
median_rate = len(trade_times) / (trade_times[-1] - trade_times[0])
tw_window = max(10.0 / median_rate, 1.0)
r2b = test_tracy_widom(trade_times, tw_window)
results['test2_tracy_widom'] = r2b
if verbose:
    print(f"  Skewness:       {r2b['skewness']:.4f}"
          f"  (should be negative)")
    print(f"  Skewness < 0:   {r2b['skewness_is_negative']}")
    print(f"  GUE ref skew:  -0.2935"
          f"  (F_0 for stationary IC differs)")

# Test 3: LOB shape
if verbose:
    print("\n" + "=" * 60)
    print("TEST 3: LOB shape fit")
    print("=" * 60)
if book_snapshots:
    avg_vol, _ = empirical_lob_shape(book_snapshots)
    r3 = fit_asep_to_lob(avg_vol)
    results['test3_shape'] = r3
    if verbose:
        print(f"  Best model: {r3['best_model']}")
        for m, v in r3.items():
            if isinstance(v, dict) and 'aic' in v:
                print(f"    {m}: AIC={v['aic']:.2f},"
                      f" loss={v['loss']:.4f}")

# Test 4: Spread
if verbose:
    print("\n" + "=" * 60)
    print("TEST 4: Spread distribution")
    print("=" * 60)
r4 = test_spread_distribution(spreads_ticks, omega_A, omega_D)
results['test4_spread'] = r4
if verbose:
    print(f"  xi_empirical = {r4['xi_empirical']:.3f}")
    print(f"  xi_predicted = {r4['xi_predicted']:.3f}")
    print(f"  R^2 (geom)   = {r4['R2_geometric']:.4f}")
    print(f"  Rel. error   = {r4['relative_error']:.3f}")

if verbose:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    _print_decision_matrix(results)

return results
```

def _print_decision_matrix(results: Dict):
“”“Print the decision matrix.”””
tests = [
(“Mid-price H=2/3”,
results.get(‘test1a_midprice_hurst’, {}).get(
‘H_variogram’, np.nan),
2 / 3, 0.10),
(“Trade Var gamma=2/3”,
results.get(‘test2_scaling’, {}).get(‘gamma’, np.nan),
2 / 3, 0.10),
(“TW skewness<0”,
results.get(‘test2_tracy_widom’, {}).get(‘skewness’, np.nan),
-0.20, 0.20),  # just check it’s negative
]

```
for name, val, target, tol in tests:
    if np.isnan(val):
        status = "NO DATA"
    elif abs(val - target) < tol:
        status = "CONSISTENT"
    elif abs(val - target) < 2 * tol:
        status = "INCONCLUSIVE"
    else:
        status = "REJECTED"
    print(f"  {name:25s}: observed={val:+.4f}"
          f"  target={target:+.4f}  -> {status}")
```

# =====================================================================

# DEMO

# =====================================================================

if **name** == ‘**main**’:
np.random.seed(42)
N = 100000
white = np.random.randn(N)
prices = 100 + np.cumsum(white) * 0.01
dt = 0.1

```
trade_times = np.sort(
    np.random.uniform(0, N * dt, size=N // 10))

# Test spread xi formula correctness
omega_A, omega_D = 5.0, 2.0
xi_correct = 1.0 / np.log(1 + omega_A / omega_D)
xi_wrong = 1.0 / np.log(1 + omega_D / omega_A)
print("=== Spread xi formula check ===")
print(f"  omega_A={omega_A}, omega_D={omega_D}")
print(f"  CORRECT: xi = 1/log(1+oA/oD) = {xi_correct:.4f}")
print(f"  WRONG:   xi = 1/log(1+oD/oA) = {xi_wrong:.4f}")
print(f"  Physical: larger oA -> smaller xi (tighter spread)")
print(f"  CORRECT gives smaller xi for larger oA: "
      f"{xi_correct < xi_wrong}")
print()

# Test TW tail
print("=== Tracy-Widom right tail check ===")
s = 3.0
tail_wrong = np.exp(-(2/3) * s**1.5)
tail_correct = np.exp(-(4/3) * s**1.5)
print(f"  At s={s}:")
print(f"  WRONG (2/3):   1-F(s) ~ {tail_wrong:.6e}")
print(f"  CORRECT (4/3): 1-F(s) ~ {tail_correct:.6e}")
print(f"  Correct tail is lighter (decays faster): "
      f"{tail_correct < tail_wrong}")
print()

# Run main tests
print("=== Running ASEP-LOB test suite on synthetic data ===\n")
r1 = test_midprice_hurst(prices, dt)
print(f"Test 1a - Mid-price Hurst: {r1['H_variogram']:.4f}"
      f" (target {r1['H_target']:.3f},"
      f" random walk gives ~0.5)")

r2 = trade_count_scaling(trade_times)
print(f"Test 2  - Trade gamma:     {r2['gamma']:.4f}"
      f" (target 0.667, Poisson gives ~1.0)")
```
