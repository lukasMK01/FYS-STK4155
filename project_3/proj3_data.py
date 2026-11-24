"""
proj3_data.py
=============
Everything that is shared by the CNN runner (proj3_run.py), the baselines (proj3_baselines.py) and the
analysis (proj3_analyse.py). No TensorFlow in here.

  1. settings and paths
  2. LUT: reading the cache, removing clear-sky runs and duplicate pure clouds
  3. splits: hold-out test set, 5-fold CV and a classic train/validation/test split
  4. noise and spectral views (wavelength range and binning)
  5. targets and scores
  6. job files (one small json per trained model, so a long run can be resumed or run in parts)
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# 1. Settings (edit the paths; they can also be set with the environment variables PROJ3_*)
# ---------------------------------------------------------------------------
SIMS = '/scratch/lukasmk/data/sims'
CACHE_PATH = os.environ.get('PROJ3_CACHE', f'{SIMS}/cnn_miniLUT/data/miniLUT_zenith_transmittance.npz')
NOISE_CSV = os.environ.get('PROJ3_NOISE', '/uio/kant/geo-geofag-u1/lukasjm/cnn_test/data/dark_noise/'
                                          'dark_noise_versions_transmittance_452bands.csv')
RESULTS_DIR = os.environ.get('PROJ3_RESULTS', f'{SIMS}/cnn_proj3')

TARGETS = ['reff_liq', 'reff_ice', 'tau_total', 'f_ice']    # order of the network outputs
LOG_TARGETS = ('tau_total',)                                 # regressed as ln(tau)
TOL_ABS = dict(reff_liq=2.0, reff_ice=10.0, f_ice=0.1)       # tolerances of the joint success rate
TOL_REL = dict(tau_total=0.2)                                # relative (20 %)

# noise, the same for every model: measured dark noise sigma_T (detrended) per channel, plus a flat 2 % gain
NOISE_COLUMN = 'detrended'
GAIN = 0.02
T_FLOOR = 1e-4                  # transmittance is clipped here before the logarithm

# splits (the same for every model)
N_FOLDS = 5
TEST_FRACTION = 0.20            # hold-out test set, untouched until the final evaluation
VAL_FRACTION = 0.15             # share of the training data used for early stopping
SPLIT_SEED = 42
EVAL_SEED = 1234                # seed of the noise on the held-out spectra

# spectral views: a range is lo <= wavelength < hi in nm; OPEN_LO / OPEN_HI mean "no limit"
OPEN_LO, OPEN_HI = 0, 10000
DETECTOR_NM = 970               # VNIR detector below, SWIR detector above: bins never cross this wavelength


# ---------------------------------------------------------------------------
# 2. LUT
# ---------------------------------------------------------------------------
def prepare(cache_path=CACHE_PATH, noise_csv=NOISE_CSV, verbose=True):
    """
    Read the cache and prepare targets, masks and noise.
    - clear-sky runs (tau = 0) are dropped
    - a pure liquid (ice) cloud appears in the LUT once per r_eff of the absent phase, with an identical
      spectrum: only one copy is kept, so that no spectrum is in the training and the test data
    - reff_liq is only defined where there is liquid, reff_ice where there is ice (mask M)
    """
    with np.load(cache_path) as f:
        raw = {k: f[k] for k in f.files}
    wl = raw['wavelength'].astype(float)
    T = raw['transmittance'].astype('float32')
    tl, ti = raw['tau_liq'].astype(float), raw['tau_ice'].astype(float)
    rl, ri = raw['reff_liq'].astype(float), raw['reff_ice'].astype(float)

    ok = (tl + ti > 0) & np.isfinite(T).all(axis=1)
    n_clear = int(np.sum(tl + ti <= 0))
    T, tl, ti, rl, ri = T[ok], tl[ok], ti[ok], rl[ok], ri[ok]

    # one row per physical cloud: the r_eff of an absent phase does not belong to the cloud
    key = np.column_stack([tl, ti, np.where(tl > 0, rl, -1.0), np.where(ti > 0, ri, -1.0)])
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    deviation = float(np.abs(T - T[first][inverse.ravel()]).max())
    keep = np.sort(first)
    n_dup = int(len(T) - len(keep))
    T, tl, ti, rl, ri = T[keep], tl[keep], ti[keep], rl[keep], ri[keep]

    M = np.column_stack([tl > 0, ti > 0, np.ones(len(T), bool), np.ones(len(T), bool)])
    Y = np.column_stack([np.where(tl > 0, rl, np.nan), np.where(ti > 0, ri, np.nan), tl + ti, ti / (tl + ti)])
    cls = np.where(ti == 0, 0, np.where(tl == 0, 1, 2))      # 0 pure liquid, 1 pure ice, 2 mixed

    noise = pd.read_csv(noise_csv)
    sigma = np.interp(wl, noise['wavelength_nm'].to_numpy(), noise[NOISE_COLUMN].to_numpy())

    ds_id = hashlib.sha1(T.tobytes() + np.nan_to_num(Y).tobytes()).hexdigest()[:10]
    ds = dict(T=T, Y=Y, M=M, cls=cls, wl=wl, sigma=sigma, n=len(T), id=ds_id)
    if verbose:
        print(f'LUT: {len(T)} clouds ({int((cls == 0).sum())} pure liquid, {int((cls == 1).sum())} pure ice, '
              f'{int((cls == 2).sum())} mixed); dropped {n_clear} clear-sky runs and {n_dup} duplicate pure clouds')
        print(f'     {len(wl)} channels, {wl.min():.0f}-{wl.max():.0f} nm; largest difference between duplicate '
              f'spectra {deviation:.1e}; data id {ds_id}')
    return ds


# ---------------------------------------------------------------------------
# 3. Splits
# ---------------------------------------------------------------------------
def make_splits(ds, n_folds=N_FOLDS, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """
    Stratified by cloud type (pure liquid / pure ice / mixed), so every part has all three.
      test    : hold-out set (indices)
      cv      : list of dicts with train / val / held for the 5-fold CV on the rest (the "pool").
                val (for early stopping) is taken from the training part of the fold, held is the CV fold.
      classic : the classic split: train / val from the pool, held = the test set
      score_scaling : unit of the scores (fitted once on the pool; the models do not use it)
    """
    idx, cls = np.arange(ds['n']), ds['cls']
    pool, test = train_test_split(idx, test_size=test_fraction, stratify=cls, random_state=seed)

    def with_val(train, held):
        tr, va = train_test_split(train, test_size=val_fraction, stratify=cls[train], random_state=seed)
        return dict(train=np.sort(tr), val=np.sort(va), held=np.sort(held))

    cv = [with_val(pool[a], pool[b]) for a, b in
          StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(pool, cls[pool])]
    classic = with_val(pool, test)
    return dict(test=np.sort(test), pool=np.sort(pool), cv=cv, classic=classic,
                score_scaling=fit_target_scaling(ds['Y'][pool], ds['M'][pool]))


# ---------------------------------------------------------------------------
# 4. Noise and spectral views
# ---------------------------------------------------------------------------
def noise_draws(shape, seed):
    """Standard-normal draws (one per channel, and one gain draw per spectrum), reproducible by seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal((shape[0], 1))


def add_noise(T, sigma, draws, gain=GAIN):
    """[T + sigma * z1] * [1 + gain * z2]; sigma has one value per channel."""
    z_abs, z_gain = draws
    return ((T + sigma * z_abs) * (1.0 + gain * z_gain)).astype('float32')


def noisy(T, sigma, seed):
    return add_noise(T, sigma, noise_draws(T.shape, seed))


def log_input(T):
    return np.log(np.maximum(T, T_FLOOR)).astype('float32')


def spectral_view(T, wl, sigma, lo=OPEN_LO, hi=OPEN_HI, bin_factor=1):
    """
    The spectra seen by an instrument that covers lo <= wavelength < hi and bins `bin_factor` neighbouring
    channels (the mean transmittance of the bin). Bins stay within the VNIR or the SWIR detector.
    The noise is the measured noise interpolated to the wavelength of every (binned) channel; it is not
    reduced by the averaging.  Returns T_view, wl_view, sigma_view.
    """
    idx = np.flatnonzero((wl >= lo) & (wl < hi))
    if idx.size == 0:
        raise ValueError(f'No channels in {lo}-{hi} nm')
    groups = []
    for part in (idx[wl[idx] < DETECTOR_NM], idx[wl[idx] >= DETECTOR_NM]):
        if part.size:
            groups += np.array_split(part, max(1, part.size // int(bin_factor)))
    T_v = np.column_stack([T[:, g].mean(axis=1) for g in groups]).astype('float32')
    wl_v = np.array([wl[g].mean() for g in groups])
    return T_v, wl_v, np.interp(wl_v, wl, sigma)


# ---------------------------------------------------------------------------
# 5. Targets and scores
# ---------------------------------------------------------------------------
def _forward(Y, log_targets=LOG_TARGETS):
    Z = np.array(Y, dtype=float, copy=True)
    for j, name in enumerate(TARGETS):
        if name in log_targets:
            Z[:, j] = np.log(Z[:, j])
    return Z


def fit_target_scaling(Y, M):
    """Mean and std of the (log-)targets over the defined entries."""
    Z = _forward(Y)
    return dict(mean=[float(Z[M[:, j], j].mean()) for j in range(len(TARGETS))],
                std=[max(float(Z[M[:, j], j].std()), 1e-12) for j in range(len(TARGETS))])


def standardize(Y, M, scaling):
    """Targets in units of the standard deviation (0 where a target is undefined)."""
    Z = (_forward(Y) - np.asarray(scaling['mean'])) / np.asarray(scaling['std'])
    return np.where(M, Z, 0.0)


def to_physical(Z, scaling):
    """Standardized network output -> physical units."""
    Y = np.asarray(Z, dtype=float) * np.asarray(scaling['std']) + np.asarray(scaling['mean'])
    for j, name in enumerate(TARGETS):
        if name in LOG_TARGETS:
            Y[:, j] = np.exp(Y[:, j])
        if name == 'f_ice':
            Y[:, j] = np.clip(Y[:, j], 0.0, 1.0)
    return Y


def scores(Y_true, M, Y_pred, scaling):
    """
    Scores on held-out spectra (each target over its defined entries only).
      std_rmse_<target>   RMSE in units of the target's standard deviation (ln tau for tau): 0 = perfect, 1 = no
                          better than the mean
      mean_std_rmse       mean of the four: THE score used to select configurations (lower is better)
      mae_<target>        mean absolute error in physical units (tau: of the value, not of ln tau)
      within_tol_pct_<t>  share of spectra within the tolerance of that target
      joint_success_pct   share of spectra with all defined targets within their tolerance (reported only)
    """
    Zt = standardize(Y_true, M, scaling)
    Zp = standardize(Y_pred, np.ones_like(M), scaling)
    out, all_ok = {}, np.ones(len(Y_true), bool)
    for j, name in enumerate(TARGETS):
        s = M[:, j]
        err = np.abs(Y_pred[s, j] - Y_true[s, j])
        tol = TOL_REL[name] * np.abs(Y_true[s, j]) if name in TOL_REL else TOL_ABS[name]
        within = err <= tol
        all_ok[s] &= within
        out[f'std_rmse_{name}'] = float(np.sqrt(np.mean((Zp[s, j] - Zt[s, j]) ** 2)))
        out[f'mae_{name}'] = float(err.mean())
        out[f'within_tol_pct_{name}'] = float(100 * within.mean())
    out['mean_std_rmse'] = float(np.mean([out[f'std_rmse_{n}'] for n in TARGETS]))
    out['joint_success_pct'] = float(100 * all_ok.mean())
    out['n'] = int(len(Y_true))
    return out


# ---------------------------------------------------------------------------
# 6. Job files: results_dir/jobs/<name>__<cv0..cv4 | classic0..>.json
# ---------------------------------------------------------------------------
def normalise_cfg(cfg):
    """Numbers as int where they are whole (970.0 -> 970), so that equal settings give equal names."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            v = float(v)
            v = int(v) if v.is_integer() else v
        out[k] = v
    return out


SHORT_NAMES = dict(n_blocks='nb', filters='f', kernel='k', n_dense='nd', units='u', dropout='do', l2='l2',
                   lr='lr', lo='lo', hi='hi', bin='b')


def config_name(prefix, cfg, keys=None):
    """Readable name from the settings in `keys` (default: the CNN's); the hash covers the whole configuration
    (also epochs, seed, ...), so a changed setting never reuses an old result."""
    cfg = normalise_cfg(cfg)
    keys = keys or list(SHORT_NAMES)
    text = '_'.join(f'{SHORT_NAMES.get(k, k)}{cfg[k]:g}' for k in keys if k in cfg)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    return f'{prefix}_{text}_{h}' if text else f'{prefix}_{h}'


def results_dir(quick=False):
    return RESULTS_DIR + ('_quick' if quick else '')


def job_path(results_dir, name, kind, k):
    return os.path.join(results_dir, 'jobs', f'{name}__{kind}{k}.json')


def save_job(results_dir, record):
    path = job_path(results_dir, record['name'], record['kind'], record['fold'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, path)       # a job file is either complete or absent


def load_jobs(results_dir):
    """All finished jobs as one table: one row per trained model."""
    rows = []
    jobs_dir = os.path.join(results_dir, 'jobs')
    for fname in sorted(os.listdir(jobs_dir)) if os.path.isdir(jobs_dir) else []:
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(jobs_dir, fname)) as f:
            r = json.load(f)
        row = dict(family=r['family'], name=r['name'], kind=r['kind'], fold=r['fold'], dataset_id=r['dataset_id'])
        row.update({f'cfg_{k}': v for k, v in r['cfg'].items()})
        row.update(r['info'])
        row.update(r['view'])
        row.update({f'score_{k}': v for k, v in r['scores'].items()})
        rows.append(row)
    return pd.DataFrame(rows)
