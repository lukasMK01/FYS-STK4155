"""
cnn_framework.py
================
Building blocks for training and evaluating 1D-CNN retrievals of
(reff_liq, reff_ice, tau_total, f_ice) from the simulated below-cloud,
zenith-viewing transmittance spectra of the mini-LUT, for many network,
training and noise configurations in one go.

Use it through run_cnn_experiments.py, which holds all settings and the list
of experiments. The only settings in this file are the defaults in
DEFAULT_CONFIG.

What run_experiments() does
---------------------------
1. Prepares the LUT once. Clear-sky runs (tau_liq = tau_ice = 0, f_ice
   undefined) are dropped. reff_liq is masked where there is no liquid and
   reff_ice where there is no ice, because the spectrum carries no
   information on them there.
2. Creates one train/validation/test split, stores it in the results folder
   and reuses it for every experiment, so all experiments are tested on the
   same runs.
3. For each experiment: trains the CNN, or reuses the saved model when the
   configuration is unchanged. Then evaluates it on the test set with a
   synthetic test noise that is identical for all experiments (EVAL_NOISE)
   and, optionally, with the experiment's own training noise, at several
   noise levels. Besides per-parameter metrics it computes four combined
   scores (mean standardized RMSE, sum of R^2, tolerance-normalized error,
   joint success rate) and makes the figures.
4. Writes a cross-experiment summary (CSV and figures) to the results folder.

Output layout
-------------
RESULTS_DIR/
    split.npz                          fixed train/val/test indices
    lut_mutual_information.npz / .png  model-free information content of the LUT
    summary_metrics.csv                all metrics of all experiments (columns
                                       test_noise = 'common' / 'own'; combined
                                       scores in the rows with target 'combined')
    summary_*.png                      comparison figures
    <experiment>/
        model.keras, config.json, history.json
        metrics.csv, predictions.npz, importance.npz
        figures/*.png

TensorFlow is only imported when a model is trained or loaded, so preparing
the data (e.g. building the LUT cache on NIRD) works without it.
"""

import os

# CPU thread limits, applied before numpy/TensorFlow start their thread pools.
# TensorFlow otherwise starts one thread per visible core, which can exceed the
# per-user thread limit on shared machines ("pthread_create() failed").
N_THREADS = int(os.environ.get('CNN_N_THREADS', '8'))
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, str(N_THREADS))

import hashlib
import itertools
import json
import re
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Targets, parameters and defaults
# ---------------------------------------------------------------------------
TARGET_NAMES = ['reff_liq', 'reff_ice', 'tau_total', 'f_ice']   # network output order
LOG_TARGETS = ('tau_total',)                                     # regressed in log space
GRID_PARAMS = ['tau_liq', 'tau_ice', 'reff_liq', 'reff_ice']     # the LUT axes
ALL_PARAMS = GRID_PARAMS + ['tau_total', 'f_ice']

LABELS = dict(
    reff_liq=r'$r_\mathrm{eff,liq}$', reff_ice=r'$r_\mathrm{eff,ice}$',
    tau_total=r'$\tau_\mathrm{total}$', f_ice=r'$f_\mathrm{ice}$',
    tau_liq=r'$\tau_\mathrm{liq}$', tau_ice=r'$\tau_\mathrm{ice}$',
)
UNITS = dict(reff_liq='µm', reff_ice='µm')
LOG_SCALE_PARAMS = {'tau_total', 'tau_liq', 'tau_ice'}

# approximate strong water-vapour absorption bands, shaded in spectral plots
H2O_BANDS = [(1330.0, 1480.0), (1780.0, 1990.0)]
# running mean (channels) applied to gradient curves in the overview plots
GRADIENT_SMOOTHING = 5

DEFAULT_CONFIG = dict(
    # network
    filters=(16, 32, 64),   # one Conv1D + MaxPooling1D(2) block per entry
    kernels=(9, 7, 5),      # kernel size (channels) per block; an int = same for all
    dense=(64, 32),         # hidden dense layers after flattening; () = none
    dropout=0.2,
    l2=1e-5,
    # training
    learning_rate=1e-3,
    batch_size=32,
    epochs=500,             # upper limit, early stopping usually ends sooner
    patience=40,            # early-stopping patience (epochs)
    seed=42,
    # synthetic training noise, in transmittance units (see add_noise). The standard noise is
    # the measured dark noise sigma_T from fenix_dark_noise.py plus the gain error, without
    # relative noise. sigma_T comes from the data, so it is set in the run script's
    # BASE_CONFIG (noise_absolute=sigma_T); 0.0 here means no absolute noise.
    noise_relative=0.0,     # 1-sigma, fraction of signal; scalar or one value per wavelength
    noise_absolute=0.0,     # 1-sigma additive; scalar or one value per wavelength (standard: sigma_T)
    noise_gain=0.02,        # 1-sigma spectrally flat gain error, one draw per spectrum
    noise_scale=1.0,        # multiplies all three terms (0 = noise-free training)
    # input preprocessing
    log_input=True,         # feed log(transmittance)
    t_floor=1e-4,           # transmittance is clipped here before the log
    # spectral range used for training AND testing: None = all wavelengths;
    # (lo, hi) in nm, either end None = open, e.g. (None, 1700) = below 1700 nm;
    # or a list of such ranges, e.g. [(380, 1300), (1500, 1700)]
    wl_range=None,
)

# short names for automatically generated experiment names (expand_grid)
_SHORT = dict(filters='f', kernels='k', dense='d', dropout='do', l2='l2',
              learning_rate='lr', batch_size='bs', epochs='ep', patience='pat',
              seed='seed', noise_relative='nrel', noise_absolute='nabs',
              noise_gain='ngain', noise_scale='noise', log_input='log',
              t_floor='floor', wl_range='wl')

# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------
# categorical colours in fixed order (validated for colour-vision deficiency)
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER_COLOR = '#b5b4ad'
UNDEFINED_COLOR = '#cfcec7'   # scatter points whose colour parameter is undefined
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID_COLOR, AXIS_COLOR, BAND_COLOR = '#e1e0d9', '#c3c2b7', '#ecebe6'

try:
    from cmcrameri import cm as _cmc
    SCATTER_CMAP = mcolors.ListedColormap(_cmc.batlow(np.linspace(0.0, 0.88, 256)), name='batlow_cut')
    HEATMAP_CMAP = _cmc.devon_r
except ImportError:
    SCATTER_CMAP = mcolors.ListedColormap(plt.get_cmap('viridis')(np.linspace(0.0, 0.92, 256)), name='viridis_cut')
    HEATMAP_CMAP = plt.get_cmap('Blues')

STYLE = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': AXIS_COLOR, 'axes.linewidth': 0.8,
    'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': GRID_COLOR, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.labelcolor': INK, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.titlecolor': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'savefig.dpi': 150,
    'lines.linewidth': 1.4,
}


def _label(name, unit=True):
    u = UNITS.get(name)
    return f'{LABELS[name]} ({u})' if (unit and u) else LABELS[name]


# ---------------------------------------------------------------------------
# TensorFlow (imported on first use)
# ---------------------------------------------------------------------------
_TF = None


def _tf():
    """Import TensorFlow/Keras once, with the CPU thread limits applied."""
    global _TF
    if _TF is None:
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(N_THREADS))
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # hide INFO/WARNING lines
        import tensorflow as tf
        from tensorflow import keras
        try:
            tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass   # already initialized; the environment variables still apply
        _TF = (tf, keras)
    return _TF


# ---------------------------------------------------------------------------
# 1. LUT reading and preparation
# ---------------------------------------------------------------------------
def normalize_radiance(radiance, solar_irradiance, wavelength, cos_sza):
    """pi * radiance / (solar_irradiance * cos(SZA)), as in miniLUT_overview.py."""
    if hasattr(solar_irradiance, 'irradiance'):
        solar_values = solar_irradiance.irradiance.values
    elif hasattr(solar_irradiance, 'values'):
        solar_values = solar_irradiance.values
    else:
        solar_values = np.asarray(solar_irradiance)
    if solar_values.shape[-1] != len(wavelength):
        raise ValueError(
            f"solar_irradiance has {solar_values.shape[-1]} wavelength points, "
            f"but the data has {len(wavelength)}; they must share a wavelength grid.")
    return np.pi * radiance / (solar_values * cos_sza)


def load_lut(cache_path, lut_path=None, solar_path=None, solar_time=None,
             cos_sza=None, rebuild=False):
    """
    Zenith-pixel transmittance spectra and cloud parameters of the LUT.

    If `cache_path` (.npz) exists, the prepared arrays are read from it.
    Otherwise the netCDF LUT and solar spectrum are read (needs xarray, plus
    fenix_functions when cos_sza is not given), normalized to transmittance
    as in miniLUT_overview.py, and written to the cache. The cache is only a
    few MB and makes training independent of xarray, fenix_functions and NIRD.
    """
    if os.path.exists(cache_path) and not rebuild:
        with np.load(cache_path) as f:
            data = {k: f[k] for k in f.files}
        print(f"LUT read from cache {cache_path}")
        return _check_lut(data)

    if lut_path is None or solar_path is None:
        raise FileNotFoundError(
            f"No LUT cache at {cache_path}. Give lut_path and solar_path to build it.")
    import xarray as xr
    if cos_sza is None:
        if solar_time is None:
            raise ValueError("Give either cos_sza or solar_time to build the LUT cache.")
        from fenix_functions import compute_solar_angles
        angles = compute_solar_angles([solar_time])
        cos_sza = float(np.cos(np.deg2rad(angles.solar_zenith.values[0])))

    combined = xr.open_dataset(lut_path)
    solar = xr.open_dataset(solar_path)
    wavelength = combined['wavelength'].values.astype(float)

    # zenith-viewing pixel: first azimuth, cos_zenith closest to -1
    cz_vals = combined['cos_zenith'].values
    cz_idx = int(np.argmin(np.abs(cz_vals - (-1))))
    radiance = (combined['radiance'].isel(azimuth=0, cos_zenith=cz_idx)
                .transpose('run', 'wavelength', ...).values)
    if radiance.ndim != 2:
        raise ValueError(f"Expected radiance with dims (run, wavelength) after selecting the "
                         f"zenith pixel, got shape {radiance.shape}.")
    T = normalize_radiance(radiance, solar, wavelength, cos_sza)

    data = dict(
        wavelength=wavelength,
        transmittance=T.astype('float32'),
        tau_liq=combined['tau_liq'].values.astype(float),
        reff_liq=combined['reff_liq'].values.astype(float),
        tau_ice=combined['tau_ice'].values.astype(float),
        reff_ice=combined['reff_ice'].values.astype(float),
        cos_sza=np.float64(cos_sza),
        view_azimuth=np.float64(combined['azimuth'].values[0]),
        view_cos_zenith=np.float64(cz_vals[cz_idx]),
    )
    combined.close()
    solar.close()
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"LUT cache written to {cache_path}")
    return _check_lut(data)


def _check_lut(data):
    need = ['wavelength', 'transmittance', 'tau_liq', 'reff_liq', 'tau_ice', 'reff_ice']
    missing = [k for k in need if k not in data]
    if missing:
        raise KeyError(f"LUT data is missing {missing}")
    n_runs, n_wl = data['transmittance'].shape
    if data['wavelength'].size != n_wl:
        raise ValueError("wavelength and transmittance have different lengths")
    for k in need[2:]:
        if data[k].shape != (n_runs,):
            raise ValueError(f"{k} has shape {data[k].shape}, expected ({n_runs},)")
    if not np.all(np.diff(data['wavelength']) > 0):
        print("NOTE: the wavelength grid is not strictly increasing. The CNN treats "
              "neighbouring array entries as spectral neighbours.")
    print(f"  {n_runs} spectra x {n_wl} wavelengths "
          f"({data['wavelength'].min():.0f}-{data['wavelength'].max():.0f} nm)")
    return data


def prepare_dataset(data):
    """Targets, masks and bookkeeping for training and evaluation."""
    T = np.asarray(data['transmittance'], dtype='float32')
    raw = {k: np.asarray(data[k], dtype=float) for k in GRID_PARAMS}
    tau_total = raw['tau_liq'] + raw['tau_ice']
    cloudy = tau_total > 0
    f_ice = np.full_like(tau_total, np.nan)
    f_ice[cloudy] = raw['tau_ice'][cloudy] / tau_total[cloudy]
    finite = np.all(np.isfinite(T), axis=1)
    keep = cloudy & finite

    print(f"  {np.sum(~cloudy)} clear-sky runs dropped (f_ice undefined)")
    if np.any(cloudy & ~finite):
        print(f"  WARNING: {np.sum(cloudy & ~finite)} runs with non-finite transmittance dropped")

    params = {k: v[keep] for k, v in raw.items()}
    params['tau_total'] = tau_total[keep]
    params['f_ice'] = f_ice[keep]
    n = int(keep.sum())
    defined = {k: np.ones(n, bool) for k in ALL_PARAMS}
    defined['reff_liq'] = params['tau_liq'] > 0   # no liquid -> reff_liq has no signal
    defined['reff_ice'] = params['tau_ice'] > 0   # no ice    -> reff_ice has no signal

    Y = np.column_stack([params[k] for k in TARGET_NAMES])
    M = np.column_stack([defined[k] for k in TARGET_NAMES])
    T = T[keep]
    fingerprint = hashlib.sha1(
        T.tobytes() + np.column_stack([params[k] for k in GRID_PARAMS]).tobytes()
    ).hexdigest()[:12]

    for j, k in enumerate(TARGET_NAMES):
        print(f"  {k:<10} defined for {int(M[:, j].sum()):5d} / {n} runs")
    return dict(T=T, Y=Y, M=M, params=params, defined=defined,
                wavelength=np.asarray(data['wavelength'], dtype=float),
                n=n, n_wl=T.shape[1], fingerprint=fingerprint)


def get_split(ds, results_dir, test_fraction=0.2, val_fraction=0.15, seed=42):
    """
    Train/validation/test indices, created once per results folder and reused
    by every experiment in it.
    """
    settings = dict(n_runs=int(ds['n']), data_fingerprint=ds['fingerprint'],
                    test_fraction=float(test_fraction), val_fraction=float(val_fraction),
                    seed=int(seed))
    path = os.path.join(results_dir, 'split.npz')
    if os.path.exists(path):
        with np.load(path) as f:
            saved = json.loads(str(f['settings']))
            split = dict(train=f['train'], val=f['val'], test=f['test'])
        if saved != settings:
            diff = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v}
            raise ValueError(
                f"{path} was made with other settings or another LUT "
                f"(saved vs. now: {diff}). The experiments in this folder were trained on "
                f"that split and can't be tested on a new one. Use a new RESULTS_DIR "
                f"(keeps the old results comparable) or delete split.npz so that all "
                f"experiments retrain.")
    else:
        if not (0 < test_fraction < 1 and 0 < val_fraction < 1):
            raise ValueError("test_fraction and val_fraction must be between 0 and 1")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ds['n'])
        n_test = int(round(test_fraction * ds['n']))
        test, rest = perm[:n_test], perm[n_test:]
        n_val = int(round(val_fraction * rest.size))
        split = dict(train=np.sort(rest[n_val:]), val=np.sort(rest[:n_val]), test=np.sort(test))
        np.savez(path, settings=json.dumps(settings), **split)
    split['id'] = hashlib.sha1(np.concatenate(
        [split['train'], [-1], split['val'], [-2], split['test']]).astype(np.int64).tobytes()
    ).hexdigest()[:12]
    print(f"Split: {split['train'].size} train / {split['val'].size} validation / "
          f"{split['test'].size} test runs")
    return split


# ---------------------------------------------------------------------------
# 2. Configurations
# ---------------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def resolve_config(base_config, experiment, n_wl, wavelength=None):
    """
    Merge DEFAULT_CONFIG <- base_config <- experiment and check the result.
    `wavelength` (the LUT grid) is needed when 'wl_range' is set.
    """
    exp = dict(experiment)
    name = exp.pop('name', None)
    if not name:
        raise ValueError(f"Every experiment needs a 'name': {experiment}")
    if not re.fullmatch(r'[A-Za-z0-9._+-]+', str(name)):
        raise ValueError(f"Experiment name '{name}' is used as a folder name; "
                         f"use only letters, digits and . _ + -")
    for src, where in ((base_config, 'BASE_CONFIG'), (exp, f"experiment '{name}'")):
        unknown = sorted(set(src) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown setting(s) {unknown} in {where}. "
                             f"Valid settings: {sorted(DEFAULT_CONFIG)}")
    cfg = {**DEFAULT_CONFIG, **base_config, **exp}

    def err(msg):
        raise ValueError(f"Experiment '{name}': {msg}")

    filters = np.atleast_1d(cfg['filters'])
    if filters.size == 0:
        err("'filters' needs at least one entry")
    cfg['filters'] = tuple(int(f) for f in filters)
    kernels = np.atleast_1d(cfg['kernels'])
    if kernels.size == 1:
        kernels = np.repeat(kernels, len(cfg['filters']))
    if kernels.size != len(cfg['filters']):
        err(f"'kernels' {tuple(int(k) for k in kernels)} needs one entry per 'filters' "
            f"entry {cfg['filters']}")
    cfg['kernels'] = tuple(int(k) for k in kernels)
    dense = () if cfg['dense'] is None else np.atleast_1d(cfg['dense'])
    cfg['dense'] = tuple(int(u) for u in dense)
    try:
        cfg['wl_range'] = _wl_ranges(cfg['wl_range'])
    except ValueError as e:
        err(str(e))
    n_use = n_wl
    if cfg['wl_range'] is not None:
        if wavelength is None:
            err("'wl_range' needs the wavelength grid (pass wavelength= to resolve_config)")
        n_use = select_channels(wavelength, cfg['wl_range']).size
        if n_use == 0:
            err(f"'wl_range' {_fmt_wl(cfg['wl_range'])} contains none of the LUT wavelengths "
                f"({np.min(wavelength):.0f}-{np.max(wavelength):.0f} nm)")
    if n_use // 2 ** len(cfg['filters']) < 1:
        err(f"{len(cfg['filters'])} pooling steps are too many for {n_use} wavelengths")
    if not 0 <= cfg['dropout'] < 1:
        err("'dropout' must be in [0, 1)")
    for key in ('learning_rate', 't_floor'):
        if cfg[key] <= 0:
            err(f"'{key}' must be positive")
    for key in ('batch_size', 'epochs', 'patience'):
        if int(cfg[key]) < 1:
            err(f"'{key}' must be at least 1")
        cfg[key] = int(cfg[key])
    for key in ('noise_relative', 'noise_absolute'):
        try:
            cfg[key] = _noise_array(cfg[key], n_wl, f"'{key}'")
        except ValueError as e:
            err(str(e))
    for key in ('noise_gain', 'noise_scale', 'dropout', 'l2', 'learning_rate', 't_floor'):
        cfg[key] = float(cfg[key])
        if cfg[key] < 0:
            err(f"'{key}' must not be negative")
    cfg['log_input'] = bool(cfg['log_input'])
    cfg['seed'] = int(cfg['seed'])
    return name, cfg


def _noise_array(value, n_wl, what):
    """
    A noise level: a scalar, or one value per wavelength in the LUT's wavelength
    order. Extra length-1 dimensions, e.g. a (n_wl, 1) DataFrame column, are dropped.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        out = float(arr)
    elif arr.shape == (n_wl,):
        out = arr.tolist()
    else:
        raise ValueError(f"{what} must be a scalar or a 1-D array with one value per wavelength, "
                         f"shape ({n_wl},); got shape {np.shape(value)}")
    if not np.all(np.isfinite(arr)):
        bad = np.flatnonzero(~np.isfinite(np.atleast_1d(arr)))
        raise ValueError(f"{what} contains NaN or inf at {bad.size} wavelength index(es), "
                         f"e.g. {bad[:5].tolist()}")
    if np.any(arr < 0):
        raise ValueError(f"{what} must not be negative")
    return out


def _wl_ranges(value):
    """None, or a list of [lo, hi] (floats or None) from a pair or a list of pairs."""
    if value is None:
        return None
    if np.isscalar(value):
        raise ValueError(f"'wl_range' must be (lo, hi) or a list of (lo, hi), got {value}; "
                         f"e.g. (None, 1700) for wavelengths up to 1700 nm")
    seq = list(value)
    pairs = [seq] if len(seq) == 2 and all(v is None or np.isscalar(v) for v in seq) else seq
    out = []
    for pair in pairs:
        if np.isscalar(pair) or len(pair) != 2:
            raise ValueError(f"'wl_range' must be (lo, hi) or a list of (lo, hi), got {value}")
        lo, hi = (None if v is None else float(v) for v in pair)
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"'wl_range': lower limit {lo:g} must be below upper limit {hi:g}")
        out.append([lo, hi])
    return out or None


def select_channels(wavelength, wl_range):
    """Indices of the wavelengths inside wl_range (all of them for None)."""
    wl = np.asarray(wavelength, dtype=float)
    ranges = _wl_ranges(wl_range)
    if ranges is None:
        return np.arange(wl.size)
    keep = np.zeros(wl.size, bool)
    for lo, hi in ranges:
        keep |= (wl >= (-np.inf if lo is None else lo)) & (wl <= (np.inf if hi is None else hi))
    return np.flatnonzero(keep)


def _fmt_wl(wl_range):
    if wl_range is None:
        return 'all'
    return '+'.join(f"{'min' if lo is None else format(lo, 'g')}-{'max' if hi is None else format(hi, 'g')}"
                    for lo, hi in _wl_ranges(wl_range))


def _noise_on_channels(cfg, channels):
    """Copy of cfg whose per-wavelength noise arrays are cut to `channels`."""
    out = dict(cfg)
    for key in ('noise_relative', 'noise_absolute'):
        if isinstance(cfg[key], (list, tuple, np.ndarray)):
            out[key] = np.asarray(cfg[key], dtype=float)[channels].tolist()
    return out


def _fmt(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return '-'.join(_fmt(x) for x in v) or 'none'
    if isinstance(v, (bool, np.bool_)):
        return 'on' if v else 'off'
    if isinstance(v, (float, np.floating)):
        return f'{v:g}'
    return str(v)


def expand_grid(prefix='', **options):
    """
    All combinations of the given settings as a list of experiments, e.g.

        expand_grid(filters=[(4, 8, 16), (16, 32, 64)], learning_rate=[1e-3, 3e-4])

    gives four experiments named 'f4-8-16_lr0.001', 'f4-8-16_lr0.0003',
    'f16-32-64_lr0.001' and 'f16-32-64_lr0.0003'. Settings not listed come
    from BASE_CONFIG.
    """
    for key, values in options.items():
        if key not in DEFAULT_CONFIG:
            raise ValueError(f"Unknown setting '{key}'. Valid settings: {sorted(DEFAULT_CONFIG)}")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"expand_grid: give a list of options for '{key}'")
        if key in ('filters', 'dense') and not all(isinstance(v, (list, tuple)) for v in values):
            raise ValueError(f"expand_grid: '{key}' needs a list of tuples, "
                             f"e.g. {key}=[(4, 8, 16), (16, 32, 64)]")
    keys = list(options)
    experiments = []
    for values in itertools.product(*(options[k] for k in keys)):
        parts = [f"{_SHORT.get(k, k)}{_fmt_wl(v) if k == 'wl_range' else _fmt(v)}"
                 for k, v in zip(keys, values)]
        name = '_'.join(([prefix] if prefix else []) + parts)
        experiments.append(dict(name=name, **dict(zip(keys, values))))
    return experiments


# ---------------------------------------------------------------------------
# 3. Noise, inputs and targets
# ---------------------------------------------------------------------------
def noise_from_config(cfg):
    return dict(relative=cfg['noise_relative'], absolute=cfg['noise_absolute'],
                gain=cfg['noise_gain'], scale=cfg['noise_scale'])


def _noise_terms(noise, n_wl):
    s = float(noise.get('scale', 1.0))
    rel = np.broadcast_to(np.asarray(noise['relative'], dtype=float), (n_wl,)) * s
    absolute = np.broadcast_to(np.asarray(noise['absolute'], dtype=float), (n_wl,)) * s
    return rel, absolute, float(noise['gain']) * s


def noise_draws(shape, seed):
    """Fixed standard-normal draws, so a noise realization can be reproduced."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape), rng.standard_normal(shape),
            rng.standard_normal((shape[0], 1)))


def add_noise(T, noise, draws):
    """
    Noisy copy of transmittance T (n, n_wl):
        [T * (1 + rel * z1) + abs * z2] * (1 + gain * z3)
    with z1, z2 per channel and z3 one draw per spectrum. Same formula as the
    training-time noise in make_train_dataset.
    """
    rel, absolute, gain = _noise_terms(noise, T.shape[1])
    z1, z2, z3 = draws
    return ((T * (1.0 + rel * z1) + absolute * z2) * (1.0 + gain * z3)).astype('float32')


def to_input(T, log_input=True, t_floor=1e-4):
    """(n, n_wl) transmittance -> (n, n_wl, 1) network input."""
    X = np.asarray(T, dtype='float32')
    if log_input:
        X = np.log(np.maximum(X, t_floor))
    return X[..., np.newaxis].astype('float32')


def _forward_targets(Y, log_targets=LOG_TARGETS):
    Z = np.array(Y, dtype=float, copy=True)
    for j, name in enumerate(TARGET_NAMES):
        if name in log_targets:
            Z[:, j] = np.log(Z[:, j])
    return Z


def fit_target_scaling(Y, M):
    """Mean/std of the (log-)targets over the defined training entries."""
    Z = _forward_targets(Y)
    mean = [float(Z[M[:, j], j].mean()) for j in range(len(TARGET_NAMES))]
    std = [max(float(Z[M[:, j], j].std()), 1e-12) for j in range(len(TARGET_NAMES))]
    return dict(target_names=list(TARGET_NAMES), log_targets=list(LOG_TARGETS),
                target_mean=mean, target_std=std)


def standardize(Y, M, scaling):
    Z = (_forward_targets(Y, scaling['log_targets']) - np.asarray(scaling['target_mean'])) \
        / np.asarray(scaling['target_std'])
    return np.where(M, Z, 0.0)


def pack_targets(Y, M, scaling):
    """Standardized targets followed by the mask: (n, 2 * n_targets). The
    masked loss receives both through y_true."""
    return np.concatenate([standardize(Y, M, scaling), M.astype(float)], axis=1).astype('float32')


def to_physical(Z, scaling):
    """Network output (standardized) -> physical units."""
    Y = np.asarray(Z, dtype=float) * np.asarray(scaling['target_std']) \
        + np.asarray(scaling['target_mean'])
    for j, name in enumerate(scaling['target_names']):
        if name in scaling['log_targets']:
            Y[:, j] = np.exp(Y[:, j])
        if name == 'f_ice':
            Y[:, j] = np.clip(Y[:, j], 0.0, 1.0)
    return Y


# ---------------------------------------------------------------------------
# 4. Model and training (TensorFlow)
# ---------------------------------------------------------------------------
def masked_mse(y_true, y_pred):
    """MSE per target over its defined entries only, averaged over targets."""
    tf, _ = _tf()
    n = len(TARGET_NAMES)
    y, m = y_true[:, :n], y_true[:, n:]
    se = tf.square(y - y_pred) * m
    per_target = tf.reduce_sum(se, axis=0) / tf.maximum(tf.reduce_sum(m, axis=0), 1.0)
    return tf.reduce_mean(per_target)


def build_model(cfg, n_wl, X_for_norm):
    tf, keras = _tf()
    layers = keras.layers
    # a single (global) mean/std keeps the spectral shape intact, unlike a
    # per-channel standardization, which would blow up noise in dark bands
    norm = layers.Normalization(axis=None, name='input_norm')
    norm.adapt(X_for_norm)

    reg = keras.regularizers.l2(cfg['l2']) if cfg['l2'] > 0 else None
    inp = keras.Input(shape=(n_wl, 1), name='spectrum')
    x = norm(inp)
    for i, (filters, kernel) in enumerate(zip(cfg['filters'], cfg['kernels'])):
        x = layers.Conv1D(filters, kernel, padding='same', activation='relu',
                          kernel_regularizer=reg, name=f'conv{i + 1}')(x)
        x = layers.MaxPooling1D(2, name=f'pool{i + 1}')(x)
    # Flatten (not global pooling): *where* in the spectrum a feature sits matters
    x = layers.Flatten(name='flatten')(x)
    if cfg['dropout'] > 0:
        x = layers.Dropout(cfg['dropout'])(x)
    for i, units in enumerate(cfg['dense']):
        x = layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f'dense{i + 1}')(x)
        if cfg['dropout'] > 0:
            x = layers.Dropout(cfg['dropout'])(x)
    out = layers.Dense(len(TARGET_NAMES), name='targets')(x)

    model = keras.Model(inp, out, name='cnn')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
                  loss=masked_mse)
    return model


def make_train_dataset(T, Yp, cfg):
    """tf.data pipeline that adds a fresh noise realization to every batch."""
    tf, _ = _tf()
    rel, absolute, gain = _noise_terms(noise_from_config(cfg), T.shape[1])
    rel_t = tf.constant(rel.astype('float32'))
    abs_t = tf.constant(absolute.astype('float32'))
    gain_f = float(gain)
    log_input, floor = cfg['log_input'], float(cfg['t_floor'])

    def _noisy_input(Tb, yb):
        shape = tf.shape(Tb)
        Tn = Tb * (1.0 + rel_t * tf.random.normal(shape))
        Tn = Tn + abs_t * tf.random.normal(shape)
        Tn = Tn * (1.0 + gain_f * tf.random.normal(tf.stack([shape[0], 1])))
        if log_input:
            Tn = tf.math.log(tf.maximum(Tn, floor))
        return Tn[..., tf.newaxis], yb

    # tf.data has its own thread pool (all visible cores by default): cap it
    options = tf.data.Options()
    options.threading.private_threadpool_size = N_THREADS
    options.threading.max_intra_op_parallelism = 1
    return (tf.data.Dataset.from_tensor_slices((T.astype('float32'), Yp))
            .shuffle(len(T), seed=cfg['seed'], reshuffle_each_iteration=True)
            .batch(cfg['batch_size'])
            .map(_noisy_input, num_parallel_calls=min(N_THREADS, 4))
            .prefetch(2)
            .with_options(options))


def train_model(ds, split, cfg, exp_dir, verbose=2, channels=None):
    """
    Train one configuration on the wavelength channels `channels` (default:
    all). The model with the lowest validation loss is written to
    exp_dir/model.keras. Returns (scaling, history, info).
    """
    tf, keras = _tf()
    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg['seed'])

    ch = np.arange(ds['n_wl']) if channels is None else np.asarray(channels)
    tr, va = split['train'], split['val']
    T, Y, M = ds['T'][:, ch], ds['Y'], ds['M']
    cfg = _noise_on_channels(cfg, ch)
    scaling = fit_target_scaling(Y[tr], M[tr])
    Yp_tr = pack_targets(Y[tr], M[tr], scaling)
    Yp_va = pack_targets(Y[va], M[va], scaling)

    # validation: one fixed noise realization with the experiment's own noise, drawn on
    # the full grid and then cut, so the kept channels get the same noise as without wl_range
    z1, z2, z3 = noise_draws((va.size, ds['n_wl']), cfg['seed'] + 1)
    T_va = add_noise(T[va], noise_from_config(cfg), (z1[:, ch], z2[:, ch], z3))
    X_va = to_input(T_va, cfg['log_input'], cfg['t_floor'])
    X_tr_clean = to_input(T[tr], cfg['log_input'], cfg['t_floor'])   # for the input normalization

    model = build_model(cfg, T.shape[1], X_tr_clean)
    if verbose:
        model.summary()
    model_path = os.path.join(exp_dir, 'model.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=max(1, cfg['patience'] // 3),
                                          min_lr=cfg['learning_rate'] / 100),
    ]
    t0 = time.time()
    fit = model.fit(make_train_dataset(T[tr], Yp_tr, cfg), validation_data=(X_va, Yp_va),
                    epochs=cfg['epochs'], callbacks=callbacks, verbose=verbose)
    history = {k: [float(v) for v in vals] for k, vals in fit.history.items()}
    val = np.asarray(history['val_loss'])
    info = dict(epochs_trained=int(val.size), best_epoch=int(np.argmin(val)) + 1,
                best_val_loss=float(val.min()), train_time_s=round(time.time() - t0, 1),
                n_params=int(model.count_params()), tensorflow=tf.__version__,
                keras=getattr(keras, '__version__', 'unknown'))
    return scaling, history, info


def load_trained_model(path):
    tf, keras = _tf()
    keras.backend.clear_session()
    return keras.models.load_model(path, compile=False)


def predict_std(model, X, batch_size=1024):
    """Standardized network output for inputs X (n, n_wl, 1)."""
    out = []
    for s in range(0, len(X), batch_size):
        xb = X[s:s + batch_size]
        if _TF is not None:
            xb = _TF[0].convert_to_tensor(xb)
        yb = model(xb, training=False)
        out.append(yb.numpy() if hasattr(yb, 'numpy') else np.asarray(yb))
    return np.concatenate(out).astype(float)


def input_gradients(model, X, batch_size=256):
    """d(standardized output j)/d(network input) per spectrum: (n_targets, n, n_wl)."""
    tf, _ = _tf()
    n_out = len(TARGET_NAMES)
    grads = np.zeros((n_out, X.shape[0], X.shape[1]), dtype='float32')
    for s in range(0, X.shape[0], batch_size):
        n_b = min(batch_size, X.shape[0] - s)
        xb = tf.convert_to_tensor(X[s:s + n_b])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xb)
            yb = model(xb, training=False)
            sums = [tf.reduce_sum(yb[:, j]) for j in range(n_out)]
        for j in range(n_out):
            grads[j, s:s + n_b] = tape.gradient(sums[j], xb).numpy()[..., 0]
        del tape
    return grads


# ---------------------------------------------------------------------------
# 5. Metrics and spectral importance
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred, relative=True):
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    pos = y_true > 0
    return dict(
        n=int(y_true.size),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        r2=float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        med_rel_err_pct=(float(np.median(np.abs(err[pos]) / y_true[pos]) * 100)
                         if (relative and pos.any()) else np.nan),
    )


# Tolerances for the combined scores. A number is an absolute tolerance in the
# parameter's units (um for r_eff, unitless for f_ice); a string ending in '%'
# is relative to the true value.
DEFAULT_TOLERANCES = dict(reff_liq=2.0, reff_ice=10.0, tau_total='20%', f_ice=0.1)


def parse_tolerances(tolerances=None):
    """
    Tolerances as {name: ('abs' | 'rel', value)}, missing parameters taken from
    DEFAULT_TOLERANCES. Accepts 2.0 (absolute), '20%' (relative) or an already
    parsed ('abs', 2.0).
    """
    raw = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    unknown = sorted(set(raw) - set(TARGET_NAMES))
    if unknown:
        raise ValueError(f"TOLERANCES: unknown parameter(s) {unknown}; use {TARGET_NAMES}")
    out = {}
    for name in TARGET_NAMES:
        v = raw[name]
        try:
            if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] in ('abs', 'rel'):
                kind, val = v[0], float(v[1])
            elif isinstance(v, str) and v.strip().endswith('%'):
                kind, val = 'rel', float(v.strip()[:-1]) / 100
            else:
                kind, val = 'abs', float(v)
        except (TypeError, ValueError):
            raise ValueError(f"TOLERANCES['{name}'] = {v!r}: use a number (absolute) "
                             f"or a string like '20%' (relative)") from None
        if not val > 0:
            raise ValueError(f"TOLERANCES['{name}'] must be positive, got {v!r}")
        if kind == 'rel' and name == 'f_ice':
            raise ValueError("TOLERANCES['f_ice'] must be absolute (the true f_ice can be 0), e.g. 0.1")
        out[name] = (kind, val)
    return out


def _fmt_tolerances(tol):
    return ', '.join(f"{n}={v * 100:g}%" if k == 'rel' else f"{n}={v:g}" for n, (k, v) in tol.items())


def combined_scores(Y_true, M, Y_pred, scaling, tolerances):
    """
    Scores of the four parameters together, each parameter over its defined
    test runs only.

    Per parameter:
      std_rmse        RMSE on the training scale (ln tau_total, the others
                      linear) divided by the training-set standard deviation,
                      i.e. the scaling of the training loss
      r2_std          R^2 on the same scale
      tol_norm_err    mean |error| / tolerance (1 = on average at the tolerance)
      within_tol_pct  share of retrievals within the tolerance
    Combined:
      mean_std_rmse     mean of std_rmse over the four parameters (lower = better)
      sum_r2            sum of r2_std, at most 4
      mean_tol_norm_err mean of tol_norm_err
      joint_success_pct share of test spectra with every defined parameter within
                        its tolerance at the same time
    """
    tol = parse_tolerances(tolerances)
    Zt = standardize(Y_true, M, scaling)
    Zp = standardize(Y_pred, np.ones_like(M, dtype=bool), scaling)
    per, ok = {}, np.ones(len(Y_true), bool)
    for j, name in enumerate(TARGET_NAMES):
        sel = M[:, j]
        if not sel.any():
            per[name] = dict(std_rmse=np.nan, r2_std=np.nan, tol_norm_err=np.nan,
                             within_tol_pct=np.nan)
            continue
        dz, zt = Zp[sel, j] - Zt[sel, j], Zt[sel, j]
        sst = np.sum((zt - zt.mean()) ** 2)
        kind, val = tol[name]
        err = np.abs(Y_pred[sel, j] - Y_true[sel, j])
        ratio = err / (val * np.abs(Y_true[sel, j]) if kind == 'rel' else val)
        ok[sel] &= ratio <= 1
        per[name] = dict(std_rmse=float(np.sqrt(np.mean(dz ** 2))),
                         r2_std=float(1.0 - np.sum(dz ** 2) / sst) if sst > 0 else np.nan,
                         tol_norm_err=float(np.mean(ratio)),
                         within_tol_pct=float(100 * np.mean(ratio <= 1)))
    comb = dict(n=int(len(Y_true)),
                mean_std_rmse=float(np.mean([p['std_rmse'] for p in per.values()])),
                sum_r2=float(np.sum([p['r2_std'] for p in per.values()])),
                mean_tol_norm_err=float(np.mean([p['tol_norm_err'] for p in per.values()])),
                joint_success_pct=float(100 * np.mean(ok)))
    return per, comb


def evaluate_predictions(Y_true, M, Y_pred, params, scaling=None, tolerances=None):
    """
    Metric rows for the four targets and the derived tau_liq / tau_ice. With
    `scaling`, the target rows also get the per-parameter parts of the combined
    scores, and a row with target 'combined' is added (see combined_scores).
    """
    rows = []
    for j, name in enumerate(TARGET_NAMES):
        sel = M[:, j]
        rows.append(dict(target=name, **_metrics(Y_true[sel, j], Y_pred[sel, j],
                                                 relative=name != 'f_ice')))
    tt = Y_pred[:, TARGET_NAMES.index('tau_total')]
    fi = Y_pred[:, TARGET_NAMES.index('f_ice')]
    rows.append(dict(target='tau_liq*', **_metrics(params['tau_liq'], tt * (1 - fi))))
    rows.append(dict(target='tau_ice*', **_metrics(params['tau_ice'], tt * fi)))
    if scaling is not None:
        tol = parse_tolerances(tolerances)
        per, comb = combined_scores(Y_true, M, Y_pred, scaling, tol)
        for r in rows[:len(TARGET_NAMES)]:
            r.update(per[r['target']])
        rows.append(dict(target='combined', **comb, tolerances=_fmt_tolerances(tol)))
    return rows


def gradient_importance(grads, T, M, scaling, log_input):
    """
    Summaries of the input gradients per target (defined test runs only).

    abs_mean : mean |d z / d ln T| (z = standardized target), for comparing
               spectral regions and targets
    resp_*   : change of the retrieved value for a +1 % change of the
               transmittance in one channel (mean, 25th and 75th percentile);
               in um for r_eff, % for tau_total, percentage points for f_ice
    """
    g = grads if log_input else grads * T[np.newaxis]   # -> d z / d ln T
    std = np.asarray(scaling['target_std'])
    out = {k: np.zeros((len(TARGET_NAMES), T.shape[1])) for k in
           ('abs_mean', 'resp_mean', 'resp_q25', 'resp_q75')}
    for j, name in enumerate(TARGET_NAMES):
        gj = g[j][M[:, j]]
        # tau_total is regressed as ln(tau): 0.01 * g * std = d ln(tau) -> x100 for %
        # f_ice: fraction -> x100 for percentage points
        factor = 0.01 * std[j] * (100.0 if name in ('tau_total', 'f_ice') else 1.0)
        resp = gj * factor
        out['abs_mean'][j] = np.mean(np.abs(gj), axis=0)
        out['resp_mean'][j] = resp.mean(axis=0)
        out['resp_q25'][j], out['resp_q75'][j] = np.percentile(resp, [25, 75], axis=0)
    return out


def band_permutation_importance(model, X, Y_true, M, scaling, window=8, repeats=3, seed=0):
    """
    Error increase when one band of `window` neighbouring channels is
    scrambled across the test spectra (each spectrum gets the band from a
    randomly chosen other spectrum). Measured as RMSE of the standardized
    targets (log space for tau_total), in % of the unscrambled RMSE.
    """
    Z_true = standardize(Y_true, M, scaling)

    def rmse(Zp):
        return np.array([np.sqrt(np.mean((Zp[M[:, j], j] - Z_true[M[:, j], j]) ** 2))
                         for j in range(len(TARGET_NAMES))])

    base = rmse(predict_std(model, X))
    n, n_wl = X.shape[0], X.shape[1]
    starts = np.arange(0, n_wl, window)
    increase = np.zeros((len(TARGET_NAMES), starts.size))
    rng = np.random.default_rng(seed)
    for w, s in enumerate(starts):
        e = min(s + window, n_wl)
        acc = np.zeros(len(TARGET_NAMES))
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, s:e] = X[rng.permutation(n), s:e]
            acc += rmse(predict_std(model, Xp))
        increase[:, w] = 100.0 * (acc / repeats - base) / base
    return dict(starts=starts, ends=np.minimum(starts + window, n_wl),
                increase_pct=increase, base_rmse=base)


def lut_mutual_information(ds, noise, results_dir, seed=0, t_floor=1e-4):
    """
    Model-free information content: mutual information between the (noisy)
    log-transmittance of each single channel and each target, over all LUT
    runs where the target is defined. Cached in results_dir.
    """
    key = json.dumps(_jsonable(dict(noise=noise, seed=seed, t_floor=t_floor,
                                    fingerprint=ds['fingerprint'])), sort_keys=True)
    path = os.path.join(results_dir, 'lut_mutual_information.npz')
    if os.path.exists(path):
        with np.load(path) as f:
            if str(f['key']) == key:
                return f['mi']
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    except ImportError:
        print("NOTE: scikit-learn is not installed, so the model-free mutual information "
              "is skipped (pip install scikit-learn to add it).")
        return None
    print("Computing the mutual information of the LUT (once per results folder) ...")
    X = to_input(add_noise(ds['T'], noise, noise_draws(ds['T'].shape, seed)),
                 True, t_floor)[..., 0]
    Z = _forward_targets(ds['Y'])
    mi = np.zeros((len(TARGET_NAMES), ds['n_wl']))
    for j, name in enumerate(TARGET_NAMES):
        sel = ds['M'][:, j]
        y = Z[sel, j]
        uniq = np.unique(y)
        if uniq.size <= 10:   # few grid values: estimator for a discrete target
            mi[j] = mutual_info_classif(X[sel], np.searchsorted(uniq, y),
                                        n_neighbors=5, random_state=seed)
        else:
            mi[j] = mutual_info_regression(X[sel], y, n_neighbors=5, random_state=seed)
    np.savez(path, mi=mi, key=key)
    return mi


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def _save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def _fig_legend(fig, handles, ncol, fallback_ax, where='upper'):
    try:
        fig.legend(handles=handles, loc=f'outside {where} center', ncol=ncol)
    except ValueError:   # matplotlib < 3.7
        fallback_ax.legend(handles=handles, loc='upper right', ncol=ncol)


def _running_mean(y, width=5):
    """Running mean along the last axis (edges averaged over fewer channels)."""
    if width <= 1:
        return y
    kernel = np.ones(width)
    valid = np.isfinite(y)
    conv = lambda a: np.apply_along_axis(lambda r: np.convolve(r, kernel, mode='same'), -1, a)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = conv(np.where(valid, y, 0.0)) / conv(valid.astype(float))
    return np.where(valid, out, np.nan)


def _shade_h2o(ax):
    for lo, hi in H2O_BANDS:
        ax.axvspan(lo, hi, color=BAND_COLOR, lw=0, zorder=0)


def _channel_edges(wl):
    return np.concatenate([[wl[0]], 0.5 * (wl[1:] + wl[:-1]), [wl[-1]]])


def _plot_bands(ax, wl, starts, ends, values, **kw):
    """Step plot of per-band values over wavelength."""
    edges = _channel_edges(wl)
    band_edges = np.append(edges[starts], edges[ends[-1]])
    if np.all(np.diff(band_edges) > 0):
        ax.stairs(values, band_edges, baseline=None, **kw)
    else:
        centers = 0.5 * (wl[starts] + wl[ends - 1])
        ax.plot(centers, values, marker='o', ms=3, **kw)


def _one_to_one(ax, yt, yp, log=False, name=''):
    both = np.concatenate([yt, yp])
    if log:
        both = both[both > 0]
        lo, hi = both.min() / 1.3, both.max() * 1.3
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif name == 'f_ice':
        lo, hi = -0.04, 1.04
    else:
        pad = 0.05 * (both.max() - both.min())
        lo, hi = both.min() - pad, both.max() + pad
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')


def _metric_text(name, m, mc, ref_scale):
    unit = UNITS.get(name, '')
    unit = f' {unit}' if unit else ''
    lines = []
    for lab, mm in ((f'noise ×{ref_scale:g}', m), ('clean', mc)):
        if name == 'tau_total':
            lines.append(f"{lab}: median |rel. err.| {mm['med_rel_err_pct']:.1f} %, "
                         f"R² {mm['r2']:.3f}")
        else:
            lines.append(f"{lab}: MAE {mm['mae']:.3g}{unit}, bias {mm['bias']:+.2g}, "
                         f"R² {mm['r2']:.3f}")
    return '\n'.join(lines)


def plot_loss(history, info, path, title):
    with plt.rc_context(STYLE):
        lr_key = next((k for k in ('learning_rate', 'lr') if k in history), None)
        nrows = 2 if lr_key else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(6.6, 4.8 if lr_key else 3.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1][:nrows]), squeeze=False)
        ax = axes[0, 0]
        ep = np.arange(1, len(history['loss']) + 1)
        ax.plot(ep, history['loss'], color=SERIES_COLORS[0], label='training (fresh noise every epoch)')
        ax.plot(ep, history['val_loss'], color=SERIES_COLORS[1], label='validation (fixed noise)')
        ax.axvline(info['best_epoch'], color=MUTED, lw=0.8)
        ax.annotate(f"best epoch {info['best_epoch']}", xy=(info['best_epoch'], 0),
                    xycoords=('data', 'axes fraction'), xytext=(4, 4),
                    textcoords='offset points', va='bottom', fontsize=8, color=INK_2)
        ax.set_yscale('log')
        ax.set_ylabel('masked MSE (standardized targets)')
        ax.legend(loc='upper right')
        ax.set_title(title, loc='left')
        if lr_key:
            axes[1, 0].plot(ep, history[lr_key], color=INK_2, lw=1.2)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_ylabel('learning rate')
        axes[-1, 0].set_xlabel('epoch')
        _save(fig, path)


def plot_pred_vs_true(Y_true, M, Y_ref, Y_clean, ref_scale, path, title):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout='constrained')
        for j, (ax, name) in enumerate(zip(axes.flat, TARGET_NAMES)):
            sel = M[:, j]
            yt, yp, yc = Y_true[sel, j], Y_ref[sel, j], Y_clean[sel, j]
            ax.scatter(yt, yp, s=10, color=SERIES_COLORS[0], alpha=0.55, lw=0, zorder=2)
            _one_to_one(ax, yt, yp, log=name in LOG_TARGETS, name=name)
            rel = name != 'f_ice'
            ax.text(0.03, 0.97,
                    _metric_text(name, _metrics(yt, yp, rel), _metrics(yt, yc, rel), ref_scale),
                    transform=ax.transAxes, va='top', ha='left', fontsize=7.5, color=INK,
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GRID_COLOR, lw=0.6))
            ax.set_xlabel('true ' + _label(name))
            ax.set_ylabel('retrieved ' + _label(name))
        fig.suptitle(title, x=0.01, ha='left', fontsize=11)
        _save(fig, path)


def _color_scale(values_all, defined_all, name):
    """Colour mapping for one parameter, built on all LUT runs so it is the
    same in every figure and experiment."""
    v = values_all[defined_all]
    uniq = np.unique(v)
    if uniq.size <= 10:   # grid parameter: one colour per grid value
        cmap = mcolors.ListedColormap(SCATTER_CMAP(np.linspace(0, 1, max(uniq.size, 2)))[:uniq.size])
        norm = mcolors.BoundaryNorm(np.arange(uniq.size + 1) - 0.5, uniq.size)
        return dict(cmap=cmap, norm=norm, value=lambda x: np.searchsorted(uniq, x),
                    ticks=np.arange(uniq.size), ticklabels=[f'{u:g}' for u in uniq])
    if name in LOG_SCALE_PARAMS and v.min() > 0:
        norm = mcolors.LogNorm(v.min(), v.max())
    else:
        norm = mcolors.Normalize(v.min(), v.max())
    return dict(cmap=SCATTER_CMAP, norm=norm, value=lambda x: x, ticks=None, ticklabels=None)


def plot_pred_vs_true_colored(Y_true, M, Y_pred, params, defined, params_all, defined_all,
                              color_by, ref_scale, path, title):
    """Rows: retrieved target. Columns: the parameter used for the colour."""
    nr, nc = len(TARGET_NAMES), len(color_by)
    scales = {c: _color_scale(params_all[c], defined_all[c], c) for c in color_by}
    rng = np.random.default_rng(0)
    any_undefined = False
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc + 0.4, 2.55 * nr + 1.4),
                                 layout='constrained', squeeze=False)
        for i, name in enumerate(TARGET_NAMES):
            idx = rng.permutation(np.where(M[:, i])[0])   # random draw order
            yt, yp = Y_true[idx, i], Y_pred[idx, i]
            for k, c in enumerate(color_by):
                ax = axes[i, k]
                if c == name:
                    ax.scatter(yt, yp, s=7, color=MUTED, lw=0, alpha=0.7, zorder=2)
                    ax.set_title('(colour = same parameter)', fontsize=8, color=MUTED)
                else:
                    cdef = defined[c][idx]
                    if np.any(~cdef):
                        any_undefined = True
                        ax.scatter(yt[~cdef], yp[~cdef], s=7, color=UNDEFINED_COLOR, lw=0, zorder=2)
                    sc = scales[c]
                    ax.scatter(yt[cdef], yp[cdef], c=sc['value'](params[c][idx][cdef]),
                               cmap=sc['cmap'], norm=sc['norm'], s=7, lw=0, alpha=0.85, zorder=3)
                _one_to_one(ax, yt, yp, log=name in LOG_TARGETS, name=name)
                ax.set_xlabel('true ' + _label(name), fontsize=8)
                if k == 0:
                    ax.set_ylabel('retrieved ' + _label(name))
                if i == 0 and c != name:
                    ax.set_title(f'colour: {LABELS[c]}', fontsize=9)
        for k, c in enumerate(color_by):
            sc = scales[c]
            cb = fig.colorbar(ScalarMappable(norm=sc['norm'], cmap=sc['cmap']), ax=axes[:, k],
                              location='bottom', shrink=0.9, aspect=22, pad=0.01)
            cb.set_label(_label(c), fontsize=9)
            cb.outline.set_linewidth(0.5)
            if sc['ticks'] is not None:
                cb.set_ticks(sc['ticks'])
                cb.set_ticklabels(sc['ticklabels'])
            cb.ax.tick_params(labelsize=7.5)
        sub = (f"\nlight grey: colour parameter undefined (no liquid for {LABELS['reff_liq']}, "
               f"no ice for {LABELS['reff_ice']})") if any_undefined else ''
        fig.suptitle(f'{title}{sub}', x=0.01, ha='left', fontsize=10)
        _save(fig, path)


def plot_error_vs_parameters(Y_true, M, Y_pred, params, path, title):
    """Error of each target (rows) grouped by the LUT grid values (columns)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES), len(GRID_PARAMS), figsize=(12, 10),
                                 sharey='row', layout='constrained')
        for i, name in enumerate(TARGET_NAMES):
            sel = M[:, i]
            yt, yp = Y_true[sel, i], Y_pred[sel, i]
            if name in LOG_TARGETS:   # symmetric for over/underestimation: +0.3 = factor 2
                err, ylab = np.log10(yp / yt), 'log$_{10}$(retrieved / true)'
            else:
                u = UNITS.get(name)
                err, ylab = yp - yt, f'error ({u})' if u else 'error'
            for k, g in enumerate(GRID_PARAMS):
                ax = axes[i, k]
                gv = params[g][sel]
                uniq = np.unique(gv)
                ax.boxplot([err[gv == u] for u in uniq], positions=np.arange(uniq.size),
                           widths=0.6, showfliers=False, patch_artist=True,
                           medianprops=dict(color=INK, lw=1.2),
                           boxprops=dict(facecolor=SERIES_COLORS[0] + '40',
                                         edgecolor=SERIES_COLORS[0], lw=0.9),
                           whiskerprops=dict(color=SERIES_COLORS[0], lw=0.9),
                           capprops=dict(color=SERIES_COLORS[0], lw=0.9))
                ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
                ax.set_xticks(np.arange(uniq.size))
                ax.set_xticklabels([f'{u:g}' for u in uniq], fontsize=7)
                ax.grid(axis='x', visible=False)
                if i == len(TARGET_NAMES) - 1:
                    ax.set_xlabel('true ' + _label(g))
                if k == 0:
                    ax.set_ylabel(f'{LABELS[name]} {ylab}')
        fig.suptitle(f'{title}\nboxes: interquartile range, whiskers: 1.5 IQR, line: median',
                     x=0.01, ha='left', fontsize=10)
        _save(fig, path)


def plot_spectral_importance(wl, T_test, imp, perm, mi, path, title):
    """Mean test spectrum plus, per target, three normalized measures of where
    the information sits: LUT mutual information (model-free), CNN gradient,
    CNN band permutation."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES) + 1, 1, figsize=(10, 11.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[0.8] + [1] * len(TARGET_NAMES)),
                                 layout='constrained')
        ax0 = axes[0]
        _shade_h2o(ax0)
        q05, q95 = np.percentile(T_test, [5, 95], axis=0)
        ax0.fill_between(wl, q05, q95, color=INK_2, alpha=0.15, lw=0)
        ax0.plot(wl, T_test.mean(axis=0), color=INK_2, lw=1.2)
        ax0.set_ylabel('transmittance\n(test mean, 5–95 %)')
        for lo, hi in H2O_BANDS:
            if wl.min() <= 0.5 * (lo + hi) <= wl.max():     # only bands inside the plotted range
                ax0.text(0.5 * (lo + hi), 0.97, 'H$_2$O', transform=ax0.get_xaxis_transform(),
                         ha='center', va='top', fontsize=7.5, color=MUTED)

        handles = []
        if mi is not None:
            handles.append(Line2D([], [], color=SERIES_COLORS[0], lw=1.4,
                                  label='LUT mutual information (model-free)'))
        handles += [Line2D([], [], color=SERIES_COLORS[1], lw=1.2,
                           label=f'CNN mean |gradient| (clean test spectra, {GRADIENT_SMOOTHING}-channel mean)'),
                    Line2D([], [], color=SERIES_COLORS[2], lw=1.6,
                           label='CNN band permutation (noisy test spectra)')]
        grad_smooth = _running_mean(imp['abs_mean'], GRADIENT_SMOOTHING)
        for j, name in enumerate(TARGET_NAMES):
            ax = axes[j + 1]
            _shade_h2o(ax)
            if mi is not None and mi[j].max() > 0:
                ax.plot(wl, mi[j] / mi[j].max(), color=SERIES_COLORS[0], lw=1.4, zorder=3)
            ga = grad_smooth[j]
            if ga.max() > 0:
                ax.plot(wl, ga / ga.max(), color=SERIES_COLORS[1], lw=1.1, zorder=4)
            pv = np.clip(perm['increase_pct'][j], 0, None)
            if pv.max() > 0:
                _plot_bands(ax, wl, perm['starts'], perm['ends'], pv / pv.max(),
                            color=SERIES_COLORS[2], lw=1.6, zorder=5)
            ax.set_ylim(0, 1.08)
            ax.set_ylabel(f'{LABELS[name]}\n(normalized)')
        axes[-1].set_xlabel('wavelength (nm)')
        axes[-1].set_xlim(wl.min(), wl.max())
        fig.suptitle(f'{title}\nwhere the information sits: each curve scaled to its maximum; '
                     f'shaded: strong H$_2$O absorption', x=0.01, ha='left', fontsize=10)
        _fig_legend(fig, handles, ncol=3, fallback_ax=axes[1], where='lower')
        _save(fig, path)


def plot_saliency(wl, imp, path, title):
    """Signed response of the retrieval to a +1 % change of transmittance in
    one channel (mean and interquartile range over the test spectra)."""
    unit = dict(reff_liq='µm', reff_ice='µm', tau_total='%', f_ice='% points')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES), 1, figsize=(10, 9), sharex=True,
                                 layout='constrained')
        for j, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
            _shade_h2o(ax)
            ax.fill_between(wl, imp['resp_q25'][j], imp['resp_q75'][j], color=SERIES_COLORS[0],
                            alpha=0.2, lw=0)
            ax.plot(wl, imp['resp_mean'][j], color=SERIES_COLORS[0], lw=1.1)
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_ylabel(f'Δ{LABELS[name]} ({unit[name]})')
        axes[-1].set_xlabel('wavelength (nm)')
        axes[-1].set_xlim(wl.min(), wl.max())
        fig.suptitle(f'{title}\nchange of the retrieved value for +1 % transmittance in one '
                     f'channel (line: mean, band: interquartile range over clean test spectra)',
                     x=0.01, ha='left', fontsize=10)
        _save(fig, path)


def plot_permutation(wl, perm, path, title):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES), 1, figsize=(10, 9), sharex=True,
                                 layout='constrained')
        width = int(perm['ends'][0] - perm['starts'][0])
        for j, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
            _shade_h2o(ax)
            _plot_bands(ax, wl, perm['starts'], perm['ends'], perm['increase_pct'][j],
                        color=SERIES_COLORS[0], lw=1.4)
            ax.axhline(0, color=INK_2, lw=0.8)
            ax.set_ylabel(f'{LABELS[name]}\nRMSE increase (%)')
        axes[-1].set_xlabel('wavelength (nm)')
        axes[-1].set_xlim(wl.min(), wl.max())
        fig.suptitle(f'{title}\nerror increase when a band of {width} channels is scrambled '
                     f'across the test spectra', x=0.01, ha='left', fontsize=10)
        _save(fig, path)


def plot_lut_mi(wl, T_all, mi, path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES) + 1, 1, figsize=(10, 10.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[0.8] + [1] * len(TARGET_NAMES)),
                                 layout='constrained')
        _shade_h2o(axes[0])
        q05, q95 = np.percentile(T_all, [5, 95], axis=0)
        axes[0].fill_between(wl, q05, q95, color=INK_2, alpha=0.15, lw=0)
        axes[0].plot(wl, T_all.mean(axis=0), color=INK_2, lw=1.2)
        axes[0].set_ylabel('transmittance\n(LUT mean, 5–95 %)')
        for j, name in enumerate(TARGET_NAMES):
            ax = axes[j + 1]
            _shade_h2o(ax)
            ax.plot(wl, mi[j], color=SERIES_COLORS[0], lw=1.3)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f'{LABELS[name]}\nMI (nats)')
        axes[-1].set_xlabel('wavelength (nm)')
        axes[-1].set_xlim(wl.min(), wl.max())
        fig.suptitle('Model-free information content of the LUT: mutual information between the '
                     'noisy log-transmittance of one channel and each target',
                     x=0.01, ha='left', fontsize=10)
        _save(fig, path)


# ---- cross-experiment summary figures ------------------------------------
HEADLINE = dict(reff_liq=('mae', 'MAE (µm)'), reff_ice=('mae', 'MAE (µm)'),
                tau_total=('med_rel_err_pct', 'median |rel. error| (%)'),
                f_ice=('mae', 'MAE'))


def _exp_colors(names):
    return {n: (SERIES_COLORS[i] if i < len(SERIES_COLORS) else OTHER_COLOR)
            for i, n in enumerate(names)}


def _exp_legend_handles(names, colors):
    handles = [Line2D([], [], color=colors[n], lw=1.6, label=n)
               for n in names[:len(SERIES_COLORS)]]
    if len(names) > len(SERIES_COLORS):
        handles.append(Line2D([], [], color=OTHER_COLOR, lw=1.0,
                              label=f'{len(names) - len(SERIES_COLORS)} further experiments'))
    return handles


def _noise_label(test_noise):
    return 'own training noise' if test_noise == 'own' else 'common test noise'


def plot_summary_metrics(df, names, ref_scale, path, noise_label='test noise'):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(13, 0.34 * len(names) + 1.9),
                                 sharey=True, layout='constrained')
        y = np.arange(len(names))
        has_clean = np.any(np.isclose(df['eval_noise_scale'], 0.0))
        for ax, name in zip(axes, TARGET_NAMES):
            metric, xlabel = HEADLINE[name]
            d = df[df['target'] == name]
            ref = (d[np.isclose(d['eval_noise_scale'], ref_scale)]
                   .set_index('experiment').reindex(names)[metric].values)
            ax.barh(y, ref, height=0.62, color=SERIES_COLORS[0], zorder=2)
            if has_clean:
                clean = (d[np.isclose(d['eval_noise_scale'], 0.0)]
                         .set_index('experiment').reindex(names)[metric].values)
                ax.scatter(clean, y, s=18, color=INK, zorder=3)
            ax.set_title(_label(name, unit=False), fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_xlim(left=0)
            ax.grid(axis='y', visible=False)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        handles = [Patch(color=SERIES_COLORS[0], label=f'{noise_label} ×{ref_scale:g}')]
        if has_clean:
            handles.append(Line2D([], [], color=INK, marker='o', lw=0, ms=5,
                                  label='clean test spectra'))
        _fig_legend(fig, handles, ncol=2, fallback_ax=axes[-1])
        _save(fig, path)


COMBINED_SCORES = (
    ('mean_std_rmse', 'mean standardized RMSE', 'lower is better'),
    ('sum_r2', 'sum of R² (max 4)', 'higher is better'),
    ('mean_tol_norm_err', 'tolerance-normalized error', '1 = at the tolerance; lower is better'),
    ('joint_success_pct', 'joint success rate (%)', 'all parameters within tolerance'),
)


def plot_summary_combined(df, names, ref_scale, path):
    """The four combined scores per experiment, for the common test noise and,
    if evaluated, each experiment's own training noise."""
    d = df[df['target'] == 'combined']
    has_own = bool((d['test_noise'] == 'own').any())
    has_clean = bool(np.any(np.isclose(d['eval_noise_scale'], 0.0)))

    def values(test_noise, scale, metric):
        sel = d[(d['test_noise'] == test_noise) & np.isclose(d['eval_noise_scale'], scale)]
        return sel.set_index('experiment').reindex(names)[metric].to_numpy(dtype=float)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(COMBINED_SCORES), figsize=(13, 0.4 * len(names) + 2.0),
                                 sharey=True, layout='constrained')
        y = np.arange(len(names))
        h = 0.38 if has_own else 0.62
        for ax, (metric, title, hint) in zip(axes, COMBINED_SCORES):
            vals = [values('common', ref_scale, metric)]
            if has_own:
                vals.append(values('own', ref_scale, metric))
                ax.barh(y - h / 2 - 0.01, vals[0], height=h, color=SERIES_COLORS[0], zorder=2)
                ax.barh(y + h / 2 + 0.01, vals[1], height=h, color=SERIES_COLORS[1], zorder=2)
            else:
                ax.barh(y, vals[0], height=h, color=SERIES_COLORS[0], zorder=2)
            if has_clean:
                vals.append(values('common', 0.0, metric))
                ax.scatter(vals[-1], y, s=18, color=INK, zorder=3)
            allv = np.concatenate(vals)
            allv = allv[np.isfinite(allv)]
            if metric == 'sum_r2':
                # axis down to -4 at most (every parameter at R^2 = -1); lower bars are cut
                # at the edge and labelled with their value
                left = max(min(0.0, allv.min()) if allv.size else 0.0, -4.0)
                ax.set_xlim(left, 4.0)
                for v_set, dy in zip(vals[:2], (-h / 2, h / 2) if has_own else (0.0,)):
                    for yy, v in zip(y + dy, v_set):
                        if np.isfinite(v) and v < left:
                            ax.text(left, yy, f' {v:.3g}', va='center', ha='left',
                                    fontsize=7, color='white', zorder=4)
            elif metric == 'joint_success_pct':
                ax.set_xlim(0.0, 100.0)
            elif allv.size and allv.min() > 0 and allv.max() / allv.min() > 30:
                ax.set_xscale('log')    # values over orders of magnitude
                ax.set_xlim(left=allv.min() / 1.5)
            else:
                ax.set_xlim(left=0.0)
            if metric == 'mean_tol_norm_err':
                ax.axvline(1.0, color=MUTED, ls='--', lw=0.9, zorder=1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(hint)
            ax.grid(axis='y', visible=False)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        handles = [Patch(color=SERIES_COLORS[0], label=f'common test noise ×{ref_scale:g}')]
        if has_own:
            handles.append(Patch(color=SERIES_COLORS[1], label=f'own training noise ×{ref_scale:g}'))
        if has_clean:
            handles.append(Line2D([], [], color=INK, marker='o', lw=0, ms=5,
                                  label='clean test spectra'))
        _fig_legend(fig, handles, ncol=len(handles), fallback_ax=axes[-1])
        _save(fig, path)


def plot_summary_noise(df, names, path, noise_label='test noise'):
    colors = _exp_colors(names)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(13, 3.8), layout='constrained')
        for ax, name in zip(axes, TARGET_NAMES):
            metric, ylabel = HEADLINE[name]
            d = df[df['target'] == name]
            for n in names[::-1]:   # first experiments drawn on top
                dn = d[d['experiment'] == n].sort_values('eval_noise_scale')
                colored = colors[n] != OTHER_COLOR
                ax.plot(dn['eval_noise_scale'], dn[metric], color=colors[n],
                        lw=1.6 if colored else 0.9, marker='o', ms=4 if colored else 0,
                        zorder=3 if colored else 2)
            ax.set_title(_label(name, unit=False), fontsize=10)
            ax.set_xlabel(f'{noise_label} scale')
            ax.set_ylabel(ylabel)
        _fig_legend(fig, _exp_legend_handles(names, colors), ncol=min(len(names), 4),
                    fallback_ax=axes[-1])
        _save(fig, path)


def plot_summary_loss(histories, names, path):
    colors = _exp_colors(names)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
        for n in names[::-1]:
            h = histories[n]
            colored = colors[n] != OTHER_COLOR
            ax.plot(np.arange(1, len(h['val_loss']) + 1), h['val_loss'], color=colors[n],
                    lw=1.4 if colored else 0.8, zorder=3 if colored else 2)
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('validation loss (masked MSE)')
        _fig_legend(fig, _exp_legend_handles(names, colors), ncol=min(len(names), 4),
                    fallback_ax=ax)
        _save(fig, path)


def plot_summary_saliency(wl, importances, names, path):
    """Normalized mean |gradient| per experiment (rows) and wavelength."""
    edges = _channel_edges(wl)
    x_edges = edges if np.all(np.diff(edges) > 0) else np.arange(wl.size + 1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(TARGET_NAMES), 1, sharex=True, layout='constrained',
                                 figsize=(10, len(TARGET_NAMES) * (0.24 * len(names) + 0.9) + 0.8))
        for j, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
            vals = _running_mean(np.array([importances[n]['abs_mean'][j] for n in names]),
                                 GRADIENT_SMOOTHING)
            vals = vals / np.maximum(np.nanmax(vals, axis=1, keepdims=True), 1e-30)
            mesh = ax.pcolormesh(x_edges, np.arange(len(names) + 1), np.ma.masked_invalid(vals),
                                 cmap=HEATMAP_CMAP,
                                 vmin=0, vmax=1, shading='flat')
            ax.set_yticks(np.arange(len(names)) + 0.5)
            ax.set_yticklabels(names, fontsize=7.5)
            ax.invert_yaxis()
            ax.grid(False)
            ax.set_title(_label(name, unit=False), loc='left', fontsize=10)
        axes[-1].set_xlabel('wavelength (nm)' if x_edges is edges else 'channel index')
        cb = fig.colorbar(mesh, ax=axes, shrink=0.6, aspect=30)
        cb.set_label(f'mean |gradient| ({GRADIENT_SMOOTHING}-channel mean), scaled to each row’s maximum')
        cb.outline.set_linewidth(0.5)
        fig.suptitle('Which wavelengths each trained CNN relies on', x=0.01, ha='left', fontsize=11)
        _save(fig, path)


# ---------------------------------------------------------------------------
# 7. Running experiments
# ---------------------------------------------------------------------------
_EXP_FILES = ('model.keras', 'config.json', 'history.json', 'metrics.csv',
              'predictions.npz', 'importance.npz')


def _train_or_load(name, cfg, ds, split, exp_dir, mode, verbose):
    cfg_path = os.path.join(exp_dir, 'config.json')
    model_path = os.path.join(exp_dir, 'model.keras')
    hist_path = os.path.join(exp_dir, 'history.json')
    saved = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
    complete = saved is not None and os.path.exists(model_path) and os.path.exists(hist_path)
    if complete:
        # settings added to the framework later count as their default for older models
        saved['config'] = {**_jsonable({k: v for k, v in DEFAULT_CONFIG.items()
                                        if k not in saved['config']}), **saved['config']}
    same_split = complete and saved.get('split_id') == split['id']
    same = same_split and saved['config'] == _jsonable(cfg)

    if mode == 'evaluate':
        if not complete:
            raise RuntimeError(f"No trained model in {exp_dir} (MODE = 'evaluate').")
        if not same_split:
            raise RuntimeError(f"'{name}' was trained on a different split; its test runs "
                               f"may have been used for training. Retrain it.")
        if not same:
            print(f"  NOTE: '{name}' was trained with a different configuration than listed "
                  f"now; evaluating the saved model with its saved configuration.")
        cfg = saved['config']
    ch = select_channels(ds['wavelength'], cfg['wl_range'])
    if ch.size < ds['n_wl']:
        print(f"  using {ch.size} of {ds['n_wl']} wavelengths "
              f"({ds['wavelength'][ch].min():.0f}-{ds['wavelength'][ch].max():.0f} nm)")
    if mode == 'auto' and same:
        print(f"  unchanged configuration: reusing the trained model")
    elif mode != 'evaluate':
        if complete and mode == 'auto':
            print(f"  configuration changed: retraining (the old model in {exp_dir} is replaced)")
        for fname in _EXP_FILES:
            fpath = os.path.join(exp_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        scaling, history, info = train_model(ds, split, cfg, exp_dir, verbose=verbose, channels=ch)
        with open(hist_path, 'w') as f:
            json.dump(history, f)
        saved = dict(name=name, config=_jsonable(cfg), split_id=split['id'],
                     data_fingerprint=ds['fingerprint'], scaling=scaling, training=info,
                     channels=ch.tolist(), wavelength=ds['wavelength'][ch].tolist())
        with open(cfg_path, 'w') as f:   # written last: marks the experiment as complete
            json.dump(saved, f, indent=2)
        print(f"  trained {info['epochs_trained']} epochs (best {info['best_epoch']}) "
              f"in {info['train_time_s']:.0f} s, {info['n_params']} parameters")

    with open(hist_path) as f:
        history = json.load(f)
    model = load_trained_model(model_path)
    return model, cfg, saved, history, ch


def _print_metrics(df, scales, ref, standard='common'):
    common = df[df['test_noise'] == standard]
    print(f"  test-set MAE at each {_noise_label(standard)} scale, R2 at x{ref:g}")
    print(f"  {'target':<10}" + ''.join(f"{'x' + format(s, 'g'):>10}" for s in scales)
          + f"{'R2':>8}")
    for tgt in TARGET_NAMES + ['tau_liq*', 'tau_ice*']:
        d = common[common['target'] == tgt].set_index('eval_noise_scale')
        print(f"  {tgt:<10}" + ''.join(f"{d.loc[s, 'mae']:10.3f}" for s in scales)
              + f"{d.loc[ref, 'r2']:8.3f}")
    print("  (* derived from the retrieved tau_total and f_ice)")

    comb = df[df['target'] == 'combined']
    if not len(comb):
        return
    cols = [('clean', 'common', 0.0), (f'common x{ref:g}', 'common', ref)]
    if (comb['test_noise'] == 'own').any():
        cols.append((f'own x{ref:g}', 'own', ref))
    print(f"\n  {'combined scores':<28}" + ''.join(f"{c[0]:>12}" for c in cols))
    for metric, label, _ in COMBINED_SCORES:
        vals = []
        for _, tn, sc in cols:
            v = comb[(comb['test_noise'] == tn) & np.isclose(comb['eval_noise_scale'], sc)][metric]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        print(f"  {label:<28}" + ''.join(f"{v:12.3f}" for v in vals))
    print(f"  tolerances: {comb['tolerances'].iloc[0]}")


def run_experiments(data, experiments, base_config, results_dir, test_fraction=0.2,
                    val_fraction=0.15, split_seed=42, eval_noise=None,
                    eval_noise_scales=(0.0, 0.5, 1.0, 2.0), eval_noise_ref=1.0, eval_seed=1234,
                    mode='auto', color_by=None, permutation_window=8, permutation_repeats=3,
                    make_figures=True, verbose=2, tolerances=None, eval_own_noise=True,
                    standard_test_noise='own'):
    """
    Train (or reuse), evaluate and plot every experiment, then write the
    cross-experiment summary. Returns the summary table (pandas DataFrame).

    mode : 'auto'     train if there is no saved model or the configuration
                      changed, otherwise reuse the saved model
           'train'    always retrain
           'evaluate' never train, only evaluate the saved models and plot
    eval_noise : common test noise, the same for every experiment
                 (None = the noise of base_config, i.e. the standard training noise)
    eval_own_noise : also test each experiment with its own training noise
                 (incl. noise_scale); same random draws as the common test noise
    tolerances : for the combined scores, see DEFAULT_TOLERANCES
    standard_test_noise : 'common' or 'own': the test noise used by the
                 per-experiment figures, the permutation importance, the MAE
                 tables, summary_metrics.png and summary_noise_robustness.png
                 (summary_combined_metrics.png always shows both)
    """
    if mode not in ('auto', 'train', 'evaluate'):
        raise ValueError("mode must be 'auto', 'train' or 'evaluate'")
    tol = parse_tolerances(tolerances)
    if standard_test_noise not in ('common', 'own'):
        raise ValueError("standard_test_noise must be 'common' or 'own'")
    if standard_test_noise == 'own' and not eval_own_noise:
        print("standard_test_noise = 'own' needs the own-noise test: switching eval_own_noise on")
        eval_own_noise = True
    std_label = _noise_label(standard_test_noise)
    base_abs = np.asarray({**DEFAULT_CONFIG, **base_config}['noise_absolute'], dtype=float)
    if not np.any(base_abs > 0):
        print("Note: base_config has no absolute noise; the standard is the measured dark noise "
              "(noise_absolute=sigma_T)")
    os.makedirs(results_dir, exist_ok=True)
    color_by = list(color_by or TARGET_NAMES)
    for c in color_by:
        if c not in ALL_PARAMS:
            raise ValueError(f"color_by: unknown parameter '{c}' (choose from {ALL_PARAMS})")

    print("\nPreparing the LUT")
    ds = prepare_dataset(data)
    wl = ds['wavelength']

    # resolve and check all configurations before any training starts
    resolved = [resolve_config(base_config, exp, ds['n_wl'], wl) for exp in experiments]
    names = [n for n, _ in resolved]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"Duplicate experiment names: {dupes}")
    split = get_split(ds, results_dir, test_fraction, val_fraction, split_seed)

    # common test noise: same draws for every experiment and noise scale
    if eval_noise is None:
        eval_noise = noise_from_config({**DEFAULT_CONFIG, **base_config})
    eval_noise = {k: eval_noise[k] for k in ('relative', 'absolute', 'gain')}
    for k in ('relative', 'absolute'):
        eval_noise[k] = _noise_array(eval_noise[k], ds['n_wl'], f"EVAL_NOISE['{k}']")
    eval_noise['gain'] = float(eval_noise['gain'])
    # clean (0) and the reference level are always evaluated
    eval_noise_ref = float(eval_noise_ref)
    scales = sorted({float(s) for s in eval_noise_scales} | {0.0, eval_noise_ref})
    te = split['test']
    T_te, Y_te, M_te = ds['T'][te], ds['Y'][te], ds['M'][te]
    params_te = {k: v[te] for k, v in ds['params'].items()}
    defined_te = {k: v[te] for k, v in ds['defined'].items()}
    draws = noise_draws(T_te.shape, eval_seed)
    T_eval = {s: add_noise(T_te, dict(eval_noise, scale=s), draws) for s in scales}
    print(f"Test noise: common (EVAL_NOISE)"
          + (" and each experiment's own training noise" if eval_own_noise else '')
          + f", scales {', '.join(format(s, 'g') for s in scales)}")
    print(f"Figures, permutation importance and MAE tables use the {std_label}")
    print(f"Tolerances for the combined scores: {_fmt_tolerances(tol)}")

    mi = lut_mutual_information(ds, dict(eval_noise, scale=eval_noise_ref), results_dir,
                                seed=eval_seed + 7)
    if mi is not None and make_figures:
        with plt.rc_context(STYLE):
            plot_lut_mi(wl, ds['T'], mi, os.path.join(results_dir, 'lut_mutual_information.png'))

    all_metrics, histories, importances, failed = [], {}, {}, []
    for k_exp, (name, cfg) in enumerate(resolved, start=1):
        print(f"\n=== Experiment {k_exp}/{len(resolved)}: {name} ===")
        exp_dir = os.path.join(results_dir, name)
        fig_dir = os.path.join(exp_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        try:
            model, cfg_used, record, history, ch = _train_or_load(name, cfg, ds, split, exp_dir,
                                                                  mode, verbose)
            scaling = record['scaling']
            log_input, t_floor = cfg_used['log_input'], cfg_used['t_floor']
            # the experiment's wavelengths; the test noise on them is the common one
            wl_x, T_te_x = wl[ch], T_te[:, ch]
            mi_x = None if mi is None else mi[:, ch]

            # predictions at every test noise level: common test noise ...
            preds, preds_own, rows = {}, {}, []
            for s in scales:
                Yp = to_physical(predict_std(model, to_input(T_eval[s][:, ch], log_input, t_floor)),
                                 scaling)
                preds[s] = Yp
                rows += [dict(experiment=name, test_noise='common', eval_noise_scale=s, **r)
                         for r in evaluate_predictions(Y_te, M_te, Yp, params_te, scaling, tol)]
            # ... and the experiment's own training noise (same draws), x1 = as in training
            T_own_ref = T_te_x
            if eval_own_noise:
                own = noise_from_config(_noise_on_channels(cfg_used, ch))
                draws_x = (draws[0][:, ch], draws[1][:, ch], draws[2])
                for s in scales:
                    if s == 0.0:
                        Yp = preds[0.0]     # clean spectra: identical to the common test
                    else:
                        T_own = add_noise(T_te_x, dict(own, scale=float(own['scale']) * s), draws_x)
                        Yp = to_physical(predict_std(model, to_input(T_own, log_input, t_floor)),
                                         scaling)
                        if np.isclose(s, eval_noise_ref):
                            T_own_ref = T_own
                    preds_own[s] = Yp
                    rows += [dict(experiment=name, test_noise='own', eval_noise_scale=s, **r)
                             for r in evaluate_predictions(Y_te, M_te, Yp, params_te, scaling, tol)]
            metrics = pd.DataFrame(rows)
            metrics.to_csv(os.path.join(exp_dir, 'metrics.csv'), index=False)
            np.savez_compressed(
                os.path.join(exp_dir, 'predictions.npz'), test_idx=te, Y_true=Y_te, mask=M_te,
                target_names=np.array(TARGET_NAMES), eval_noise_scales=np.array(scales),
                **{f'Y_pred_noise{s:g}': preds[s] for s in scales},
                **{f'Y_pred_own_noise{s:g}': preds_own[s] for s in preds_own})
            _print_metrics(metrics, scales, eval_noise_ref, standard_test_noise)
            std_preds = preds_own if standard_test_noise == 'own' else preds
            T_ref = T_own_ref if standard_test_noise == 'own' else T_eval[eval_noise_ref][:, ch]

            # where in the spectrum the CNN finds its information
            X_clean = to_input(T_te_x, log_input, t_floor)
            X_ref = to_input(T_ref, log_input, t_floor)
            imp = gradient_importance(input_gradients(model, X_clean), T_te_x, M_te, scaling,
                                      log_input)
            perm = band_permutation_importance(model, X_ref, Y_te, M_te, scaling,
                                               permutation_window, permutation_repeats,
                                               seed=eval_seed)
            np.savez_compressed(os.path.join(exp_dir, 'importance.npz'), wavelength=wl_x, channels=ch,
                                **{f'grad_{k}': v for k, v in imp.items()},
                                **{f'perm_{k}': v for k, v in perm.items()},
                                target_names=np.array(TARGET_NAMES))

            if make_figures:
                ttl = f'{name}: test set'
                Y_ref, Y_clean = std_preds[eval_noise_ref], preds[0.0]
                ref_ttl = f'{ttl}, {std_label} ×{eval_noise_ref:g}'
                plot_loss(history, record['training'], os.path.join(fig_dir, 'loss.png'), name)
                plot_pred_vs_true(Y_te, M_te, Y_ref, Y_clean, eval_noise_ref,
                                  os.path.join(fig_dir, 'pred_vs_true.png'), ref_ttl)
                # the other test noise as pred_vs_true_<other>.png; a file named after the
                # standard noise is left over from a run with the other setting
                other = 'own' if standard_test_noise == 'common' else 'common'
                stale = os.path.join(fig_dir, f'pred_vs_true_{standard_test_noise}.png')
                if os.path.exists(stale):
                    os.remove(stale)
                if eval_own_noise:
                    other_preds = preds_own if other == 'own' else preds
                    plot_pred_vs_true(Y_te, M_te, other_preds[eval_noise_ref], Y_clean,
                                      eval_noise_ref, os.path.join(fig_dir, f'pred_vs_true_{other}.png'),
                                      f'{ttl}, {_noise_label(other)} ×{eval_noise_ref:g}')
                plot_pred_vs_true_colored(Y_te, M_te, Y_ref, params_te, defined_te,
                                          ds['params'], ds['defined'], color_by, eval_noise_ref,
                                          os.path.join(fig_dir, 'pred_vs_true_colored.png'),
                                          ref_ttl)
                plot_error_vs_parameters(Y_te, M_te, Y_ref, params_te,
                                         os.path.join(fig_dir, 'error_vs_parameters.png'),
                                         ref_ttl)
                plot_spectral_importance(wl_x, T_te_x, imp, perm, mi_x,
                                         os.path.join(fig_dir, 'spectral_importance.png'), name)
                plot_saliency(wl_x, imp, os.path.join(fig_dir, 'saliency.png'), name)
                plot_permutation(wl_x, perm, os.path.join(fig_dir, 'permutation_importance.png'),
                                 f'{name}, {std_label} ×{eval_noise_ref:g}')
                print(f"  figures written to {fig_dir}")

            info = record.get('training', {})
            for col in ('epochs_trained', 'best_epoch', 'best_val_loss', 'train_time_s', 'n_params'):
                metrics[col] = info.get(col)
            for key, val in cfg_used.items():
                if key == 'wl_range':
                    val = _fmt_wl(val)
                elif isinstance(val, (list, tuple)):
                    val = 'per-wavelength' if len(val) > 10 else _fmt(val)
                metrics[f'cfg_{key}'] = val
            all_metrics.append(metrics)
            # on the full grid (NaN outside the experiment's range) for the summary figure
            imp_full = {}
            for k, v in imp.items():
                full = np.full((v.shape[0], ds['n_wl']), np.nan)
                full[:, ch] = v
                imp_full[k] = full
            histories[name], importances[name] = history, imp_full
            del model
        except Exception:
            traceback.print_exc()
            print(f"  Experiment '{name}' FAILED, continuing with the next one.")
            failed.append(name)

    done = [n for n in names if n in histories]
    if not done:
        print("\nNo experiment finished.")
        return None
    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(os.path.join(results_dir, 'summary_metrics.csv'), index=False)
    standard = summary[summary['test_noise'] == standard_test_noise]
    if make_figures:
        plot_summary_metrics(standard, done, eval_noise_ref,
                             os.path.join(results_dir, 'summary_metrics.png'), std_label)
        plot_summary_combined(summary, done, eval_noise_ref,
                              os.path.join(results_dir, 'summary_combined_metrics.png'))
        if len(scales) > 1:
            plot_summary_noise(standard, done, os.path.join(results_dir, 'summary_noise_robustness.png'),
                               std_label)
        plot_summary_loss(histories, done, os.path.join(results_dir, 'summary_loss.png'))
        plot_summary_saliency(wl, importances, done, os.path.join(results_dir, 'summary_saliency.png'))

    print(f"\nSummary written to {results_dir}")
    ref = standard[np.isclose(standard['eval_noise_scale'], eval_noise_ref)]
    table = ref.pivot(index='experiment', columns='target', values='mae').reindex(done)
    print(f"MAE at {std_label} x{eval_noise_ref:g}:")
    print(table[TARGET_NAMES].to_string(float_format=lambda v: f'{v:.3f}'))
    comb = summary[(summary['target'] == 'combined')
                   & np.isclose(summary['eval_noise_scale'], eval_noise_ref)]
    short = dict(mean_std_rmse='std_RMSE', sum_r2='sum_R2', mean_tol_norm_err='tol_err',
                 joint_success_pct='joint_%')
    parts = []
    for tn in ('common', 'own'):
        c = comb[comb['test_noise'] == tn]
        if len(c):
            parts.append(c.set_index('experiment').reindex(done)[list(short)]
                         .rename(columns=lambda m: f'{short[m]} {tn}'))
    print(f"\nCombined scores at test noise x{eval_noise_ref:g} (tolerances: {_fmt_tolerances(tol)}):")
    print(pd.concat(parts, axis=1).to_string(float_format=lambda v: f'{v:.3f}'))
    if failed:
        print(f"\nFailed experiments: {failed}")
    return summary


def predict_parameters(model, T, scaling, log_input=True, t_floor=1e-4, channels=None):
    """
    Apply a trained CNN to new spectra (e.g. measurements).

    T : (n_spectra, n_wl) solar-normalized transmittance on the LUT wavelength
        grid. `scaling`, `log_input` and `t_floor` come from the experiment's
        config.json ('scaling' and 'config' entries).
    channels : for a model trained with 'wl_range', the 'channels' entry of its
        config.json; full-grid spectra are then cut to these wavelengths.
    Returns reff_liq, reff_ice, tau_total, f_ice and the derived tau_liq, tau_ice.
    """
    T = np.atleast_2d(np.asarray(T, dtype='float32'))
    if channels is not None and T.shape[1] != len(channels):
        T = T[:, np.asarray(channels)]
    Y = to_physical(predict_std(model, to_input(T, log_input, t_floor)), scaling)
    out = {name: Y[:, j] for j, name in enumerate(scaling['target_names'])}
    out['tau_liq'] = out['tau_total'] * (1 - out['f_ice'])
    out['tau_ice'] = out['tau_total'] * out['f_ice']
    return out