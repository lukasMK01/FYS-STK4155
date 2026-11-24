#%%
"""
Run several CNN configurations on the mini-LUT and compare them.

Edit the settings below, list the experiments in EXPERIMENTS, and run:

    MPLBACKEND=Agg python miniLUT_runCNNexperiments.py 2>&1 | tee experiments.log

All machinery lives in cnn_framework.py (keep it in the same folder).
Each experiment gets its own folder in RESULTS_DIR with the model, metrics
and figures; comparison figures and summary_metrics.csv go to RESULTS_DIR.

Re-running is cheap: with MODE = 'auto', experiments whose configuration is
unchanged are not retrained, only re-evaluated and re-plotted. Adding an
experiment to the list trains only that one.
"""
import os

os.environ['CNN_N_THREADS'] = '8'   # CPU threads; lower to 4 if you get "pthread_create() failed"

import pandas as pd
import numpy as np
import miniLUT_CNNframework as cf

#%%
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# The prepared LUT (zenith-pixel transmittance + cloud parameters) is cached
# in a small .npz file. If the cache exists, only it is read (no xarray,
# fenix_functions or NIRD access needed). If not, it is built from the
# netCDF files below.
CACHE_PATH = './data/miniLUT_zenith_transmittance.npz'
LUT_PATH = './combined_miniLUT.nc'
SOLAR_PATH = './NewGuey2003_convolved.nc'
# time passed to compute_solar_angles, as in miniLUT_overview.py
# SOLAR_TIME = pd.Timestamp('2026-02-15T11:25:00') + pd.Timedelta(hours=1)
# cosine of the solar zenith angle of the LUT simulations (here 60 deg -> 0.5).
# Only used when the cache is built: after changing it, delete CACHE_PATH.
COS_ZEN = np.cos(np.deg2rad(60))

# True: build the cache and stop (e.g. on NIRD), then copy the .npz to the
# machine where you train
ONLY_BUILD_CACHE = False

RESULTS_DIR = './cnn_results_sigmaT'  # new folder: sigma_T is now the standard training noise

#%%
# ---------------------------------------------------------------------------
# Split and evaluation (shared by all experiments in RESULTS_DIR)
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.20   # share of runs held out for testing
VAL_FRACTION = 0.15    # share of the remaining runs used for early stopping
SPLIT_SEED = 42
# The split is stored in RESULTS_DIR. Changing these three values later
# requires a new RESULTS_DIR (or deleting split.npz and retraining everything).

# Test noise: identical for every experiment, so differences between
# experiments come from the models and not from the test data. Every model is
# evaluated at all EVAL_NOISE_SCALES (clean = 0 is always added); the
# per-experiment figures use EVAL_NOISE_REF.
# EVAL_NOISE (this common test noise) is set further down, after the measured
# dark noise sigma_T has been read.
EVAL_NOISE_SCALES = [0.0, 0.5, 1.0, 2.0]
EVAL_NOISE_REF = 1.0

# Also test every experiment with its own training noise (incl. noise_scale):
# x1 is then exactly the noise it was trained with. Reported next to the common
# test noise (column test_noise = 'common' / 'own' in metrics.csv and
# summary_metrics.csv), in summary_combined_metrics.png and in
# figures/pred_vs_true_own.png.
EVAL_OWN_NOISE = True

# Which test noise the per-experiment figures (pred_vs_true.png, ...), the
# permutation importance, the MAE tables, summary_metrics.png and
# summary_noise_robustness.png use: 'own' (each experiment's training noise)
# or 'common' (EVAL_NOISE). summary_combined_metrics.png always shows both,
# and the other one is drawn as figures/pred_vs_true_<other>.png.
STANDARD_TEST_NOISE = 'own'

# Tolerances for the combined scores (tolerance-normalized error and joint
# success rate): a number is absolute (um for r_eff, unitless for f_ice), a
# string with % is relative to the true value.
TOLERANCES = dict(reff_liq=2.0, reff_ice=10.0, tau_total='20%', f_ice=0.1)

#%%
# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
# 'auto'    : train new or changed experiments, reuse unchanged ones
# 'train'   : retrain everything in EXPERIMENTS
# 'evaluate': no training; re-evaluate saved models and redo all figures
MODE = 'auto'

# Parameters used to colour the retrieved-vs-true scatter plots (one column
# each). Also possible: 'tau_liq', 'tau_ice'.
COLOR_BY = ['reff_liq', 'reff_ice', 'tau_total', 'f_ice']

# Band-permutation importance: channels per scrambled band, and repeats
PERMUTATION_WINDOW = 8
PERMUTATION_REPEATS = 3

#%%
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# def read_noise():
#     """Read noise from precalculated file
#     This is the median of the standard deviations of the detrended dark measurements per wavelength channel."""
#     noise_file = '/uio/kant/geo-geofag-u1/lukasjm/cnn_test/data/dark_noise/dark_noise_std_radiance_452bands.csv'
#     noise_df = pd.read_csv(noise_file, header=0, index_col=0)
#     median_noise = np.nanmedian(noise_df.values[:], axis=0)
#     return median_noise

# LUT: read from the cache, or build the cache if it does not exist yet
data = cf.load_lut(CACHE_PATH, LUT_PATH, SOLAR_PATH, cos_sza=COS_ZEN)
cache_cos = float(data.get('cos_sza', np.nan))
print(f"cos(SZA) used for the transmittance in the cache: {cache_cos:.3f}")
if not np.isclose(cache_cos, COS_ZEN):
    raise ValueError(f"The cache was built with cos_sza = {cache_cos:.3f}, not COS_ZEN = "
                     f"{COS_ZEN:.3f}. Delete {CACHE_PATH} so that it is rebuilt.")

# measured dark noise in transmittance units from fenix_dark_noise.py (TO_TRANSMITTANCE = True),
# three versions, each combined over all dark periods, shape (452,):
#   sigma_floor : noise floor at zero signal, from adjacent frame differences
#   sigma_T     : detrended standard deviation (the version used so far)
#   sigma_max   : largest reasonable noise (no detrending, 95th percentile over pixels and periods)
dark = pd.read_csv('/uio/kant/geo-geofag-u1/lukasjm/cnn_test/data/dark_noise/dark_noise_versions_transmittance_452bands.csv')
wl_noise = dark['wavelength_nm'].to_numpy()
wl_lut = data['wavelength']


def on_lut_grid(values):
    """The noise on the LUT wavelength grid (interpolated only if the grids differ)."""
    if len(wl_noise) == len(wl_lut) and np.allclose(wl_noise, wl_lut, atol=0.5):
        return values
    return np.interp(wl_lut, wl_noise, values)


sigma_floor = on_lut_grid(dark['floor'].to_numpy())
sigma_T = on_lut_grid(dark['detrended'].to_numpy())
sigma_max = on_lut_grid(dark['max'].to_numpy())


#%%
# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# Settings shared by all experiments. Each experiment below only lists what
# it changes. See cf.DEFAULT_CONFIG for all settings and their meaning.
# Standard noise for every experiment: the measured (detrended) dark noise sigma_T plus
# a 2 % calibration gain, no relative noise. This is the same as the common test noise
# EVAL_NOISE below, so for the CNN-configuration experiments 'own' and 'common' agree.
BASE_CONFIG = dict(
    filters=(16, 32, 64),
    kernels=(9, 7, 5),
    dense=(64, 32),
    dropout=0.2,
    l2=1e-5,
    learning_rate=1e-3,
    batch_size=32,
    epochs=500,
    patience=40,
    noise_relative=0.0,
    noise_absolute=sigma_T,
    noise_gain=0.02,
    noise_scale=1.0,
)

# Constant absolute noise slightly above sigma_T: 10 % above its 95th percentile over the
# wavelengths (the maximum would be set by the few noisiest channels at the spectrum's ends)
SIGMA_CONST = 1.1 * np.percentile(sigma_T, 95)
print(f"sigma_T: median {np.median(sigma_T):.3g}, 95th percentile {np.percentile(sigma_T, 95):.3g}, "
      f"max {np.max(sigma_T):.3g} -> constant noise {SIGMA_CONST:.3g}")

EXPERIMENTS = [
    # CNN configurations, all trained with the standard noise (sigma_T)
    dict(name='baseline'),
    dict(name='small_filters', filters=(4, 8, 16)),
    dict(name='tiny_filters', filters=(2, 4, 6)),
    dict(name='large_kernels', kernels=(15, 11, 7)),
    dict(name='lr_3e-4', learning_rate=3e-4),
    dict(name='no_log', log_input=False),
    dict(name='lt1700', wl_range=(None, 1700)),           # only wavelengths up to 1700 nm
    # relative noise on top of sigma_T
    dict(name='noise_rel_2pct', noise_relative=0.02),     # as in the earlier experiments
    dict(name='noise_rel_10pct', noise_relative=0.10),    # large
    # other dark-noise levels instead of sigma_T
    dict(name='noise_floor', noise_absolute=sigma_floor),
    dict(name='noise_max', noise_absolute=sigma_max),
    dict(name='noise_sigmaT_x2', noise_absolute=2 * sigma_T),
    dict(name='noise_const', noise_absolute=SIGMA_CONST),   # one value at every wavelength
]

# common test noise for all experiments
EVAL_NOISE = dict(relative=0.0, absolute=sigma_T, gain=0.02)

# A full grid of combinations can be added with expand_grid, e.g. 2 x 2 = 4
# experiments named 'f4-8-16_lr0.001', 'f4-8-16_lr0.0003', ...:
# EXPERIMENTS += cf.expand_grid(filters=[(4, 8, 16), (16, 32, 64)],
#                               learning_rate=[1e-3, 3e-4])

#%%
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if ONLY_BUILD_CACHE:
    print(f"Cache ready: {CACHE_PATH}")
else:
    summary = cf.run_experiments(
        data, EXPERIMENTS, BASE_CONFIG, RESULTS_DIR,
        test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
        eval_noise=EVAL_NOISE, eval_noise_scales=EVAL_NOISE_SCALES,
        eval_noise_ref=EVAL_NOISE_REF, mode=MODE, color_by=COLOR_BY,
        permutation_window=PERMUTATION_WINDOW, permutation_repeats=PERMUTATION_REPEATS,
        tolerances=TOLERANCES, eval_own_noise=EVAL_OWN_NOISE,
        standard_test_noise=STANDARD_TEST_NOISE,
    )