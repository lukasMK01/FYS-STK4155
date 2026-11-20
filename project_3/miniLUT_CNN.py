#%%
"""
1D CNN retrieval of (reff_liq, reff_ice, tau_total, f_ice) from the
simulated below-cloud, zenith-viewing transmittance spectra of the mini-LUT.

Pipeline
--------
1. Read and solar-normalize the LUT (same as miniLUT_overview.py /
   jacobian_sensitivity_miniLUT.py)
2. Build the targets and drop clear-sky runs (tau_liq = tau_ice = 0, where
   f_ice = tau_ice / tau_total is undefined)
3. Random train/test split (TEST_FRACTION); a validation subset is carved
   out of the training part for early stopping, so the test set is never
   seen during training
4. Train a small 1D CNN with a fresh synthetic noise realization every epoch,
   then save model, target scaling, split indices and training history
5. Evaluate on a noisy and a clean copy of the test set and plot

Set TRAIN_MODEL = False to skip step 4 and re-run only the evaluation and
plots with the saved model, on exactly the same test runs.

Masking of undefined effective radii
------------------------------------
When tau_liq = 0 there is no liquid in the cloud, so reff_liq has no effect
on the spectrum: the LUT then contains identical spectra carrying different
reff_liq labels. Training on those would force the network to predict the
average reff_liq and add irreducible error to the loss. These entries are
therefore masked out of the loss and the metrics (same for reff_ice when
tau_ice = 0). tau_total and f_ice are defined for every remaining run.

Note on the split: with a regular LUT grid, a random split tests
interpolation between grid nodes, not extrapolation beyond the grid.
"""

import os
import json
 
# ---------------------------------------------------------------------------
# CPU thread limit -- must be set BEFORE numpy and TensorFlow are imported.
# By default TensorFlow starts one thread per CPU core it can see. On a large
# shared node this can exceed the per-user thread limit and crash with
# "Thread tf_numa_..._Eigen creation via pthread_create() failed".
# ---------------------------------------------------------------------------
N_THREADS = 8   # lower to 4 if the thread error persists
os.environ['TF_NUM_INTRAOP_THREADS'] = str(N_THREADS)
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)   # also caps numpy's BLAS threads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          # hide TensorFlow INFO/WARNING lines (errors still shown)
 
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
try:
    tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass  # TensorFlow was already initialized; the environment variables above still apply

from fenix_functions import compute_solar_angles

#%%
# ---------------------------------------------------------------------------
# 0. User settings
# ---------------------------------------------------------------------------
# True : train a new model and save it (overwrites the previous one in out_dir)
# False: load the saved model and only redo the evaluation and plots.
#        The test runs are read from the saved split; the test noise is
#        regenerated with the NOISE_* settings below, so changing NOISE_SCALE
#        here evaluates the same model at a different noise level.
TRAIN_MODEL = True

SEED = 42
TEST_FRACTION = 0.20   # share of all (non-clear-sky) runs held out for testing
VAL_FRACTION = 0.15    # share of the *training* runs used for early stopping

# Synthetic measurement noise, added in transmittance units BEFORE any
# preprocessing. Training spectra get a new noise realization every epoch;
# validation and test spectra get one fixed realization (reproducible via SEED).
#   NOISE_RELATIVE : 1-sigma Gaussian noise as a fraction of the signal,
#                    independent per channel. Scalar, or an array with one
#                    value per wavelength (e.g. 1 / SNR(lambda) of the sensor).
#   NOISE_ABSOLUTE : 1-sigma additive Gaussian noise in transmittance units
#                    (noise floor, matters most in dark absorption bands).
#                    Scalar or per-wavelength array.
#   NOISE_GAIN     : 1-sigma spectrally flat multiplicative error, drawn once
#                    per spectrum (absolute calibration uncertainty).
#   NOISE_SCALE    : multiplies all three terms: 0 = noise-free, 2 = doubled.
# The defaults are placeholders, NOT AisaFENIX specifications.
NOISE_RELATIVE = 0.02
NOISE_ABSOLUTE = 0.0
NOISE_GAIN = 0.02
NOISE_SCALE = 1.0

# Input preprocessing
LOG_INPUT = True   # feed log(transmittance) to the network
T_FLOOR = 1e-4     # transmittance is clipped here before the log (noise can push it <= 0)

# Targets (column order of the network output)
TARGET_NAMES = ['reff_liq', 'reff_ice', 'tau_total', 'f_ice']
LOG_TARGETS = ['tau_total']  # regressed in log space (spans orders of magnitude)

# Network and training
CNN_CONFIG = dict(
    filters=(16, 32, 64),   # one Conv1D + MaxPool(2) block per entry
    kernels=(9, 7, 5),      # kernel size (in wavelength channels) per block
    dense=(64, 32),         # hidden dense layers after flattening
    dropout=0.2,
    l2=1e-5,
    learning_rate=1e-3,
    batch_size=32,
    epochs=500,             # upper limit, early stopping usually ends sooner
    patience=40,            # early-stopping patience (epochs)
)

main_path = '/uio/kant/geo-geofag-u1/lukasjm/cnn_test/'
out_dir = os.path.join(main_path, 'cnn_output')
os.makedirs(out_dir, exist_ok=True)

model_path = os.path.join(out_dir, 'cnn_miniLUT.keras')
scaling_path = os.path.join(out_dir, 'cnn_miniLUT_scaling.json')
split_path = os.path.join(out_dir, 'cnn_miniLUT_split.npz')
history_path = os.path.join(out_dir, 'cnn_miniLUT_history.json')

if not TRAIN_MODEL:
    # preprocessing and target settings must match the trained model,
    # so they are taken from the saved run instead of from above
    with open(scaling_path) as f:
        saved_scaling = json.load(f)
    TARGET_NAMES = saved_scaling['target_names']
    LOG_TARGETS = saved_scaling['log_targets']
    LOG_INPUT = saved_scaling['log_input']
    T_FLOOR = saved_scaling['t_floor']
    print(f"Evaluation only: using saved model from {out_dir}")

keras.utils.set_random_seed(SEED)

#%%
# ---------------------------------------------------------------------------
# 1. Data reading and normalization -- reused from miniLUT_overview.py
# ---------------------------------------------------------------------------
combined = xr.open_dataset(main_path + 'combined_miniLUT.nc')

solar_aisafenix = xr.open_dataset(
    main_path + 'NewGuey2003_convolved.nc'
)

local_noon = pd.Timestamp('2026-02-15T12:43:58')
utc_time = pd.Timestamp('2026-02-15T11:25:00')
local_time = utc_time + pd.Timedelta(hours=1)
# angles = compute_solar_angles([local_time])
cos_zenith_val = 60 #np.cos(np.deg2rad(angles.solar_zenith.values[0]))


def _normalize_radiance(radiance_all, solar_irradiance, wavelength, cos_zenith):
    """pi * radiance / (solar_irradiance * cos(solar_zenith)) -- copied
    from miniLUT_overview.py."""
    if hasattr(solar_irradiance, 'irradiance'):
        solar_values = solar_irradiance.irradiance.values
    elif hasattr(solar_irradiance, 'values'):
        solar_values = solar_irradiance.values
    else:
        solar_values = np.asarray(solar_irradiance)
    if solar_values.shape[-1] != len(wavelength):
        raise ValueError(
            f"solar_irradiance has {solar_values.shape[-1]} wavelength points, "
            f"but the data has {len(wavelength)} -- must share a wavelength grid."
        )
    return np.pi * radiance_all / (solar_values * cos_zenith)


wavelength = combined['wavelength'].values

# zenith-viewing pixel, same selection as in miniLUT_overview.py
azimuth = float(combined['azimuth'].values[0])
cz_vals = combined['cos_zenith'].values
cos_zenith = float(cz_vals[np.argmin(np.abs(cz_vals - (-1)))])
az_idx = int(np.argmin(np.abs(combined['azimuth'].values - azimuth)))
cz_idx = int(np.argmin(np.abs(combined['cos_zenith'].values - cos_zenith)))

radiance_all = combined['radiance'].isel(azimuth=az_idx, cos_zenith=cz_idx).values
transmittance_all = _normalize_radiance(radiance_all, solar_aisafenix, wavelength, cos_zenith_val)

tau_liq = combined['tau_liq'].values.astype(float)
reff_liq = combined['reff_liq'].values.astype(float)
tau_ice = combined['tau_ice'].values.astype(float)
reff_ice = combined['reff_ice'].values.astype(float)

n_wl = wavelength.size
print(f"Loaded {transmittance_all.shape[0]} spectra x {n_wl} wavelengths")

if not TRAIN_MODEL and not np.allclose(wavelength, saved_scaling['wavelength']):
    raise ValueError("Wavelength grid differs from the one the saved model was trained on.")

#%%
# ---------------------------------------------------------------------------
# 2. Targets, clear-sky removal and masks
# ---------------------------------------------------------------------------
tau_total = tau_liq + tau_ice
with np.errstate(divide='ignore', invalid='ignore'):
    f_ice = np.where(tau_total > 0, tau_ice / tau_total, np.nan)

keep = tau_total > 0
print(f"Dropping {np.sum(~keep)} clear-sky runs (f_ice undefined)")

finite = np.all(np.isfinite(transmittance_all), axis=1)
if not finite.all():
    print(f"WARNING: dropping {np.sum(keep & ~finite)} runs with non-finite transmittance")
keep &= finite

target_values = dict(reff_liq=reff_liq, reff_ice=reff_ice, tau_total=tau_total, f_ice=f_ice)
target_defined = dict(reff_liq=tau_liq > 0, reff_ice=tau_ice > 0,
                      tau_total=np.ones_like(keep), f_ice=np.ones_like(keep))

T_all = transmittance_all[keep].astype('float32')                            # (N, n_wl)
Y_phys = np.column_stack([target_values[n] for n in TARGET_NAMES])[keep]     # (N, 4)
M = np.column_stack([target_defined[n] for n in TARGET_NAMES])[keep].astype('float32')
tau_liq_k, tau_ice_k = tau_liq[keep], tau_ice[keep]
N = T_all.shape[0]

for j, name in enumerate(TARGET_NAMES):
    print(f"  {name:<10} defined for {int(M[:, j].sum()):5d} / {N} runs")

#%%
# ---------------------------------------------------------------------------
# 3. Train / validation / test split
# ---------------------------------------------------------------------------
if TRAIN_MODEL:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_test = int(round(TEST_FRACTION * N))
    test_idx, trainval_idx = perm[:n_test], perm[n_test:]
    n_val = int(round(VAL_FRACTION * trainval_idx.size))
    val_idx, train_idx = trainval_idx[:n_val], trainval_idx[n_val:]
else:
    split = np.load(split_path)
    if int(split['n_runs']) != N:
        raise ValueError(f"Saved split was made for {int(split['n_runs'])} runs, "
                         f"but {N} runs are loaded now -- has the LUT changed?")
    train_idx, val_idx, test_idx = split['train_idx'], split['val_idx'], split['test_idx']
print(f"Split: {train_idx.size} train / {val_idx.size} validation / {test_idx.size} test")

#%%
# ---------------------------------------------------------------------------
# 4. Target scaling (log for LOG_TARGETS, then standardization using only
#    the defined entries of the training set)
# ---------------------------------------------------------------------------
def forward_targets(Y, target_names=TARGET_NAMES, log_targets=LOG_TARGETS):
    Z = np.array(Y, dtype=float, copy=True)
    for j, name in enumerate(target_names):
        if name in log_targets:
            Z[:, j] = np.log(Z[:, j])
    return Z


if TRAIN_MODEL:
    Z_train = forward_targets(Y_phys[train_idx])
    M_train = M[train_idx] > 0
    scaling = dict(
        target_names=TARGET_NAMES,
        log_targets=LOG_TARGETS,
        target_mean=[float(Z_train[M_train[:, j], j].mean()) for j in range(len(TARGET_NAMES))],
        target_std=[float(Z_train[M_train[:, j], j].std()) for j in range(len(TARGET_NAMES))],
        log_input=LOG_INPUT,
        t_floor=T_FLOOR,
        wavelength=wavelength.tolist(),
    )
else:
    scaling = saved_scaling

t_mean = np.asarray(scaling['target_mean'])
t_std = np.asarray(scaling['target_std'])


def pack_targets(idx):
    """Standardized targets (undefined entries set to 0) followed by the mask,
    so the masked loss receives both through y_true: shape (n, 2 * n_targets)."""
    Z = (forward_targets(Y_phys[idx]) - t_mean) / t_std
    Z = np.where(M[idx] > 0, Z, 0.0)
    return np.concatenate([Z, M[idx]], axis=1).astype('float32')


def inverse_targets(Z, scaling):
    """Network output (standardized) -> physical units."""
    Y = np.asarray(Z) * np.asarray(scaling['target_std']) + np.asarray(scaling['target_mean'])
    for j, name in enumerate(scaling['target_names']):
        if name in scaling['log_targets']:
            Y[:, j] = np.exp(Y[:, j])
        if name == 'f_ice':
            Y[:, j] = np.clip(Y[:, j], 0.0, 1.0)
    return Y


Yp_train, Yp_val, Yp_test = pack_targets(train_idx), pack_targets(val_idx), pack_targets(test_idx)

#%%
# ---------------------------------------------------------------------------
# 5. Synthetic noise and input preprocessing
# ---------------------------------------------------------------------------
def _per_channel(value):
    return tf.constant(np.array(np.broadcast_to(np.asarray(value, dtype='float32'), (n_wl,))))


_noise_rel = _per_channel(NOISE_RELATIVE) * NOISE_SCALE
_noise_abs = _per_channel(NOISE_ABSOLUTE) * NOISE_SCALE
_noise_gain = float(NOISE_GAIN) * NOISE_SCALE


def add_noise(T):
    """T: (batch, n_wl) transmittance -> noisy transmittance, same shape."""
    T = tf.convert_to_tensor(T, tf.float32)
    shape = tf.shape(T)
    T_noisy = T * (1.0 + _noise_rel * tf.random.normal(shape))
    T_noisy = T_noisy + _noise_abs * tf.random.normal(shape)
    gain = 1.0 + _noise_gain * tf.random.normal(tf.stack([shape[0], 1]))
    return T_noisy * gain


def to_model_input(T, log_input=LOG_INPUT, floor=T_FLOOR):
    """(batch, n_wl) transmittance -> (batch, n_wl, 1) network input."""
    T = tf.convert_to_tensor(T, tf.float32)
    if log_input:
        T = tf.math.log(tf.maximum(T, floor))
    return T[..., tf.newaxis]


# fixed noise realizations for validation and test, plus a clean test copy
X_val = to_model_input(add_noise(T_all[val_idx])).numpy()
X_test_noisy = to_model_input(add_noise(T_all[test_idx])).numpy()
X_test_clean = to_model_input(T_all[test_idx]).numpy()
X_train_clean = to_model_input(T_all[train_idx]).numpy()   # only used to fit the input normalization

# training data: fresh noise for every batch in every epoch
train_ds = (
    tf.data.Dataset.from_tensor_slices((T_all[train_idx], Yp_train))
    .shuffle(train_idx.size, seed=SEED, reshuffle_each_iteration=True)
    .batch(CNN_CONFIG['batch_size'])
    .map(lambda T, y: (to_model_input(add_noise(T)), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

#%%
# ---------------------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------------------
N_OUT = len(TARGET_NAMES)


def masked_mse(y_true, y_pred):
    """MSE per target over its defined entries only, averaged over targets."""
    y, m = y_true[:, :N_OUT], y_true[:, N_OUT:]
    se = tf.square(y - y_pred) * m
    per_target = tf.reduce_sum(se, axis=0) / tf.maximum(tf.reduce_sum(m, axis=0), 1.0)
    return tf.reduce_mean(per_target)


def build_model(n_wl, cfg, X_for_norm):
    # a single (global) mean/std keeps the spectral shape intact, unlike a
    # per-channel standardization, which would blow up noise in dark bands
    norm = layers.Normalization(axis=None, name='input_norm')
    norm.adapt(X_for_norm)

    reg = keras.regularizers.l2(cfg['l2'])
    inp = keras.Input(shape=(n_wl, 1), name='spectrum')
    x = norm(inp)
    for filters, kernel in zip(cfg['filters'], cfg['kernels']):
        x = layers.Conv1D(filters, kernel, padding='same', activation='relu',
                          kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
    # Flatten (not global pooling): *where* in the spectrum a feature sits matters
    x = layers.Flatten()(x)
    x = layers.Dropout(cfg['dropout'])(x)
    for units in cfg['dense']:
        x = layers.Dense(units, activation='relu', kernel_regularizer=reg)(x)
        x = layers.Dropout(cfg['dropout'])(x)
    out = layers.Dense(N_OUT, name='targets')(x)

    model = keras.Model(inp, out, name='cnn_miniLUT')
    model.compile(optimizer=keras.optimizers.Adam(cfg['learning_rate']), loss=masked_mse)
    return model


#%%
# ---------------------------------------------------------------------------
# 7. Training and saving -- or loading the saved model
# ---------------------------------------------------------------------------
if TRAIN_MODEL:
    model = build_model(n_wl, CNN_CONFIG, X_train_clean)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=CNN_CONFIG['patience'],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=CNN_CONFIG['patience'] // 3, min_lr=1e-5),
    ]
    history = model.fit(
        train_ds,
        validation_data=(X_val, Yp_val),
        epochs=CNN_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=2,
    )
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # save everything needed to redo the evaluation or apply the model later
    model.save(model_path)
    scaling['noise'] = dict(relative=np.asarray(NOISE_RELATIVE).tolist(),
                            absolute=np.asarray(NOISE_ABSOLUTE).tolist(),
                            gain=NOISE_GAIN, scale=NOISE_SCALE)
    scaling['cnn_config'] = CNN_CONFIG
    with open(scaling_path, 'w') as f:
        json.dump(scaling, f, indent=2)
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, n_runs=N)
    with open(history_path, 'w') as f:
        json.dump(history_dict, f)
    print(f"\nSaved model, scaling, split and history to {out_dir}")
else:
    model = keras.models.load_model(model_path, compile=False)
    with open(history_path) as f:
        history_dict = json.load(f)
    print(f"Model was trained with noise settings: {scaling.get('noise')}")

#%%
# ---------------------------------------------------------------------------
# 8. Evaluation (physical units, defined entries only)
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred, relative=True):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    bias = np.mean(err)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan
    pos = y_true > 0
    med_rel = np.median(np.abs(err[pos]) / y_true[pos]) * 100 if (relative and pos.any()) else np.nan
    return mae, rmse, bias, r2, med_rel


def evaluate(X, idx, label):
    Y_pred = inverse_targets(model.predict(X, verbose=0), scaling)
    Y_true, mask = Y_phys[idx], M[idx] > 0

    # derived tau_liq / tau_ice (needed for LWP / IWP)
    tt, fi = Y_pred[:, TARGET_NAMES.index('tau_total')], Y_pred[:, TARGET_NAMES.index('f_ice')]
    rows = [(name, Y_true[mask[:, j], j], Y_pred[mask[:, j], j], name != 'f_ice')
            for j, name in enumerate(TARGET_NAMES)]
    rows += [('tau_liq*', tau_liq_k[idx], tt * (1 - fi), True),
             ('tau_ice*', tau_ice_k[idx], tt * fi, True)]

    print(f"\n--- {label} test set ---")
    print(f"{'target':<10}{'n':>6}{'MAE':>10}{'RMSE':>10}{'bias':>10}{'R2':>8}{'med|rel|%':>11}")
    for name, yt, yp, rel in rows:
        mae, rmse, bias, r2, med_rel = _metrics(yt, yp, relative=rel)
        rel_str = f"{med_rel:11.1f}" if np.isfinite(med_rel) else f"{'-':>11}"
        print(f"{name:<10}{yt.size:6d}{mae:10.3f}{rmse:10.3f}{bias:10.3f}{r2:8.3f}{rel_str}")
    print("(* derived from predicted tau_total and f_ice; med|rel| excludes true values of 0)")
    return Y_pred


Y_pred_noisy = evaluate(X_test_noisy, test_idx, f'NOISY (NOISE_SCALE={NOISE_SCALE})')
Y_pred_clean = evaluate(X_test_clean, test_idx, 'CLEAN')

#%%
# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history_dict['loss'], label='train (noisy, fresh each epoch)')
ax.plot(history_dict['val_loss'], label='validation (fixed noise)')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Masked MSE (standardized targets)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'cnn_loss.png'), dpi=150)
plt.show()

units = dict(reff_liq=r'$\mu$m', reff_ice=r'$\mu$m', tau_total='', f_ice='')
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
Y_true_test, mask_test = Y_phys[test_idx], M[test_idx] > 0
for j, (ax, name) in enumerate(zip(axes.flat, TARGET_NAMES)):
    sel = mask_test[:, j]
    yt, yp = Y_true_test[sel, j], Y_pred_noisy[sel, j]
    ax.scatter(yt, yp, s=12, alpha=0.5, edgecolor='none')
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)
    if name in LOG_TARGETS:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(f'True {name} {units[name]}')
    ax.set_ylabel(f'Predicted {name} {units[name]}')
    ax.set_title(f'{name}: MAE = {np.mean(np.abs(yp - yt)):.3g} (noisy test)')
fig.suptitle(f'Test set, NOISE_SCALE = {NOISE_SCALE:g}')
plt.tight_layout()
# file name carries the noise level, so evaluations at other levels don't overwrite it
plt.savefig(os.path.join(out_dir, f'cnn_pred_vs_true_noise{NOISE_SCALE:g}.png'), dpi=150)
plt.show()

#%%
# ---------------------------------------------------------------------------
# 10. Applying the model to new spectra
# ---------------------------------------------------------------------------
def predict_parameters(model, T, scaling):
    """
    Apply the trained CNN to new spectra.

    T : (n_spectra, n_wl) solar-normalized transmittance on the SAME
        wavelength grid as the LUT (scaling['wavelength']).
    Returns a dict with reff_liq, reff_ice, tau_total, f_ice and the
    derived tau_liq, tau_ice.
    """
    T = np.atleast_2d(np.asarray(T, dtype='float32'))
    if T.shape[1] != len(scaling['wavelength']):
        raise ValueError(f"Expected {len(scaling['wavelength'])} wavelengths, got {T.shape[1]}")
    X = to_model_input(T, scaling['log_input'], scaling['t_floor'])
    Y = inverse_targets(model.predict(X, verbose=0), scaling)
    out = {name: Y[:, j] for j, name in enumerate(scaling['target_names'])}
    out['tau_liq'] = out['tau_total'] * (1 - out['f_ice'])
    out['tau_ice'] = out['tau_total'] * out['f_ice']
    return out


# Example (in a new session):
# model = keras.models.load_model(os.path.join(out_dir, 'cnn_miniLUT.keras'), compile=False)
# with open(os.path.join(out_dir, 'cnn_miniLUT_scaling.json')) as f:
#     scaling = json.load(f)
# retrieved = predict_parameters(model, measured_transmittance, scaling)