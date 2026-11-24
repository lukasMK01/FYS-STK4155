"""
proj3_cnn.py
============
The 1D-CNN: build, train (with fresh noise on every batch) and predict. TensorFlow is imported when the first
model is trained, so the rest of the code works without it.

The only function the rest of the code calls is train_cnn(). It returns a predict function that takes
(noisy) transmittance and returns the four parameters in physical units.
"""
import os

# CPU thread limits, set before TensorFlow starts its thread pools (lower the number if you get
# "pthread_create() failed" on the shared server, or when running several jobs in parallel)
N_THREADS = int(os.environ.get('CNN_N_THREADS', '8'))
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, str(N_THREADS))

import time

import numpy as np

import proj3_data as D

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


def masked_mse(y_true, y_pred):
    """MSE per target over its defined entries only (y_true = targets followed by the mask), mean of the targets."""
    tf, _ = _tf()
    n = len(D.TARGETS)
    y, m = y_true[:, :n], y_true[:, n:]
    per_target = tf.reduce_sum(tf.square(y - y_pred) * m, axis=0) / tf.maximum(tf.reduce_sum(m, axis=0), 1.0)
    return tf.reduce_mean(per_target)


def build_model(cfg, n_channels, X_for_norm):
    """
    n_blocks x [Conv1D + MaxPooling1D(2)] (filters double with every block), Flatten, n_dense dense layers
    of `units`, linear output with one value per target. Pooling is skipped once the spectrum is only
    one value long (very coarse binning).
    """
    tf, keras = _tf()
    layers = keras.layers
    norm = layers.Normalization(axis=None)      # one global mean/std keeps the spectral shape intact
    norm.adapt(X_for_norm)
    reg = keras.regularizers.l2(cfg['l2']) if cfg['l2'] > 0 else None

    x = inp = keras.Input(shape=(n_channels, 1))
    x = norm(x)
    length, filters = n_channels, int(cfg['filters'])
    for _ in range(int(cfg['n_blocks'])):
        x = layers.Conv1D(filters, int(cfg['kernel']), padding='same', activation='relu', kernel_regularizer=reg)(x)
        if length >= 2:
            x = layers.MaxPooling1D(2)(x)
            length //= 2
        filters *= 2
    x = layers.Flatten()(x)                     # not global pooling: where a feature sits in the spectrum matters
    if cfg['dropout'] > 0:
        x = layers.Dropout(cfg['dropout'])(x)
    for _ in range(int(cfg['n_dense'])):
        x = layers.Dense(int(cfg['units']), activation='relu', kernel_regularizer=reg)(x)
        if cfg['dropout'] > 0:
            x = layers.Dropout(cfg['dropout'])(x)
    out = layers.Dense(len(D.TARGETS))(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg['lr']), loss=masked_mse)
    return model


def make_train_dataset(T, Yp, sigma, cfg):
    """tf.data pipeline: every batch gets a new noise realization, then the logarithm."""
    tf, _ = _tf()
    sig = tf.constant(np.asarray(sigma, dtype='float32'))
    gain, floor = float(D.GAIN), float(D.T_FLOOR)

    def _noisy(Tb, yb):
        shape = tf.shape(Tb)
        Tn = Tb + sig * tf.random.normal(shape)
        Tn = Tn * (1.0 + gain * tf.random.normal(tf.stack([shape[0], 1])))
        return tf.math.log(tf.maximum(Tn, floor))[..., tf.newaxis], yb

    options = tf.data.Options()                 # tf.data would otherwise use all visible cores
    options.threading.private_threadpool_size = N_THREADS
    options.threading.max_intra_op_parallelism = 1
    return (tf.data.Dataset.from_tensor_slices((np.asarray(T, dtype='float32'), Yp))
            .shuffle(len(T), seed=int(cfg['seed']), reshuffle_each_iteration=True)
            .batch(int(cfg['batch']))
            .map(_noisy, num_parallel_calls=min(N_THREADS, 4))
            .prefetch(2)
            .with_options(options))


def train_cnn(T_tr, Y_tr, M_tr, T_va, Y_va, M_va, sigma, cfg, verbose=0):
    """
    Train one CNN on clean transmittance T_tr (n, n_channels); the noise (sigma per channel + gain) is added
    on the fly. The validation spectra get one fixed noise realization and decide the early stopping.
    Returns predict(T) -> (n, 4) physical values for (noisy) transmittance, and a dict with the training info.
    """
    tf, keras = _tf()
    keras.backend.clear_session()
    keras.utils.set_random_seed(int(cfg['seed']))

    scaling = D.fit_target_scaling(Y_tr, M_tr)

    def pack(Y, M):     # standardized targets followed by the mask, as the masked loss expects
        return np.concatenate([D.standardize(Y, M, scaling), M.astype(float)], axis=1).astype('float32')

    X_va = D.log_input(D.noisy(T_va, sigma, int(cfg['seed']) + 1))[..., np.newaxis]
    X_norm = D.log_input(T_tr)[..., np.newaxis]
    model = build_model(cfg, T_tr.shape[1], X_norm)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(cfg['patience']), restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, int(cfg['patience']) // 3),
                                          min_lr=cfg['lr'] / 100),
    ]
    t0 = time.time()
    fit = model.fit(make_train_dataset(T_tr, pack(Y_tr, M_tr), sigma, cfg), validation_data=(X_va, pack(Y_va, M_va)),
                    epochs=int(cfg['epochs']), callbacks=callbacks, verbose=verbose)
    val = np.asarray(fit.history['val_loss'])
    info = dict(n_params=int(model.count_params()), epochs_trained=int(val.size), best_epoch=int(np.argmin(val)) + 1,
                best_val_loss=float(val.min()), train_time_s=round(time.time() - t0, 1))

    def predict(T):
        Z = model.predict(D.log_input(T)[..., np.newaxis], batch_size=1024, verbose=0)
        return D.to_physical(Z, scaling)

    return predict, info
