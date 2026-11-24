"""
proj3_baselines.py
==================
Four classic baselines on exactly the same data, folds and held-out noise as the CNN (proj3_run.py):

  knn        nearest neighbours in the LUT: a look-up of the clean spectra, i.e. "retrieval without learning"
  pca_ridge  PCA of the spectra, then ridge regression: the linear baseline
  rf         PCA, then random forest
  gbm        PCA, then gradient boosting (HistGradientBoostingRegressor)

  python proj3_baselines.py                    all four
  python proj3_baselines.py knn rf             a selection
  python proj3_baselines.py --lo 970 --bin 4   on another spectral view (the same options as the CNN stages)
  python proj3_baselines.py --quick            fast test (own results folder)

Each option of a baseline is one "configuration", scored with the 5-fold CV and once with the classic split
(train -> test set). The results go to the same RESULTS_DIR/jobs as the CNN, so proj3_analyse.py compare
puts them next to the CNN. Finished jobs are skipped.

How the noise is handled: the models that learn (pca_ridge, rf, gbm) are trained on N_REPLICAS noisy copies of the
training spectra, every copy with its own noise draw. knn searches the clean training spectra (the LUT) with the
noisy spectrum. The held-out spectra get the same noise as for the CNN.
A target of the absent phase (r_eff of liquid in a pure ice cloud) is only fitted on the clouds where it is defined.
"""
import argparse
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor

import proj3_data as D

N_REPLICAS = 5          # noisy copies of the training spectra
N_JOBS = int(os.environ.get('CNN_N_THREADS', '8'))

# the options of every baseline (each one is run and scored like a CNN configuration)
OPTIONS = dict(
    knn=[dict(k=1), dict(k=5), dict(k=15)],                       # number of neighbours
    pca_ridge=[dict(n_pca=10), dict(n_pca=20), dict(n_pca=40)],   # number of principal components
    rf=[dict(n_pca=20, leaf=1), dict(n_pca=20, leaf=5)],          # minimum samples per leaf
    gbm=[dict(n_pca=20, depth=3), dict(n_pca=20, depth=6)],       # maximum tree depth
)


def regressor(family, opt, seed):
    """One regressor per target (fitted on one column)."""
    if family == 'knn':
        return KNeighborsRegressor(n_neighbors=opt['k'])
    if family == 'pca_ridge':
        return RidgeCV(alphas=np.logspace(-3, 3, 13))
    if family == 'rf':
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=opt['leaf'], n_jobs=N_JOBS, random_state=seed)
    return HistGradientBoostingRegressor(max_depth=opt['depth'], max_iter=300, learning_rate=0.05, random_state=seed)


def make_train_fn(family, opt):
    """Same interface as proj3_cnn.train_cnn: returns predict(T) and an info dict (the validation set is not needed)."""
    def train_fn(T_tr, Y_tr, M_tr, T_va, Y_va, M_va, sigma, cfg):
        t0, seed = time.time(), int(cfg['seed'])
        scaling = D.fit_target_scaling(Y_tr, M_tr)
        Z = D.standardize(Y_tr, M_tr, scaling)
        X_clean = D.log_input(T_tr)
        mu, sd = X_clean.mean(), X_clean.std()               # one global mean/std (a per-channel one would blow up dark bands)

        def scale(X):
            return (X - mu) / sd

        if family == 'knn':
            X_fit, Z_fit, M_fit, pca = scale(X_clean), Z, M_tr, None
        else:
            X_noisy = scale(np.vstack([D.log_input(D.noisy(T_tr, sigma, 1000 + r)) for r in range(N_REPLICAS)]))
            pca = PCA(opt['n_pca'], random_state=seed).fit(X_noisy)
            X_fit = pca.transform(X_noisy)
            Z_fit, M_fit = np.tile(Z, (N_REPLICAS, 1)), np.tile(M_tr, (N_REPLICAS, 1))
        models = []
        for j in range(len(D.TARGETS)):
            sel = M_fit[:, j]
            models.append(regressor(family, opt, seed).fit(X_fit[sel], Z_fit[sel, j]))

        def predict(T):
            X = scale(D.log_input(T))
            X = X if pca is None else pca.transform(X)
            return D.to_physical(np.column_stack([m.predict(X) for m in models]), scaling)

        return predict, dict(n_params=0, epochs_trained=0, best_epoch=0, best_val_loss=0.0,
                             train_time_s=round(time.time() - t0, 1))
    return train_fn


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('families', nargs='*', help=f'any of {list(OPTIONS)} (default: all)')
    p.add_argument('--lo', type=float, default=D.OPEN_LO)
    p.add_argument('--hi', type=float, default=D.OPEN_HI)
    p.add_argument('--bin', type=int, default=1)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args(argv)

    import proj3_run as R       # the same run_job as for the CNN: only the training function differs
    results_dir = D.results_dir(args.quick)
    ds = D.prepare()
    splits = D.make_splits(ds)
    families = args.families or list(OPTIONS)
    unknown = [f for f in families if f not in OPTIONS]
    if unknown:
        raise SystemExit(f'unknown baseline {unknown}; choose from {list(OPTIONS)}')
    n_cv = 2 if args.quick else D.N_FOLDS

    t_start, n_done = time.time(), 0
    for family in families:
        for opt in OPTIONS[family][:1 if args.quick else None]:
            cfg = D.normalise_cfg(dict(opt, lo=args.lo, hi=args.hi, bin=args.bin, seed=42))
            name = D.config_name(family, cfg, keys=list(opt) + ['lo', 'hi', 'bin'])
            train_fn = make_train_fn(family, opt)
            for kind, k in [('cv', k) for k in range(n_cv)] + [('classic', 0)]:
                if os.path.exists(D.job_path(results_dir, name, kind, k)):
                    continue
                info, sc, view = R.run_job(cfg, kind, k, ds, splits, train_fn)
                D.save_job(results_dir, dict(family=family, name=name, kind=kind, fold=k, cfg=cfg, info=info,
                                             scores=sc, view=view, dataset_id=ds['id']))
                n_done += 1
                print(f'{name} {kind}{k}: mean_std_rmse {sc["mean_std_rmse"]:.4f}, {info["train_time_s"]:.0f} s',
                      flush=True)
    print(f'{n_done} jobs run in {(time.time() - t_start) / 60:.1f} min; results in {results_dir}')


if __name__ == '__main__':
    main()
