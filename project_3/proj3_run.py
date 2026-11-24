"""
proj3_run.py
============
Cross-validation of the CNN configurations, stage by stage. Every trained model is one small json file in
RESULTS_DIR/jobs, so the run can be stopped and restarted (finished jobs are skipped) and split over processes.

  python proj3_run.py A                 comparison A: conv blocks x filters x kernel size      (27 configs, 5-fold CV)
  python proj3_analyse.py A             -> table, 1-SE choice, heat maps, writes best_A.json (edit it to choose yourself)
  python proj3_run.py B                 comparison B: dense layers x units x learning rate     (uses best_A.json)
  python proj3_analyse.py B
  python proj3_run.py C                 ablation: dropout x L2                                 (uses best_A, best_B)
  python proj3_analyse.py C
  python proj3_run.py spectral          wavelength range and binning: CV and the classic split (uses best_A/B/C)
  python proj3_analyse.py spectral

Options (any stage):
  --quick       3 epochs, a few configs and folds, own results folder: a fast test of the whole chain
  --list        only list the jobs and how many are finished
  --part 2/4    run only every 4th job, starting with the 2nd: start four processes with 1/4 ... 4/4 and
                CNN_N_THREADS=2 each to use 8 cores
"""
import argparse
import itertools
import json
import os
import time

import numpy as np

import proj3_data as D

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
BASE = dict(
    n_blocks=3, filters=16, kernel=9,           # conv blocks; filters = first block, doubling in every block
    n_dense=1, units=64,                        # dense layers after the flatten
    dropout=0.2, l2=1e-5,
    lr=1e-3, batch=32, epochs=500, patience=40, seed=42,
    lo=D.OPEN_LO, hi=D.OPEN_HI, bin=1,          # spectral view: full spectrum, native resolution
)

# the options of every stage: a full grid of these (3 x 3 x 3), everything else from BASE and the earlier winners
STAGES = dict(
    A=dict(n_blocks=[2, 3, 4], filters=[8, 16, 32], kernel=[5, 9, 15]),
    B=dict(n_dense=[1, 2, 3], units=[32, 64, 128], lr=[3e-4, 1e-3, 3e-3]),
    C=dict(dropout=[0, 0.2, 0.4], l2=[0, 1e-5, 1e-4]),
)
WINNERS = {'B': ['A'], 'C': ['A', 'B'], 'spectral': ['A', 'B', 'C']}    # stages whose best_*.json are used
REQUIRED = {'B': ['A'], 'C': ['A', 'B'], 'spectral': ['A', 'B']}        # ... and that must exist

# spectral stage: (a) lower x upper limit at native resolution, (b) binning x a few ranges
LOWER = [D.OPEN_LO, 700, 970, 1500]                                      # nm (open = 380)
UPPER = [970, 1300, 1700, 2000, D.OPEN_HI]                               # nm (open = 2500)
MIN_WIDTH_NM = 150
RANGE_SETS = [(D.OPEN_LO, D.OPEN_HI), (D.OPEN_LO, 1700), (D.OPEN_LO, 970), (970, D.OPEN_HI)]   # full, <1700, VNIR, SWIR
BINS = [1, 2, 4, 8, 16]
CLASSIC_SEEDS = 3                                                        # models per configuration for the classic split


def best_path(results_dir, stage):
    return os.path.join(results_dir, f'best_{stage}.json')


def get_base(stage, quick=False):
    """BASE updated with the winners of the earlier stages (best_A.json, ...). Edit those files to choose yourself."""
    results_dir = D.results_dir(quick)
    base = dict(BASE)
    for prev in WINNERS.get(stage, []):
        path = best_path(results_dir, prev)
        if os.path.exists(path):
            with open(path) as f:
                base.update(json.load(f))
        elif prev in REQUIRED.get(stage, []) and not quick:
            raise SystemExit(f'{path} is missing: run "python proj3_run.py {prev}" and '
                             f'"python proj3_analyse.py {prev}" first')
        else:
            print(f'note: {path} not found, using the BASE values for {list(STAGES[prev])}')
    if quick:
        base.update(epochs=3, patience=3)
    return D.normalise_cfg(base)


def stage_configs(stage, base):
    """The list of configurations of a stage (dicts)."""
    if stage in STAGES:
        opts = STAGES[stage]
        combos = [dict(zip(opts, values)) for values in itertools.product(*opts.values())]
    else:
        ranges = {(lo, hi, 1) for lo in LOWER for hi in UPPER if hi - lo >= MIN_WIDTH_NM}            # (a)
        ranges |= {(lo, hi, b) for (lo, hi) in RANGE_SETS for b in BINS}                              # (b)
        combos = [dict(lo=lo, hi=hi, bin=b) for lo, hi, b in sorted(ranges)]
    return [D.normalise_cfg({**base, **c}) for c in combos]


def make_jobs(stage, base, quick):
    configs = stage_configs(stage, base)
    if quick:
        configs = configs[::max(1, len(configs) // 3)][:3]
    jobs = []
    for cfg in configs:
        n_cv = 2 if quick else D.N_FOLDS
        jobs += [(cfg, 'cv', k) for k in range(n_cv)]
        if stage == 'spectral':
            jobs += [(cfg, 'classic', k) for k in range(1 if quick else CLASSIC_SEEDS)]
    return jobs


# ---------------------------------------------------------------------------
# One job = train one model on one split and score it on the held-out spectra
# ---------------------------------------------------------------------------
def run_job(cfg, kind, k, ds, splits, train_fn):
    sp = splits['cv'][k] if kind == 'cv' else splits['classic']
    T_v, wl_v, sig_v = D.spectral_view(ds['T'], ds['wl'], ds['sigma'], cfg['lo'], cfg['hi'], cfg['bin'])
    run_cfg = dict(cfg, seed=cfg['seed'] + (k if kind == 'classic' else 0))   # classic: k = model seed
    tr, va, held = sp['train'], sp['val'], sp['held']
    predict, info = train_fn(T_v[tr], ds['Y'][tr], ds['M'][tr], T_v[va], ds['Y'][va], ds['M'][va], sig_v, run_cfg)
    seed = D.EVAL_SEED + (k if kind == 'cv' else 100)        # the same noise on the held-out spectra for every model
    Y_pred = predict(D.noisy(T_v[held], sig_v, seed))
    sc = D.scores(ds['Y'][held], ds['M'][held], Y_pred, splits['score_scaling'])
    view = dict(n_channels=int(T_v.shape[1]), wl_min=float(wl_v.min()), wl_max=float(wl_v.max()))
    return info, sc, view


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['A', 'B', 'C', 'spectral'])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--list', action='store_true')
    p.add_argument('--part', default='1/1', help='i/n: run every n-th job starting with the i-th')
    args = p.parse_args(argv)

    results_dir = D.results_dir(args.quick)
    base = get_base(args.stage, args.quick)
    i_part, n_part = (int(x) for x in args.part.split('/'))
    jobs = make_jobs(args.stage, base, args.quick)[i_part - 1::n_part]

    def done(job):
        cfg, kind, k = job
        return os.path.exists(D.job_path(results_dir, D.config_name('cnn', cfg), kind, k))

    todo = [j for j in jobs if not done(j)]
    print(f'stage {args.stage}: {len(jobs)} jobs in this part, {len(jobs) - len(todo)} already finished, '
          f'results in {results_dir}')
    if args.list or not todo:
        return

    import proj3_cnn
    ds = D.prepare()
    splits = D.make_splits(ds)
    t_start = time.time()
    for n, (cfg, kind, k) in enumerate(todo, start=1):
        name = D.config_name('cnn', cfg)
        t0 = time.time()
        info, sc, view = run_job(cfg, kind, k, ds, splits, proj3_cnn.train_cnn)
        D.save_job(results_dir, dict(family='cnn', name=name, kind=kind, fold=k, cfg=cfg, info=info, scores=sc,
                                     view=view, dataset_id=ds['id']))
        eta = (time.time() - t_start) / n * (len(todo) - n) / 3600
        print(f'[{n}/{len(todo)}] {name} {kind}{k}: mean_std_rmse {sc["mean_std_rmse"]:.4f}, '
              f'{info["epochs_trained"]} epochs, {time.time() - t0:.0f} s, {info["n_params"]} parameters, '
              f'about {eta:.1f} h left', flush=True)


if __name__ == '__main__':
    main()
