"""
proj3_analyse.py
================
Tables and figures of the finished jobs (proj3_run.py, proj3_baselines.py).

  python proj3_analyse.py A | B | C     table, 1-SE choice, heat maps; writes best_<stage>.json
  python proj3_analyse.py spectral      heat maps of wavelength range and binning (CV and classic split)
  python proj3_analyse.py compare       CNN next to the baselines (CV and test set)
  add --quick to read the results of "proj3_run.py --quick"

Selection (stages A, B, C): the score is mean_std_rmse of the 5 CV folds (lower is better). A configuration is
"within 1 SE" of the best if its mean difference to the best, computed fold by fold (the same folds and the
same noise), is smaller than the standard error of that difference. Of those, the simplest one (fewest
parameters) is chosen. best_<stage>.json holds this choice: edit the file to choose another cell yourself.
Heat maps: red frame = lowest score, dashed black = the 1-SE choice, bold numbers = within 1 SE.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import proj3_data as D
import proj3_run as R

### fontsizes and figure settings (all figures are 12 inches wide)
label_fontsize = 14
title_fontsize = 16
tick_fontsize = 13
legend_fontsize = 11
FIG_DIR = './figures'
SCORE = 'score_mean_std_rmse'
SCORE_LABEL = 'mean standardized RMSE'

# what the heat maps of stages A and B show: (row option, label), (column option, label), (panel option, label)
AXES = dict(
    A=(('n_blocks', 'conv blocks'), ('filters', 'filters in first block'), ('kernel', 'kernel size')),
    B=(('n_dense', 'dense layers'), ('units', 'units per layer'), ('lr', 'learning rate')),
)
STAGE_KEYS = {s: list(o) for s, o in R.STAGES.items()}
INT_KEYS = {'n_blocks', 'filters', 'kernel', 'n_dense', 'units', 'batch', 'epochs', 'patience', 'seed', 'bin'}
RANGE_LABEL = {D.OPEN_LO: '380', D.OPEN_HI: '2500'}
SET_LABELS = ['full (380-2500)', '380-1700', 'VNIR (380-970)', 'SWIR (970-2500)']
BASELINES = ['knn', 'pca_ridge', 'rf', 'gbm']


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def select(df, base, vary, kind):
    """CNN jobs of one kind whose settings equal `base`, except for the options in `vary`."""
    m = (df['family'] == 'cnn') & (df['kind'] == kind)
    for key, value in base.items():
        if key not in vary:
            m &= df[f'cfg_{key}'] == value
    return df[m]


def cv_table(frame, metric=SCORE, complexity='n_params'):
    """
    One row per configuration: CV mean and SE, paired difference to the best, within-1-SE flag.
    Returns the table, the name of the best and the name of the 1-SE choice (the simplest within 1 SE).
    """
    P = frame.pivot(index='name', columns='fold', values=metric)       # rows: configurations, columns: folds
    complete = P.notna().all(axis=1)
    if (~complete).any():
        print(f'  {int((~complete).sum())} configurations have missing folds and are left out')
    P = P[complete]
    if P.empty:
        raise SystemExit('no configuration has all folds yet')
    k = P.shape[1]
    mean, se = P.mean(axis=1), P.std(axis=1, ddof=1) / np.sqrt(k)
    best = mean.idxmin()
    d = P.sub(P.loc[best], axis=1)                                      # fold-wise difference to the best
    diff, diff_se = d.mean(axis=1), d.std(axis=1, ddof=1) / np.sqrt(k)
    within = diff <= diff_se                                            # (the best itself: 0 <= 0)
    info = frame.drop_duplicates('name').set_index('name').loc[P.index]
    tab = info[[c for c in info.columns if c.startswith('cfg_')]].copy()
    tab['complexity'] = info[complexity]
    tab['mean'], tab['se'], tab['diff'], tab['diff_se'], tab['within'] = mean, se, diff, diff_se, within
    choice = tab[tab['within']].sort_values(['complexity', 'mean']).index[0]
    return tab, best, choice


def print_table(tab, best, choice, keys, n=12):
    cols = [f'cfg_{k}' for k in keys] + ['complexity', 'mean', 'se', 'diff', 'diff_se', 'within']
    print(tab.sort_values('mean')[cols].head(n).rename(columns=lambda c: c.replace('cfg_', ''))
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print(f'lowest score: {best}\n1-SE choice : {choice}  ({int(tab["within"].sum())} configurations within 1 SE)')


# ---------------------------------------------------------------------------
# Heat maps (plain matplotlib)
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f'proj3_{name}.png')
    fig.savefig(path, dpi=500, bbox_inches='tight')
    print('figure:', path)


def fmt(v):
    return f'{v:g}'


def heatmap(ax, grid, row_labels, col_labels, title, ylabel, xlabel, norm, best=None, choice=None, bold=None):
    """Values in the cells; best / choice = (row, column) of the red / dashed black frame; bold = boolean grid."""
    cmap = plt.get_cmap('viridis_r').copy()
    cmap.set_bad('0.85')                                       # cells that do not exist
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect='auto')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                r, g, b, _ = cmap(norm(grid[i, j]))
                ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center', fontsize=legend_fontsize,
                        color='white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black',
                        fontweight='bold' if bold is not None and bold[i, j] else 'normal')
    for cell, color, style in ((best, 'red', '-'), (choice, 'black', '--')):
        if cell is not None:
            ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, fill=False, edgecolor=color,
                                       lw=3, linestyle=style, clip_on=False))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    return im


def add_colorbar(fig, im, axes, label=SCORE_LABEL):
    cb = fig.colorbar(im, ax=axes)
    cb.set_label(label, fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)


def stage_heatmaps(stage, tab, best, choice):
    """Stages A and B: one panel per value of the third option."""
    (rk, rl), (ck, cl), (pk, pl) = AXES[stage]
    rows, cols, panels = (list(R.STAGES[stage][k]) for k in (rk, ck, pk))
    G = np.full((len(panels), len(rows), len(cols)), np.nan)
    W = np.zeros(G.shape, bool)
    pos = {}
    for name, r in tab.iterrows():
        p, i, j = panels.index(r[f'cfg_{pk}']), rows.index(r[f'cfg_{rk}']), cols.index(r[f'cfg_{ck}'])
        G[p, i, j], W[p, i, j], pos[name] = r['mean'], r['within'], (p, i, j)
    norm = Normalize(vmin=np.nanmin(G), vmax=np.nanmax(G))
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 4.4), constrained_layout=True)
    for p, ax in enumerate(axes):
        here = lambda name: pos[name][1:] if pos[name][0] == p else None
        im = heatmap(ax, G[p], [fmt(v) for v in rows], [fmt(v) for v in cols], f'{"abc"[p]}) {pl} {fmt(panels[p])}',
                     rl, cl, norm, best=here(best), choice=here(choice), bold=W[p])
    add_colorbar(fig, im, axes)
    save_fig(fig, f'stage{stage}_heatmaps')


def ablation_heatmap(tab, best, choice):
    """Stage C: dropout x L2 in a single panel."""
    rows, cols = list(R.STAGES['C']['dropout']), list(R.STAGES['C']['l2'])
    G, W, pos = np.full((len(rows), len(cols)), np.nan), np.zeros((len(rows), len(cols)), bool), {}
    for name, r in tab.iterrows():
        i, j = rows.index(r['cfg_dropout']), cols.index(r['cfg_l2'])
        G[i, j], W[i, j], pos[name] = r['mean'], r['within'], (i, j)
    fig, ax = plt.subplots(figsize=(12, 4.4), constrained_layout=True)
    # ax.set_title('Dropout and L2 regularisation', fontsize=title_fontsize)   # single panel: left to the caption
    im = heatmap(ax, G, [fmt(v) for v in rows], [fmt(v) for v in cols], '', 'dropout', 'L2 weight',
                 Normalize(np.nanmin(G), np.nanmax(G)), best=pos[best], choice=pos[choice], bold=W)
    add_colorbar(fig, im, ax)
    save_fig(fig, 'stageC_heatmap')


# ---------------------------------------------------------------------------
# Stages A, B, C
# ---------------------------------------------------------------------------
def analyse_stage(stage, df, quick):
    base = R.get_base(stage, quick)
    frame = select(df, base, STAGE_KEYS[stage], 'cv')
    print(f'stage {stage}: {frame["name"].nunique()} configurations, 5-fold CV, score = {SCORE_LABEL}')
    tab, best, choice = cv_table(frame)
    print_table(tab, best, choice, STAGE_KEYS[stage])
    results_dir = D.results_dir(quick)
    tab.to_csv(os.path.join(results_dir, f'summary_{stage}.csv'))
    if stage == 'C':
        ablation_heatmap(tab, best, choice)
    else:
        stage_heatmaps(stage, tab, best, choice)
    row = tab.loc[choice]
    winner = {k: (int(row[f'cfg_{k}']) if k in INT_KEYS else float(row[f'cfg_{k}'])) for k in STAGE_KEYS[stage]}
    path = R.best_path(results_dir, stage)
    with open(path, 'w') as f:
        json.dump(winner, f, indent=1)
    print(f'chosen settings written to {path}: {winner}\n(edit the file to choose another configuration)')


# ---------------------------------------------------------------------------
# Spectral range and binning
# ---------------------------------------------------------------------------
def spectral_table(df, base, metric):
    """Per configuration (lo, hi, bin): CV mean, classic-split mean."""
    out = {}
    for kind, name in (('cv', 'cv'), ('classic', 'classic')):
        frame = select(df, base, ['lo', 'hi', 'bin'], kind)
        for key, g in frame.groupby(['cfg_lo', 'cfg_hi', 'cfg_bin']):
            out.setdefault(tuple(key), {})[name] = g[metric].mean()
    return out


def spectral_heatmaps(df, base, metric, name, frames):
    """
    Range (lower x upper limit) and binning (range x binning factor) in a) CV and b) classic split (test set).
    Red = lowest value of the panel. With frames (all targets): dashed black = the instrument with the fewest
    channels within 1 SE of the best CV score of ALL spectral configurations, bold = within that 1 SE.
    """
    tab = spectral_table(df, base, metric)
    choice_cell, within = None, set()
    if frames:
        cv_tab, _, choice = cv_table(select(df, base, ['lo', 'hi', 'bin'], 'cv'), metric, complexity='n_channels')
        keys = ['cfg_lo', 'cfg_hi', 'cfg_bin']
        choice_cell = tuple(cv_tab.loc[choice, keys])
        within = {tuple(v) for v in cv_tab.loc[cv_tab['within'], keys].to_numpy()}

    for kind in ('range', 'bin'):
        if kind == 'range':
            rows, cols = R.LOWER, R.UPPER
            cells = [[(lo, hi, 1) for hi in cols] for lo in rows]
            rl, cl = [RANGE_LABEL.get(v, fmt(v)) for v in rows], [RANGE_LABEL.get(v, fmt(v)) for v in cols]
            ylabel, xlabel = 'lower limit (nm)', 'upper limit (nm)'
        else:
            rows, cols = R.RANGE_SETS, R.BINS
            cells = [[(lo, hi, b) for b in cols] for lo, hi in rows]
            rl, cl = SET_LABELS, [fmt(v) for v in cols]
            ylabel, xlabel = 'wavelength range (nm)', 'binning factor (channels per bin)'
        grids = {w: np.array([[tab.get(c, {}).get(w, np.nan) for c in row] for row in cells], dtype=float)
                 for w in ('cv', 'classic')}
        shown = [w for w in ('cv', 'classic') if np.isfinite(grids[w]).any()]
        if 'cv' not in shown:
            print(f'no CV results for the {kind} comparison yet')
            continue
        values = np.concatenate([grids[w][np.isfinite(grids[w])] for w in shown])
        norm = Normalize(values.min(), values.max())
        fig, axes = plt.subplots(1, len(shown), figsize=(12, 4.4), constrained_layout=True, squeeze=False)
        for p, w in enumerate(shown):
            g = grids[w]
            best = np.unravel_index(np.nanargmin(g), g.shape)
            use_frames = frames and w == 'cv'
            mark = next(((i, j) for i in range(len(rows)) for j in range(len(cols))
                         if use_frames and cells[i][j] == choice_cell), None)
            bold = np.array([[use_frames and c in within for c in row] for row in cells])
            title = 'a) 5-fold CV' if w == 'cv' else f'{"ab"[p]}) classic split, test set'
            im = heatmap(axes[0, p], g, rl, cl, title, ylabel, xlabel, norm, best=best, choice=mark, bold=bold)
        add_colorbar(fig, im, axes[0], label=SCORE_LABEL if name == 'all_targets' else f'standardized RMSE, {name}')
        save_fig(fig, f'spectral_{kind}_{name}')


def analyse_spectral(df, quick):
    base = R.get_base('spectral', quick)
    tab, best, choice = cv_table(select(df, base, ['lo', 'hi', 'bin'], 'cv'), SCORE, complexity='n_channels')
    print(f'spectral: {len(tab)} configurations (5-fold CV); sorted by number of channels, '
          f'1-SE choice = fewest channels within 1 SE of the best:')
    cols = ['cfg_lo', 'cfg_hi', 'cfg_bin', 'complexity', 'mean', 'se', 'within']
    print(tab.sort_values('complexity')[cols].rename(columns=lambda c: c.replace('cfg_', ''))
          .to_string(float_format=lambda v: f'{v:.4f}'))
    c = tab.loc[choice]
    print(f'1-SE choice: {fmt(c["cfg_lo"])}-{fmt(c["cfg_hi"])} nm, binning {fmt(c["cfg_bin"])}, '
          f'{int(c["complexity"])} channels')
    tab.to_csv(os.path.join(D.results_dir(quick), 'summary_spectral.csv'))
    spectral_heatmaps(df, base, SCORE, 'all_targets', frames=True)
    for t in D.TARGETS:         # the same maps for every parameter: where in the spectrum is its information?
        spectral_heatmaps(df, base, f'score_std_rmse_{t}', t, frames=False)


# ---------------------------------------------------------------------------
# CNN against the baselines
# ---------------------------------------------------------------------------
def analyse_compare(df, quick):
    """The final CNN (best_A/B/C, full spectrum) and the best option of every baseline (lowest CV score)."""
    base = R.get_base('spectral', quick)
    full = lambda f: f[(f['cfg_lo'] == base['lo']) & (f['cfg_hi'] == base['hi']) & (f['cfg_bin'] == base['bin'])]
    groups = [('CNN', full(select(df, base, ['lo', 'hi', 'bin'], 'cv')),
               full(select(df, base, ['lo', 'hi', 'bin'], 'classic')))]
    for fam in BASELINES:
        f = full(df[(df['family'] == fam) & (df['kind'] == 'cv')])
        if f.empty:
            continue
        best = f.groupby('name')[SCORE].mean().idxmin()
        groups.append((fam, f[f['name'] == best], df[(df['name'] == best) & (df['kind'] == 'classic')]))

    rows = []
    for label, cv, cl in groups:
        v = cv.sort_values('fold')[SCORE]
        row = dict(model=label, option=cv['name'].iloc[0] if len(cv) else '', cv_mean=v.mean(),
                   cv_se=v.std(ddof=1) / np.sqrt(max(len(v), 1)),
                   cv_joint_success_pct=cv['score_joint_success_pct'].mean(),
                   test_mean=cl[SCORE].mean(), test_joint_success_pct=cl['score_joint_success_pct'].mean())
        row.update({f'test_std_rmse_{t}': cl[f'score_std_rmse_{t}'].mean() for t in D.TARGETS})
        rows.append(row)
    table = pd.DataFrame(rows).set_index('model')
    print(table.drop(columns='option').to_string(float_format=lambda v: f'{v:.4f}'))
    table.to_csv(os.path.join(D.results_dir(quick), 'summary_compare.csv'))

    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].bar(x, table['cv_mean'], yerr=table['cv_se'], capsize=4, color='0.6')
    axes[0].set_title('a) 5-fold CV', fontsize=title_fontsize)
    axes[1].bar(x, table['test_mean'], color='0.6')
    axes[1].set_title('b) test set (classic split)', fontsize=title_fontsize)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(table.index, rotation=40, ha='right', fontsize=tick_fontsize)
        ax.set_ylabel(SCORE_LABEL, fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    save_fig(fig, 'compare_models')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['A', 'B', 'C', 'spectral', 'compare'])
    p.add_argument('--quick', action='store_true')
    args = p.parse_args(argv)
    df = D.load_jobs(D.results_dir(args.quick))
    if df.empty:
        raise SystemExit(f'no finished jobs in {D.results_dir(args.quick)}/jobs')
    if df['dataset_id'].nunique() > 1:
        print(f'WARNING: the jobs come from {df["dataset_id"].nunique()} different data sets')
    if args.what in STAGE_KEYS:
        analyse_stage(args.what, df, args.quick)
    elif args.what == 'spectral':
        analyse_spectral(df, args.quick)
    else:
        analyse_compare(df, args.quick)


if __name__ == '__main__':
    main()
