"""Plot helpers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#function from part d but tweaked
def heatmap_nopen(ax, grid, row_labels, col_labels, title, ylabel, xlabel,vmin=0.01, vmax=0.2, grid_se=None,
                  label_fontsize=14, title_fontsize=16, tick_fontsize=13, annot_fontsize=11):
    """Heat map of CV MSE (log colour scale) with the value written in every cell;
    the best cell gets a red frame."""
    cmap = plt.get_cmap("viridis_r")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")          # row 0 is drawn at the top
    
    i_best = np.unravel_index(np.argmin(grid), grid.shape)
    within = grid <= grid[i_best] + grid_se[i_best]

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            r, g, b, _ = cmap(norm(grid[i, j]))                   # colour of this cell
            text_color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"   # readable on dark/light
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color=text_color, fontsize=annot_fontsize,
                    fontweight="bold" if within is not None and within[i, j] else "normal")
 
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
 
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=3))   # cell centres are at integers
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)