import re
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure
from matplotlib.text import TextPath
from matplotlib.font_manager import FontProperties, findSystemFonts
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.collections import PatchCollection
from cycler import cycler
from numba import jit
from typing import Optional, Union
from numpy.typing import NDArray
from utils import alphabet

from datetime import datetime

font_colors = {
    'A': '#33cc00',
    'R': '#cc0000',
    'N': '#6600cc',
    'D': '#0033ff',
    # 'C': '#ffff00',
    # 'C': '#f0e68c',
    'C': '#ffd700',
    'Q': '#6600cc',
    'E': '#0033ff',
    'G': '#33cc00',
    'H': '#009900',
    'I': '#33cc00',
    'L': '#33cc00',
    'K': '#cc0000',
    'M': '#33cc00',
    'F': '#009900',
    'P': '#33cc00',
    'S': '#0099ff',
    'T': '#0099ff',
    'W': '#009900',
    'Y': '#009900',
    'V': '#33cc00',
    '-': '#cccccc', # '#e0e0e0',
}


def get_font_paths(font_weight='normal', char_margin=0.):
    system_font_names = [os.path.splitext(os.path.basename(i))[0] for i in findSystemFonts()]
    for font_family in ['Menlo', 'Inconsolata', 'SourceCodePro', 'monospace']:
        if font_family in system_font_names:
            break
    font_paths = {}
    fp = FontProperties(family=font_family, weight=font_weight)
    for a in alphabet:
        text_path = TextPath((0,0), a, size=10, prop=fp)
        vertices = text_path.vertices.copy()
        codes = text_path.codes.copy()
        path = Path(vertices, codes, readonly=False)
        path.vertices[:, 0] -= path.vertices[:, 0].min()
        path.vertices[:, 1] -= path.vertices[:, 1].min()
        path.vertices[:, 0] /= path.vertices[:, 0].max()
        path.vertices[:, 1] /= path.vertices[:, 1].max() / (1 - char_margin)
        # path.vertices[:, 1] = 1 - path.vertices[:, 1]
        path.vertices[:, 1] += char_margin / 2
        font_paths[a] = path
    return font_paths


@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def select_clusters(N_k: NDArray, max_clusters: int | None=None, pinned_clusters: list[int] | None=None, cluster_size_threshold: float | None=None):
    M = len(N_k)
    if max_clusters is None:
        K = M
    else:
        K = min(M, max_clusters)

    used_clusters = np.zeros(K, dtype=int)

    i = K - 2
    if pinned_clusters is not None:
        j = 0
        while (i < K) and (j < len(pinned_clusters)):
            used_clusters[i] = pinned_clusters[j]
            i -= 1
            j += 1
    
    num_used_big_clusters = K - 1 - i
    t = 0 if cluster_size_threshold is None else N_k.max() * cluster_size_threshold        

    j = 1
    while (i < K) and (j < K):
        if j not in used_clusters:
            used_clusters[i] = j
            i -= 1
            if N_k[j] >= t:
                num_used_big_clusters += 1
        j += 1

    return used_clusters, num_used_big_clusters


def calculate_letter_heights(f_kia, N_k, used_clusters, used_columns, scaling_func):
    D = f_kia.shape[2]
    L = len(used_columns)
    K = len(used_clusters)

    used_N_k = N_k[used_clusters][:, np.newaxis, np.newaxis]
    letter_heights = np.empty((K + 1, L, D))
    letter_heights[:-1, :, :] = f_kia[np.ix_(used_clusters, used_columns, np.arange(D, dtype=int))] * scaling_func(used_N_k / used_N_k.max())
    letter_heights[-1, :, :] = np.average(f_kia, axis=0, weights=N_k)[used_columns, :]

    return letter_heights



def draw(output_file, f_kia, N_k, cluster_labels, lead_seq, column_order_by:Optional[NDArray]=None, 
    curve_y:Optional[NDArray]=None, title=None, max_clusters=None, cluster_size_threshold=None, pinned_clusters=None,
    row_margin=1., column_margin=1., char_margin=1., char_stroke_width=1., xscale=1., yscale=1., tick_size=1., top_panel_height=1.,
    scaling_func=np.sqrt, hspace=0.045, dS_ki_fraction=None, show_gap=False, 
):
    xscale = 0.4 * xscale
    yscale = 1.8 * yscale
    tick_size = 32 * tick_size
    row_margin = 0.07 * row_margin
    column_margin = 0.4 * column_margin
    char_stroke_width = 6 * char_stroke_width
    char_margin = char_margin * 2 * (char_stroke_width / 72 / yscale)
    hspace = 0.00013 * hspace
    top_panel_height = 2.6 * top_panel_height

    limit1 = 0
    limit2 = 0.06
    limit3 = 0.1

    print('begin', datetime.now())

    used_clusters, num_kept_big_clusters = select_clusters(N_k, max_clusters=max_clusters, pinned_clusters=pinned_clusters, cluster_size_threshold=cluster_size_threshold)
    K = len(used_clusters)
    used_cluster_labels = cluster_labels[used_clusters]
    used_cluster_sizes = N_k[used_clusters]

    num_cols = len(lead_seq.seq)
    used_columns = np.arange(num_cols, dtype=int)
    resn_labels = np.fromiter(lead_seq.seq, dtype='U1')
    m = re.match(r'^[^/]+/(\d+)-\d+', lead_seq.id)
    if m:
        resi_labels = np.arange(int(m[1]), int(m[1]) + num_cols, dtype=int)
    else:
        resi_labels = np.arange(1, num_cols + 1, dtype=int)
    
    if curve_y is None:
        curve_y = np.zeros(num_cols)
    
    if column_order_by is not None:
        sorted_idx = np.argsort(column_order_by)
        used_columns = used_columns[sorted_idx]

        resn_labels = resn_labels[used_columns]
        resi_labels = resi_labels[used_columns]
        curve_y = curve_y[used_columns]

    if dS_ki_fraction is not None:
        f = dS_ki_fraction[used_clusters, used_columns]

    letter_heights = calculate_letter_heights(f_kia, N_k, used_clusters, used_columns, scaling_func)

    col_widths = np.full(num_cols, 1 + column_margin)
    x_tick_locations = np.cumsum(col_widths)
    x_tick_locations -= col_widths/2
    assert len(x_tick_locations) == len(resi_labels)
    curve_x = x_tick_locations

    row_heights = np.zeros(letter_heights.shape[0])
    y_tick_locations = np.empty_like(row_heights)

    logo_patches = []
    font_paths = get_font_paths(char_margin=char_margin)
    paths = [font_paths[a] for a in alphabet]
    colors = [font_colors[a] for a in alphabet]
    col_widths[:-1].cumsum()
    xbase = x_tick_locations - col_widths/2 + column_margin/2
    ybase = row_margin/2
    gap_color = colors[alphabet.index('-')]
    letter_order = letter_heights.argsort(axis=-1)
    for k in range(letter_heights.shape[0]):
        for i, x in enumerate(xbase):
            y = ybase
            prev_type = None
            gap_height = 0
            for a in letter_order[k, i, :]:
                if alphabet[a] == '-':
                    gap_height = letter_heights[k, i, a]
                    continue
                if round(letter_heights[k, i, a], 2) <= limit2:
                    logo_patches.append(Rectangle((x, y), col_widths[i] - column_margin, letter_heights[k, i, a], linewidth=0, color=colors[a]))
                    y += letter_heights[k, i, a]
                else:
                    path = paths[a].deepcopy()
                    path.vertices[:, 1] *= letter_heights[k, i, a]
                    path.vertices += [x, y]
                    logo_patches.append(PathPatch(path, facecolor=colors[a], linewidth=0, edgecolor='none', )) # capstyle='butt', joinstyle='miter'
                    y += letter_heights[k, i, a]
            if show_gap and (gap_height > 0):
                # logo_patches.append(Rectangle((x, y), col_widths[i] - column_margin, h[a], linewidth=0, color=colors[a]))
                logo_patches.append(Rectangle((x, y), col_widths[i] - column_margin, gap_height, linewidth=tick_size/10, edgecolor=gap_color, facecolor='none'))
            y += gap_height

            if row_heights[k] < y - ybase:
                row_heights[k] = y - ybase
        
        y_tick_locations[k] = ybase + row_heights[k]/2
        row_heights[k] += row_margin
        ybase += row_heights[k]
        if k == K - 1:
            ybase += row_margin
        if k == K - num_kept_big_clusters - 1:
            ybase += row_margin
        if pinned_clusters is not None:
            if k == K - len(pinned_clusters) - 1:
                ybase += row_margin

    bottom_panel_height = ybase - row_margin/2

    
    y_tick_labels = [f'{l}: {s}' for l, s in zip(used_cluster_labels, used_cluster_sizes)]
    y_tick_labels.append(f'{len(N_k)} clusters: {N_k.sum()}')


    title_panel_height = 0.1 * tick_size
    if title is not None:
        title_size = 2 * tick_size
        title_panel_height = top_panel_height
    figure_height = (bottom_panel_height + top_panel_height + title_panel_height) * yscale 
    figure_width = col_widths.sum() * xscale

    fig = matplotlib.figure.Figure(figsize=(figure_width, figure_height), layout='constrained', frameon=False)
    # print('bottom_panel_height', bottom_panel_height)
    hspace = hspace * tick_size / (bottom_panel_height + top_panel_height + title_panel_height)
    gs = matplotlib.gridspec.GridSpec(2, 1, figure=fig, wspace=0., hspace=hspace, height_ratios=[title_panel_height, bottom_panel_height])
    if title is not None:
        fig.suptitle(title, x=0, y=1, horizontalalignment='left', fontsize=title_size)

    ax0 = fig.add_subplot(gs[1])
    # if highlight_resi is not None:
    #     for r in highlight_resi:
    #         i = find_first(r, resi_labels)
    #         if i >= 0:
    #             ax0.axvspan(x_tick_locations[i] - col_widths[i]/2, x_tick_locations[i] + col_widths[i]/2, facecolor='lightgray')

    if curve_y is not None:
        ax0_df = pd.DataFrame(dict(column_types=column_types, curve_x=curve_x, curve_y=curve_y))
        bottom_y = curve_y.min()
        for gn, g in ax0_df.groupby('column_types'):
            markerline, stemlines, baseline = ax0.stem(g['curve_x'].to_numpy(), g['curve_y'].to_numpy(), bottom=bottom_y, basefmt=' ')
            markerline.set(markersize=tick_size, color=COLUMN_TYPE_COLOR[gn], **top_panel_marker_styles[gn])
            # markerline.set_markeredgewidth(tick_size / 3)
            stemlines.set(color=COLUMN_TYPE_COLOR[gn])
            # stemlines.set_color(s[gn]['color'])
            stemlines.set_linewidth(tick_size / 10)

    #     ax0.xaxis.set_visible(False)
    #     ax0.tick_params(axis='y', which='both', left=False, labelleft=False)
    #     ax0.set_ylabel('ΔS')
    ax0.spines.top.set_visible(False)
    ax0.spines.bottom.set_visible(False)
    ax0.spines.left.set_visible(False)
    ax0.spines.right.set_visible(False)
    ax0.tick_params(labeltop=True, bottom=False, labelbottom=False, left=False, labelleft=False, right=False, labelsize=tick_size)
    ax0.set_xticks(x_tick_locations, range(len(x_tick_locations)), rotation='vertical')
    for r in ax0.get_xticklabels():
        r.set_color('grey')


    ax1 = fig.add_subplot(gs[2], sharex=ax0)
    ax1.set_xlim((0, col_widths.sum()))
    ax1.set_ylim((0, bottom_panel_height))
    # ax1.tick_params(left=False, bottom=False, labelbottom=False, labelsize=tick_size)
    ax1.set_yticks(y_tick_locations, y_tick_labels)
    ax1.spines.top.set_visible(False)
    ax1.spines.bottom.set_visible(False)
    ax1.spines.left.set_visible(False)
    ax1.spines.right.set_visible(False)

    ax1.add_collection(PatchCollection(logo_patches, match_original=True))

    if lead_seq is not None:
        ax2 = ax1.twiny()
        ax2.set_zorder(0)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.spines.top.set_visible(False)
        ax2.spines.bottom.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax2.spines.right.set_visible(False)

        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        ax3.spines.top.set_position(("outward", 1.2 * tick_size))
        ax3.spines.top.set_visible(False)
        ax3.spines.bottom.set_visible(False)
        ax3.spines.left.set_visible(False)
        ax3.spines.right.set_visible(False)

        ax2.set_xticks(x_tick_locations, resn_labels, ha='center')
        ax3.set_xticks(x_tick_locations, resi_labels, rotation='vertical')
        ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False, top=False, labelsize=tick_size)
        ax3.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False, top=False, labelsize=tick_size)

    ax1.tick_params(left=False, bottom=False, labelbottom=False, labelsize=1.25 * tick_size)
    print('begin save', datetime.now())
    fig.savefig(output_file, bbox_inches='tight')
    print('end save', datetime.now())

    return letter_heights
