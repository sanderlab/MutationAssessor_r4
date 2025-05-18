import sys
import os
import re
import io_helpers

from Bio import SeqIO
from Bio.Align import substitution_matrices

import numpy as np
import numba

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import PercentFormatter

alphabet = b'ACDEFGHIKLMNPQRSTVWY'
encoding_map = np.full(256, -1, dtype=np.int8)
for i, a in enumerate(alphabet):
    encoding_map[ord(chr(a).upper())] = i
    encoding_map[ord(chr(a).lower())] = i

@numba.jit(nopython=True, parallel=True, nogil=True)
def _tokenize(a):
    num_threads = numba.get_num_threads()
    b = a.ravel()
    N = len(b)
    chunks = np.linspace(0, N, num_threads + 1).astype(np.int64)
    for t in numba.prange(num_threads):
        for i in range(chunks[t], chunks[t + 1]):
            b[i] = encoding_map[b[i]]
    return a

def encode_alignment(aln):
    a = np.array(aln, dtype='S1').view(np.int8)
    return _tokenize(a)


@numba.jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def identities_histogram(a, bin_width=1):
    num_threads = numba.get_num_threads()
    N = a.shape[0]
    S = np.arange(0., num_threads + 1) / num_threads * N * (N - 1) / 2
    chunks = np.round(N - 3/2 - np.sqrt((N - 1/2) * (N - 1/2) - 2 * S)).astype(np.int64)
    chunks[0] = -1
    chunks[-1] = N - 2
    target_hist = np.zeros(int(100 / bin_width) + 1, dtype=np.int64)
    tmp_hist = np.zeros((num_threads, int(100 / bin_width) + 1), dtype=np.int64)
    for t in numba.prange(num_threads):
        for i in range(chunks[t] + 1, chunks[t + 1] + 1):
            for j in range(i + 1, N):
                identities_without_gaps = 0
                effective_lengths = a.shape[1]
                for k in range(a.shape[1]):
                    # == v1 ==
                    # if (a[i, k] < 0) or (a[j, k] < 0): # gap
                    #     effective_lengths -= 1
                    # elif a[i, k] == a[j, k]:
                    #     identities_without_gaps += 1

                    # == v2 ==
                    if a[i, k] == a[j, k]:
                        if a[i, k] < 0: # gap
                            effective_lengths -= 1
                        else:
                            identities_without_gaps += 1
                bin_index = int((identities_without_gaps / effective_lengths * 100) / bin_width)
                tmp_hist[t, bin_index] += 1
            if i == 0:
                target_hist[:] = tmp_hist[t].copy()
    pairwise_hist = tmp_hist.sum(axis=0)
    edges = np.arange(len(pairwise_hist) + 1) * bin_width
    return target_hist, pairwise_hist, edges


def _biopython_score_maxtrix_to_numpy(matrix_name, alphabet):
    biopython_matrix = substitution_matrices.load(matrix_name)
    scores = np.array(biopython_matrix.select(tuple(chr(a) for a in alphabet)))
    return scores


@numba.jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def similarities_histogram(a, scoring_matrix, bin_width=1):
    num_threads = numba.get_num_threads()
    N = a.shape[0]
    S = np.arange(0., num_threads + 1) / num_threads * N * (N - 1) / 2
    chunks = np.round(N - 3/2 - np.sqrt((N - 1/2) * (N - 1/2) - 2 * S)).astype(np.int64)
    chunks[0] = -1
    chunks[-1] = N - 2
    min_score = scoring_matrix.min()
    max_score = scoring_matrix.max()
    target_hist = np.zeros(int((max_score - min_score) / bin_width) + 1, dtype=np.int64)
    tmp_hist = np.zeros((num_threads, int((max_score - min_score) / bin_width) + 1), dtype=np.int64)
    for t in numba.prange(num_threads):
        for i in range(chunks[t] + 1, chunks[t + 1] + 1):
            for j in range(i + 1, N):
                score_without_gaps = 0
                effective_lengths = a.shape[1]
                for k in range(a.shape[1]):
                    # == v1 ==
                    # if (a[i, k] < 0) or (a[j, k] < 0): # gap
                    #     effective_lengths -= 1
                    # else:
                    #     score_without_gaps += scoring_matrix[a[i, k], a[j, k]]

                    # == v2 ==
                    if (a[i, k] < 0) or (a[j, k] < 0): # gap
                        if a[i, k] == a[j, k]:
                            effective_lengths -= 1
                    else:
                        score_without_gaps += scoring_matrix[a[i, k], a[j, k]]
                bin_index = int((score_without_gaps / effective_lengths - min_score) / bin_width)
                tmp_hist[t, bin_index] += 1
            if i == 0:
                target_hist[:] = tmp_hist[t].copy()
    
    pairwise_hist = tmp_hist.sum(axis=0)
    edges = np.arange(len(pairwise_hist) + 1) * bin_width + min_score

    min_index = 0
    for i in range(len(pairwise_hist)):
        if pairwise_hist[i] > 0:
            min_index = i
            break

    max_index = len(pairwise_hist) - 1
    for i in range(len(pairwise_hist) - 1, -1, -1):
        if pairwise_hist[i] > 0:
            max_index = i
            break
    return target_hist[min_index : max_index + 1], pairwise_hist[min_index : max_index + 1], edges[min_index : max_index + 2]


@numba.jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def histograms(a, scoring_matrix, identities_bin_width=1, similarities_bin_width=0.05):
    num_threads = numba.get_num_threads()
    N = a.shape[0]
    S = np.arange(0., num_threads + 1) / num_threads * N * (N - 1) / 2
    _chunks = np.empty(num_threads + 1, dtype=np.float64)
    np.round(N - 3/2 - np.sqrt((N - 1/2) * (N - 1/2) - 2 * S), 0, _chunks)
    chunks = _chunks.astype(np.int64)
    assert chunks[0] == -1
    assert chunks[-1] == N - 2

    target_identities_hist = np.zeros(int(100 / identities_bin_width) + 1, dtype=np.int64)
    tmp_identities_hist = np.zeros((num_threads, int(100 / identities_bin_width) + 1), dtype=np.int64)

    min_score = scoring_matrix.min()
    max_score = scoring_matrix.max()
    target_similarities_hist = np.zeros(int((max_score - min_score) / similarities_bin_width) + 1, dtype=np.int64)
    tmp_similarities_hist = np.zeros((num_threads, int((max_score - min_score) / similarities_bin_width) + 1), dtype=np.int64)
    
    for t in numba.prange(num_threads):
        for i in range(chunks[t] + 1, chunks[t + 1] + 1):
            for j in range(i + 1, N):
                identities_without_gaps = 0
                similarities_score_without_gaps = 0.
                effective_lengths = a.shape[1]
                for k in range(a.shape[1]):
                    if (a[i, k] < 0) or (a[j, k] < 0): # gap
                        # == v1 ==
                        # effective_lengths -= 1

                        # == v2 ==
                        if a[i, k] == a[j, k]:
                            effective_lengths -= 1
                    else:
                        if a[i, k] == a[j, k]:
                            identities_without_gaps += 1
                        similarities_score_without_gaps += scoring_matrix[a[i, k], a[j, k]]
                identities_bin_index = int((identities_without_gaps / effective_lengths * 100) / identities_bin_width)
                tmp_identities_hist[t, identities_bin_index] += 1
                similarities_bin_index = int((similarities_score_without_gaps / effective_lengths - min_score) / similarities_bin_width)
                tmp_similarities_hist[t, similarities_bin_index] += 1
            if i == 0:
                target_identities_hist[:] = tmp_identities_hist[t]
                target_similarities_hist[:] = tmp_similarities_hist[t]
    
    pairwise_identities_hist = tmp_identities_hist.sum(axis=0)
    identities_edges = np.arange(len(pairwise_identities_hist) + 1) * identities_bin_width
    pairwise_similarities_hist = tmp_similarities_hist.sum(axis=0)
    similarities_edges = np.arange(len(pairwise_similarities_hist) + 1) * similarities_bin_width + min_score

    min_index = 0
    for i in range(len(pairwise_similarities_hist)):
        if pairwise_similarities_hist[i] > 0:
            min_index = i
            break

    max_index = len(pairwise_similarities_hist) - 1
    for i in range(len(pairwise_similarities_hist) - 1, -1, -1):
        if pairwise_similarities_hist[i] > 0:
            max_index = i
            break
    
    return (
        target_identities_hist, 
        pairwise_identities_hist, 
        identities_edges,
        target_similarities_hist[min_index : max_index + 1], 
        pairwise_similarities_hist[min_index : max_index + 1], 
        similarities_edges[min_index : max_index + 2],
    )


def draw_alignment_stats(aln_fn, min_cdf_threshold=0.01):
    fn_base = os.path.splitext(aln_fn)[0]
    npz_fn = fn_base + '.stats.npz'
    png_fn = fn_base + '.stats.png'

    aln = io_helpers.read_msa(aln_fn, 'fasta')
    a = encode_alignment(aln)

    if os.path.isfile(npz_fn):
        data = np.load(npz_fn)
        target_identities_hist = data['target_identities_hist']
        pairwise_identities_hist = data['pairwise_identities_hist']
        identities_hist_edges = data['identities_hist_edges']
        target_similarities_hist = data['target_similarities_hist']
        pairwise_similarities_hist = data['pairwise_similarities_hist']
        similarities_hist_edges = data['similarities_hist_edges']
        site_coverages = data['site_coverages']
    else:
        # blosum45 = _biopython_score_maxtrix_to_numpy('BLOSUM45', alphabet)
        blosum62 = _biopython_score_maxtrix_to_numpy('BLOSUM62', alphabet)
        (
            target_identities_hist, 
            pairwise_identities_hist, 
            identities_hist_edges,
            target_similarities_hist, 
            pairwise_similarities_hist, 
            similarities_hist_edges,
        ) = histograms(a, blosum62, identities_bin_width=1, similarities_bin_width=0.05)

        site_coverages = np.count_nonzero(a >= 0, axis=0) / a.shape[0] * 100

        with io_helpers.file_guard(npz_fn) as tmp_npz_fn:
            np.savez(tmp_npz_fn, 
                target_identities_hist=target_identities_hist,
                pairwise_identities_hist=pairwise_identities_hist,
                identities_hist_edges=identities_hist_edges,
                target_similarities_hist=target_similarities_hist,
                pairwise_similarities_hist=pairwise_similarities_hist,
                similarities_hist_edges=similarities_hist_edges,
                site_coverages=site_coverages,
            )

    cutoff = 70
    is_site_covered = (site_coverages >= cutoff)
    coverage = np.count_nonzero(is_site_covered) / len(site_coverages) * 100

    fig = plt.figure(figsize=(6.4 * 2.3 / 1.5, 4.8 * 3 / 1.5))
    title = '{}, {} sequences, {:.0f}% coverage'.format(os.path.basename(fn_base), a.shape[0], coverage)
    fig.suptitle(title, fontsize='x-large', fontweight='regular')

    ax = fig.add_subplot(3, 2, 1)
    width = identities_hist_edges[1] - identities_hist_edges[0]
    ax.bar(identities_hist_edges[:-1], target_identities_hist, width=width, align='edge', color='#000000', edgecolor='none', alpha=0.3)
    ax.set_xlim(identities_hist_edges[0], identities_hist_edges[-1])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel('Identity % (All vs Target)')
    ax.set_ylabel('Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax = ax.twinx()
    cdf = target_identities_hist.cumsum()
    # ax.plot(identities_hist_edges[:-1] + width / 2, cdf)
    # ax.plot(identities_hist_edges[:-1], cdf, color='#000000', drawstyle="steps-pre")
    ax.stairs(cdf, identities_hist_edges, color='#000000')
    ax.set_ylim(0, max(1, cdf[-1]))
    ax.set_ylabel('Cumulative Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        auc_cdf = float(cdf.sum() * width / (identities_hist_edges[-1] - identities_hist_edges[np.argmax(cdf >= min_cdf_threshold)]))
    else:
        auc_cdf = 0.0
    ax.text(0.02, 0.98, f'auc_cdf = {auc_cdf:.3g}', transform=ax.transAxes, ha='left', va='top')

    ax = fig.add_subplot(3, 2, 3)
    width = identities_hist_edges[1] - identities_hist_edges[0]
    ax.bar(identities_hist_edges[:-1], pairwise_identities_hist, width=width, align='edge', color='#000000', edgecolor='none', alpha=0.3)
    ax.set_xlim(identities_hist_edges[0], identities_hist_edges[-1])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel('Identity % (All vs All)')
    ax.set_ylabel('Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax = ax.twinx()
    cdf = pairwise_identities_hist.cumsum()
    # ax.plot(identities_hist_edges[:-1] + width / 2, cdf)
    # ax.plot(identities_hist_edges[:-1], cdf, color='#000000', drawstyle="steps-pre")
    ax.stairs(cdf, identities_hist_edges, color='#000000')
    ax.set_ylim(0, max(1, cdf[-1]))
    ax.set_ylabel('Cumulative Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        auc_cdf = float(cdf.sum() * width / (identities_hist_edges[-1] - identities_hist_edges[np.argmax(cdf >= min_cdf_threshold)]))
    else:
        auc_cdf = 0.0
    ax.text(0.02, 0.98, f'auc_cdf = {auc_cdf:.3g}', transform=ax.transAxes, ha='left', va='top')

    ax = fig.add_subplot(3, 2, 2)
    width = similarities_hist_edges[1] - similarities_hist_edges[0]
    ax.bar(similarities_hist_edges[:-1], target_similarities_hist, width=width, align='edge', color='#000000', edgecolor='none', alpha=0.3)
    ax.set_xlim(similarities_hist_edges[0], similarities_hist_edges[-1])
    ax.set_xlabel('BLOSUM62/L (All vs Target)')
    ax.set_ylabel('Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax = ax.twinx()
    cdf = target_similarities_hist.cumsum()
    # ax.plot(similarities_hist_edges[:-1] + width / 2, cdf)
    # ax.plot(similarities_hist_edges[:-1], cdf, color='#000000', drawstyle="steps-pre")
    ax.stairs(cdf, similarities_hist_edges, color='#000000')
    ax.set_ylim(0, max(1, cdf[-1]))
    ax.set_ylabel('Cumulative Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        auc_cdf = float(cdf.sum() * width / (similarities_hist_edges[-1] - similarities_hist_edges[np.argmax(cdf >= min_cdf_threshold)]))
    else:
        auc_cdf = 0.0
    ax.text(0.02, 0.98, f'auc_cdf = {auc_cdf:.3g}', transform=ax.transAxes, ha='left', va='top')

    ax = fig.add_subplot(3, 2, 4)
    width = similarities_hist_edges[1] - similarities_hist_edges[0]
    ax.bar(similarities_hist_edges[:-1], pairwise_similarities_hist, width=width, align='edge', color='#000000', edgecolor='none', alpha=0.3)
    ax.set_xlim(similarities_hist_edges[0], similarities_hist_edges[-1])
    ax.set_xlabel('BLOSUM62/L (All vs All)')
    ax.set_ylabel('Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax = ax.twinx()
    cdf = pairwise_similarities_hist.cumsum()
    # ax.plot(similarities_hist_edges[:-1] + width / 2, cdf)
    # ax.plot(similarities_hist_edges[:-1], cdf, color='#000000', drawstyle="steps-pre")
    ax.stairs(cdf, similarities_hist_edges, color='#000000')
    ax.set_ylim(0, max(1, cdf[-1]))
    ax.set_ylabel('Cumulative Count')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        auc_cdf = float(cdf.sum() * width / (similarities_hist_edges[-1] - similarities_hist_edges[np.argmax(cdf >= min_cdf_threshold)]))
    else:
        auc_cdf = 0.0
    ax.text(0.02, 0.98, f'auc_cdf = {auc_cdf:.3g}', transform=ax.transAxes, ha='left', va='top')

    m = re.match(r'^[^\s/]+/(\d+)-(\d+)', aln[0].id)
    resi_start = int(m[1]) if m else 1
    ax = fig.add_subplot(3, 1, 3)
    seg_ends = np.nonzero(np.diff(is_site_covered))[0].tolist()
    seg_ends.append(len(is_site_covered) - 1)
    coverage_patches_height = 4
    coverage_patches_sep = 4
    coverage_patches_base_y = -(coverage_patches_sep + coverage_patches_height)
    patches = []
    seg_start = 0
    for seg_end in seg_ends:
        if is_site_covered[seg_start]:
            patches.append(Rectangle((seg_start + resi_start, coverage_patches_base_y), seg_end - seg_start, coverage_patches_height))
        # if not is_site_covered[seg_start]:
        #     ax.axvspan(seg_start + resi_start, seg_end + resi_start, color='#000000', alpha=0.1)
        seg_start = seg_end + 1
    ax.add_collection(PatchCollection(patches, color='#000000', edgecolor='none', alpha=0.3))
    ax.axhline(cutoff, linestyle='--', color='#000000')
    ax.plot(np.arange(resi_start, len(site_coverages) + resi_start), site_coverages, 'o', color='#000000', markeredgecolor='none')
    ax.set_xlim(resi_start, len(site_coverages) + resi_start - 1)
    ax.set_ylim(coverage_patches_base_y, 100)
    # ax.set_ylim(0, 100)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Column Coverage %')

    fig.tight_layout()
    with io_helpers.file_guard(png_fn) as tmp_png_fn:
        fig.savefig(tmp_png_fn)
    plt.close()


if __name__ == '__main__':
    aln_fn = sys.argv[1]
    draw_alignment_stats(aln_fn)
