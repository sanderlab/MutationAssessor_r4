import argparse
import yaml
import os
import re
import numpy as np
from numpy.typing import NDArray
import subprocess
import io_helpers
import gc
import math
import sys
import psutil
import utils
import make_msa

from scipy.special import loggamma
import numba
import clusterlogos
import matplotlib
import matplotlib.figure
import matplotlib.colors
import matplotlib.patches

ceo_clustering_bin = '/n/groups/marks/users/su/mar4/ceo-clustering/build/bin/ceo-clustering'
pdb_renum_bin = '/n/groups/marks/users/su/mar4/pipeline/pdb_renum.sh'
sifts_mapping_table = '/n/groups/marks/databases/SIFTS/pdb_chain_uniprot_plus_current.o2.csv'
size_type = np.uint16
entropy_type = np.float64
default_fis_c = 1
default_clusterlogos_c = -1


NPZ_onehot = "onehot"
NPZ_to_alignment_column = "to_alignment_column"
NPZ_N_ia = "N_ia"
NPZ_A = "A"
NPZ_Stilde_km_cache = "Stilde_km_cache"
NPZ_dS0 = "dS0"
NPZ_dS_km = "dS_km"
NPZ_dQ_km = "dQ_km"
NPZ_merge_km = "merge_km"

class CEOClusteringResult:
    def __init__(self, clustering_npz_file, alphabet):
        clustering_npz: dict[str, NDArray] = dict(np.load(clustering_npz_file))
        self.to_alignment_column = clustering_npz[NPZ_to_alignment_column]
        self.alphabet = alphabet
        self.onehot = np.swapaxes(clustering_npz[NPZ_onehot], 1, 2)
        self.N, self.L, self.D = self.onehot.shape
        self.N_ia = self.onehot.sum(axis=0)
        self.f_ia = self.N_ia / self.N

        self.merge_km = clustering_npz[NPZ_merge_km]
        self.full_dS0 = clustering_npz[NPZ_dS0]
        self._begin_steps = self.full_dS0.reshape(-1, self.N - 1).argmin(axis=0) * (self.N - 1)
        self.dS0_over_opt = self.full_dS0[self._begin_steps + np.arange(self.N - 1, dtype=int)] / self.full_dS0.min()

    def _select_clustering_steps(self, c):
        dS0 = self.full_dS0.reshape(-1, self.N - 1)
        begin = np.arange(0, len(self.full_dS0), self.N - 1, dtype=int)
        if c < 0:
            p = dS0.argmin(axis=1)
            x = np.linspace(-p/(self.N - 2 - p), 1.0, self.N - 1).T # start is chosen such that x[p] == 0 and x[N-2] == 1
            end = np.argmin(x * dS0, axis=1)
            num_clusters = self.N - 1 - end
            end += begin
        elif c <= 1.0:
            m = dS0 <= c * self.full_dS0.min()
            p = np.any(m, axis=1)
            num_clusters = 1 + np.argmax(m[p, ::-1], axis=1)
            begin = begin[p]
            end = self.N - 1 - num_clusters + begin
        else:
            num_clusters = np.full_like(begin, int(c))
            end = self.N - 1 - num_clusters + begin

        return begin, end, num_clusters

    def assign_cluster(self, c=1.0, max_specificity_cluster_size=0, onehot=None):
        @numba.jit(nopython=True, nogil=True, fastmath=True, error_model='numpy')
        def numba_assign_clusters(N, merge_km, begin, end, max_specificity_cluster_size=0):
            cluster_assignments = np.arange(N, dtype=size_type)
            cluster_sizes = np.ones(N, dtype=size_type)
            new_cluster = N - 2
            for i in range(begin, end + 1):
                k = merge_km[i, 0]
                m = merge_km[i, 1]
                
                new_cluster_size = cluster_sizes[k] + cluster_sizes[m]
                if (k == cluster_assignments[0]) or (m == cluster_assignments[0]):
                    if (max_specificity_cluster_size > 0) and (new_cluster_size > max_specificity_cluster_size):
                        break

                for j in range(k + 1, m):
                    cluster_sizes[j - 1] = cluster_sizes[j]
                for j in range(m + 1, new_cluster + 2):
                    cluster_sizes[j - 2] = cluster_sizes[j]
                cluster_sizes[new_cluster] = new_cluster_size

                for j in range(N):
                    if (cluster_assignments[j] == k) or (cluster_assignments[j] == m):
                        cluster_assignments[j] = new_cluster
                    elif (cluster_assignments[j] > k) and (cluster_assignments[j] < m):
                        cluster_assignments[j] -= 1
                    elif cluster_assignments[j] > m:
                        cluster_assignments[j] -= 2
                
                new_cluster -= 1
            cluster_sizes = cluster_sizes[0 : new_cluster + 2]
            num_clusters = len(cluster_sizes)

            # relabel clusters by descending size
            # ensure lead sequence is in cluster 0
            cluster_sizes[cluster_assignments[0]] += N
            sorted_indices = cluster_sizes.argsort()
            cluster_sizes[cluster_assignments[0]] -= N
            relabel_mapping = np.empty(num_clusters, dtype=size_type)
            for i in range(num_clusters):
                relabel_mapping[sorted_indices[i]] = num_clusters - 1 - i
            
            for i in range(N):
                cluster_assignments[i] = relabel_mapping[cluster_assignments[i]]

            N_k = np.empty(num_clusters, dtype=size_type)
            for i in range(num_clusters):
                N_k[relabel_mapping[i]] = cluster_sizes[i]
            
            return cluster_assignments, N_k

        @numba.jit(nopython=True, nogil=True, error_model='numpy')
        def calc_N_kia(onehot, cluster_assignments, M):
            N, L, D = onehot.shape
            N_kia = np.zeros((M, L, D), dtype=entropy_type)
            for n in range(N):
                N_kia[cluster_assignments[n]] += onehot[n, :, :]
            return N_kia

        if max_specificity_cluster_size > 0:
            begin = np.arange(0, len(self.full_dS0), self.N - 1, dtype=int)
            end = begin + np.argmin(self.full_dS0.reshape(-1, self.N - 1), axis=1)
            cluster_assignments = None
            N_k = None
            dS0 = 0.0
            for i in range(len(begin)):
                tmp_cluster_assignments, tmp_N_k = numba_assign_clusters(self.N, self.merge_km, begin[i], end[i], max_specificity_cluster_size)
                tmp_dS0 = self.full_dS0[begin[i] + self.N - 1 - len(tmp_N_k)]
                if tmp_dS0 < dS0:
                    cluster_assignments = tmp_cluster_assignments
                    N_k = tmp_N_k
                    dS0 = tmp_dS0
        else:
            begin, end, _ = self._select_clustering_steps(c)
            i = np.argmin(self.full_dS0[end])
            cluster_assignments, N_k = numba_assign_clusters(self.N, self.merge_km, begin[i], end[i], max_specificity_cluster_size)
        num_clusters = len(N_k)
        N_kia = calc_N_kia(self.onehot if onehot is None else onehot, cluster_assignments, num_clusters)
        f_kia = N_kia / N_k[:, np.newaxis, np.newaxis]
        return cluster_assignments, num_clusters, N_k, self.N_ia, self.f_ia, N_kia, f_kia


def extract_msa_cols(msa_file, alphabet, max_col_gap):
    @numba.jit(nopython=True, nogil=True)
    def find_duplicate_rows(a):
        is_dup = np.full(a.shape[0], False, numba.types.bool_)
        for i in range(a.shape[0]):
            if is_dup[i]:
                continue
            for j in range(i + 1, a.shape[0]):
                if is_dup[j]:
                    continue
                where_diff = -1
                for k in range(a.shape[1]):
                    if a[i, k] != a[j, k]:
                        where_diff = k
                        break
                if where_diff == -1:
                    is_dup[j] = True
        return is_dup

    aln = io_helpers.read_msa(msa_file, 'fasta')
    a = np.array(aln, dtype='S1').view(np.int8)
    to_alignment_column = np.nonzero(
        (np.count_nonzero(a == 45, axis=0) <= a.shape[0] * max_col_gap) &
        np.isin(a[0], alphabet)
    )[0].astype(size_type)
    return a[:, to_alignment_column], to_alignment_column


def parse_A(A_str):
    A_begin, A_end, A_step = map(float, A_str.split(','))
    lenA = math.floor((A_end - A_begin) / A_step) + 1
    return np.arange(A_begin, A_begin + lenA * A_step, A_step)


def estimate_memory_usage(msa_file, max_col_gap, alphabet, A_str, cpu):
    size_type_size = np.dtype(size_type).itemsize
    entropy_type_size = np.dtype(entropy_type).itemsize
    
    _alphabet = np.fromiter(alphabet, dtype='S1').view(np.int8)
    N, L = extract_msa_cols(msa_file, _alphabet, max_col_gap)[0].shape
    D = len(alphabet)
    lenA = len(parse_A(A_str))
    return max(5, math.ceil(1 + (
        psutil.Process().memory_info().rss
        + N * D * L * size_type_size # onehot
        + lenA * entropy_type_size # A
        + N * D * L * size_type_size # m_N_iak
        + D * L * size_type_size # m_N_ia
        + lenA * entropy_type_size # m_A
        + 2 * (N - 1) * lenA * size_type_size # m_result_merge_km
        + lenA * (N - 1) * entropy_type_size # m_result_dS0
        + lenA * (N - 1) * entropy_type_size # m_result_dS_km
        + lenA * (N - 1) * entropy_type_size # m_result_dQ_km
        + (N + 1) * entropy_type_size # m_logfactorial_cache
        + (N + 1) * entropy_type_size # m_log_cache
        + N * (N - 1) / 2 * entropy_type_size # m_dS_km_cache
        + (N + 1) * entropy_type_size # m_Stilde_km_cache
        + cpu * (N + 1) * D * L * size_type_size # per-thread N_iak
        + cpu * (N + 1) * size_type_size # per-thread N_k
        + cpu * (N + 1) * entropy_type_size # per-thread dS_k
        + cpu * N * (N - 1) / 2 * entropy_type_size # per-thread dS_km_cache
    ) / 1024 / 1024 / 1024))


def estimate_runtime(msa_file):
    N = sum(1 for _ in io_helpers.read_msa_iter(msa_file, 'fasta'))
    if N <= 2500:
        return 0.5
    elif N <= 5000:
        return 1.0
    elif N <= 10000:
        return 2.0
    elif N <= 15000:
        return 6.0
    else:
        return 13.0


def ceo_clustering(msa_file: str, ceo_dir: str, clustering_npz_file: str, max_col_gap: float, alphabet: str, A_str: str, force: bool, plot: bool, cpu: int):
    if force or (not os.path.isfile(clustering_npz_file)):
        _alphabet = np.fromiter(alphabet, dtype='S1').view(np.int8)
        a, to_alignment_column = extract_msa_cols(msa_file, _alphabet, max_col_gap)
        with io_helpers.file_guard(clustering_npz_file) as tmp_clustering_npz_file:
            in_npz_file = os.path.join(ceo_dir, '_clustering_in.npz')
            np.savez_compressed(in_npz_file, **{
                NPZ_A: parse_A(A_str), 
                NPZ_onehot: (a[:, np.newaxis, :] == _alphabet[np.newaxis, :, np.newaxis]).astype(size_type), 
                NPZ_to_alignment_column: to_alignment_column,
            })
            del a, to_alignment_column, _alphabet
            gc.collect()

            out_npz_file = os.path.join(ceo_dir, '_clustering_out.npz')
            subprocess.run([ceo_clustering_bin, '--threads', str(cpu), '--in-npz', in_npz_file, '--out-npz', out_npz_file], check=True)
            np.savez_compressed(tmp_clustering_npz_file, **dict(np.load(in_npz_file)), **dict(np.load(out_npz_file)))
        os.remove(in_npz_file)
        os.remove(out_npz_file)
    
    if plot:
        ceo_clustering_plot_file = io_helpers.splitext(clustering_npz_file)[0] + '.pdf'
        if force or (not os.path.isfile(ceo_clustering_plot_file)):
            pass


def fis(ceo_result: CEOClusteringResult, fis_c: float, ceo_dir: str, ceo_prefix: str, seq_id_ver: str, seq_start: int, seq_end: int, force: bool, plot: bool):
    fis_labels = {0: 'Neutral', 1: 'Low', 2: 'Medium', 3: 'High'}
    fis_colors = {0: '#FFFFFF', 1: '#E8E894', 2: '#C79060', 3: '#C83C3C'}
    norm = matplotlib.colors.BoundaryNorm([-8, 2.6, 5.25, 7, 9], ncolors=4, clip=True, extend='neither')
    from_aa_idx = ceo_result.onehot[0, :, :].argmax(axis=-1)
    all_idx = np.arange(ceo_result.L, dtype=size_type)

    fis_npz_file = os.path.join(ceo_dir, ceo_prefix + '_FIS.npz')
    fis_csv_file = os.path.join(ceo_dir, ceo_prefix + '_FIS.csv')
    if force or (not os.path.isfile(fis_npz_file)) or (not os.path.isfile(fis_csv_file)):
        _, _, _, N_ia, _, N_kia, _ = ceo_result.assign_cluster(fis_c)
        conservation_score = np.log(N_ia[all_idx, from_aa_idx][:, np.newaxis] / (1. + N_ia))
        conservation_score[all_idx, from_aa_idx] = 0.
        specificity_score = np.log(N_kia[0, all_idx, from_aa_idx][:, np.newaxis] / (1. + N_kia[0]))
        specificity_score[all_idx, from_aa_idx] = 0.
        fis = (conservation_score + specificity_score) / 2

        to_aa = np.fromiter(ceo_result.alphabet, dtype='U1')
        from_aa = to_aa[from_aa_idx]
        residue = seq_start + ceo_result.to_alignment_column
        
        with io_helpers.file_guard(fis_npz_file) as tmp_fis_npz_file:
            np.savez_compressed(
                tmp_fis_npz_file, 
                FIS=fis, 
                conservation_score=conservation_score, 
                specificity_score=specificity_score, 
                from_aa=from_aa, 
                to_aa=to_aa, 
                residue_num=residue
            )

        gap_idx = ceo_result.alphabet.index('-')
        with io_helpers.file_guard(fis_csv_file) as tmp_fis_csv_file:
            with open(tmp_fis_csv_file, 'w') as f:
                f.write('residue,from,to,FIS,conservation score,specificity score,functional impact\n')
                for i in range(ceo_result.L):
                    for j in range(ceo_result.D):
                        if j != gap_idx:
                            f.write(f'{residue[i]},{from_aa[i]},{to_aa[j]},{fis[i, j]},{conservation_score[i, j]},{specificity_score[i, j]},{fis_labels[norm(fis[i, j])]}\n')
    else:
        npz = dict(np.load(fis_npz_file))
        fis = npz['FIS']
        residue = npz['residue_num']
        from_aa = npz['from_aa']
        to_aa = npz['to_aa']

    if plot:
        fis_plot_file = os.path.join(ceo_dir, ceo_prefix + '_FIS.pdf')
        if force or (not os.path.isfile(fis_plot_file)):
            figure_width = ceo_result.D * 1/4 + 11/16
            figure_height = ceo_result.L * 1/4 + 19/16
            fig = matplotlib.figure.Figure(figsize=(figure_width, figure_height), frameon=False)
            ax = fig.add_subplot()
            ax.set_title(ceo_prefix, pad=40)
            cmap = matplotlib.colors.ListedColormap(list(fis_colors.values()))
            handles = [matplotlib.patches.Rectangle((0, 0), 0, 0, facecolor=fis_colors[i], edgecolor='#000000') for i in fis_labels.keys()]
            ax.legend(
                handles, fis_labels.values(), bbox_to_anchor=(-0.5, -2.5, ceo_result.D, 1), bbox_transform=ax.transData, loc="lower left", 
                ncol=len(fis_labels), frameon=False, facecolor='none', mode="expand", borderpad=0, handlelength=1, handleheight=1#, alignment='left'
            )#, borderaxespad=0
            ax.imshow(fis, aspect='equal', norm=norm, cmap=cmap)
            ax.plot(from_aa_idx, all_idx, 'o', markersize=3, color='dimgray')
            ax.set_xticks(np.arange(ceo_result.D + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(ceo_result.L + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="#000000", linestyle='-', linewidth=1, clip_on=False)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticks(np.arange(ceo_result.D), to_aa)
            ax.set_yticks(np.arange(ceo_result.L), [f'{i[0]} {i[1]}' for i in zip(residue, from_aa)])
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='both', which='both', length=0)
            ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
            ax.spines[:].set_visible(False)
            with io_helpers.file_guard(fis_plot_file) as tmp_fis_plot_file:
                fig.savefig(tmp_fis_plot_file, bbox_inches='tight')

        
def calc_dS_i(N_ia, N_kia, N, N_k):
    return (loggamma(1. + N_ia[np.newaxis, :, :] * N_k[:, np.newaxis, np.newaxis] / N) - loggamma(1. + N_kia)).sum(axis=(0, 2))


def draw_clusterlogos(ceo_result: CEOClusteringResult, clusterlogos_c: float, msa_file: str, ceo_dir: str, ceo_prefix: str, seq_id_ver: str, seq_start: int, seq_end: int, force: bool):
    clusterlogos_file = os.path.join(ceo_dir, ceo_prefix + f'_CEO@{clusterlogos_c}_clusterlogos.pdf')
    if force or (not os.path.isfile(clusterlogos_file)):
        aln = io_helpers.read_msa(msa_file, 'fasta')
        a = np.array(aln, dtype='S1').view(np.int8)
        _alphabet = np.fromiter(ceo_result.alphabet, dtype='S1').view(np.int8)
        onehot_full = (a[:, :, np.newaxis] == _alphabet[np.newaxis, np.newaxis, :]).astype(size_type)
        _, _, _, _, _, _, f_kia_full = ceo_result.assign_cluster(clusterlogos_c, onehot=onehot_full)

        _, num_clusters, N_k, _, _, _, _ = ceo_result.assign_cluster(clusterlogos_c)

        cluster_labels = [f'Cluster {i}' for i in range(1, num_clusters + 1)]
        cluster_labels = np.array(cluster_labels)


        title = ceo_prefix + f' CEO@{clusterlogos_c} {num_clusters} Clusters'
        with io_helpers.file_guard(clusterlogos_file) as tmp_clusterlogos_file:
            clusterlogos.draw(
                tmp_clusterlogos_file, f_kia_full, N_k, cluster_labels, lead_seq=aln[0], column_order_by=None, title=title, 
                max_clusters=50, cluster_size_threshold=0.02, row_margin=2, column_margin=1.4, char_margin=0.3, char_stroke_width=1.5, xscale=1.4, yscale=1, 
                tick_size=1.5, top_panel_height=0.25, hspace=0.045, dS_ki_fraction=None, highlight_resi=None, pinned_clusters=None, 
            )



def ceo(job_name, msa_file, ceo_dir, ceo_prefix, max_col_gap, alphabet, A_str, actions, fis_c, clusterlogos_c, plot, cpu, runtime, mem):
    clustering_npz_file = os.path.join(ceo_dir, ceo_prefix + '_ceo_clustering.npz')

    force = ('CLUSTERING' in actions)
    if force or (len(actions) == 0):
        utils.set_job_status(ceo_dir, 'CLUSTERING')
        ceo_clustering(msa_file, ceo_dir, clustering_npz_file, max_col_gap, alphabet, A_str, force, plot, cpu)

    ceo_result = CEOClusteringResult(clustering_npz_file, alphabet)

    for target_seqrec in io_helpers.read_msa_iter(msa_file, 'fasta'):
        break
    assert target_seqrec
    m = re.match(r'^([^/]+)/(\d+)-(\d+)', target_seqrec.id)
    if m:
        seq_id_ver, seq_start, seq_end = m[1], int(m[2]), int(m[3])
        seq_id = seq_id_ver
        if '.' in seq_id_ver:
            seq_id = seq_id_ver[:seq_id_ver.index('.')]
    else:
        m = re.match(r'^(?:sp|tr)\|([^|]+)\|.+ SV=(\d+)', target_seqrec.id)
        if m:
            seq_id_ver, seq_start, seq_end = m[1], int(m[2]), int(m[3])
            seq_id = seq_id_ver
            if '.' in seq_id_ver:
                seq_id = seq_id_ver[:seq_id_ver.index('.')]
        else:
            seq_id_ver = target_seqrec.id
            seq_id = seq_id_ver
            seq_start = 1
            seq_end = len(target_seqrec.seq)

    force = ('FIS' in actions)
    if force or (len(actions) == 0):
        utils.set_job_status(ceo_dir, 'FIS')
        fis(ceo_result, fis_c, ceo_dir, ceo_prefix, seq_id_ver, seq_start, seq_end, force, plot)
    
    force = ('CLUSTERLOGOS' in actions)
    if force or (len(actions) == 0):
        utils.set_job_status(ceo_dir, 'CLUSTERLOGOS')
        draw_clusterlogos(ceo_result, clusterlogos_c, msa_file, ceo_dir, ceo_prefix, seq_id_ver, seq_start, seq_end, force)

    utils.set_job_status(ceo_dir, 'DONE')


def submit_slurm_job(job_name, msa_file, ceo_dir, ceo_prefix, max_col_gap, alphabet, A_str, actions, fis_c, clusterlogos_c, plot, cpu, runtime, mem):
    active_job_ids = utils.get_active_slurm_jobs()
    slurm_info = utils.get_slurm_info(ceo_dir)
    if 'SLURM_JOBID' in slurm_info:
        if slurm_info['SLURM_JOBID'] in active_job_ids:
            return 'NOT SUBMITTED'
    
    cmd_list = [
        'timeout', 
        '-k', 
        '0.05h', 
        f'{runtime - 0.1}h', 
        'python', os.path.abspath(__file__),
        '--local',
        '--job-name', job_name,
        '--msa-file', msa_file,
        '--output-dir', ceo_dir, 
        '--ceo-prefix', ceo_prefix, 
        '--cpu', str(cpu),
        '--mem', str(mem),
        '--time', str(runtime),
    ]

    if max_col_gap is not None:
        cmd_list.extend(['--max-col-gap', str(max_col_gap)])
    
    if A_str:
        cmd_list.extend(['-A', A_str])
    
    if 'CLUSTERING' in actions:
        cmd_list.append('--clustering')

    if 'FIS' in actions:
        cmd_list.extend(['--fis', str(fis_c)])

    if 'CLUSTERLOGOS' in actions:
        cmd_list.extend(['--clusterlogos', str(clusterlogos_c)])

    if plot:
        cmd_list.append('--plot')

    cmd_str = subprocess.list2cmdline(filter(None, cmd_list))

    partition = 'short' if runtime <= 12 else 'medium' if runtime <= 120 else 'long'
    days = int(runtime / 24)
    hours = int(runtime - days * 24)
    minutes = int((runtime - days * 24 - hours) * 60)
    seconds = int((runtime - days * 24 - hours) * 3600 - minutes * 60)
    sbatch_runtime = f'{days}-{hours:02}:{minutes:02}:{seconds:02}'
    stdout_file = os.path.join(ceo_dir, 'stdout_%j.txt')
    stderr_file = os.path.join(ceo_dir, 'stderr_%j.txt')

    if not os.path.isdir(ceo_dir):
        os.makedirs(ceo_dir)

    slurm_script_file = os.path.join(ceo_dir, 'slurm_ceo.sh')
    with open(slurm_script_file, 'w') as f:
        f.write(f'''#!/bin/bash 
#SBATCH -c {cpu}                      # Request one core
#SBATCH -t {sbatch_runtime}               # Runtime in D-HH:MM format
#SBATCH --mem={mem}G
#SBATCH -p {partition}
#SBATCH -o "{stdout_file}"     # Standard output and error log
#SBATCH -e "{stderr_file}"     # Standard output and error log

module load gcc/9.2.0

# source /n/groups/marks/software/anaconda_o2/bin/activate py310ys
source /n/groups/marks/users/su/mambaforge/etc/profile.d/conda.sh
conda activate py310
cd "{ceo_dir}"

echo JOB: {job_name}
echo CPU: {cpu}
echo MEMORY: {mem}G
echo RUNTIME: {runtime}
echo COMMAND: {cmd_str}

{cmd_str}
exitcode=$?
if [[ ${{exitcode}} -eq 124 ]]; then
  echo exit_code=0 >> slurm.txt
  echo "*** Task timed out DUE TO TIME LIMIT ***" >> "{stderr_file.replace('%j', '${SLURM_JOB_ID}')}"
else
  echo exit_code=${{exitcode}} >> slurm.txt
fi
echo MaxRSS=$(sstat -j ${{SLURM_JOB_ID}} --format=MaxRSS -a -n | sort -h -r | head -n 1) >> slurm.txt
exit ${{exitcode}}
''')

    r = subprocess.run(['sbatch', '-J', job_name, slurm_script_file], capture_output=True, text=True, check=True)
    m = re.search(r'\d+', r.stdout)
    slurm_job_id = m[0] if m else '0'

    slurm_info_file = os.path.join(ceo_dir, 'slurm.txt')
    stdout_file = os.path.basename(stdout_file).replace('%j', slurm_job_id)
    stderr_file = os.path.basename(stderr_file).replace('%j', slurm_job_id)
    with open(slurm_info_file, 'w') as f:
        f.write(f'''SLURM_JOBID={slurm_job_id}
SLURM_MEM_PER_NODE={mem}
SLURM_CPUS_PER_TASK={cpu}
time={runtime}
stdout={stdout_file}
stderr={stderr_file}
''')

    return slurm_job_id


def run(args_list):
    parser = argparse.ArgumentParser(description="CEO")
    parser.add_argument('msa_file_or_name', type=str, nargs='?')
    parser.add_argument('--job-name', type=str)
    parser.add_argument('--msa-file', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--ceo-prefix', type=str)
    parser.add_argument('--max-col-gap', type=float, default=0.3)
    parser.add_argument('-A', type=str, default='0.5,0.975,0.025')
    parser.add_argument('--clustering', action='store_true')
    parser.add_argument('--fis', type=float, nargs='?', const=default_fis_c)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--clusterlogos', type=float, nargs='?', const=default_clusterlogos_c)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--mem', type=int)
    parser.add_argument('--time', type=float)
    parser.add_argument('--estimate-mem', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--slurm-info', action='store_true')
    parser.add_argument('--msa-min-id', type=float, default=30.0)
    parser.add_argument('--msa-aa7', action='store_true')
    parser.add_argument('--msa-vertebrate', action='store_true')
    parser.add_argument('--find-ceo', action='store_true')
    args = parser.parse_args(args_list)

    root_dir = '/n/groups/marks/users/su/mar4'
    pdb_repo_dir = os.path.join(root_dir, 'pdb')
    alphafold_repo_dir = os.path.join(root_dir, 'alphafold')

    msa_file_or_name = args.msa_file or args.msa_file_or_name
    if os.path.isfile(msa_file_or_name):
        msa_file = msa_file_or_name
        msa_name = io_helpers.splitext(os.path.basename(msa_file))[0]
        ceo_dir = args.output_dir
    else:
        msa_name = msa_file_or_name
        msa_file = None
        msa_name_base = msa_name if '/' not in msa_name else msa_name[:msa_name.index('/')]
        ceo_dir = os.path.join(
            root_dir, 
            'ceo' + str(args.msa_min_id) + ('_vertebrate' if args.msa_vertebrate else '') + ('_aa7' if args.msa_aa7 else ''), 
            *(msa_name_base[2:5]), 
            msa_name_base, 
            msa_name.replace('/', '_')
        )

    if not ceo_dir:
        ceo_dir = os.getcwd()

    if args.status:
        return utils.get_job_status(ceo_dir)

    if args.slurm_info:
        return utils.get_slurm_info(ceo_dir)

    if msa_file is None:
        cmd = ['--find-msa', msa_name, '--min-id', str(args.msa_min_id)]
        if args.msa_aa7:
            cmd.append('--aa7')
        if args.msa_vertebrate:
            cmd.append('--vertebrate')
        msa_file_info = make_msa.run(cmd)
        if msa_file_info is None:
            raise FileNotFoundError(msa_name, msa_file_info)
        else:
            msa_file = msa_file_info['file']

    if not os.path.isfile(msa_file) and not os.path.isfile(msa_file.replace('.fa.xz', '.desc.fa.br')):
        raise FileNotFoundError(msa_file)

    if not os.path.isdir(ceo_dir):
        os.makedirs(ceo_dir)

    job_name = args.job_name if args.job_name is not None else 'ceo{}{}:{}'.format(
        ('_vertebrate' if args.msa_vertebrate else ''), 
        ('_aa7' if args.msa_aa7 else ''), 
        msa_name
    )
    ceo_prefix = args.ceo_prefix if args.ceo_prefix is not None else io_helpers.splitext(os.path.basename(msa_file))[0]

    if args.find_ceo:
        if utils.get_job_status(ceo_dir) != 'DONE':
            return None
        return os.path.join(ceo_dir, ceo_prefix)

    fis_c = args.fis or default_fis_c
    clusterlogos_c = args.clusterlogos or default_clusterlogos_c
    actions = []
    if args.clustering:
        actions.append('CLUSTERING')
    if args.fis:
        actions.append('FIS')
    if args.clusterlogos:
        actions.append('CLUSTERLOGOS')

    cpu = args.cpu or len(os.sched_getaffinity(0))

    mem = args.mem
    if (not mem) or args.estimate_mem:
        if (utils.get_job_status(ceo_dir) in ['', 'CLUSTERING']) or ('CLUSTERING' in actions):
            mem = estimate_memory_usage(msa_file, args.max_col_gap, utils.alphabet, args.A, cpu)
        else:
            mem = 3
    
    if args.estimate_mem:
        return mem

    runtime = args.time
    if not runtime:
        runtime = estimate_runtime(msa_file)

    run = ceo if args.local else submit_slurm_job
    return run(
        job_name=job_name, 
        msa_file=msa_file, 
        ceo_dir=ceo_dir, 
        ceo_prefix=ceo_prefix, 
        max_col_gap=args.max_col_gap, 
        alphabet=utils.alphabet, 
        A_str=args.A, 
        actions=actions, 
        fis_c=fis_c, 
        clusterlogos_c=clusterlogos_c, 
        plot=args.plot,
        cpu=cpu, 
        runtime=runtime, 
        mem=mem
    )

if __name__ == '__main__':
    r = run(sys.argv[1:])
    if r is not None:
        print(r)
