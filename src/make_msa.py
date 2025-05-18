import argparse
import yaml
import os
import sys
import gc
import re
import math
import numpy as np
import json
import subprocess
import signal
import io_helpers
from msa_stats_mt import draw_alignment_stats
import utils
from typing import Optional, Callable
from collections.abc import Iterable, Mapping
import itertools
import shutil
import filelock
import glob

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

hhblits_db = '/n/groups/marks/users/su/mar4/uniclust30/UniRef30_2022_02_hhsuite/UniRef30_2022_02'
uniref100_vertebrata_db = '/n/groups/marks/users/su/mar4/msa/uniref100.id.vertebrata.txt.xz'

jackhmmer_bin = '/n/groups/marks/pipelines/evcouplings/software/hmmer-3.1b2-linux-intel-x86_64/binaries/jackhmmer'
hhblits_bin = '/programs/x86_64-linux/system/biogrids_bin/hhblits'
reformat_pl = '/programs/x86_64-linux/system/biogrids_bin/reformat.pl'
hmmbuild_bin = '/programs/x86_64-linux/system/biogrids_bin/hmmbuild'
hmmsearch_bin = '/programs/x86_64-linux/system/biogrids_bin/hmmsearch'
hmmalign_bin = '/programs/x86_64-linux/system/biogrids_bin/hmmalign'
valgrind_bin = '/usr/bin/valgrind'

rng = np.random.default_rng()
_target_identity_histogram_bin_edges = np.arange(102)

# to be initialized when first used
uniref100_vertebrata = None
aa7_mapping = None


class MsaRepo:
    def __init__(self, msa_dir, seq_file, max_row_gap, init_b, b_step_sizes, min_ids, vertebrata_min_ids, cpu, mem, runtime):
        self.msa_dir = msa_dir
        self.repo_file = os.path.join(msa_dir, 'summary.json')
        self.seq_file = seq_file
        self.L = len(io_helpers.read_seq(seq_file, 'fasta'))
        self.max_row_gap = max_row_gap
        self.init_b = init_b
        self.b_step_sizes = b_step_sizes
        self.cpu = cpu
        self.mem = mem
        self.runtime = runtime
        self.msa_repo = {
            'msa': {}, 
            'min_id': {},
            'aa7_min_id': {},
            'vertebrata_min_id': {},
            'vertebrata_aa7_min_id': {},
            'history': [],
        }
        if os.path.isfile(self.repo_file):
            try:
                with open(self.repo_file) as f:
                    repo = json.load(f)
                    is_valid_repo = True
                    for key in ['msa', 'min_id', 'aa7_min_id', 'vertebrata_min_id', 'vertebrata_aa7_min_id', 'history']:
                        if key not in repo:
                            is_valid_repo = False
                            break
                    
                    if is_valid_repo:
                        for b in repo['msa']:
                            for key in ['raw', 'filtered', 'aa7_filtered', 'vertebrata_filtered', 'vertebrata_aa7_filtered']:
                                if key not in repo['msa'][b]:
                                    is_valid_repo = False
                                    break

                    if is_valid_repo:
                        self.msa_repo.update(repo)
                    elif 'msa' in repo:
                        self.add_msa([float(b) for b in repo['msa'] if repo['msa'][b]['raw']['N'] == 0])

                    for section in ['msa', 'min_id', 'aa7_min_id', 'vertebrata_min_id', 'vertebrata_aa7_min_id']:
                        for k in list(self.msa_repo[section].keys()):
                            self.msa_repo[section][float(k)] = self.msa_repo[section][k]
                            del self.msa_repo[section][k]
                    if 'history' not in self.msa_repo:
                        self.msa_repo['history'] = []
                        if 'last_b' in self.msa_repo:
                            self.msa_repo['history'].append(self.msa_repo['last_b'])
                            del self.msa_repo['last_b']
            except:
                pass
        self.set_min_ids(min_ids, vertebrata=False, aa7=False)
        self.set_min_ids(min_ids, vertebrata=True, aa7=False)
        self.set_min_ids(min_ids, vertebrata=False, aa7=True)
        self.set_min_ids(min_ids, vertebrata=True, aa7=True)

    def set_min_ids(self, min_ids, vertebrata=False, aa7=False):
        _, min_id_type = self.get_msa_and_min_id_type(vertebrata, aa7)
        modified = False
        for b in min_ids:
            if b not in self.msa_repo[min_id_type]:
                self.msa_repo[min_id_type][b] = 0
                modified = True
        if modified:
            self.save()
    
    def add_to_history(self, b):
        if (len(self.msa_repo['history']) == 0) or (self.msa_repo['history'][-1] != abs(b)):
            self.msa_repo['history'].append(b)
        self.save()
    
    def add_msa(self, bitscores, entry=None):
        def _recursive_update(d, u):
            try:
                for k, v in u.items():
                    assert k in d, print('Key', k, 'not found', flush=True)
                    if isinstance(v, Mapping):
                        assert isinstance(d[k], Mapping), print('Key', k, 'is not Mapping', flush=True)
                        _recursive_update(d[k], v)
                    else:
                        d[k] = v
            except:
                print(k)
                print(d)
                print(u)
                raise

        if not isinstance(bitscores, Iterable):
            bitscores = [bitscores]
        
        for b in bitscores:
            r = {}
            for k in ['raw', 'filtered', 'aa7_filtered', 'vertebrata_filtered', 'vertebrata_aa7_filtered']:
                r[k] = {
                    'b': b,
                    'file': '',
                    'N': 0,
                    'L': self.L,
                    'min_id': 0,
                    'auc_cdf': 0.0,
                    'n_cpus': self.cpu,
                    'mem': self.mem,
                    'runtime': self.runtime,
                }
            for k in ['filtered', 'aa7_filtered', 'vertebrata_filtered', 'vertebrata_aa7_filtered']:
                r[k]['max_row_gap'] = self.max_row_gap

            if entry is not None:
                _recursive_update(r, entry)

            self.add_to_history(b)
            self.msa_repo['msa'][b] = r

        self.update_min_id_links(vertebrata=False, aa7=False)
        self.update_min_id_links(vertebrata=False, aa7=True)
        self.update_min_id_links(vertebrata=True, aa7=False)
        self.update_min_id_links(vertebrata=True, aa7=True)
        self.save()
    
    def get_msa_and_min_id_type(self, vertebrata=False, aa7=False):
        msa_type = 'filtered'
        min_id_type = 'min_id'
        
        if aa7:
            min_id_type = 'aa7_' + min_id_type
            msa_type = 'aa7_' + msa_type
        
        if vertebrata:
            min_id_type = 'vertebrata_' + min_id_type
            msa_type = 'vertebrata_' + msa_type
        
        return msa_type, min_id_type
    
    def update_min_id_links(self, vertebrata=False, aa7=False):
        msa_type, min_id_type = self.get_msa_and_min_id_type(vertebrata, aa7)
        for desired_m, b in self.msa_repo[min_id_type].items():
            for test_b in self.msa_repo['msa']:
                if test_b == b:
                    continue
                
                if b == 0:
                    self.msa_repo[min_id_type][desired_m] = test_b
                    continue

                if self.msa_repo['msa'][test_b][msa_type]['N'] < 2:
                    if (self.msa_repo['msa'][b][msa_type]['N'] < 2) and (test_b < b):
                        self.msa_repo[min_id_type][desired_m] = test_b
                    continue

                if self.msa_repo['msa'][b][msa_type]['N'] < 2:
                    self.msa_repo[min_id_type][desired_m] = test_b
                    continue

                m = self.msa_repo['msa'][b][msa_type]['min_id']
                diff = abs(desired_m - m)
                test_m = self.msa_repo['msa'][test_b][msa_type]['min_id']
                test_diff = abs(desired_m - test_m)

                if (
                    (test_diff < diff)
                    or ((test_diff == diff) and (test_m < m))
                    or ((test_diff == diff) and (test_m == m) and (m < desired_m) and (test_b > b))
                    or ((test_diff == diff) and (test_m == m) and (m > desired_m) and (test_b < b))
                ):
                    self.msa_repo[min_id_type][desired_m] = test_b

    def find_msa_by_min_id(self, min_id=None, vertebrata=False, aa7=False):
        msa_type, min_id_type = self.get_msa_and_min_id_type(vertebrata, aa7)
        if min_id is None:
            return [self.msa_repo['msa'][b][msa_type] for b in set(self.msa_repo[min_id_type].values())]
        else:
            b = self.msa_repo[min_id_type][min_id]
            return self.msa_repo['msa'][b][msa_type]
    
    def get_last_bitscore(self):
        if len(self.msa_repo['history']) > 0:
            return self.msa_repo['history'][-1]
        else:
            return None
    
    def get_last_msa(self, nonempty=False):
        for b in reversed(self.msa_repo['history']):
            entry = self.msa_repo['msa'][b]
            if (not nonempty) or (entry['filtered']['N'] > 0):
                return entry
        return None

    def get_pending_bitscores(self, recovery_mode=False):
        pending_bitscores = []
        if recovery_mode:
            for msa_file in glob.glob(os.path.join(self.msa_dir, '*.fa.xz')):
                m = re.search(r'_b([^_]+)\.fa\.xz', msa_file)
                if m is None:
                    continue
                b = float(m[1])
                if b not in self.msa_repo['msa']:
                    pending_bitscores.append(b)
        else:
            # if len(self.msa_repo['msa']) == 0:
            if self.init_b not in self.msa_repo['msa']:
                pending_bitscores.append(self.init_b)
            else:
                for desired_m in self.msa_repo['min_id']:
                    b = self.msa_repo['min_id'][desired_m]
                    m = self.msa_repo['msa'][b]['filtered']['min_id']
                    if desired_m == m:
                        continue

                    if (m > desired_m) or (self.msa_repo['msa'][b]['filtered']['N'] == 0):
                        sign = -1
                    else:
                        sign = 1
                    
                    exponent = math.floor(math.log10(b))
                    if (sign == -1) and (b == 10 ** exponent):
                        exponent -= 1
                    exponent = min(-1, max(-2, exponent))
                    prev_step_size = 0
                    for step_size in self.b_step_sizes:
                        new_b = round(b + sign * step_size * (10 ** exponent), -exponent)
                        if (new_b in self.msa_repo['msa']) or (new_b == 0):
                            break
                        prev_step_size = step_size
                    if prev_step_size != 0:
                        new_b = round(b + sign * prev_step_size * (10 ** exponent), -exponent)
                        if (new_b > 0) and (new_b not in pending_bitscores):
                            pending_bitscores.append(new_b)
        return pending_bitscores

    def save(self):
        with open(self.repo_file, 'w') as f:
            self.msa_repo['msa'] = dict(sorted(self.msa_repo['msa'].items()))
            self.msa_repo['min_id'] = dict(sorted(self.msa_repo['min_id'].items()))
            json.dump(self.msa_repo, f, indent=2)
        

def batched_map(func: Callable, iterable: Iterable, batch_size: int=100000, args: tuple=()):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield from func(chunk, *args)
        del chunk
        gc.collect()


def make_focus_msa(target: SeqRecord, msa_iter: Iterable[SeqRecord]):
    target_np = np.array(target, dtype="S1").view(np.int8)
    use_cols = np.nonzero(target_np >= 65)[0]
    insertion_cols = np.nonzero(target_np >= 97)[0]
    target_np[insertion_cols] -= 32
    target.letter_annotations.clear()
    target.seq = Seq(target_np[use_cols].view(f'S{len(use_cols)}').item())
    yield target
    
    def _filter(aln, use_cols, insertion_cols):
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        a[:, insertion_cols] = 45
        a = a[:, use_cols].ravel()
        assert np.all(a < 97) and np.all(a != 46)
        a = a.view(f'S{len(use_cols)}')
        for i, rec in enumerate(aln):
            rec.letter_annotations.clear()
            rec.seq = Seq(a[i])
        return aln

    yield from batched_map(_filter, (rec for rec in msa_iter if rec.id != target.id), args=(use_cols, insertion_cols))


def filter_msa(msa_iter: Iterable[SeqRecord], alphabet: str='ACDEFGHIKLMNPQRSTVWY-', max_row_gap: float=0.3):
    target = next(msa_iter)
    target_np = np.array(target, dtype="S1").view(np.int8)
    alphabet_np = np.fromiter(alphabet, dtype='S1').view(np.int8)
    use_cols = np.isin(target_np, alphabet_np)
    yield target

    def _filter(aln, alphabet_np, use_cols, max_row_gap):
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        kept_idx = np.nonzero(
            (np.count_nonzero(a == 45, axis=1) <= a.shape[1] * max_row_gap)
            & (np.isin(a[:, use_cols], alphabet_np).all(axis=1))
        )[0]
        return (aln[i] for i in kept_idx)

    yield from batched_map(_filter, (rec for rec in msa_iter if rec.id != target.id), args=(alphabet_np, use_cols, max_row_gap))


def filter_msa_vertebrata(target: SeqRecord, msa_iter: Iterable[SeqRecord]):
    global uniref100_vertebrata_db
    global uniref100_vertebrata
    if uniref100_vertebrata is None:
        with io_helpers.universal_open(uniref100_vertebrata_db, 'rt') as f:
            uniref100_vertebrata = set(line.strip() for line in f)

    yield target
    for rec in msa_iter:
        if (rec.id != target.id) and (rec.id[0:rec.id.index('/')] in uniref100_vertebrata):
            yield rec


def filter_msa_above_id(target: SeqRecord, msa_iter: Iterable[SeqRecord], min_id_threshold: float=30):
    target_np = np.array(target, dtype="S1").view(np.int8)
    yield target

    def _filter(aln, target_np, min_id_threshold):
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        target_identities = np.count_nonzero(a == target_np, axis=1) / a.shape[1] * 100
        kept_idx = np.nonzero(target_identities >= min_id_threshold)[0]
        return (aln[i] for i in kept_idx)

    yield from batched_map(_filter, (rec for rec in msa_iter if rec.id != target.id), args=(target_np, min_id_threshold))


def filter_msa_by_id_weighting(target: SeqRecord, msa_iter: Iterable[SeqRecord], min_id_threshold: float=30):
    target_np = np.array(target, dtype="S1").view(np.int8)
    yield target

    def _filter(aln, target_np, min_id_threshold):
        global rng
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        target_identities = np.count_nonzero(a == target_np, axis=1) / a.shape[1] * 100
        kept_idx = np.nonzero(rng.uniform(0, 100, size=a.shape[0]) <= target_identities)[0]
        return (aln[i] for i in kept_idx)

    yield from batched_map(_filter, (rec for rec in msa_iter if rec.id != target.id), args=(target_np, min_id_threshold))


def filter_msa_to_aa7(msa_iter: Iterable[SeqRecord]):
    global aa7_mapping
    if aa7_mapping is None:
        aa7_mapping = np.zeros(256, dtype=np.int8)
        aa7_mapping[[ord('A'), ord('C'), ord('S'), ord('T')]] = ord('A')
        aa7_mapping[[ord('D'), ord('E'), ord('N'), ord('Q')]] = ord('D')
        aa7_mapping[[ord('F'), ord('H'), ord('W'), ord('Y')]] = ord('F')
        aa7_mapping[ord('G')] = ord('G')
        aa7_mapping[[ord('I'), ord('L'), ord('M'), ord('V')]] = ord('I')
        aa7_mapping[[ord('K'), ord('R')]] = ord('K')
        aa7_mapping[ord('P')] = ord('P')
        aa7_mapping[ord('-')] = ord('-')

    def _filter(aln):
        global aa7_mapping
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        L = a.shape[1]
        a = aa7_mapping[a.ravel()]
        a = a.view(f'S{L}')
        for i, rec in enumerate(aln):
            rec.letter_annotations.clear()
            rec.seq = Seq(a[i])
        return aln

    yield from batched_map(_filter, msa_iter, args=())


def min_id_and_auc(target: SeqRecord, msa_iter: Iterable[SeqRecord], threshold: float=0.01):
    global _target_identity_histogram_bin_edges

    def _filter(aln, target_np):
        a = np.array(aln, dtype='S1', order='C').view(np.int8)
        target_identities = np.count_nonzero(a == target_np, axis=1) / a.shape[1] * 100
        h, _ = np.histogram(target_identities, _target_identity_histogram_bin_edges, density=False)
        return (h,)
    
    target_np = np.array(target, dtype='S1', order='C').view(np.int8)
    hist = np.zeros(len(_target_identity_histogram_bin_edges) - 1, dtype=np.int_)
    for h in batched_map(_filter, (rec for rec in msa_iter if rec.id != target.id), args=(target_np,)):
        hist += h
    cdf = hist.cumsum()
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
        min_id = float(_target_identity_histogram_bin_edges[np.argmax(cdf >= threshold)])
        auc_cdf = float(cdf.sum() / (_target_identity_histogram_bin_edges[-1] - min_id))
    else:
        min_id = 0.0
        auc_cdf = 0.0
    return min_id, auc_cdf


def jackhmmer(seq_file: str, bitscore: float, msa_file: str, sequence_db: str, num_cpus: int):
    if os.path.isfile(msa_file):
        return

    with io_helpers.file_guard(msa_file) as tmp_msa_file:
        jackhmmer_msa_prefix = io_helpers.splitext(tmp_msa_file)[0]
        num_iterations = 5
        cmd = [
            jackhmmer_bin, '-N', str(num_iterations), '--cpu', str(num_cpus), 
            '-o', os.devnull, '--noali', '--notextw', 
            '-A', tmp_msa_file, 
            '-T', str(bitscore), '--domT', str(bitscore), '--incT', str(bitscore), '--incdomT', str(bitscore), 
            #'--tblout', os.devnull, '--domtblout', os.devnull, 
            #'--chkhmm', jackhmmer_msa_prefix, 
            # '--chkali', jackhmmer_msa_prefix, 
            seq_file, sequence_db
        ]
        clean_up = []
        jackhmmer_result_files = [tmp_msa_file]
        for i in range(num_iterations, 0, -1):
            fn = jackhmmer_msa_prefix + f'-{i}.sto'
            jackhmmer_result_files.append(fn)
            clean_up.append(fn)
            clean_up.append(jackhmmer_msa_prefix + f'-{i}.hmm')
        try:
            subprocess.run(cmd, stderr=subprocess.PIPE, check=True)
        finally:
            for fn in clean_up:
                if os.path.isfile(fn):
                    os.remove(fn)
            

def hmmbuild(msa_file: str, hmm_file: str, num_cpus: int):
    if os.path.isfile(hmm_file):
        return

    with io_helpers.file_guard(hmm_file) as tmp_hmm_file:
        subprocess.run([hmmbuild_bin, '--cpu', str(num_cpus), '-o', os.devnull, '--symfrac', '0', '--fragthresh', '1', tmp_hmm_file, msa_file], check=True)


def hmmsearch(hmm_file: str, msa_sto_file: str, bitscore: float, sequence_db: str, num_cpus: int):
    if os.path.isfile(msa_sto_file):
        return

    with io_helpers.file_guard(msa_sto_file) as tmp_msa_sto_file:
        subprocess.run([
            hmmsearch_bin, '--domT', str(bitscore), '-T', str(bitscore), '--incdomT', str(bitscore), '--incT', str(bitscore), 
            '--noali', '--notextw', '--cpu', str(num_cpus), '-o', os.devnull, '-A', tmp_msa_sto_file, hmm_file, sequence_db
        ], check=True)


def hmmalign(seq_file: str, hmm_file: str, msa_file_for_hmm: str, msa_file_with_target: str):
    if os.path.isfile(msa_file_with_target):
        return

    with io_helpers.file_guard(msa_file_with_target) as tmp_msa_file_with_target:
        subprocess.run([
            hmmalign_bin, '--mapali', msa_file_for_hmm, '-o', tmp_msa_file_with_target, hmm_file, seq_file
        ], check=True)


def hhblits(seq_file: str, seed_msa_file: str, sequence_db: str, num_cpus: int):
    if os.path.isfile(seed_msa_file):
        return
    
    seed_msa_a3m = os.path.splitext(seed_msa_file)[0] + '.a3m'
    if not os.path.isfile(seed_msa_a3m):
        # hhblits_e_value = '1e-12'
        hhblits_e_value = '1e-6'
        hhblits_num_iterations = '4'
        hhblits_cov = '70'
        with io_helpers.file_guard(seed_msa_a3m) as tmp_seed_msa_a3m:
            subprocess.run([
                hhblits_bin, '-cpu', str(num_cpus), '-i', seq_file, '-d', sequence_db, '-o', os.devnull, 
                '-oa3m', tmp_seed_msa_a3m, '-n', hhblits_num_iterations, '-e', hhblits_e_value, '-all', #'-cov', hhblits_cov
            ], check=True)

    def _filter(aln):
        for rec in aln:
            rec.seq = Seq(re.sub(r'[a-z]', '', str(rec.seq)))
        return aln

    with io_helpers.file_guard(seed_msa_file) as tmp_seed_msa_file:
        it = io_helpers.read_msa_iter(seed_msa_a3m, 'fasta')
        io_helpers.write_msa(batched_map(_filter, it), tmp_seed_msa_file, 'fasta')

    os.remove(seed_msa_a3m)


def iterative_hmmsearch(seq_file: str, seed_msa_file: Optional[str], seed_hmm_file: Optional[str], bitscore: float, msa_file: str, sequence_db: str, num_cpus: int, num_iterations: int=5):
    if os.path.isfile(msa_file):
        return

    assert seed_msa_file or seed_hmm_file
    msa_file_prefix = io_helpers.splitext(msa_file)[0]
    clean_up = []
    query_msa_file = [seed_msa_file]
    for i in range(1, num_iterations + 2):
        iter_msa_prefix = msa_file_prefix + f'_it{i}'

        if query_msa_file[-1] is None:
            query_hmm_file = seed_hmm_file
        else:
            query_hmm_file = iter_msa_prefix + '.hmm'
            clean_up.append(query_hmm_file)
            hmmbuild(query_msa_file[-1], query_hmm_file, num_cpus)

        if i > num_iterations:
            break
        
        iter_msa_sto = iter_msa_prefix + '.sto'
        clean_up.append(iter_msa_sto)
        hmmsearch(query_hmm_file, iter_msa_sto, bitscore, sequence_db, num_cpus)

        if os.path.isfile(iter_msa_sto) and (os.path.getsize(iter_msa_sto) > 0):
            query_msa_file.append(iter_msa_sto)
        else: # empty hmmsearch result sto file
            break

    if query_msa_file[-1] is None:
        io_helpers.write_msa((), msa_file, 'fasta')
    else:
        hmmalign(seq_file, query_hmm_file, query_msa_file[-1], msa_file)

    for fn in clean_up:
        if os.path.isfile(fn):
            os.remove(fn)


def make_msa(seq_file, b, L, msa_dir, msa_prefix, method, sequence_db, seed_msa_file, seed_hmm_file, max_row_gap, alphabet, sample_size, cpu):
    global rng

    tmp_sequence_db = os.path.join('/tmp/mar4/db', os.path.basename(sequence_db))
    if os.path.isfile(tmp_sequence_db):
        sequence_db = tmp_sequence_db
    elif shutil.disk_usage('/tmp').free > 1.5 * os.path.getsize(sequence_db):
        try:
            if not os.path.isdir(os.path.dirname(tmp_sequence_db)):
                os.makedirs(os.path.dirname(tmp_sequence_db), exist_ok=True)
            with filelock.FileLock(tmp_sequence_db + '.lock', timeout=0):
                with io_helpers.file_guard(tmp_sequence_db) as tf:
                    shutil.copy2(sequence_db, tf)
                sequence_db = tmp_sequence_db
        except filelock.Timeout:
            pass
    del tmp_sequence_db

    msa_file = os.path.join(msa_dir, f'{msa_prefix}_b{b}.fa.xz')
    raw_msa_file = os.path.join(msa_dir, f'{msa_prefix}_b{b}_raw.sto')
    if (not os.path.isfile(raw_msa_file)) and (not os.path.isfile(msa_file)):
        if method == 'jackhmmer':
            jackhmmer(seq_file, b * L, raw_msa_file, sequence_db, cpu)
        else:
            iterative_hmmsearch(seq_file, seed_msa_file, seed_hmm_file, b * L, raw_msa_file, sequence_db, cpu)

    if os.path.isfile(raw_msa_file):
        target_seq = None
        it = io_helpers.read_msa_iter(raw_msa_file, 'stockholm')
        if method == 'jackhmmer':
            # first sequence is target sequence
            for target_seq in it:
                break
        else:
            # last sequence is target sequence
            for target_seq in it:
                continue
        
        with io_helpers.file_guard(msa_file) as tmp_msa_file:
            if target_seq:
                focus_msa_it = make_focus_msa(target_seq, io_helpers.read_msa_iter(raw_msa_file, 'stockholm'))
                io_helpers.write_msa(focus_msa_it, tmp_msa_file, 'fasta')
            else:
                io_helpers.write_msa((), tmp_msa_file, 'fasta')

        os.remove(raw_msa_file)

    target_seq = None
    it = io_helpers.read_msa_iter(msa_file, 'fasta')
    for target_seq in it:
        break
    N = 1 + sum(1 for _ in it) if target_seq else 0

    if N > 0:
        result = {}
        min_id, auc_cdf = min_id_and_auc(target_seq, io_helpers.read_msa_iter(msa_file, 'fasta'), threshold=0.01)
        result['raw'] = {
            'b': b,
            'file': msa_file,
            'N': N,
            'L': L,
            'min_id': min_id,
            'auc_cdf': auc_cdf,
        }

        filtered_msa_file = io_helpers.splitext(msa_file)[0] + f'_rg{max_row_gap}.fa.xz'
        if (not os.path.isfile(filtered_msa_file)) or (os.path.getmtime(msa_file) > os.path.getmtime(filtered_msa_file)):
            filtered_msa_it = filter_msa(io_helpers.read_msa_iter(msa_file, 'fasta'), alphabet=alphabet, max_row_gap=max_row_gap)
            with io_helpers.file_guard(filtered_msa_file) as tmp_filtered_msa_file:
                io_helpers.write_msa(filtered_msa_it, tmp_filtered_msa_file, 'fasta')
        N = sum(1 for _ in io_helpers.read_msa_iter(filtered_msa_file, 'fasta'))

        if N > sample_size:
            sampled_filtered_msa_file = io_helpers.splitext(filtered_msa_file)[0] + '_sample.fa.xz'
            if (not os.path.isfile(sampled_filtered_msa_file)) or (os.path.getmtime(filtered_msa_file) > os.path.getmtime(sampled_filtered_msa_file)):
                selectors = np.full(N, False)
                selectors[0] = True
                selectors[rng.choice(np.arange(1, N), sample_size - 1, replace=False)] = True            
                with io_helpers.file_guard(sampled_filtered_msa_file) as tmp_sampled_filtered_msa_file:
                    io_helpers.write_msa(itertools.compress(io_helpers.read_msa_iter(filtered_msa_file, 'fasta'), selectors), tmp_sampled_filtered_msa_file, 'fasta')
            N = sample_size
        else:
            sampled_filtered_msa_file = filtered_msa_file

        min_id, auc_cdf = min_id_and_auc(target_seq, io_helpers.read_msa_iter(sampled_filtered_msa_file, 'fasta'), threshold=0.01)
        result['filtered'] = {
            'b': b,
            'file': sampled_filtered_msa_file,
            'N': N,
            'L': L,
            'min_id': min_id,
            'auc_cdf': auc_cdf,
            'max_row_gap': max_row_gap,
        }

        aa7_sampled_filtered_msa_file = io_helpers.splitext(sampled_filtered_msa_file)[0] + '_aa7.fa.xz'
        if (not os.path.isfile(aa7_sampled_filtered_msa_file)) or (os.path.getmtime(sampled_filtered_msa_file) > os.path.getmtime(aa7_sampled_filtered_msa_file)):
            aa7_sampled_filtered_msa_it = filter_msa_to_aa7(io_helpers.read_msa_iter(sampled_filtered_msa_file, 'fasta'))
            with io_helpers.file_guard(aa7_sampled_filtered_msa_file) as tmp_aa7_sampled_filtered_msa_file:
                io_helpers.write_msa(aa7_sampled_filtered_msa_it, tmp_aa7_sampled_filtered_msa_file, 'fasta')
        N = sum(1 for _ in io_helpers.read_msa_iter(aa7_sampled_filtered_msa_file, 'fasta'))

        target_seq_aa7 = None
        it = io_helpers.read_msa_iter(aa7_sampled_filtered_msa_file, 'fasta')
        for target_seq_aa7 in it:
            break

        min_id, auc_cdf = min_id_and_auc(target_seq_aa7, io_helpers.read_msa_iter(aa7_sampled_filtered_msa_file, 'fasta'), threshold=0.01)
        result['aa7_filtered'] = {
            'b': b,
            'file': aa7_sampled_filtered_msa_file,
            'N': N,
            'L': L,
            'min_id': min_id,
            'auc_cdf': auc_cdf,
            'max_row_gap': max_row_gap,
        }

        vertebrata_filtered_msa_file = io_helpers.splitext(filtered_msa_file)[0] + '_vertebrata.fa.xz'
        if (not os.path.isfile(vertebrata_filtered_msa_file)) or (os.path.getmtime(filtered_msa_file) > os.path.getmtime(vertebrata_filtered_msa_file)):
            vertebrata_filtered_msa_it = filter_msa_vertebrata(target_seq, io_helpers.read_msa_iter(filtered_msa_file, 'fasta'))
            with io_helpers.file_guard(vertebrata_filtered_msa_file) as tmp_vertebrata_filtered_msa_file:
                io_helpers.write_msa(vertebrata_filtered_msa_it, tmp_vertebrata_filtered_msa_file, 'fasta')
        N = sum(1 for _ in io_helpers.read_msa_iter(vertebrata_filtered_msa_file, 'fasta'))

        if N > sample_size:
            sampled_vertebrata_filtered_msa_file = io_helpers.splitext(vertebrata_filtered_msa_file)[0] + '_sample.fa.xz'
            if (not os.path.isfile(sampled_vertebrata_filtered_msa_file)) or (os.path.getmtime(vertebrata_filtered_msa_file) > os.path.getmtime(sampled_vertebrata_filtered_msa_file)):
                selectors = np.full(N, False)
                selectors[0] = True
                selectors[rng.choice(np.arange(1, N), sample_size - 1, replace=False)] = True            
                with io_helpers.file_guard(sampled_vertebrata_filtered_msa_file) as tmp_sampled_vertebrata_filtered_msa_file:
                    io_helpers.write_msa(itertools.compress(io_helpers.read_msa_iter(vertebrata_filtered_msa_file, 'fasta'), selectors), tmp_sampled_vertebrata_filtered_msa_file, 'fasta')
            N = sample_size
        else:
            sampled_vertebrata_filtered_msa_file = vertebrata_filtered_msa_file

        min_id, auc_cdf = min_id_and_auc(target_seq, io_helpers.read_msa_iter(sampled_vertebrata_filtered_msa_file, 'fasta'), threshold=0.01)
        result['vertebrata_filtered'] = {
            'b': b,
            'file': sampled_vertebrata_filtered_msa_file,
            'N': N,
            'L': L,
            'min_id': min_id,
            'auc_cdf': auc_cdf,
            'max_row_gap': max_row_gap,
        }

        aa7_sampled_vertebrata_filtered_msa_file = io_helpers.splitext(sampled_vertebrata_filtered_msa_file)[0] + '_aa7.fa.xz'
        if (not os.path.isfile(aa7_sampled_vertebrata_filtered_msa_file)) or (os.path.getmtime(sampled_vertebrata_filtered_msa_file) > os.path.getmtime(aa7_sampled_vertebrata_filtered_msa_file)):
            aa7_sampled_vertebrata_filtered_msa_it = filter_msa_to_aa7(io_helpers.read_msa_iter(sampled_vertebrata_filtered_msa_file, 'fasta'))
            with io_helpers.file_guard(aa7_sampled_vertebrata_filtered_msa_file) as tmp_aa7_sampled_vertebrata_filtered_msa_file:
                io_helpers.write_msa(aa7_sampled_vertebrata_filtered_msa_it, tmp_aa7_sampled_vertebrata_filtered_msa_file, 'fasta')
        N = sum(1 for _ in io_helpers.read_msa_iter(aa7_sampled_vertebrata_filtered_msa_file, 'fasta'))

        min_id, auc_cdf = min_id_and_auc(target_seq_aa7, io_helpers.read_msa_iter(aa7_sampled_vertebrata_filtered_msa_file, 'fasta'), threshold=0.01)
        result['vertebrata_aa7_filtered'] = {
            'b': b,
            'file': aa7_sampled_vertebrata_filtered_msa_file,
            'N': N,
            'L': L,
            'min_id': min_id,
            'auc_cdf': auc_cdf,
            'max_row_gap': max_row_gap,
        }
    else:
        result = {
            'raw': {
                'b': b,
                'file': '',
                'N': 0,
                'L': L,
                'min_id': 0,
                'auc_cdf': 0,
            },
            'filtered': {
                'b': b,
                'file': '',
                'N': 0,
                'L': L,
                'min_id': 0,
                'auc_cdf': 0,
                'max_row_gap': max_row_gap,
            },
            'aa7_filtered': {
                'b': b,
                'file': '',
                'N': 0,
                'L': L,
                'min_id': 0,
                'auc_cdf': 0,
                'max_row_gap': max_row_gap,
            },
            'vertebrata_filtered': {
                'b': b,
                'file': '',
                'N': 0,
                'L': L,
                'min_id': 0,
                'auc_cdf': 0,
                'max_row_gap': max_row_gap,
            },
            'vertebrata_aa7_filtered': {
                'b': b,
                'file': '',
                'N': 0,
                'L': L,
                'min_id': 0,
                'auc_cdf': 0,
                'max_row_gap': max_row_gap,
            },
        }
    
    return result


def make_msa_repo(msa_repo: MsaRepo, job_name, seq_file, msa_dir, msa_prefix, database, method, seed_msa_file, seed_hmm_file, min_id, max_row_gap, sample_size, cpu, runtime, mem, recovery_mode):
    global hhblits_db

    utils.set_job_status(msa_dir, 'MSA')
    
    if cpu is None:
        cpu = len(os.sched_getaffinity(0))

    if method == 'hhblits':
        seed_msa_file = os.path.join(msa_dir, msa_prefix + '_hhblits.fa')
        hhblits(seq_file, seed_msa_file, hhblits_db, cpu)
    
    while True:
        pending_bitscores = msa_repo.get_pending_bitscores(recovery_mode)
        if len(pending_bitscores) == 0:
            break
        pending_bitscores.sort()
        while len(pending_bitscores) > 0:
            b = pending_bitscores.pop()
            msa_repo.add_to_history(b)
            msa_repo.add_msa(b, make_msa(seq_file, b, msa_repo.L, msa_dir, msa_prefix, method, database[1], seed_msa_file, seed_hmm_file, max_row_gap, utils.alphabet, sample_size, cpu))

    utils.set_job_status(msa_dir, 'STATS')

    for entry in sorted(msa_repo.find_msa_by_min_id(vertebrata=False, aa7=False), key=lambda entry: entry['N'], reverse=True):
        draw_alignment_stats(entry['file'])

    for entry in sorted(msa_repo.find_msa_by_min_id(vertebrata=True, aa7=False), key=lambda entry: entry['N'], reverse=True):
        draw_alignment_stats(entry['file'])

    for entry in sorted(msa_repo.find_msa_by_min_id(vertebrata=False, aa7=True), key=lambda entry: entry['N'], reverse=True):
        draw_alignment_stats(entry['file'])

    for entry in sorted(msa_repo.find_msa_by_min_id(vertebrata=True, aa7=True), key=lambda entry: entry['N'], reverse=True):
        draw_alignment_stats(entry['file'])

    utils.set_job_status(msa_dir, 'DONE')


def submit_slurm_job(msa_repo, job_name, seq_file, msa_dir, msa_prefix, database, method, seed_msa_file, seed_hmm_file, min_id, max_row_gap, sample_size, cpu, runtime, mem, recovery_mode):
    active_job_ids = utils.get_active_slurm_jobs()
    slurm_info = utils.get_slurm_info(msa_dir)
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
        '--seq-file', seq_file,
        '--output-dir', msa_dir, 
        '--msa-prefix', msa_prefix, 
        '--cpu', str(cpu),
        '--mem', str(mem),
        '--time', str(runtime),
    ]

    if method == 'jackhmmer':
        cmd_list.append('--jackhmmer')
    elif method == 'hhblits':
        cmd_list.append('--hhblits')

    if database:
        cmd_list.extend(['--database', '::'.join(database)])

    if seed_msa_file:
        cmd_list.extend(['--seed-msa', seed_msa_file])

    if seed_hmm_file:
        cmd_list.extend(['--seed-hmm', seed_hmm_file])

    if min_id:
        cmd_list.extend(['--min-id', min_id])
    
    if max_row_gap:
        cmd_list.extend(['--max-row-gap', str(max_row_gap)])
    
    if sample_size:
        cmd_list.extend(['--sample-size', str(sample_size)])
    
    if recovery_mode:
        cmd_list.append('--recovery-mode')
        
    cmd_str = subprocess.list2cmdline(filter(None, cmd_list))

    partition = 'short' if runtime <= 12 else 'medium' if runtime <= 120 else 'long'
    days = int(runtime / 24)
    hours = int(runtime - days * 24)
    minutes = int((runtime - days * 24 - hours) * 60)
    seconds = int((runtime - days * 24 - hours) * 3600 - minutes * 60)
    sbatch_runtime = f'{days}-{hours:02}:{minutes:02}:{seconds:02}'
    stdout_file = os.path.join(msa_dir, 'stdout_%j.txt')
    stderr_file = os.path.join(msa_dir, 'stderr_%j.txt')

    if not os.path.isdir(msa_dir):
        os.makedirs(msa_dir)

    slurm_script_file = os.path.join(msa_dir, 'slurm_make_msa.sh')
    with open(slurm_script_file, 'w') as f:
        f.write(f'''#!/bin/bash 
#SBATCH -c {cpu}                      # Request one core
#SBATCH -t {sbatch_runtime}               # Runtime in D-HH:MM format
#SBATCH --mem={mem}G
#SBATCH -p {partition}
#SBATCH -o "{stdout_file}"     # Standard output and error log
#SBATCH -e "{stderr_file}"     # Standard output and error log

# source /n/groups/marks/software/anaconda_o2/bin/activate py310ys
source /n/groups/marks/users/su/mambaforge/etc/profile.d/conda.sh
conda activate py310
cd "{msa_dir}"

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

    slurm_info_file = os.path.join(msa_dir, 'slurm.txt')
    stdout_file = os.path.basename(stdout_file).replace('%j', slurm_job_id)
    stderr_file = os.path.basename(stderr_file).replace('%j', slurm_job_id)
    with open(slurm_info_file, 'w') as f:
        f.write(f'''SLURM_JOBID={slurm_job_id}
SLURM_MEM_PER_NODE={mem}
SLURM_CPUS_PER_TASK={cpu}
time={runtime}
stdout={stdout_file}
stderr={stderr_file}
database={'::'.join(database)}
''')

    return slurm_job_id


def run(args_list: list):
    parser = argparse.ArgumentParser(description="Make MSA")
    parser.add_argument('seq_file_or_name', nargs='?', type=str, default='')
    parser.add_argument('--job-name', type=str)
    parser.add_argument('--seq-file', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--msa-prefix', type=str)
    parser.add_argument('--database', type=str, default='ur100')
    parser.add_argument('--jackhmmer', action='store_true')
    parser.add_argument('--hhblits', action='store_true')
    parser.add_argument('--seed-msa', type=str)
    parser.add_argument('--seed-hmm', type=str)
    parser.add_argument('--min-id', type=str, default='20,30,40')
    parser.add_argument('--max-row-gap', type=float, default=0.3)
    parser.add_argument('--sample-size', type=int, default=20000)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--mem', type=int, default=4)
    parser.add_argument('--time', type=float, default=48)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--find-msa', action='store_true')
    parser.add_argument('--slurm-info', action='store_true')
    parser.add_argument('--empty-msa-at', type=str)
    parser.add_argument('--fail-last', action='store_true')
    parser.add_argument('--aa7', action='store_true')
    parser.add_argument('--vertebrate', action='store_true')
    parser.add_argument('--recovery-mode', action='store_true')
    parser.add_argument('--get-pending-bitscores', action='store_true')
    args = parser.parse_args(args_list)

    root_dir = '/n/groups/marks/users/su/mar4'

    method = ''
    if args.jackhmmer:
        method = 'jackhmmer'
    elif args.hhblits:
        method = 'hhblits'
    elif (args.seed_msa is None) and (args.seed_hmm is None):
        method = 'jackhmmer'

    database = args.database
    if '::' in database:
        database = database.split('::', maxsplit=1)
    elif os.path.isfile(database):
        database = ('custom::', database)
    elif database == 'ur100':
        database = ('ur100', '/n/groups/marks/databases/jackhmmer/uniref100/uniref100_current.o2.fasta')
    elif database == 'uniprot':
        database = ('uniprot', '/n/groups/marks/databases/jackhmmer/uniprot/uniprot_current.o2.fasta')

    seq_file_or_name = args.seq_file or args.seq_file_or_name
    if os.path.isfile(seq_file_or_name):
        seq_file = seq_file_or_name
        seqrec = io_helpers.read_seq(seq_file, 'fasta')
        seq_name = seqrec.id
        msa_dir = args.output_dir
    else:
        seq_name = seq_file_or_name
        seq_name_base = seq_name if '/' not in seq_name else seq_name[:seq_name.index('/')]
        _seq_name = seq_name.replace('/', '_')
        msa_dir = os.path.join(root_dir, 'msa', method, database[0], *(seq_name_base[2:5]), seq_name_base, _seq_name)
        seq_file = os.path.join(root_dir, 'seqs', *(seq_name_base[2:5]), _seq_name + '.fa')
        if not os.path.isfile(seq_file):
            seq_file = os.path.join(root_dir, 'seqs', _seq_name + '.fa')

    if not msa_dir:
        msa_dir = os.getcwd()

    if not os.path.isdir(msa_dir):
        os.makedirs(msa_dir)

    if args.status:
        return utils.get_job_status(msa_dir)

    if args.slurm_info:
        return utils.get_slurm_info(msa_dir)

    init_b = 0.9
    b_step_sizes = [1, 2]
    min_ids = list(map(float, args.min_id.split(',')))
    msa_repo = MsaRepo(msa_dir, seq_file, args.max_row_gap, init_b, b_step_sizes, min_ids, min_ids, args.cpu, args.mem, args.time)

    if args.get_pending_bitscores:
        pending_bitscores = msa_repo.get_pending_bitscores(args.recovery_mode)
        if len(pending_bitscores) > 0:
            return pending_bitscores
        else:
            return None

    if args.find_msa:
        if utils.get_job_status(msa_dir) != 'DONE':
            return None
        try:
            return msa_repo.find_msa_by_min_id(float(args.min_id), vertebrata=args.vertebrate, aa7=args.aa7)
        except:
            return None

    if args.empty_msa_at:
        for b in map(float, args.empty_msa_at.split(',')):
            with io_helpers.file_guard(os.path.join(msa_dir, msa_prefix + f'_b{b}[._]*')) as tmp_pattern:
                for f in glob.glob(tmp_pattern):
                    os.remove(f)
            msa_repo.add_msa(b)
    
    cpu = args.cpu
    runtime = args.time
    mem = args.mem
    job_name = args.job_name if args.job_name is not None else f'msa:{seq_name}'
    msa_prefix = args.msa_prefix if args.msa_prefix else io_helpers.splitext(os.path.basename(seq_file))[0]
    run = make_msa_repo if args.local else submit_slurm_job

    if args.fail_last:
        b = msa_repo.get_last_bitscore()
        assert b is not None
        msa_repo.add_msa(b)

        with io_helpers.file_guard(os.path.join(msa_dir, msa_prefix + f'_b{b}[._]*')) as tmp_pattern:
            for f in glob.glob(tmp_pattern):
                os.remove(f)

        if len(msa_repo.get_pending_bitscores(args.recovery_mode)) == 0:
            mem = 4
            runtime = 0.25
        else:
            entry = msa_repo.get_last_msa(nonempty=True)
            if entry is None:
                entry = msa_repo.get_last_msa(nonempty=False)
            cpu = entry['filtered']['n_cpus']
            mem = entry['filtered']['mem']
            runtime = entry['filtered']['runtime']

    return run(
        msa_repo=msa_repo,
        job_name=job_name, 
        seq_file=seq_file, 
        msa_dir=msa_dir, 
        msa_prefix=msa_prefix, 
        database=database, 
        method=method, 
        seed_msa_file=args.seed_msa, 
        seed_hmm_file=args.seed_hmm, 
        min_id=args.min_id, 
        max_row_gap=args.max_row_gap, 
        sample_size=args.sample_size, 
        cpu=cpu, 
        runtime=runtime, 
        mem=mem,
        recovery_mode=args.recovery_mode
    )


if __name__ == '__main__':
    r = run(sys.argv[1:])
    if r is not None:
        print(r)
