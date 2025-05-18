import os
import lzma
import gzip
import typing
import itertools
from contextlib import contextmanager
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

@contextmanager
def file_guard(fn):
    if os.path.isfile(fn):
        os.remove(fn)
    tmpfn = os.path.join(os.path.dirname(fn), '_' + os.path.basename(fn))
    yield tmpfn
    if os.path.isfile(tmpfn):
        os.replace(tmpfn, fn)


def splitext(fn: str) -> typing.Tuple[str, str]:
    prefix, ext = os.path.splitext(fn)
    if ext in ['.xz']:
        prefix, ext2 = os.path.splitext(prefix)
        ext = ext2 + ext
    return prefix, ext


def universal_open(fn: str, *args, **kwargs):
    if fn.endswith('.xz'):
        return lzma.open(fn, *args, **kwargs)
    elif fn.endswith('.gz'):
        return gzip.open(fn, *args, **kwargs)
    else:
        return open(fn, *args, **kwargs)


def read_seq(seq_file, format):
    with universal_open(seq_file, 'rt') as f:
        return SeqIO.read(f, format)


def read_msa_iter(msa_file, format):
    with universal_open(msa_file, 'rt') as f:
        if format == 'stockholm':
            yield from simple_stockholm_iter(f)
        else:
            yield from SeqIO.parse(f, format)

    
def read_msa(msa_file, format, max_len=1000000):
    with universal_open(msa_file, 'rt') as f:
        aln = list(itertools.islice(SeqIO.parse(f, format), max_len))
    return aln


def write_msa(aln, msa_file, format):
    with universal_open(msa_file, 'wt') as f:
        SeqIO.write(aln, f, format)


def simple_stockholm_iter(fp):
    for line in fp:
        if (line[0] == '#') or (line == '\n') or (line == '//\n'):
            continue
        seq_id, seq_str = line.split()
        yield SeqRecord(Seq(seq_str), id=seq_id)

