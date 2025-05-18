import os
import glob
import subprocess

alphabet = 'ACDEFGHIKLMNPQRSTVWY-'


def get_job_status(job_dir):
    try:
        with open(os.path.join(job_dir, 'status.txt')) as f:
            return f.read().strip()
    except:
        return ''


def set_job_status(job_dir, status):
    with open(os.path.join(job_dir, 'status.txt'), 'w') as f:
        f.write(status)
    
    
def get_active_slurm_jobs():
    r = subprocess.run(['squeue', '-u', str(os.getuid()), '-h', '-O', 'JobID'], capture_output=True, text=True, check=True)
    return r.stdout.split()


def get_slurm_info(slurm_dir):
    slurm_info = {
        'stdout': '', 
        'stderr': '', 
        'time': 0.,
        'submissions': 0,
        'exit_code': 0,
        'out_of_memory': False, 
        'out_of_memory_count': 0, 
        'out_of_time': False, 
        'out_of_time_count': 0,
        'new_msa': False,
        'error_message': '',
        'SLURM_CPUS_PER_TASK': 0,
    }

    slurm_file = os.path.join(slurm_dir, 'slurm.txt')
    if os.path.isfile(slurm_file):
        with open(slurm_file) as f:
            for line in f:
                try:
                    k, v = line.strip().split('=', maxsplit=1)
                    slurm_info[k] = v
                except ValueError:
                    pass
        slurm_info['SLURM_MEM_PER_NODE'] = int(slurm_info['SLURM_MEM_PER_NODE'])
        slurm_info['SLURM_CPUS_PER_TASK'] = int(slurm_info['SLURM_CPUS_PER_TASK'])
        slurm_info['time'] = float(slurm_info['time'])
        slurm_info['exit_code'] = int(slurm_info['exit_code'])
        slurm_info['stdout'] = os.path.join(slurm_dir, slurm_info['stdout'])
        slurm_info['stderr'] = os.path.join(slurm_dir, slurm_info['stderr'])

        stdout_files = glob.glob(os.path.join(slurm_dir, 'stdout_*.txt'))
        slurm_info['submissions'] = len(stdout_files)
        if slurm_info['stdout'] not in stdout_files:
            slurm_info['submissions'] += 1 # latest submission is still pending

    if os.path.isfile(slurm_info['stdout']):
        mtime = os.path.getmtime(slurm_info['stdout'])
        for msa_file in glob.glob(os.path.join(slurm_dir, '*.fa.xz')):
            if os.path.getmtime(msa_file) > mtime:
                slurm_info['new_msa'] = True
                break

    if os.path.isfile(slurm_info['stderr']):
        if os.path.getsize(slurm_info['stderr']) > 0:
            with open(slurm_info['stderr']) as f:
                slurm_info['error_message'] = f.read()
                if ('oom-kill' in slurm_info['error_message']) or ('oom_kill' in slurm_info['error_message']):
                    slurm_info['out_of_memory'] = True
                if 'DUE TO TIME LIMIT ***' in slurm_info['error_message']:
                    slurm_info['out_of_time'] = True

    return slurm_info
