"""Capture runtime provenance for reproducibility.

Each trained run should call `capture_provenance()` and store the returned
dict alongside its experiment metadata. This lets future readers (or the
harvest script) identify exactly which code/host/job produced a result.
"""

import os
import socket
import subprocess
import sys
import time


def capture_provenance():
    """Capture git commit, host, SLURM job id, user, command line.

    Returns a dict safe to JSON-serialize. All fields default to '' if
    unavailable (e.g. running outside a git checkout or without SLURM).
    """
    return {
        'git_commit': _git('rev-parse', '--short', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'host': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
        'user': os.environ.get('USER', ''),
        'cli_argv': ' '.join(sys.argv),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def _git(*args):
    try:
        out = subprocess.check_output(
            ['git', *args], stderr=subprocess.DEVNULL, cwd=None)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''
