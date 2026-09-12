"""Best-effort hardware probing; missing tools never mean zero memory."""
import csv
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys


def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def parse_nvidia(text):
    devices = []
    for row in csv.reader((text or '').splitlines()):
        if len(row) != 5:
            continue
        try:
            devices.append(dict(index=int(row[0]), name=row[1].strip(),
                                vram_gib=round(float(row[2])/1024, 2),
                                free_vram_gib=round(float(row[3])/1024, 2),
                                driver=row[4].strip()))
        except ValueError:
            continue
    return devices


def detect(path='.'):
    warnings = []
    ram = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram = {'total_gib': round(mem.total/2**30, 2), 'available_gib': round(mem.available/2**30, 2)}
    except ImportError:
        warnings.append('psutil unavailable: RAM unknown. Install DehCode dependencies.')
    query = command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
    gpus = parse_nvidia(query)
    if query is None:
        warnings.append('NVIDIA probe unavailable. Other vendors are not supported by the first adapter.')
    smi = command(['nvidia-smi']) or ''
    cuda = re.search(r'CUDA Version:\s*([\d.]+)', smi)
    toolkit = command(['nvcc', '--version'])
    torch = {'installed': False, 'cuda_available': False}
    if importlib.util.find_spec('torch'):
        # Isolate driver initialization and broken binary imports from doctor.
        result = command([sys.executable, '-c',
            'import torch,json; print(json.dumps(dict(installed=True,version=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),architectures=torch.cuda.get_arch_list())))'], timeout=45)
        try:
            torch = json.loads(result)
        except (TypeError, ValueError):
            torch = {'installed': True, 'cuda_available': False, 'error': 'PyTorch probe failed or timed out'}
    disk = shutil.disk_usage(path)
    cpu = platform.processor() or platform.machine()
    if platform.system() == 'Linux':
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
        except OSError:
            pass
    elif platform.system() == 'Darwin':
        cpu = command(['sysctl', '-n', 'machdep.cpu.brand_string']) or cpu
    return dict(os=platform.system(), os_version=platform.release(), python=platform.python_version(),
                cpu=cpu, cpu_threads=os.cpu_count(), ram=ram,
                gpus=gpus, driver_cuda_max=cuda.group(1) if cuda else None,
                cuda_toolkit=toolkit, torch=torch, disk_free_gib=round(disk.free/2**30, 2),
                warnings=warnings)
