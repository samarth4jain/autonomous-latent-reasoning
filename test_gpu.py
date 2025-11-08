#!/usr/bin/env python3
"""Quick GPU test script.
Runs nvidia-smi (if available) and prints PyTorch CUDA availability and device info.
"""
import subprocess
import sys

def run_nvidia_smi():
    try:
        out = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        print('===== nvidia-smi =====')
        print(out.decode())
    except Exception as e:
        print('nvidia-smi not available or failed:', e)

def torch_info():
    try:
        import torch
        print('===== PyTorch CUDA info =====')
        print('torch.__version__ =', torch.__version__)
        print('cuda_available =', torch.cuda.is_available())
        try:
            print('cuda_count =', torch.cuda.device_count())
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print('device_name =', torch.cuda.get_device_name(0))
        except Exception as e:
            print('Error querying device count/name:', e)
    except Exception as e:
        print('PyTorch import failed:', e)

def main():
    run_nvidia_smi()
    torch_info()

if __name__ == '__main__':
    main()
