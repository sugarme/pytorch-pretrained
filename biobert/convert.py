from pathlib import Path
import shutil
import os
import numpy as np
import torch
import subprocess

target_path = Path().absolute()
os.makedirs(str(target_path), exist_ok=True)
weights = torch.load("./pytorch_model.bin", map_location='cpu')
nps = {}
for k, v in weights.items():
    k = k.replace("gamma", "weight").replace("beta", "bias")
    nps[k] = np.ascontiguousarray(v.cpu().numpy())

np.savez(target_path / 'model.npz', **nps)
