from pathlib import Path
import importlib.util
import sys
import json
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_orca as replay
import numpy as np
from scipy.spatial.transform import Rotation
spec = importlib.util.spec_from_file_location('axis_parent', HERE.parent/'ppo_axis_specialists/train.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
base = parent.base
torch = base.torch
P = json.loads((HERE/'protocol.json').read_text())

def goal(axis, angle):
    return Rotation.from_rotvec(np.eye(3)['XYZ'.index(axis)] * np.deg2rad(angle)).as_quat().astype('float32').tolist()

def starts(bank):
    x, _ = replay.initial_cases(64, bank, 'tilt', 5.)
    yz = np.random.default_rng(np.random.SeedSequence([bank,1])).uniform(-10,10,(64,2))
    return Rotation.from_euler('xyz',np.column_stack((x,yz)),degrees=True).as_quat().astype('float32')

def path_for(axis, arm, seed):
    if arm == 'parent':
        return parent.checkpoint(axis,seed,'dwell')
    epoch = {'curr':5512,'normal':6024,'mixed':6024,'pilot':1000}[arm]
    run = f'{axis}_{arm}_{seed}'
    return unique(HERE/'runs'/run/'nn',f'last_{run}_ep_{epoch}_rew__*.pth')

def unique(folder, pattern):
    hits = list(folder.glob(pattern))
    if len(hits)!=1:
        raise FileNotFoundError((str(folder),pattern,hits))
    return hits[0]

def save(file, **items):
    file.parent.mkdir(parents=True,exist_ok=True)
    tmp=file.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**items)
    tmp.replace(file)
