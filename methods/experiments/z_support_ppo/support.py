"""Apply the same 4 mm collision-hull and body-force evidence used in evaluation."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'z_ppo_smoothness'))
from geometry import gap

def contact_evidence(poses, forces, names, millimeters=4., newtons=.1):
    obj=poses[:,-1]
    bits=np.zeros(len(obj),np.uint32)
    housing=np.zeros(len(obj),bool)
    for b, name in enumerate(names[:-1]):
        loads=(forces[:,b]>=newtons)&(forces[:,-1]>=newtons)
        if loads.any():
            touches=gap(name,poses[loads,b],obj[loads])<=millimeters
            if name in ('right_tower','right_forearm_mount'):
                housing[loads]|=touches
            else:
                bits[loads]|=touches.astype(np.uint32)<<b
    return bits,housing
