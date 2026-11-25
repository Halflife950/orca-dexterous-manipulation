"""Reconstruct the slip signal on trajectories, without future information."""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# Units are metres and native 20 Hz policy decisions.
DEAD_ZONE = .050
WINDOW = 3
MIN_GAIN = .003
WEIGHT = 30.
MAX_STEP = .010


def replay(poses, names):
    palm = poses[:, :, names.index('right_palm')]
    delta = poses[:, :, names.index('object'), :3] - palm[:, :, :3]
    local = Rotation.from_quat(palm[:,:,3:].reshape(-1,4)).inv().apply(delta.reshape(-1,3)).reshape(delta.shape)
    excursion = np.linalg.norm(local-local[:1], axis=2)
    radius = np.linalg.norm(local, axis=2)
    outward = (np.diff(excursion,axis=0)>0) & (np.diff(radius,axis=0)>0)
    sustained = np.lib.stride_tricks.sliding_window_view(outward, WINDOW, axis=0).all(-1)
    gate = np.zeros_like(excursion,dtype=bool)
    gate[WINDOW:] = (sustained & (excursion[WINDOW:]>DEAD_ZONE) &
                     (excursion[WINDOW:]-excursion[:-WINDOW]>=MIN_GAIN))
    excess = np.maximum(excursion-DEAD_ZONE,0)
    cost = np.zeros_like(excursion)
    cost[1:] = WEIGHT*np.clip(np.diff(excess,axis=0),0,MAX_STEP)*gate[1:]
    return cost


def evaluate(sources):
    rows=[]
    for label,seed,raw,folder,prefix in sources:
        with np.load(raw/f'roll_{seed}.npz') as z:
            cost=replay(z['poses'],list(z['names']))
        with np.load(folder/f'{prefix}_{seed}.npz') as z:
            live=z['live']
        events=list(csv.DictReader((folder/f'events_{prefix}_{seed}.csv').open()))
        trial=[r for r in csv.DictReader((folder/f'{prefix}_{seed}.csv').open()) if r['seconds']=='25']
        banks=sorted({int(r['bank']) for r in trial})
        exchange=np.zeros_like(live)
        for e in events:
            if int(e['active']):
                idx=banks.index(int(e['bank']))*64+int(e['slot'])
                lo=min(int(e['on']),int(e['release']))
                hi=max(int(e['on']),int(e['release']))
                exchange[max(0,lo-3):hi+5,idx]=True
        trig=cost>0
        safe=np.array([not int(r['drop']) for r in trial])
        assert np.array_equal(safe,live[-1])
        successful=np.array([not int(r['drop']) and int(r['supported45']) and int(r['repeated_handoff']) for r in trial],bool)
        lost=np.flatnonzero(~safe)
        warned=sum(trig[max(0,(n:=int(live[:,i].sum()))-20):n,i].any() for i in lost)
        ex=exchange & live
        rows.append(dict(cohort=label,seed=seed,episodes=live.shape[1],live_frames=int(live.sum()),
                         triggered_live_frames=int((live&trig).sum()),exchange_frames=int(ex.sum()),
                         triggered_exchange_frames=int((ex&trig).sum()),drops=len(lost),warned_last_second=int(warned),
                         joint_success=int(successful.sum()),successful_ever_triggered=int(trig[:,successful].any(axis=0).sum())))
    return rows
