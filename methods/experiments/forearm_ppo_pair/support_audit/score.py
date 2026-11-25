"""Re-score original seed/start pairs with framewise contact proxies."""
import os
import argparse
import csv
import ctypes as C
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('project',type=Path)
p.add_argument('scratch',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
study=a.project/'experiments/forearm_ppo_pair'
root=study.parent/'forearm_rotation'
protocol=json.loads((study/'protocol.json').read_text())
xml=ET.parse(root/'forearm.xml')
lib=C.CDLL(os.environ.get('ORCA_GAP_LIBRARY',str(here/'gap.so')))
lib.create.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int];lib.create.restype=C.c_void_p
lib.distances.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p];lib.distances.restype=None
models={m.get('name'):m for m in xml.findall('.//asset/mesh')}
geometry={}
mount=xml.find(".//body[@name='right_forearm_mount']/geom")
mount_offset=np.fromstring(mount.get('pos'),sep=' ')
mount_radius=float(mount.get('size'))
for body in xml.findall('.//worldbody//body'):
    for geom in body.findall('geom'):
        if geom.get('mesh') is None:continue
        spec=models[geom.get('mesh')]
        stl=np.fromfile(root/spec.get('file'),dtype=[('n','<f4',3),('v','<f4',(3,3)),('extra','<u2')],offset=84)
        vertices=np.unique(stl['v'].reshape(-1,3),axis=0)*np.fromstring(spec.get('scale'),sep=' ')
        rot=R.from_quat(np.roll(np.fromstring(geom.get('quat','1 0 0 0'),sep=' '),-1))
        vertices=np.ascontiguousarray(rot.apply(vertices)+np.fromstring(geom.get('pos','0 0 0'),sep=' '))
        triangles=np.ascontiguousarray(ConvexHull(vertices).simplices,dtype='i4')
        h=lib.create(vertices.ctypes.data,len(vertices),triangles.ctypes.data,len(triangles))
        geometry.setdefault(body.get('name'),[]).append((h,vertices.min(0),vertices.max(0)))

def gap(name,bodies,objects):
    inv=R.from_quat(bodies[:,3:]).inv()
    rot=inv*R.from_quat(objects[:,3:])
    pos=inv.apply(objects[:,:3]-bodies[:,:3])
    if name=='right_forearm_mount':
        local=rot.inv().apply(mount_offset-pos)
        return np.maximum(0,np.linalg.norm(np.maximum(np.abs(local)-.03,0),axis=1)-mount_radius)*1000
    # Broad phase only omits mesh pairs strictly beyond the largest tested distance.
    q=rot.as_quat()
    radius=.03*np.abs(rot.as_matrix()).sum(axis=2)
    dist=np.full(len(pos),np.inf,dtype='f4')
    for handle,lower,upper in geometry[name]:
        near=np.all((pos-radius <= upper+.006001)&(pos+radius >= lower-.006001),axis=1)
        if not np.any(near):continue
        xyz=np.ascontiguousarray(pos[near]);xyzw=np.ascontiguousarray(q[near])
        g=np.empty(len(xyz),dtype='f4')
        lib.distances(handle,len(g),xyz.ctypes.data,xyzw.ctypes.data,g.ctypes.data)
        dist[near]=np.minimum(dist[near],g)
    return dist

conditions={'primary':(4,.1),'tight':(2,.2),'loose':(6,.05),'geometry':(4,0)}
original={}
with (study/'results/trials.csv').open() as f:
    for r in csv.DictReader(f):
        original[r['axis'],r['mode'],int(r['seed']),int(r['bank']),int(r['slot'])]=r

def streaks(mask):
    w=np.lib.stride_tricks.sliding_window_view(mask,16,axis=0).all(-1)
    return w.any(0),w[-1],np.where(w.any(0),np.argmax(w,axis=0)+15,-1)

def score(axis,mode,seed):
    name=f'{axis}_{mode}_{seed}'
    path=a.scratch/(name+'.npz')
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as rec, np.load(Path(os.environ.get('ORCA_EVAL_OUTPUT',study/'evaluation'))/path.name) as ref:
        names=list(rec['names']);poses=rec['poses'];force=rec['forces'];dead=ref['drop']
        sha=str(rec['checksum']);assert sha==str(ref['digest'])
        qdiff=float(rec['max_q_diff'])
        errors=(R.from_quat(ref['target'])*R.from_quat(ref['q'][:,:,3,:].reshape(-1,4)).inv()).magnitude().reshape(dead.shape)
    live=~dead;inside=live&(errors<.1);obj=poses[:,:,-1];objforce=force[:,:,-1]
    shell={k:np.zeros_like(live) for k in conditions}
    support={k:np.zeros_like(live) for k in conditions}
    palm=np.zeros_like(live)
    for i,n in enumerate(names[:-1]):
        casing=n in ('right_forearm_mount','right_tower')
        keep=live if casing else inside
        if not keep.any():continue
        distance=gap(n,poses[:,:,i][keep],obj[keep])
        magn=force[:,:,i][keep]
        cube=objforce[keep]
        for k,(mm,newtons) in conditions.items():
            near=(distance<=mm)&(magn>=newtons)&(cube>=newtons)
            group=shell if casing else support
            group[k][keep]|=near
            if n=='right_palm' and k=='primary':
                palm[keep]=near
    old,old_end,old_first=streaks(inside)
    records=[]
    for k in conditions:
        clear=~shell[k]
        accepted=inside&clear&support[k]
        held,end,first=streaks(accepted)
        shellclear,clearend,_=streaks(inside&clear)
        historyclear,historyend,_=streaks(accepted&~np.maximum.accumulate(shell[k],axis=0))
        for i in range(inside.shape[1]):
            b,slot=divmod(i,64);bank=protocol['test_banks'][b]
            prior=original[axis,mode,seed,bank,slot]
            assert (old[i],old_end[i],old_first[i])==(int(prior['held']),int(prior['final16']),int(prior['first_hold_frame']))
            records.append(dict(axis=axis,mode=mode,seed=seed,bank=bank,slot=slot,rule=k,
                    old=int(old[i]),old_end=int(old_end[i]),shell_free=int(shellclear[i]),shell_free_end=int(clearend[i]),
                    held=int(held[i]),final16=int(end[i]),first=int(first[i]),old_first=int(old_first[i]),
                    clean_history=int(historyclear[i]),clean_history_end=int(historyend[i]),
                    shell_in_old_first=int(shell[k][old_first[i]-15:old_first[i]+1,i].any()) if old[i] else '',
                    shell_in_final16=int(shell[k][-16:,i].any()) if old_end[i] else ''))
    fmt=a.out/'frames'
    fmt.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(fmt/(name+'.npz'),inside=inside,live=live,shell=shell['primary'],hand_support=support['primary'],
                        palm=palm,digest=sha,max_quat_difference=qdiff)
    print(name,'verified; max quaternion difference',qdiff,flush=True)
    return records

rows=[]
for seed in ([int(os.environ['ORCA_AUDIT_SEED'])] if 'ORCA_AUDIT_SEED' in os.environ else protocol['seeds']):
    for axis in ([os.environ['ORCA_AUDIT_AXIS']] if 'ORCA_AUDIT_AXIS' in os.environ else protocol['axes']):
        for mode in ([os.environ['ORCA_AUDIT_MODE']] if 'ORCA_AUDIT_MODE' in os.environ else protocol['modes']):
            rows+=score(axis,mode,seed)
a.out.mkdir(parents=True,exist_ok=True)
with (a.out/'trial_scores.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
