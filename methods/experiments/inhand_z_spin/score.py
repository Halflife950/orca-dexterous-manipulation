"""Score net in-palm twist, independent swing and support from measured body poses."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'z_support_ppo'))
from support import contact_evidence
from geometry import geometry
P=json.loads((HERE/'protocol.json').read_text())
DIGITS=('thumb','index','middle','ring','pinky')

def angles(rot,axis):
    d=rot*rot[0].inv()
    q=d.as_quat()
    twist=np.degrees(np.unwrap(2*np.arctan2(q[:,:3]@axis,q[:,3])))
    swing=np.degrees(np.arccos(np.clip(d.apply(axis)@axis,-1,1)))
    step=rot[1:]*rot[:-1].inv()
    integrated=np.r_[0,np.cumsum(np.degrees(step.as_rotvec()@axis))]
    return twist-twist[0],swing,integrated

def first(mask):
    hits=np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1

def hold(mask):
    hits=np.lib.stride_tricks.sliding_window_view(mask,16).all(-1)
    i=first(hits)
    return i+15 if i>=0 else -1

def coordination(p,names,bits,shell,angle,swing,end):
    a=np.arange(0,end-3,4);b=a+4
    if not len(a):return 0,0
    progress=(angle[b]-angle[a]>=1)&(swing[a]<=20)&(swing[b]<=20)&~np.maximum.reduce([shell[a+i] for i in range(5)])
    a=a[progress];b=b[progress]
    if not len(a):return 0,0
    palm=R.from_quat(p[:,names.index('right_palm'),3:])
    axis=palm[0].inv().apply([0,0,1]); inv=palm.inv()
    cube=p[:,-1,:3];moving=np.zeros(len(a),int)
    for digit in DIGITS:
        best=np.full(len(a),np.inf);follow=np.zeros(len(a),bool)
        for j,name in enumerate(names):
            if '_'+digit+'_' not in name:continue
            contact=((bits[a]&bits[b])&(1<<j))!=0
            if not contact.any():continue
            boxes=geometry[name]
            center=(np.min([v[1] for v in boxes],0)+np.max([v[2] for v in boxes],0))/2
            pos=p[:,j,:3]+R.from_quat(p[:,j,3:]).apply(center)
            rel=inv.apply(pos-cube);d=np.linalg.norm(rel[a],axis=1)
            update=contact&(d<best)
            tangent=np.cross(axis,rel[a]);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
            follow[update]=(np.sum((rel[b]-rel[a])*tangent,axis=1)[update]>.0003)
            best[update]=d[update]
        moving+=follow
    return int(moving.max()),int((moving>=3).sum())

def score(seed,raw):
    with np.load(raw/f'roll_{seed}.npz') as data:
        p=data['poses'];f=data['force'];live=data['live'];names=list(data['names']);starts=data['starts']
        sha=str(data['sha']);dt=float(data['dt'])
    T,N=live.shape
    assert (T,N)==(600,192) and not (live[1:]&~live[:-1]).any() and live[0].all()
    assert np.isfinite(p[live]).all() and np.isfinite(f[live]).all()
    bits=np.zeros((T,N),np.uint32);shell=np.zeros((T,N),bool)
    for start in range(0,T,25):
        sl=slice(start,start+25);mask=live[sl]
        if mask.any():bits[sl][mask],shell[sl][mask]=contact_evidence(p[sl][mask],f[sl][mask],names)
    fields=['twist','swing','integrated','world_twist','palm_excursion']
    curves={s:np.full((T,N),np.nan,dtype=np.float32) for s in fields}
    rows=[]
    for i in range(N):
        last=int(live[:,i].sum())-1;cut=slice(0,last+1);pose=p[cut,i]
        palm=R.from_quat(pose[:,names.index('right_palm'),3:]);cube=R.from_quat(pose[:,-1,3:])
        rel=palm.inv()*cube; axis=palm[0].inv().apply([0,0,1])
        tw,sw,integ=angles(rel,axis);world=angles(cube,np.array([0,0,1]))[0]
        excursion=np.degrees((palm*palm[0].inv()).magnitude())
        for k,values in zip(fields,[tw,sw,integ,world,excursion]):curves[k][cut,i]=values
        hand=bits[cut,i]!=0;casing=shell[cut,i];good=hand&~np.logical_or.accumulate(casing)&(sw<=20)&~np.logical_or.accumulate(sw>=170)
        inside=lambda a: (tw>=a)&(sw<=20)&~np.logical_or.accumulate(sw>=170)
        h45=hold(good&(tw>=45)) if last>=15 else -1
        h90=hold(good&(tw>=90)) if last>=15 else -1
        h180=hold(good&(tw>=180)) if last>=15 else -1
        end=h45 if h45>=0 else last
        fingers,windows=coordination(pose,names,bits[cut,i],casing,tw,sw,end)
        reliable=~np.logical_or.accumulate(sw>=170)
        # Among directions within the 20-degree swing cone; initial angle is zero.
        in_cone=(sw<=20)&reliable
        rows.append(dict(seed=seed,bank=P['test_banks'][i//64],slot=i%64,initial=' '.join(map(str,starts[i])),sha=sha,
                         raw45=first((tw>=45)&reliable),cone45=first(inside(45)),hold45=h45,hold90=h90,hold180=h180,
                         nondrop=last==T-1,drop=int(last<T-1),qualified=int(h45>=0 and last==T-1 and fingers>=3),
                         fingers=fingers,windows3=windows,last_live=last,shell_ever=int(casing.any()),singular=int((~reliable).any()),
                         peak_cone_twist=float(tw[in_cone].max()),last_twist=float(tw[last]) if reliable[last] else '',
                         max_palm=float(excursion.max()),world_at45=float(world[h45]) if h45>=0 else '',
                         integrated_at45=float(integ[h45]) if h45>=0 else '',t45=round(h45*dt,5) if h45>=0 else ''))
    out=HERE/'results';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'angles_{seed}.npz',**curves,live=live,shell=shell,contact_bits=bits,starts=starts,dt=dt,names=names)
    with (out/f'trials_{seed}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(seed,'qualified',sum(x['qualified'] for x in rows),'supported 45',sum(x['hold45']>=0 for x in rows),'drop',sum(x['drop'] for x in rows),flush=True)

if __name__=='__main__':score(int(sys.argv[1]),Path(sys.argv[2]))
