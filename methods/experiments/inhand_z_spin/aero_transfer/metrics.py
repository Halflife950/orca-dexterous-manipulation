"""Evaluate support, rotation and active digit exchanges on a common 20 Hz grid."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import score
OUT=HERE/'results'
ORDINARY=[991009,991027,991043]
WIDE=[996101,996103,996107]
DIGITS=['index','middle','ring','pinky','thumb']
JOINTS=[[6,7,8],[9,10,11],[12,13,14],[15,16,17],[2,3,4,5]]
GROUPS=[[0],[1],[2],[3],[4,5,6]]
def write(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader();w.writerows(rows)
def first_hold(mask,n=6):
    if len(mask)<n:return -1
    spans=np.lib.stride_tricks.sliding_window_view(mask,n).all(1)
    good=np.flatnonzero(spans)
    return int(good[0]+n-1) if len(good) else -1
def exchanges(bits,ang,swing,shell,q,targets,actions,is_aero,masks):
    contact=(bits[:,None]&masks)!=0
    if len(contact)<5:return []
    pre=contact[:-3]&contact[1:-2]
    post=contact[2:-1]&contact[3:]
    off=(~contact[:-3])&(~contact[1:-2])
    lost=(~contact[2:-1])&(~contact[3:])
    gaining=np.argwhere(off&post)
    releasing=np.argwhere(pre&lost)
    examples=[]
    for idx,new in gaining:
        on=int(idx+2)
        for j,old in releasing[abs(releasing[:,0]-idx)<=5]:
            if old==new:continue
            release=int(j+2)
            l=max(0,min(on,release)-3);h=min(len(ang)-1,max(on,release)+4)
            if shell[l:h+1].any() or (bits[l:h+1]!=0).mean()<.9 or swing[l:h+1].max()>20:continue
            gain=float(ang[h]-ang[l])
            if gain<5:continue
            # Require the incoming digit's actual motion to follow a target change.
            # A command may lead the contact, so include a short pre-onset window.
            start=max(0,on-5)
            joint_ids=JOINTS[new];act_ids=GROUPS[new] if is_aero else joint_ids
            active=False
            for joint in joint_ids:
                chan=act_ids if is_aero else [joint]
                if is_aero and new==4:chan=[{3:4,2:5,4:6,5:6}[joint]]
                series=targets[start:h+1,joint]
                for t in range(len(series)-1):
                    diff=series[t+1:]-series[t]
                    hits=np.flatnonzero(abs(diff)>=np.deg2rad(3))
                    if not len(hits):continue
                    base=start+t
                    for step in hits:
                        end=base+1+int(step)
                        if np.ptp(actions[base:end+1][:,chan],axis=0).max()<=1e-4:continue
                        sign=np.sign(diff[step])
                        if np.max(sign*(q[base:min(h+1,end+4),joint]-q[base,joint]))>=np.deg2rad(2):
                            active=True;break
                    if active:break
                if active:break
            examples.append(dict(on=on,release=release,incoming=DIGITS[new],outgoing=DIGITS[old],
                                 gain=gain,active=int(active)))
    return sorted(examples,key=lambda x:(x['on'],-x['active'],-x['gain']))
def distinct(events):
    frames=[]
    for e in events:
        if not frames or e['on']-frames[-1]>=6:frames.append(e['on'])
    return len(frames)
def evaluate(arm,seed,raw):
    arch=HERE.parent
    aero=arm in ('aero','challenge')
    source=raw if arm=='aero' else raw/arm
    with np.load(source/f'roll_{seed}.npz') as z:
        dt=float(z['dt']);step=round(.05/dt)
        assert abs(dt*step-.05)<.0001
        starts=z['starts'];sha=str(z['sha']);names=list(z['names'])
        native=z['live'];pos=z['poses'];forces=z['force'][::step]
        original_xyz=pos[:,:,-1,:3]
        if aero:distance=z['distance']
        elif arm=='original':distance=np.linalg.norm(original_xyz-np.array([.06,-.17,.57]),axis=2)
        else:
            with np.load(source/f'signals_{seed}.npz') as d:distance=d['distance']
        # Thresholds are defined against the physical nominal spawn location.
        if arm=='original':height=.57
        else:
            from scipy.spatial.transform import Rotation
            ref=np.array([.01135,-.03195,.0094]);axis=np.cross(ref,[0.,0.,1.]);axis/=np.linalg.norm(axis)
            alpha=-16.69-np.degrees(np.arcsin(ref[2]/np.linalg.norm(ref)))
            pivot=np.array([.0386,-.1422,.5197])
            height=(pivot+Rotation.from_rotvec(axis*np.radians(alpha)).apply(np.array([.06,-.17,.57])-pivot))[2]
        alive=native&~np.logical_or.accumulate((distance>.20)|(original_xyz[:,:,2]<height-.10),axis=0)
        live=alive[::step];poses=pos[::step];native=native[::step]
        # Capture loss events in the final two 60 Hz source samples for the 10-s drop count.
        end_live={10:alive[min(len(alive)-1,round(10/dt)-1)]}
        if aero:end_live[25]=alive[-1]
        if aero:
            q=z['joint'];tar=z['target'];cmd=z['cmd']
        elif arm=='legacy':
            with np.load(source/f'signals_{seed}.npz') as d:
                q=d['joint'][::step];tar=d['target'][::step];cmd=d['cmd'][::step]
        elif seed:
            with np.load(arch/'first_stop'/'results'/f'joints_{seed}.npz') as j:
                q=j['actual'][::step];tar=j['target'][::step];cmd=j['action'][::step]
        else:
            q=tar=cmd=np.zeros((len(live),192,18),dtype=np.float32)
    assert live.shape[1]==192 and not (live[1:]&~live[:-1]).any()
    if arm!='challenge':
        with np.load(arch/'results'/'angles_990011.npz') as old:
            orig=old['starts']
            if arm=='legacy':
                change=R.from_quat(starts)*R.from_quat(orig).inv()
                assert (change*change[0].inv()).magnitude().max()<1e-5
                assert abs(np.degrees(change[0].magnitude())-32)<1
            else:np.testing.assert_array_equal(starts,orig)
    elif not np.allclose(np.linalg.norm(starts,axis=-1),1):raise ValueError('Invalid quaternions')
    if seed and aero:
        ck,=list((HERE/'runs'/f'aero_{seed}'/'nn').glob('last_*_ep_3072_rew__*.pth'))
        assert sha==hashlib.sha256(ck.read_bytes()).hexdigest()
    T,N=live.shape;bits=np.zeros_like(live,dtype=np.uint32);shell=np.zeros_like(live)
    for b in range(0,T,32):
        sl=slice(b,b+32);m=live[sl]
        if m.any():bits[sl][m],shell[sl][m]=score.contact_evidence(poses[sl][m],forces[sl][m],names)
    masks=np.array([sum(1<<j for j,n in enumerate(names) if '_'+d+'_' in n) for d in DIGITS],dtype=np.uint32)
    rows=[];events=[];banks=WIDE if arm=='challenge' else ORDINARY
    angles=np.full((T,N,3),np.nan,dtype='float32')
    for i in range(N):
        last=int(live[:,i].sum())
        # Seed horizons start at the first observed post-reset pose.
        if last:
            p=poses[:last,i]
            palm=R.from_quat(p[:,names.index('right_palm'),3:])
            cube=R.from_quat(p[:,-1,3:]);axis=palm[0].inv().apply([0,0,1])
            if last==1:
                tw,sw,ag=(np.zeros(1) for _ in range(3))
            else:
                tw,sw,ag=score.angles(palm.inv()*cube,axis)
            angles[:last,i]=np.column_stack([tw,sw,ag])
            hand=bits[:last,i]!=0;casing=shell[:last,i]
            reliable=~np.logical_or.accumulate(sw>=170)
            cone=reliable&(sw<=20)
            good=hand&cone&~np.logical_or.accumulate(casing)
            ev=exchanges(bits[:last,i],ag,sw,casing,q[:last,i],tar[:last,i],cmd[:last,i],aero,masks)
        else:
            tw=sw=ag=np.array([]);hand=casing=cone=good=np.zeros(0,bool);ev=[]
        key=dict(arm=arm,seed=seed,bank=banks[i//64],slot=i%64)
        for e in ev:events.append({**key,**e})
        for horizon in end_live:
            count=min(round(horizon/(dt*step)),last)
            ok=good[:count];angle=tw[:count]
            i45=first_hold(ok&(angle>=45));i90=first_hold(ok&(angle>=90))
            # Re-evaluate events at the endpoint so no post-horizon motion can qualify.
            ee=[e for e in ev if max(e['on'],e['release'])+4<count]
            active=[e for e in ee if e['active']]
            survival=bool(end_live[horizon][i]);h0=round(horizon/(dt*step))-100
            late=float(ag[count-1]-ag[h0]) if survival and count>h0 else np.nan
            rows.append({**key,'seconds':horizon,'drop':int(not survival),'native_drop':int(not native[min(len(native)-1,round(horizon/(dt*step))-1),i]),
                         'angle45':int(np.any(angle[cone[:count]]>=45)),'supported45':int(i45>=0),'supported90':int(i90>=0),
                         'rotation_handoff':int(bool(ee)),'active_handoff':int(bool(active)),'repeated_handoff':int(distinct(active)>=2),
                         'clean45_handoff_survival':int(i45>=0 and bool(active) and survival),
                         'handoff_count':distinct(active),'peak_twist':float(max(angle[cone[:count]],default=0.)),
                         'net_axial':float(ag[count-1]) if count else np.nan,'last5_axial_survivor':late,
                         'casing':int(casing[:count].any()),'last_sample':count-1,'sha':sha})
    path=OUT/f'{arm}_{seed}'
    write(path.with_suffix('.csv'),rows)
    write(OUT/f'events_{arm}_{seed}.csv',events,['arm','seed','bank','slot','on','release','incoming','outgoing','gain','active'])
    np.savez_compressed(path.with_suffix('.npz'),angles=angles,bits=bits,shell=shell,live=live,starts=starts,dt=step*dt)
    print(arm,seed,'n',len(rows),'10s drops',sum(x['drop'] for x in rows if x['seconds']==10),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['aero','legacy','original','challenge']);p.add_argument('seed',type=int);p.add_argument('raw',type=Path)
    opt=p.parse_args();evaluate(opt.arm,opt.seed,opt.raw)
