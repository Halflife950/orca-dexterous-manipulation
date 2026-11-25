"""Capture contact readouts and rigid poses without changing any control commands."""
import os
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('project',type=Path)
p.add_argument('axis',choices=['Y','Z'])
p.add_argument('mode',choices=['fixed','free'])
p.add_argument('seed',type=int)
p.add_argument('scratch',type=Path)
a=p.parse_args()
parent=a.project/'experiments/forearm_ppo_pair'
sys.path.insert(0,str(parent))
from study import P,starts,settings,base,make,torch,replay,checkpoint
from isaacgym import gymtorch

quats=np.concatenate([starts(b) for b in P['test_banks']])
n=len(quats)
conf,run=settings(a.axis,a.mode,a.seed,n)
base.set_seed(P['test_banks'][0],torch_deterministic=True)
conf['task']['sim']['physx']['contact_collection']=2
env=make(conf,a.mode)
env.test_quats=torch.tensor(quats,device=env.device)
net=gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim)).view(n,-1,3)
ck=checkpoint(run)
actor,epoch=replay.player_from_checkpoint(conf['train']['params'],env,ck,env.device)
ref=np.load(Path(os.environ.get('ORCA_EVAL_OUTPUT',parent/'evaluation'))/f'{run}.npz')
digest=hashlib.sha256(ck.read_bytes()).hexdigest()
assert epoch==P['epochs'] and digest==str(ref['digest'])
ptr=env.envs[0]
hand=env.gym.find_actor_handle(ptr,'hand')
names=list(env.gym.get_actor_rigid_body_names(ptr,hand))+['object']
ids=[names.index(k) for k in ('right_forearm_mount','right_tower','right_palm','object')]
object_id=env.gym.find_actor_rigid_body_handle(ptr,env.gym.find_actor_handle(ptr,'object'),'object')
assert object_id==len(names)-1
env.reset_buf.fill_(True);env.reset_goal_buf.fill_(True)
obs=env.reset()
np.testing.assert_array_equal(env.resetted_object_states[:,3:7].cpu().numpy(),ref['initial'])
torch.testing.assert_close(env.goal_rot,torch.tensor(ref['target'],device=env.device).expand_as(env.goal_rot))
live=torch.ones(n,device=env.device,dtype=torch.bool)
poses=[]; forces=[]; dead=[]
with torch.no_grad():
    for frame in range(P['horizon']):
        live &= ~env.dropped_buf
        env.gym.refresh_net_contact_force_tensor(env.sim)
        pose=env.rigid_body_states[:,:len(names),:7].clone()
        pose[:,-1,:3]=env.object_pos
        pose[:,-1,3:]=env.object_rot
        poses.append(pose)
        forces.append(net[:,:len(names)].norm(dim=-1))
        dead.append((~live).clone())
        if frame+1<P['horizon']:
            env.reset_buf[live]=False
            env.reset_goal_buf[live]=False
            obs,_,_,_=env.step(actor.get_action(obs['obs'],is_deterministic=True))
poses=torch.stack(poses).cpu().numpy()
force=torch.stack(forces).cpu().numpy()
d=torch.stack(dead).cpu().numpy()
np.testing.assert_array_equal(d,ref['drop'])
q=poses[:,:,ids,3:]
err=np.abs(q[~d]-ref['q'][~d]).max()
assert err<1e-5 and np.isfinite(force[~d]).all(),err
a.scratch.mkdir(parents=True,exist_ok=True)
tmp=a.scratch/f'{run}.partial.npz'
np.savez(tmp,poses=poses,forces=force,names=names,checksum=digest,max_q_diff=err)
tmp.replace(a.scratch/f'{run}.npz')
print('Replay verified:',run,'max quaternion difference',err,flush=True)
