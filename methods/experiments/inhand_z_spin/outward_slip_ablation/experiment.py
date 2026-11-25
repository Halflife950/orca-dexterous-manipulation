"""Pure PPO with a gated cost for sustained outward translation."""
from pathlib import Path
import os
import argparse
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'command_cost_ablation'
sys.path.insert(0,str(HERE.parent/'aero_transfer'))
FRESH = (999211, 999223, 999239)
SEEDS = (990011, 990023, 990037)


def simulate(mode, seed, raw, fresh, arm):
    import transfer
    from transfer import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from slip_audit import DEAD_ZONE, WINDOW, MIN_GAIN, WEIGHT, MAX_STEP
    import numpy as np

    assert mode != 'train' or (not fresh and arm=='slip')
    transfer.HERE = HERE if arm=='slip' else PREVIOUS
    if fresh:
        transfer.BANKS = FRESH
    original_config = transfer.config
    def config(seed,n,training=False):
        c=original_config(seed,n,training)
        c['task']['command_change_weight']=.1
        if arm=='slip':
            c['task']['slip_cost']=dict(dead_zone_m=DEAD_ZONE,window_steps=WINDOW,
                                        min_gain_m=MIN_GAIN,weight_per_m=WEIGHT,max_step_m=MAX_STEP)
        return c
    trace=[]
    class SlipHand(transfer.AeroORCA):
        def __init__(self,*args):
            super().__init__(*args)
            self.anchor=torch.zeros((self.num_envs,3),device=self.device)
            self.new_grasp=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
            self.past=torch.zeros((self.num_envs,WINDOW),device=self.device)
            self.radius=torch.zeros(self.num_envs,device=self.device)
            self.streak=torch.zeros(self.num_envs,dtype=torch.int32,device=self.device)
        def reset_idx(self,env_ids,goal_env_ids):
            super().reset_idx(env_ids,goal_env_ids)
            if hasattr(self,'new_grasp'):
                self.new_grasp[env_ids]=True
                self.past[env_ids]=0.
                self.streak[env_ids]=0
        def slip_cost(self):
            palm=self.rigid_body_states[:,self.palm]
            local=quat_rotate_inverse(palm[:,3:7],self.object_pos-palm[:,:3])
            self.anchor=torch.where(self.new_grasp[:,None],local,self.anchor)
            distance=(local-self.anchor).norm(dim=1)
            radius=local.norm(dim=1)
            prior=self.past[:,-1]
            moving=(distance>prior)&(radius>self.radius)&(~self.new_grasp)
            self.streak=torch.where(moving,self.streak+1,0)
            active=(self.streak>=WINDOW)&(distance>DEAD_ZONE)&(distance-self.past[:,0]>=MIN_GAIN)
            beyond=(distance-DEAD_ZONE).clamp_min(0)-(prior-DEAD_ZONE).clamp_min(0)
            cost=WEIGHT*beyond.clamp(0,MAX_STEP)*active
            self.past=torch.cat((self.past[:,1:],distance[:,None]),dim=1)
            self.radius=radius
            self.new_grasp.zero_()
            return cost
        def compute_reward(self):
            change=(self.command-self.previous).square().sum(1)
            self.rew_buf.copy_(self.control_dt*(self.object_angvel[:,2]-.1*change-100*self.dropped_buf))
            if arm=='slip':
                cost=self.slip_cost()
                self.rew_buf.sub_(cost)
                if mode=='test':
                    trace.append(cost.clone())
            self.previous.copy_(self.command)
    transfer.AeroORCA=SlipHand
    transfer.config=config
    if mode=='train':
        transfer.train(seed)
        return
    dest=raw/'challenge' if fresh else raw
    transfer.capture(seed,dest)
    if trace:
        values=torch.stack(trace).cpu().numpy()
        assert values.shape[0]>=500
        np.save(dest/f'cost_{seed}.npy',values[-500:])


def score(seed,raw,fresh,arm):
    import metrics
    metrics.HERE=HERE if arm=='slip' else PREVIOUS
    metrics.OUT=HERE/'results'
    if fresh:
        metrics.WIDE=FRESH
        metrics.OUT=metrics.OUT/'fresh'/arm
    metrics.OUT=Path(os.environ.get('ORCA_SCORE_OUTPUT', metrics.OUT))
    metrics.OUT.mkdir(exist_ok=True,parents=True)
    metrics.evaluate('challenge' if fresh else 'aero',seed,raw)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['train','test','score'])
    p.add_argument('seed',type=int,choices=SEEDS)
    p.add_argument('raw',type=Path)
    p.add_argument('--arm',choices=['baseline','slip'],default='slip')
    p.add_argument('--fresh',action='store_true')
    opt=p.parse_args()
    if opt.mode=='score':
        score(opt.seed,opt.raw,opt.fresh,opt.arm)
    else:
        simulate(opt.mode,opt.seed,opt.raw,opt.fresh,opt.arm)
