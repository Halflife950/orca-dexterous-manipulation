"""ORCA adaptation of the Aero continuous spin task. Seven kinematic actuator channels."""
from pathlib import Path
import argparse
import copy
import json
import hashlib
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import spin
from spin import torch, gymtorch
from gym import spaces
from isaacgym.torch_utils import quat_mul
SEEDS=spin.P['seeds']
BANKS=spin.P['test_banks']
DIGITS=('index','middle','ring','pinky')
# Measured forward direction in the ordinary ORCA grasp, wrist frame origin to
# midpoint between the index and small finger bases (rounding below 0.01 deg).
FORWARD=np.array([.01135,-.03195,.0094])
PIVOT=np.array([.0386,-.1422,.5197])
axis=np.cross(FORWARD,[0.,0.,1.]);axis/=np.linalg.norm(axis)
pitch=np.degrees(np.arcsin(FORWARD[2]/np.linalg.norm(FORWARD)))
TILT=R.from_rotvec(axis*np.radians(-16.69-pitch))
def mount(c):
    env=c['task']['env']
    p=np.array(env['hand_start_p'])
    env['hand_start_p']=(PIVOT+TILT.apply(p-PIVOT)).tolist()
    env['hand_start_r']=(TILT*R.from_quat(env['hand_start_r'])).as_quat().tolist()
    env['object_start_offset']=TILT.apply(env['object_start_offset']).tolist()
    return c
def config(seed,n,training=False):
    c=mount(spin.setup(seed,n))
    t=c['task']
    t['env'].update(controlFrequencyInv=3,max_target_delta=0.,episode_length_s=25.)
    t['transfer_training']=training
    t['transfer_seed']=seed
    if training:
        t['reset_noise']['dof_pos']=.045
    cfg=c['train']['params']['config']
    cfg.update(name=f'aero_{seed}',full_experiment_name=f'aero_{seed}',max_epochs=3072,
               save_frequency=0,train_dir=str(HERE/'runs'))
    core={k:cfg[k] for k in ('minibatch_size','mini_epochs','learning_rate','lr_schedule','schedule_type',
                             'kl_threshold','clip_value','normalize_input','truncate_grads')}
    core['network']=dict(name='actor_critic',central_value=True,mlp=copy.deepcopy(c['train']['params']['network']['mlp']))
    cfg['central_value_config']=core
    return c

class AeroORCA(spin.SpinHand):
    def _prepare_observations(self):
        super()._prepare_observations()
        return 14,96
    def _create_envs(self,*args,**kwargs):
        super()._create_envs(*args,**kwargs)
        if not self.cfg['transfer_training']:return
        rng=np.random.default_rng(self.cfg['transfer_seed']+221)
        for ptr in self.envs:
            obj=self.gym.find_actor_handle(ptr,'object');hand=self.gym.find_actor_handle(ptr,'hand')
            props=self.gym.get_actor_rigid_body_properties(ptr,obj)
            props[0].mass*=rng.uniform(.8,1.2)
            self.gym.set_actor_rigid_body_properties(ptr,obj,props,True)
            shapes=self.gym.get_actor_rigid_shape_properties(ptr,obj)
            for s in shapes:s.friction=float(rng.uniform(.1,.5))
            self.gym.set_actor_rigid_shape_properties(ptr,obj,shapes)
            shapes=self.gym.get_actor_rigid_shape_properties(ptr,hand)
            ranges=self.gym.get_actor_rigid_body_shape_indices(ptr,hand)
            fn=self.gym.get_actor_rigid_body_names(ptr,hand)
            friction=float(rng.uniform(.5,1.))
            for k,name in enumerate(fn):
                if name in ['right_'+w+'_ip' for w in DIGITS]+['right_thumb_dp']:
                    for j in range(ranges[k].start,ranges[k].start+ranges[k].count):
                        shapes[j].friction=friction
            self.gym.set_actor_rigid_shape_properties(ptr,hand,shapes)
    def __init__(self,*args):
        super().__init__(*args)
        self.num_actions=7
        self.act_space=spaces.Box(-1.,1.,shape=(7,),dtype=np.float32)
        names=self.joint_names
        # Each column is one actuator. Coupling uses equal fractional excursion,
        # so the physical angle of each linked joint still respects its own limits.
        groups=[['right_'+w+'_mcp','right_'+w+'_pip'] for w in DIGITS]
        groups += [['right_thumb_abd'],['right_thumb_mcp'],['right_thumb_pip','right_thumb_dip']]
        mat=torch.zeros((18,7),device=self.device)
        for j,g in enumerate(groups):
            for name in g:mat[names[name],j]=1.
        self.coupling=mat
        self.readout=mat/mat.sum(0)
        lo,hi=self.actuated_dof_lower_limits,self.actuated_dof_upper_limits
        self.home=(2*self.hand_dof_default_pos-lo-hi)/(hi-lo)
        # Center each flexor about a half-closed reference to keep opening and closing available.
        self.home=torch.where(mat.sum(1)>0,torch.zeros_like(self.home),self.home).clamp(-1,1)
        self.home_q=(lo+hi)/2+(hi-lo)/2*self.home
        self.command=torch.zeros((self.num_envs,7),device=self.device)
        self.previous=self.command.clone()
        self.test_tilt=torch.tensor(TILT.as_quat(),device=self.device,dtype=torch.float)
        self.omega_drop_height=float(self.cfg['env']['hand_start_p'][2]+self.cfg['env']['object_start_offset'][2]-.10)
        self.tips=[self.gym.find_actor_rigid_body_handle(self.envs[0],self.gym.find_actor_handle(self.envs[0],'hand'),'right_'+name)
                   for name in ['index_ip','middle_ip','ring_ip','pinky_ip','thumb_dp']]
        self.obs_noise=self.cfg['transfer_training']
    def reset_object_states(self,ids):
        super().reset_object_states(ids)
        states=self.resetted_object_states
        if self.cfg['transfer_training']:
            q=torch.randn((len(ids),4),device=self.device)
            states[:,3:7]=torch.nn.functional.normalize(q,dim=-1)
            states[:,:3]+=(torch.rand_like(states[:,:3])*2-1)*.01
        elif hasattr(self,'test_quats'):
            states[:,3:7]=quat_mul(self.test_tilt.expand(len(ids),4),self.test_quats[ids])
    def reset_idx(self,env_ids,goal_env_ids):
        super().reset_idx(env_ids,goal_env_ids)
        if hasattr(self,'previous'):self.previous[env_ids]=0.
    def pre_physics_step(self,actions):
        self.command=actions.to(self.device).clamp(-1,1)
        joint=(self.home+self.command@self.coupling.T).clamp(-1,1)
        super().pre_physics_step(joint)
    def compute_observations(self):
        lo,hi=self.actuated_dof_lower_limits,self.actuated_dof_upper_limits
        q=self.hand_dof_pos[:,self.actuated_dof_indices]
        normalized=(2*q-lo-hi)/(hi-lo)
        signals=normalized@self.readout
        if self.obs_noise:
            signals=signals+(2*torch.rand_like(signals)-1)*.025
        obs=torch.cat([signals,self.command],dim=1)
        self.obs_buf.copy_(obs)
        tips=self.rigid_body_states[:,self.tips,:3]-self.object_pos[:,None,:]
        cube=torch.cat([self.object_pos-self.rigid_body_states[:,self.palm,:3],self.object_rot,
                        self.object_linvel,self.object_angvel],dim=1)
        state=torch.cat([obs,normalized,self.hand_dof_vel,cube,tips.flatten(1),self.dof_force_tensor[:,:18]],dim=1)
        self.states_buf.copy_(state.clamp(-100,100))
    def check_termination(self):
        self.dropped_buf=self.object_pos[:,2]<self.omega_drop_height
        self.reset_buf=self.dropped_buf|(self.progress_buf>=500)
        self.reset_goal_buf=self.reset_buf.clone()
        self.success_buf=torch.zeros_like(self.dropped_buf)
    def compute_reward(self):
        self.rew_buf.copy_(self.control_dt*(self.object_angvel[:,2]-(self.command-self.previous).square().sum(1)-100*self.dropped_buf))
        self.previous.copy_(self.command)

def make(c):
    return AeroORCA(c['task'],c['rl_device'],c['sim_device'],-1,True,False,False)
def policy(seed):
    paths=list((HERE/'runs'/f'aero_{seed}'/'nn').glob('last_*_ep_3072_rew__*.pth'))
    assert len(paths)==1,paths
    return paths[0]
def train(seed):
    c=config(seed,512,True)
    (HERE/'results'/f'config_{seed}.json').write_text(json.dumps(c,indent=1))
    spin.base.set_seed(seed,torch_deterministic=c['torch_deterministic'])
    spin.base.vecenv.register('RLGPU',lambda config_name,num_actors,**kw:spin.base.RLGPUEnv(config_name,num_actors,**kw))
    spin.base.env_configurations.register('rlgpu',{'vecenv_type':'RLGPU','env_creator':lambda **kw:make(c)})
    runner=spin.base.Runner(spin.base.RLGPUAlgoObserver());runner.load(c['train']);runner.reset()
    runner.run({'train':True,'play':False,'checkpoint':'','sigma':None})
    print('Verified',policy(seed),flush=True)
def capture(seed,raw,probe=False):
    starts=np.concatenate([spin.starts(k) for k in BANKS])
    n=8 if probe else len(starts)
    c=config(seed,n)
    c['task']['sim']['physx']['contact_collection']=2
    spin.base.set_seed(BANKS[0],torch_deterministic=True)
    env=make(c)
    env.test_quats=torch.tensor(starts[:n],device=env.device)
    ptr=env.envs[0];h=env.gym.find_actor_handle(ptr,'hand')
    names=list(env.gym.get_actor_rigid_body_names(ptr,h))+['object']
    net=gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim)).view(n,-1,3)
    checkpoint=None
    if seed and not probe:
        checkpoint=policy(seed)
        actor,epoch=spin.replay.player_from_checkpoint(c['train']['params'],env,checkpoint,env.device)
        assert epoch==3072
    env.reset_buf.fill_(True);env.reset_goal_buf.fill_(True)
    obs=env.reset()
    np.testing.assert_allclose(env.resetted_object_states[:,3:7].cpu().numpy(),(TILT*R.from_quat(starts[:n])).as_quat(),atol=2e-6)
    live=torch.ones(n,device=env.device,dtype=torch.bool)
    records={k:[] for k in ('poses','force','live','cmd','joint','target','distance')}
    with torch.no_grad():
        for t in range(8 if probe else 500):
            live &= ~env.dropped_buf
            env.gym.refresh_net_contact_force_tensor(env.sim)
            p=env.rigid_body_states[:,:len(names),:7].clone()
            p[:,-1]=env.object_pose
            values=(p,net[:,:len(names)].norm(dim=-1),live,env.command,env.hand_dof_pos,env.cur_targets,
                    (env.object_pos-env.object_init_states[:,:3]).norm(dim=1))
            for k,v in zip(records,values):records[k].append(v.clone())
            if t+1<(8 if probe else 500):
                env.reset_buf[live]=False;env.reset_goal_buf[live]=False
                a=actor.get_action(obs['obs'],is_deterministic=True) if checkpoint else torch.zeros_like(env.command)
                obs,_,_,_=env.step(a)
    arr={k:torch.stack(v).cpu().numpy() for k,v in records.items()}
    p=arr['poses'][0];u=(p[:,names.index('right_index_mp'),:3]+p[:,names.index('right_pinky_mp'),:3])/2-p[:,names.index('right_palm'),:3]
    pitch=np.degrees(np.arcsin(u[:,2]/np.linalg.norm(u,axis=1)))
    print('pitch median',np.median(pitch),'live0',arr['live'][0].sum(), 'obs',obs['obs'].shape,flush=True)
    assert np.abs(np.median(pitch)+16.69)<2 and arr['live'][0].all()
    assert all(np.isfinite(a).all() for a in arr.values())
    if not probe:
        raw.mkdir(parents=True,exist_ok=True)
        np.savez(raw/f'roll_{seed}.npz',**arr,names=names,starts=starts,dt=env.control_dt,pitch=pitch,
                 sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else 'fixed')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['probe','train','test']);p.add_argument('seed',type=int);p.add_argument('raw',type=Path)
    a=p.parse_args()
    if a.mode=='train':train(a.seed)
    else:capture(a.seed,a.raw,a.mode=='probe')
