"""Pure PPO training and frozen deterministic rollouts for ORCA in-hand positive Z rotation."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forearm_ppo_pair'))
from study import StudyHand,base,torch,settings,starts,replay
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz
P=json.loads((HERE/'protocol.json').read_text())

class SpinHand(StudyHand):
    def __init__(self,*args):
        super().__init__(*args,mode='free')
        ptr=self.envs[0];actor=self.gym.find_actor_handle(ptr,'hand')
        self.palm=self.gym.find_actor_rigid_body_handle(ptr,actor,'right_palm')
        ids=self.actuated_dof_indices.tolist()
        frozen=[self.joint_names[x] for x in ('right_wrist','right_forearm_roll')]
        self.locked=[ids.index(k) for k in frozen]
        lo=self.actuated_dof_lower_limits[self.locked];hi=self.actuated_dof_upper_limits[self.locked]
        self.zero_action=-(hi+lo)/(hi-lo)
        for ptr in self.envs:
            actor=self.gym.find_actor_handle(ptr,'hand')
            props=self.gym.get_actor_dof_properties(ptr,actor)
            for k in frozen:
                props['lower'][k]=props['upper'][k]=0
                props['stiffness'][k]=500000.
                props['damping'][k]=200.
                props['effort'][k]=1000.
            self.gym.set_actor_dof_properties(ptr,actor,props)
        self.last_cmd=torch.zeros((self.num_envs,self.num_actions),device=self.device)
        self.is_first=torch.ones_like(self.reset_buf,dtype=torch.bool)
    def reset_idx(self,env_ids,goal_env_ids):
        super().reset_idx(env_ids,goal_env_ids)
        self.is_first[env_ids]=True
    def reset_object_states(self,ids):
        super().reset_object_states(ids)
        if not hasattr(self,'test_quats'):
            noise=(2*torch.rand((len(ids),3),device=self.device)-1)*torch.tensor([5,10,10],device=self.device)*(np.pi/180)
            self.resetted_object_states[:,3:7]=quat_from_euler_xyz(*noise.unbind(1))
    def pre_physics_step(self,actions):
        actions=actions.clone()
        actions[:,self.locked]=self.zero_action
        super().pre_physics_step(actions)
    def check_termination(self):
        self.dropped_buf=(self.object_pos-self.object_init_states[:,:3]).norm(dim=1)>self.cfg['rewards']['fall_dist_threshold']
        self.reset_buf=self.dropped_buf|(self.progress_buf>=self.max_episode_length)
        self.reset_goal_buf=self.reset_buf.clone()
        self.success_buf=torch.zeros_like(self.dropped_buf)
    def compute_reward(self):
        axial=self.object_angvel[:,2]-self.rigid_body_states[:,self.palm,12]
        change=(self.actions-self.last_cmd).square().sum(dim=1)*(7/16)
        self.rew_buf.copy_(self.control_dt*((axial-change)*(~self.is_first)-100*self.dropped_buf))
        self.last_cmd.copy_(self.actions)
        self.is_first.zero_()

def setup(seed,n):
    cfg,_=settings('Z','free',seed,n)
    cfg['task']['env']['episode_length_s']=10
    return cfg
def make(c):
    return SpinHand(c['task'],c['rl_device'],c['sim_device'],-1,True,False,False)
def policy(seed):
    candidates=list((HERE/'runs'/f'spin_{seed}'/'nn').glob(f'last_*_ep_{P["epochs"]}_rew__*.pth'))
    assert len(candidates)==1,candidates
    return candidates[0]
def train(seed):
    c=setup(seed,P['envs']);cfg=c['train']['params']['config'];name=f'spin_{seed}'
    cfg.update(name=name,full_experiment_name=name,max_epochs=P['epochs'],save_frequency=0,train_dir=str(HERE/'runs'))
    base.set_seed(seed,torch_deterministic=c['torch_deterministic'])
    base.vecenv.register('RLGPU',lambda config_name,num_actors,**kw:base.RLGPUEnv(config_name,num_actors,**kw))
    base.env_configurations.register('rlgpu',{'vecenv_type':'RLGPU','env_creator':lambda **kw:make(c)})
    runner=base.Runner(base.RLGPUAlgoObserver());runner.load(c['train']);runner.reset()
    runner.run({'train':True,'play':False,'checkpoint':'','sigma':None})
    print('Verified',policy(seed))
def capture(seed,raw):
    quats=np.concatenate([starts(k) for k in P['test_banks']])
    n=len(quats);c=setup(seed,n);base.set_seed(P['test_banks'][0],torch_deterministic=True)
    c['task']['sim']['physx']['contact_collection']=2
    env=make(c);env.test_quats=torch.tensor(quats,device=env.device)
    ptr=env.envs[0];h=env.gym.find_actor_handle(ptr,'hand')
    names=list(env.gym.get_actor_rigid_body_names(ptr,h))+['object']
    assert env.gym.find_actor_rigid_body_handle(ptr,env.gym.find_actor_handle(ptr,'object'),'object')==len(names)-1
    net=gymtorch.wrap_tensor(env.gym.acquire_net_contact_force_tensor(env.sim)).view(n,-1,3)
    if seed:
        ck=policy(seed);actor,epoch=replay.player_from_checkpoint(c['train']['params'],env,ck,env.device)
        assert epoch==P['epochs']
    env.reset_buf.fill_(True);env.reset_goal_buf.fill_(True);obs=env.reset()
    np.testing.assert_array_equal(env.resetted_object_states[:,3:7].cpu().numpy(),quats)
    live=torch.ones(n,dtype=torch.bool,device=env.device)
    poses=[];force=[];lives=[]
    # Constant initial joint targets for the passive reference.
    lo=env.actuated_dof_lower_limits;hi=env.actuated_dof_upper_limits
    neutral=(2*env.cur_targets[:,env.actuated_dof_indices]-lo-hi)/(hi-lo)
    with torch.no_grad():
        for t in range(P['horizon']):
            live &= ~env.dropped_buf
            env.gym.refresh_net_contact_force_tensor(env.sim)
            pose=env.rigid_body_states[:,:len(names),:7].clone()
            pose[:,-1,:3]=env.object_pos;pose[:,-1,3:]=env.object_rot
            poses.append(pose);force.append(net[:,:len(names)].norm(dim=-1));lives.append(live.clone())
            if t+1<P['horizon']:
                env.reset_buf[live]=False;env.reset_goal_buf[live]=False
                action=actor.get_action(obs['obs'],is_deterministic=True) if seed else neutral
                obs,_,_,_=env.step(action)
    arr=lambda x: torch.stack(x).cpu().numpy()
    raw.mkdir(exist_ok=True,parents=True)
    target=raw/f'roll_{seed}.npz'
    np.savez(target.with_suffix('.tmp.npz'),poses=arr(poses),force=arr(force),live=arr(lives),names=names,
             starts=quats,dt=env.control_dt,sha=hashlib.sha256(ck.read_bytes()).hexdigest() if seed else 'neutral')
    target.with_suffix('.tmp.npz').replace(target)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['train','test']);a.add_argument('seed',type=int);a.add_argument('scratch',type=Path,nargs='?')
    args=a.parse_args()
    if args.mode=='train':train(args.seed)
    else:capture(args.seed,args.scratch)
