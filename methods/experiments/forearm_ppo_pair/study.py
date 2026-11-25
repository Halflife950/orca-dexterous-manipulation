import os
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'forearm_rotation'))
from model import ForearmHand, configuration, base, torch
from common import replay, starts, unique, save
import numpy as np

P = json.loads((HERE / 'protocol.json').read_text())


class StudyHand(ForearmHand):
    def __init__(self, *args, mode, stiff_brace=False, **kwargs):
        self.fixed = mode == 'fixed'
        super().__init__(*args, **kwargs)
        self.roll = self.joint_names['right_forearm_roll']
        self.roll_action = self.actuated_dof_indices.tolist().index(self.roll)
        if self.fixed:
            for ptr in self.envs:
                actor = self.gym.find_actor_handle(ptr, 'hand')
                props = self.gym.get_actor_dof_properties(ptr, actor)
                props['lower'][self.roll] = props['upper'][self.roll] = 0
                props['stiffness'][self.roll] = 500000. if stiff_brace else 100000.
                props['damping'][self.roll] = 200. if stiff_brace else 100.
                props['effort'][self.roll] = 1000.
                self.gym.set_actor_dof_properties(ptr, actor, props)

    def pre_physics_step(self, actions):
        if self.fixed:
            actions = actions.clone()
            actions[:, self.roll_action] = 0.
        super().pre_physics_step(actions)

    def reset_object_states(self, env_ids):
        super().reset_object_states(env_ids)
        if hasattr(self,'test_quats'):
            self.resetted_object_states[:,3:7] = self.test_quats[env_ids]


def settings(axis, mode, seed, n):
    run = f'{axis}_{mode}_{seed}'
    conf = configuration(n,axis,seed,P['epochs'])
    cfg = conf['train']['params']['config']
    cfg.update(name=run, full_experiment_name=run, train_dir=str(HERE/'runs'),
               save_frequency=0)
    return conf, run


def checkpoint(run):
    return unique(HERE/'runs'/run/'nn',f'last_{run}_ep_{P["epochs"]}_rew_*.pth')


def make(conf,mode,stiff_brace=False):
    return StudyHand(conf['task'],conf['rl_device'],conf['sim_device'],-1,True,False,False,mode=mode,stiff_brace=stiff_brace)


def train(a):
    conf, run = settings(a.axis,a.mode,a.seed,P['envs'])
    base.set_seed(a.seed,torch_deterministic=conf['torch_deterministic'])
    base.vecenv.register('RLGPU',lambda config_name,num_actors,**kw:base.RLGPUEnv(config_name,num_actors,**kw))
    base.env_configurations.register('rlgpu',{'vecenv_type':'RLGPU','env_creator':lambda **kw:make(conf,a.mode)})
    runner = base.Runner(base.RLGPUAlgoObserver())
    runner.load(conf['train'])
    runner.reset()
    runner.run({'train':True,'play':False,'checkpoint':'','sigma':None})
    print('Verified final checkpoint',checkpoint(run))


def evaluate(a):
    import hashlib
    bank_quats = np.concatenate([starts(b) for b in P['test_banks']])
    conf,run = settings(a.axis,a.mode,a.seed,len(bank_quats))
    base.set_seed(P['test_banks'][0],torch_deterministic=True)
    sensitivity=a.phase=='test_stiff'
    if sensitivity and a.mode!='fixed': raise ValueError("stiff check requires fixed policy")
    env = make(conf,a.mode,stiff_brace=sensitivity)
    env.test_quats = torch.tensor(bank_quats,device=env.device)
    ck = checkpoint(run)
    player, epoch = replay.player_from_checkpoint(conf['train']['params'],env,ck,env.device)
    assert epoch == P['epochs']
    ptr=env.envs[0]
    actor=env.gym.find_actor_handle(ptr,'hand')
    bodies=env.gym.get_actor_rigid_body_dict(ptr,actor)
    ids=[bodies[name] for name in ('right_forearm_mount','right_tower','right_palm')]
    env.reset_buf.fill_(True)
    env.reset_goal_buf.fill_(True)
    obs=env.reset()
    np.testing.assert_array_equal(env.resetted_object_states[:,3:7].cpu().numpy(),bank_quats)
    target=torch.tensor(conf['task']['env']['goal_start_quat'],device=env.device)
    torch.testing.assert_close(env.goal_rot,target.expand_as(env.goal_rot))
    live=torch.ones(len(bank_quats),device=env.device,dtype=torch.bool)
    rotations,roll,dead=[],[],[]
    with torch.no_grad():
        for frame in range(P['horizon']):
            live &= ~env.dropped_buf
            rotations.append(torch.cat((env.rigid_body_states[:,ids,3:7],env.object_rot[:,None,:]),dim=1).clone())
            roll.append(env.hand_dof_pos[:,env.roll].clone())
            dead.append((~live).clone())
            if frame+1 < P['horizon']:
                env.reset_buf[live]=False
                env.reset_goal_buf[live]=False
                obs,_,_,_=env.step(player.get_action(obs['obs'],is_deterministic=True))
    q=torch.stack(rotations).cpu().numpy()
    d=torch.stack(dead).cpu().numpy()
    r=torch.stack(roll).cpu().numpy()
    assert np.isfinite(q[~d]).all() and np.isfinite(r[~d]).all()
    if a.mode=='fixed':
        max_roll=float(np.degrees(np.abs(r[~d])).max())
        print('Brace check (0.25 deg):',max_roll, 'PASS' if max_roll<.25 else 'FLAG')
    out = Path(os.environ.get('ORCA_EVAL_OUTPUT', HERE/('sensitivity' if sensitivity else 'evaluation')))
    save(out/f'{run}.npz',q=q,drop=d,roll=r,initial=bank_quats,target=target.cpu().numpy(),
         epoch=epoch,digest=hashlib.sha256(ck.read_bytes()).hexdigest(),dt=env.control_dt)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','test','test_stiff'])
    parser.add_argument('axis',choices=P['axes'])
    parser.add_argument('mode',choices=P['modes'])
    parser.add_argument('seed',type=int,choices=P['seeds'])
    a=parser.parse_args()
    (train if a.phase=='train' else evaluate)(a)
