"""Matched continuation of the original +45 degree hold-16 actors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent/'ppo_factorial'
sys.path.insert(0,str(FACTORIAL))
import importlib.util
spec = importlib.util.spec_from_file_location('factorial_training', FACTORIAL/'train.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
torch = original.torch
P = json.loads((HERE/'protocol.json').read_text())


class ContinuingHand(original.FactorialHand):
    def __init__(self, *args, dwell, **kwargs):
        super().__init__(*args, **kwargs)
        self.paid = torch.zeros_like(self.progress_buf, dtype=torch.bool)
        self.dwell = dwell

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)
        self.paid[env_ids] = False

    def check_termination(self):
        displacement = (self.object_pos-self.object_init_states[:, :3]).norm(dim=-1)
        self.dropped_buf = displacement > self.cfg['rewards']['fall_dist_threshold']
        inside = original.distance(self.object_rot,self.goal_rot) < self.cfg['rewards']['success_tolerance']
        self.streak = torch.where(inside, self.streak+1, 0)
        self.success_buf = (self.streak >= self.hold) & ~self.paid
        self.paid |= self.success_buf
        self.successes += self.success_buf
        self.reset_buf = self.dropped_buf | (self.progress_buf > self.max_episode_length-1)
        self.reset_goal_buf = self.reset_buf.clone()
        n = self.reset_buf.sum()
        finished = (self.reset_buf*self.successes).sum()
        self.consecutive_successes = torch.where(n > 0, 0.1*finished/n + 0.9*self.consecutive_successes, self.consecutive_successes)
        self.successes[self.reset_buf] = 0

    def compute_reward(self):
        super().compute_reward()
        if self.dwell:
            bonus = P['dwell_scale'] * (self.streak.float()/P['dwell_samples']).clamp(max=1)
            self.rew_buf += bonus * ~self.dropped_buf


def parent(seed):
    matches = list((FACTORIAL/'runs'/f'proximity_16_{seed}'/'nn').glob('last_*_ep_5000_rew__*.pth'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one final proximity/hold-16 parent for seed {seed}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=P['arms'])
    parser.add_argument('seed',type=int,choices=P['training_seeds'])
    a = parser.parse_args()
    end = P['parent_epoch'] + P['extra_epochs']
    cfg = original.OmegaConf.load(original.CONFIG)
    cfg.seed, cfg.max_iterations, cfg.experiment = a.seed, end, f'{a.arm}_{a.seed}'
    config = original.OmegaConf.to_container(cfg,resolve=True)
    settings = config['train']['params']['config']
    settings['train_dir'] = str(HERE/'runs')
    start = parent(a.seed)
    state = torch.load(start,map_location='cpu',weights_only=False)
    assert state['epoch']==P['parent_epoch'] and 'optimizer' in state
    state['model'] = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k:v for k,v in state['model'].items()}
    original.set_seed(a.seed,torch_deterministic=cfg.torch_deterministic)

    def make_env(**kwargs):
        args = (config['task'],cfg.rl_device,cfg.sim_device,cfg.graphics_device_id,True,False,False)
        common = dict(shaping='proximity',hold=16)
        if a.arm=='reset':
            return original.FactorialHand(*args,**common)
        return ContinuingHand(*args,dwell=a.arm=='dwell',**common)
    original.vecenv.register('RLGPU',lambda config_name,num_actors,**kw: original.RLGPUEnv(config_name,num_actors,**kw))
    original.env_configurations.register('rlgpu',{'vecenv_type':'RLGPU','env_creator':make_env})
    with TemporaryDirectory() as tmp:
        ck = Path(tmp)/'parent.pth'
        torch.save(state,ck)
        runner = original.Runner(original.RLGPUAlgoObserver())
        runner.load(config['train'])
        runner.reset()
        runner.run({'train':True,'play':False,'checkpoint':str(ck),'sigma':None})


if __name__=='__main__':
    main()
