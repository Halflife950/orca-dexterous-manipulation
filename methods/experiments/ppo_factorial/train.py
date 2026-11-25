"""Train one factorial cell with the archived native rl-games PPO configuration."""
import argparse
from pathlib import Path

import isaacgym
import isaacgymenvs
import torch
from faive_gym.robot_hand import RobotHand, quat_mul, quat_conjugate
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from isaacgymenvs.utils.utils import set_seed
from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[1] / 'orca_reversal/model/config.yaml'


def distance(q, goal):
    delta = quat_mul(q, quat_conjugate(goal))
    return 2 * torch.asin(delta[:, :3].norm(dim=-1).clamp(max=1.0))


class FactorialHand(RobotHand):
    def __init__(self, *args, shaping, hold, **kwargs):
        self.shaping = shaping
        self.hold = hold
        super().__init__(*args, **kwargs)
        self.streak = torch.zeros_like(self.progress_buf)
        self.previous_score = torch.zeros_like(self.rew_buf)

    def reset_idx(self, env_ids, goal_env_ids):
        super().reset_idx(env_ids, goal_env_ids)
        if len(env_ids):
            self.streak[env_ids] = 0
            if self.shaping == 'improvement':
                error = distance(self.resetted_object_states[:, 3:7], self.goal_states[env_ids, 3:7])
                self.previous_score[env_ids] = 1 / (error + 0.1)

    def check_termination(self):
        if self.hold == 1:
            return super().check_termination()
        inside = distance(self.object_rot, self.goal_rot) < self.cfg['rewards']['success_tolerance']
        self.streak = torch.where(inside, self.streak + 1, 0)
        self.success_buf = self.streak >= self.hold
        self.successes += self.success_buf
        displacement = (self.object_pos - self.object_init_states[:, :3]).norm(dim=-1)
        self.dropped_buf = displacement > self.cfg['rewards']['fall_dist_threshold']
        self.reset_buf = self.success_buf | self.dropped_buf | (self.progress_buf > self.max_episode_length - 1)
        self.reset_goal_buf = self.reset_buf.clone()
        n = self.reset_buf.sum()
        completed = (self.reset_buf * self.successes).sum()
        self.consecutive_successes = torch.where(n > 0, 0.1 * completed / n + 0.9 * self.consecutive_successes, self.consecutive_successes)
        self.successes[self.reset_buf] = 0

    def _reward_reorienttask_obj_rot(self):
        score = super()._reward_reorienttask_obj_rot()
        if self.shaping == 'proximity':
            return score
        improvement = score - self.previous_score
        self.previous_score.copy_(score)
        return improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shaping', choices=['improvement', 'proximity'])
    parser.add_argument('hold', type=int, choices=[1, 16])
    parser.add_argument('seed', type=int)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--output', type=Path, default=HERE/'runs')
    args = parser.parse_args()
    cfg = OmegaConf.load(CONFIG)
    cfg.seed, cfg.max_iterations = args.seed, args.epochs
    cfg.experiment = f'{args.shaping}_{args.hold}_{args.seed}'
    config = OmegaConf.to_container(cfg, resolve=True)
    assert config['checkpoint'] == '' and config['num_envs'] == 512
    settings = config['train']['params']['config']
    settings['train_dir'] = str(args.output.resolve())
    folder = args.output / cfg.experiment
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    set_seed(args.seed, torch_deterministic=cfg.torch_deterministic)

    def make_env(**kwargs):
        return FactorialHand(config['task'], cfg.rl_device, cfg.sim_device,
                             cfg.graphics_device_id, True, False, False,
                             shaping=args.shaping, hold=args.hold)

    vecenv.register('RLGPU', lambda config_name, num_actors, **kw: RLGPUEnv(config_name, num_actors, **kw))
    env_configurations.register('rlgpu', {'vecenv_type': 'RLGPU', 'env_creator': make_env})
    runner = Runner(RLGPUAlgoObserver())
    runner.load(config['train'])
    runner.reset()
    runner.run({'train': True, 'play': False, 'checkpoint': '', 'sigma': None})


if __name__ == '__main__':
    main()
