"""Matched fixed-axis PPO specialists. Same training environments as the archived X arm."""
import argparse
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hold_training', HERE.parent/'ppo_sustained_hold/train.py')
hold_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hold_training)
base = hold_training.original
torch = base.torch
P = json.loads((HERE/'protocol.json').read_text())


def checkpoint(axis, seed, stage):
    name = f'{axis}_{stage}_{seed}'
    epoch = P['base_epochs'] + (P['dwell_epochs'] if stage == 'dwell' else 0)
    files = list((HERE/'runs'/name/'nn').glob(f'last_{name}_ep_{epoch}_rew__*.pth'))
    if len(files) != 1:
        raise FileNotFoundError(f'Expected one final checkpoint for {name}: {files}')
    return files[0]


def goal_for_axis(axis):
    quat = [0., 0., 0., 0.9238795]
    quat['XYZ'.index(axis)] = 0.3826834
    return quat


def train(axis, seed, stage):
    cfg = base.OmegaConf.load(base.CONFIG)
    cfg.task.env.goal_start_quat = goal_for_axis(axis)
    cfg.seed = seed
    cfg.max_iterations = P['base_epochs'] + (P['dwell_epochs'] if stage == 'dwell' else 0)
    cfg.experiment = f'{axis}_{stage}_{seed}'
    conf = base.OmegaConf.to_container(cfg, resolve=True)
    settings = conf['train']['params']['config']
    settings['train_dir'] = str(HERE/'runs')
    settings['save_frequency'] = 1000
    assert conf['num_envs'] * settings['horizon_length'] == P['steps_per_epoch']
    obs_keys = conf['task']['observations']['actor_observations']
    assert 'goal_quat' in obs_keys and 'goal_quat_diff' in obs_keys
    folder = HERE/'runs'/cfg.experiment
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'config.yaml').write_text(base.OmegaConf.to_yaml(base.OmegaConf.create(conf)))
    base.set_seed(seed, torch_deterministic=cfg.torch_deterministic)

    def make_env(**kwargs):
        args = (conf['task'], cfg.rl_device, cfg.sim_device, cfg.graphics_device_id, True, False, False)
        shared = dict(shaping='proximity', hold=16)
        if stage == 'base':
            return base.FactorialHand(*args, **shared)
        return hold_training.ContinuingHand(*args, dwell=True, **shared)

    base.vecenv.register('RLGPU', lambda config_name, num_actors, **kw: base.RLGPUEnv(config_name, num_actors, **kw))
    base.env_configurations.register('rlgpu', {'vecenv_type':'RLGPU', 'env_creator':make_env})
    with TemporaryDirectory() as tmp:
        resume = ''
        if stage == 'dwell':
            state = torch.load(checkpoint(axis, seed, 'base'), map_location='cpu', weights_only=False)
            assert state['epoch'] == P['base_epochs'] and 'optimizer' in state
            state['model'] = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k:v for k,v in state['model'].items()}
            resume = str(Path(tmp)/'parent.pth')
            torch.save(state, resume)
        runner = base.Runner(base.RLGPUAlgoObserver())
        runner.load(conf['train'])
        runner.reset()
        runner.run({'train':True, 'play':False, 'checkpoint':resume, 'sigma':None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('axis', choices=['Y','Z'])
    parser.add_argument('seed', type=int, choices=P['seeds'])
    parser.add_argument('stage', choices=['base','dwell'])
    args = parser.parse_args()
    train(args.axis, args.seed, args.stage)
