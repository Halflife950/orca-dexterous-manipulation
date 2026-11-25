"""Apply a frozen actor to four predeclared banks without resetting on success."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_orca as replay
import numpy as np
from omegaconf import OmegaConf

torch = replay.torch
P = json.loads((HERE / 'protocol.json').read_text())


def score(error, dropped):
    alive = ~np.logical_or.accumulate(dropped)
    inside = alive & (error < P['threshold_rad'])
    entries = np.flatnonzero(inside)
    hold = P['hold_samples']
    entered = bool(len(entries))
    start = int(entries[0]) if entered else len(error)
    exits = np.flatnonzero(~inside[start:])
    length = int(exits[0]) if len(exits) else len(inside) - start
    late = entered and start + hold > len(inside)
    return dict(entered=int(entered), first_step=start + 1 if entered else '',
                first_streak=length, first_hold=int(length >= hold),
                any_hold=int(replay.longest(inside) >= hold), final_hold=int(inside[-hold:].all()),
                dropped=int(dropped.any()), late_entry=int(late),
                censored=int(late and inside[start:].all()),
                min_error_rad=float(error[alive].min()) if alive.any() else '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('bank', choices=['fixed', 'tilt'])
    p.add_argument('seed', type=int)
    p.add_argument('output', type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda:0'])
    args = p.parse_args()
    cfg = OmegaConf.load(ROOT / 'orca_reversal/model/config.yaml')
    cfg.num_envs = P['cases_per_bank']
    cfg.pipeline = 'cpu' if args.device == 'cpu' else 'gpu'
    cfg.sim_device = cfg.rl_device = args.device
    config = OmegaConf.to_container(cfg, resolve=True)
    torch.set_num_threads(1)
    replay.set_seed(args.seed, torch_deterministic=True)
    angles, quats = replay.initial_cases(cfg.num_envs, args.seed, args.bank, 5)
    env = replay.ReplayHand(config['task'], args.device, args.device, -1, True, False, False)
    env.initial_quats = torch.tensor(quats, device=env.device)
    player, epoch = replay.player_from_checkpoint(config['train']['params'], env, args.checkpoint, args.device)
    frames = int(env.max_episode_length)
    assert frames == 600
    obs = env.reset()
    torch.testing.assert_close(env.resetted_object_states[:, 3:7], env.initial_quats)
    initial = hashlib.sha256()
    for array in (env.initial_quats, env.object_pose, env.goal_states, env.hand_dof_pos, env.hand_dof_vel, env.cur_targets, obs['obs']):
        initial.update(array.detach().cpu().contiguous().numpy().tobytes())
    errors, drops = [], []
    active = torch.ones(cfg.num_envs, device=env.device, dtype=torch.bool)
    with torch.inference_mode():
        for step in range(1, frames + 1):
            errors.append(replay.angle_error(env.object_rot, env.goal_rot).cpu().numpy().copy())
            drops.append(env.dropped_buf.cpu().numpy().copy())
            active &= ~env.dropped_buf
            if step < frames:
                env.reset_buf[active] = False
                env.reset_goal_buf[active] = False
                action = player.get_action(obs['obs'], is_deterministic=True)
                obs, _, _, _ = env.step(action)
    errors, drops = np.stack(errors), np.stack(drops)
    assert np.isfinite(errors).all()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = f'{args.bank}_{args.seed}'
    np.savez_compressed(args.output / (prefix + '.npz'), initial_xyzw=quats, error_rad=errors, dropped=drops)
    rows = [dict(bank=args.bank, bank_seed=args.seed, slot=i, init_x_deg=ang,
                 **score(errors[:, i], drops[:, i])) for i, ang in enumerate(angles)]
    with (args.output / (prefix + '.csv')).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    meta = dict(initial_state_sha256=initial.hexdigest(), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                checkpoint=str(args.checkpoint.resolve()), epoch=epoch, horizon=frames, device=args.device)
    meta_path = args.output / (prefix + '.json')
    temp = meta_path.with_suffix('.tmp')
    temp.write_text(json.dumps(meta, indent=2))
    temp.replace(meta_path)


if __name__ == '__main__':
    main()
