"""Replay one batch of first episodes from the trusted local ORCA PPO checkpoint.

Run once per condition, in separate processes with identical environment slots.
traces.npz uses [physics/control step, case, ...], with NaN after the trial ends.
Step 1 is the zero-action physics step that RobotHand.reset() uses to obtain obs.
"""
import argparse
from collections import Counter
import csv
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys

import isaacgym  # Must precede torch.
import isaacgymenvs  # Registers the OmegaConf resolvers used by the saved config.
import numpy as np
import torch
from isaacgym import gymtorch
from faive_gym import robot_hand
from isaacgymenvs.utils.utils import set_seed
from omegaconf import OmegaConf
from rl_games.algos_torch.players import PpoPlayerContinuous


def angle_error(q, goal):
    """Shortest geodesic angle, invariant to quaternion sign."""
    q = torch.nn.functional.normalize(q, dim=-1)
    goal = torch.nn.functional.normalize(goal, dim=-1)
    a = torch.norm(q - goal, dim=-1)
    b = torch.norm(q + goal, dim=-1)
    return 4 * torch.atan2(torch.minimum(a, b), torch.maximum(a, b))


class ReplayHand(robot_hand.RobotHand):
    def reset_goal_states(self, env_ids):
        super().reset_goal_states(env_ids)
        goal_quats = getattr(self, 'trial_goal_quats', None)
        if goal_quats is not None:
            self.goal_states[env_ids, 3:7] = goal_quats[env_ids]
            self.resetted_visual_goal_states[:, 3:7] = goal_quats[env_ids]

    def reset_object_states(self, env_ids):
        super().reset_object_states(env_ids)
        self.resetted_object_states[:, 3:7] = self.initial_quats[env_ids]


class MiddlePipFloorHand(ReplayHand):
    """Apply one post-policy target constraint from the step after near entry."""

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        target = self.cur_targets[:, self.pip_index]
        self.pip_before_floor = target.clone()
        affected = self.floor_enabled & (target < 0)
        if affected.any():
            target[affected] = 0
            self.prev_targets[affected, self.pip_index] = 0
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.cur_targets))


class ThumbHoldHand(ReplayHand):
    """On first hit, latch the selected thumb reference for subsequent steps."""

    def prepare_thumb_hold(self, use_actual):
        ptr = self.envs[0]
        actor = self.gym.find_actor_handle(ptr, 'hand')
        dofs = self.gym.get_actor_dof_dict(ptr, actor)
        self.thumb_indices = [dofs['right_thumb_pip'], dofs['right_thumb_dip']]
        self.use_actual = use_actual
        self.holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.latched = torch.zeros((self.num_envs, 2), device=self.device)

    def latch_after_hit(self, inside, active):
        new = active & inside & ~self.holding
        reference = self.hand_dof_pos if self.use_actual else self.cur_targets
        self.latched[new] = reference[new][:, self.thumb_indices]
        self.holding = active & (self.holding | new)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.policy_targets = self.cur_targets[:, self.actuated_dof_indices].clone()
        if self.holding.any():
            for column, joint in enumerate(self.thumb_indices):
                self.cur_targets[self.holding, joint] = self.latched[self.holding, column]
                self.prev_targets[self.holding, joint] = self.latched[self.holding, column]
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.cur_targets))


def initial_cases(count, seed, condition, degrees):
    # Stratified uniform signed X rotations; shuffle their assignment to slots.
    rng = np.random.default_rng(seed)
    angles = degrees * (2 * (np.arange(count) + rng.random(count)) / count - 1)
    rng.shuffle(angles)
    if condition == 'fixed':
        angles.fill(0)
    half = np.deg2rad(angles) / 2
    quats = np.column_stack((np.sin(half), np.zeros((count, 2)), np.cos(half)))
    return angles, quats.astype(np.float32)


def set_goal_x(task, degrees):
    """Retain the exact reference quaternion for the original +45 degree task."""
    if degrees != 45:
        half = np.deg2rad(degrees) / 2
        task['env']['goal_start_quat'] = [float(np.sin(half)), 0., 0., float(np.cos(half))]


def player_from_checkpoint(params, env, path, device):
    params['config'].update(env_info=dict(action_space=env.action_space,
                                          observation_space=env.observation_space),
                            device_name=device)
    player = PpoPlayerContinuous(params)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    prefix = '_orig_mod.'
    weights = {k[len(prefix):] if k.startswith(prefix) else k: v
               for k, v in checkpoint['model'].items()}
    player.model.load_state_dict(weights, strict=True)
    if 'running_mean_std' in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
    player.model.eval()
    player.has_batch_dimension = True
    player.reset()
    return player, checkpoint['epoch']


def longest(mask):
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int(max(changes[1::2] - changes[::2], default=0))


def summarize(tr, angles, end, dt, threshold, near, hold_samples):
    rows = []
    for i, angle in enumerate(angles):
        n = int(end[i])
        err = tr['error_rad'][:n, i]
        drop = tr['drop'][:n, i].astype(bool)
        inside = (err < threshold) & ~drop
        hits = np.flatnonzero(inside)
        first = int(hits[0]) if hits.size else None
        held = longest(inside) >= hold_samples
        outcome = 'drop' if drop[-1] else ('held' if held else 'timeout')
        legacy = 'hit' if first is not None else outcome
        band = (err >= threshold) & (err <= near)
        # Count adjacent sample intervals, so a single passing sample has zero dwell.
        dwell = max(0, longest(band) - 1) * dt
        outside = np.flatnonzero(~band)
        tail = max(0, n - (outside[-1] + 1 if outside.size else 0) - 1) * dt
        kind = ''
        if outcome == 'timeout':
            if first is not None:
                kind = 'hit_without_hold'
            elif not band.any():
                kind = 'never_near'
            elif tail >= 1:
                kind = 'near_at_end_1s'
            elif dwell >= 1:
                kind = 'near_1s_then_left_or_returned'
            else:
                kind = 'brief_visits_only'
        row = dict(case=i, initial_x_deg=angle, original_outcome=legacy,
                   hold_outcome=outcome, timeout_class=kind, steps=n,
                   first_hit_s='' if first is None else (first + 1) * dt,
                   minimum_error_rad=err.min(), final_error_rad=err[-1],
                   near_samples_fraction=band.mean(), longest_near_s=dwell,
                   final_near_s=tail)
        # Original-task return stops at first hit, even in an extended hold trial.
        stop = n if first is None else first + 1
        row['original_return'] = tr['reward'][:stop, i].sum(dtype=np.float64)
        for name in tr:
            if name.startswith('weighted_'):
                row[name] = tr[name][:stop, i].sum(dtype=np.float64)
        rows.append(row)
    return rows


def capture(env, active, fields):
    def save(name, tensor):
        value = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        if not np.isfinite(value[active]).all():
            raise ValueError('Nonfinite active data in ' + name)
        value[~active] = np.nan
        fields.setdefault(name, []).append(value)

    err = angle_error(env.object_rot, env.goal_rot)
    distance = torch.norm(env.object_pos - env.object_init_states[:, :3], dim=-1)
    dropped = distance > env.cfg['rewards']['fall_dist_threshold']
    inside = err < env.cfg['rewards']['success_tolerance']
    if not torch.equal(dropped, env.dropped_buf):
        raise ValueError('Drop classification differs from task')
    save('error_rad', err)
    save('drop', dropped)
    save('distance_m', distance)
    save('object_xyz', env.object_pos)
    save('object_xyzw', env.object_rot)
    save('object_angvel_rad_s', env.object_angvel)
    save('goal_xyzw', env.goal_rot)
    save('action', env.actions)
    indices = env.actuated_dof_indices
    save('joint_rad', env.hand_dof_pos[:, indices])
    save('joint_speed_rad_s', env.hand_dof_vel[:, indices])
    save('target_rad', env.cur_targets[:, indices])
    if isinstance(env, MiddlePipFloorHand):
        save('floor_enabled', env.floor_enabled)
        save('pip_before_floor_rad', env.pip_before_floor)
    if isinstance(env, ThumbHoldHand):
        save('thumb_hold_enabled', env.holding)
        save('policy_target_rad', env.policy_targets)
    save('reward', env.rew_buf)
    total = torch.zeros_like(env.rew_buf)
    for name, fn in zip(env.reward_names, env.reward_functions):
        weight = env.reward_scales[name]
        if weight:
            raw = fn()
            save('raw_' + name, raw)
            save('weighted_' + name, weight * raw)
            total += weight * raw
    torch.testing.assert_close(total, env.rew_buf)
    return inside.cpu().numpy(), dropped.cpu().numpy()


def main(observer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['fixed', 'tilt'], required=True)
    parser.add_argument('--out', type=Path, required=True, help='New result directory')
    run = Path.home() / 'orcahand/src/faive_gym/faive_gym/runs/OrcaHandXPlus45Direct'
    parser.add_argument('--checkpoint', type=Path, default=run/'nn/OrcaHandXPlus45Direct.pth')
    parser.add_argument('--config', type=Path, help='Defaults to the run config beside nn/')
    parser.add_argument('--cases', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--degrees', type=float, default=5)
    goals = parser.add_mutually_exclusive_group()
    goals.add_argument('--goal-x-deg', type=float, default=45.)
    goals.add_argument('--goal-x-range', type=float, nargs=2, metavar=('LOW','HIGH'),
                       help='Use a stratified continuous bank of trial goals about world X')
    parser.add_argument('--cross-axis-degrees', type=float, default=0,
                        help='For tilt tests, add independent uniform Y and Z initial rotations in degrees')
    parser.add_argument('--near-rad', type=float, default=0.20)
    parser.add_argument('--hold-s', type=float, default=0.25)
    parser.add_argument('--continue-after-hold', action='store_true',
                        help='Keep sampling after the hold criterion until drop or the episode horizon')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument('--grasp-policy', type=Path,
                              help='Use fitted grasp offsets and wrist feedback following the PPO approach')
    intervention.add_argument('--middle-pip-floor', action='store_true',
                              help='After first near entry, floor only right_middle_pip target at zero')
    intervention.add_argument('--thumb-hold', action='store_true',
                              help='After first hit, hold thumb PIP and DIP at their crossing targets')
    intervention.add_argument('--thumb-hold-actual', action='store_true',
                              help='After first hit, hold thumb PIP and DIP targets at the measured crossing angles')
    args = parser.parse_args()
    if args.cases < 1 or not 0 < args.degrees < 180 or args.hold_s < 0:
        parser.error('Check the number of cases, perturbation and hold duration')
    config_path = args.config or args.checkpoint.resolve().parents[1]/'config.yaml'
    cfg = OmegaConf.load(config_path)
    cfg.num_envs = args.cases
    cfg.pipeline = 'cpu' if args.device == 'cpu' else 'gpu'
    cfg.rl_device = cfg.sim_device = args.device
    set_goal_x(cfg.task, args.goal_x_deg)
    if observer is not None:
        observer.configure(cfg.task)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    task = resolved['task']
    threshold = task['rewards']['success_tolerance']
    if args.near_rad <= threshold:
        parser.error('Near band must extend beyond the success threshold')
    if (any(task['reset_noise'].values()) or task['task']['randomize']
            or task['env']['object_start_quat'] != [0, 0, 0, 1]
            or task['env']['object_fix_base']):
        raise ValueError('This protocol requires the fixed identity-start free-object baseline')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('Output directory must be empty')
    torch.set_num_threads(1)
    set_seed(args.seed, torch_deterministic=True)
    angles, quats = initial_cases(args.cases, args.seed, args.condition, args.degrees)
    if args.cross_axis_degrees:
        if args.condition != 'tilt' or not 0 < args.cross_axis_degrees < 180:
            parser.error('Cross-axis perturbations require tilt and a range in (0,180) degrees')
        from scipy.spatial.transform import Rotation
        cross_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
        yz = cross_rng.uniform(-args.cross_axis_degrees, args.cross_axis_degrees, (args.cases, 2))
        quats = Rotation.from_euler('xyz', np.column_stack((angles, yz)), degrees=True).as_quat().astype(np.float32)
    hand = MiddlePipFloorHand if args.middle_pip_floor else ReplayHand
    if args.thumb_hold or args.thumb_hold_actual:
        hand = ThumbHoldHand
    env = hand(task, args.device, args.device, -1, True, False, False)
    if isinstance(env, ThumbHoldHand):
        env.prepare_thumb_hold(use_actual=args.thumb_hold_actual)
    if args.middle_pip_floor:
        ptr = env.envs[0]
        actor = env.gym.find_actor_handle(ptr, 'hand')
        env.pip_index = env.gym.get_actor_dof_dict(ptr, actor)['right_middle_pip']
        env.floor_enabled = torch.zeros(args.cases, dtype=torch.bool, device=env.device)
    env.initial_quats = torch.tensor(quats, device=env.device)
    goal_degrees = args.goal_x_deg
    if args.goal_x_range:
        low, high = args.goal_x_range
        if not -180 < low < high < 180:
            parser.error('Expected an ascending goal range inside (-180,180) degrees')
        goal_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 2]))
        goal_degrees = low + (high-low)*(np.arange(args.cases)+goal_rng.random(args.cases))/args.cases
        goal_rng.shuffle(goal_degrees)
        half = np.deg2rad(goal_degrees)/2
        trial_quats = np.column_stack([np.sin(half), np.zeros((args.cases,2)), np.cos(half)])
        env.trial_goal_quats = torch.tensor(trial_quats, dtype=torch.float32, device=env.device)
    player, epoch = player_from_checkpoint(resolved['train']['params'].copy(),
                                           env, args.checkpoint, args.device)
    grasp = None
    if args.grasp_policy:
        from grasp_policy import GraspPolicy, goal_offsets
        params = json.loads(args.grasp_policy.read_text())
        for key, path in [('checkpoint_sha256', args.checkpoint), ('training_config_sha256', config_path)]:
            if key in params and params[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'Policy and {key} do not match')
        offsets = goal_offsets(params, goal_degrees)
        if 'finger_feedback' in params:
            from pose_feedback import feedback_policy
            grasp = feedback_policy(player, params, goal_degrees, offsets)
        else:
            grasp = GraspPolicy(player, offsets, params['switch_step'], params['wrist_gain'])
    norm = player.model.running_mean_std
    norm_before = {k: v.clone() for k, v in norm.state_dict().items()}
    dt = env.control_dt
    hold_samples = 1 + math.ceil(args.hold_s / dt)
    horizon = int(env.max_episode_length)
    if hold_samples > horizon:
        raise ValueError('Requested hold exceeds the episode horizon')
    active = np.ones(args.cases, dtype=bool)
    end = np.zeros(args.cases, dtype=int)
    streak = np.zeros_like(end)
    fields = {}
    if observer is not None:
        observer.attach(env)
    obs = env.reset()
    torch.testing.assert_close(env.resetted_object_states[:, 3:7], env.initial_quats)
    for step in range(1, horizon + 1):
        np.testing.assert_array_equal(env.progress_buf.cpu().numpy()[active], step)
        inside, drop = capture(env, active, fields)
        streak = np.where(inside & ~drop, streak + 1, 0)
        reached_hold = (streak >= hold_samples) & (not args.continue_after_hold)
        finished = active & (drop | reached_hold | (step == horizon))
        end[finished] = step
        active &= ~finished
        if not active.any():
            break
        # The task resets both the hand and the goal after a hit. Defer just
        # these resets for ongoing trials, to measure uninterrupted holding.
        keep = torch.as_tensor(active, device=env.device)
        if isinstance(env, ThumbHoldHand):
            env.latch_after_hit(torch.as_tensor(inside, device=env.device), keep)
        if args.middle_pip_floor:
            err = angle_error(env.object_rot, env.goal_rot)
            env.floor_enabled |= (err >= threshold) & (err <= args.near_rad)
            env.floor_enabled &= keep
        env.reset_buf[keep] = False
        env.reset_goal_buf[keep] = False
        actions = (grasp.action(obs['obs'], env, step) if grasp is not None else
                   player.get_action(obs['obs'], is_deterministic=True))
        obs, _, _, _ = env.step(actions)
    for name, value in norm.state_dict().items():
        assert torch.equal(value, norm_before[name]), 'Observation statistics changed'
    traces = {name: np.stack(values) for name, values in fields.items()}
    np.savez_compressed(args.out/'traces.npz', initial_xyzw=quats, **traces)
    rows = summarize(traces, angles, end, dt, threshold, args.near_rad, hold_samples)
    for row, goal_deg in zip(rows, np.broadcast_to(goal_degrees, (args.cases,))):
        row['goal_x_deg'] = float(goal_deg)
    with (args.out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: dict(Counter(row[key] for row in rows if row[key]))
               for key in ['original_outcome', 'hold_outcome', 'timeout_class']}
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    info = dict(args=vars(args), epoch=int(epoch), checkpoint_sha256=sha(args.checkpoint),
                training_config_sha256=sha(config_path), config_path=config_path,
                task_source_sha256=sha(robot_hand.__file__), script_sha256=sha(__file__),
                python=sys.version, os=platform.platform(), torch=torch.__version__,
                rl_games=metadata.version('rl-games'), isaacgym=metadata.version('isaacgym'),
                device=env.device, control_dt=dt, horizon=horizon, hold_samples=hold_samples,
                joint_indices=env.actuated_dof_indices.tolist(), outcomes=summary)
    if args.grasp_policy:
        info['grasp_policy_sha256'] = sha(args.grasp_policy)
        info['grasp_controller_sha256'] = sha(Path(__file__).with_name('grasp_policy.py'))
        if 'finger_feedback' in params:
            info['finger_feedback_sha256'] = sha(Path(__file__).with_name('pose_feedback.py'))
    (args.out/'manifest.json').write_text(json.dumps(info, indent=2, default=str))
    (args.out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    if observer is not None:
        observer.finish(args.out, dict(initial_xyzw=quats, **traces))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
