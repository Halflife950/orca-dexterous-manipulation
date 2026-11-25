# PPO training

Training and inference use the ORCA simulator sources in this directory. They require the external runtime described in the repository README. X starts from a fixed-goal proximity-reward parent and then continues training. Y starts a free-forearm world-Y policy from scratch. Z starts a continuous-rotation policy from scratch with a central critic. Seed lists and training horizons are in the corresponding task protocols. The base configuration is in `orca_reversal/model/config.yaml`.

Use a separate writable checkout for retraining. The workshop checkout includes symlinked evaluator checkpoints at the paths expected by the archived scripts. Running a trainer under the same name could overwrite these links. In the training checkout first copy the methods while excluding existing run outputs, as below. The working directory for these commands is the repository root. The shell variable `TRAIN` points to a newly created directory.

```sh
TRAIN=$(mktemp -d)
mkdir -p "$TRAIN/methods"
tar -C methods --exclude=runs --exclude=__pycache__ -cf - . | tar -C "$TRAIN/methods" -xf -
cd "$TRAIN/methods"
export PYTHONPATH="$PWD/orca_reversal/simulator${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
python experiments/ppo_factorial/train.py proximity 16 615101
python experiments/ppo_sustained_hold/train.py dwell 615101
python experiments/forearm_ppo_pair/study.py train Y free 976043
mkdir -p experiments/inhand_z_spin/outward_slip_ablation/results
python experiments/inhand_z_spin/outward_slip_ablation/experiment.py train 990037 "$TRAIN/scratch"
```

The X continuation locates the regenerated epoch-5000 parent from its seed. Y can be trained in fixed mode by changing `free` to `fixed`. Z's gated cost is implemented in `outward_slip_ablation/experiment.py`. The native SDK and dependencies must be installed beforehand. The release was checked by frozen-policy replay; reproducing training on different hardware or drivers can change the learned weights.

For the matched Z baseline, create `experiments/inhand_z_spin/command_cost_ablation/results` inside the training checkout and run `python experiments/inhand_z_spin/command_cost_ablation/experiment.py train 990037 "$TRAIN/scratch"`. Repeat both Z conditions with seeds 990011 and 990023. Both arms use command cost 0.1 and drop cost 100. The slip condition adds the displacement gate.
