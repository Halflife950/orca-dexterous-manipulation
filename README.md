# Reinforcement Learning for Cube Reorientation and Continuous Rotation with a Simulated Dexterous Hand

This repository follows a cube manipulation study from physical ORCA Hand assembly and motion calibration through failure diagnosis and controlled PPO retraining. Three released checkpoints address different questions.

| Policy | Task and held-out test | Recorded result |
|---|---|---|
| X, seed 615101 | +45° world X, entire final 5 s in tolerance, 192 fresh starts | 187/192, compared with 39/192 for the matched reset continuation |
| Y, seed 976043 | +45° world Y, terminal 16 supported samples, 192 starts | 110/192 for this checkpoint. Across all three seeds forearm roll improved terminal holds from 0 to 252/576 |
| Z, seed 990037 | Continuous in-hand rotation, 25 s, 192 fresh starts | 133/192 joint successes for this checkpoint in the archived evaluation. Across seeds the displacement cost raised joint success from 58 to 208/576 |

The Z joint endpoint requires survival to 25 s, a supported crossing of +45° and at least two command-linked finger exchanges. A target-angle stopping task is proposed in the report.

## Reading the release

- [Workshop abstract](paper/abstract.txt)
- [English technical report (PDF)](paper/technical_report.pdf) and [editable Word version](paper/technical_report.docx)
- [Detailed technical summary in Chinese](paper/technical_summary_zh.md)
- [Reversal failure analysis](figures/reversal_failure.png) and [fresh Z evaluation](figures/z_outcomes.png)
- [Evaluation tables](evidence/), [checkpoint files](policies/) and [simulator methods](methods/)

## Reproduce the numbers

From the repository root, run the following. The first command needs only standard Python 3. The second also needs NumPy and Matplotlib. The verifier checks paired episode identifiers, published endpoint counts and file hashes.

```sh
python3 scripts/verify.py
python3 scripts/plot_results.py
```

The three selected checkpoints, configuration, task code, hand geometry and simulator notices are included. The report and numerical verification are independent of the former training workspace. GPU simulation additionally requires the NVIDIA Isaac Gym Preview 4 SDK and its licensed native libraries, IsaacGymEnvs, an NVIDIA driver, and a compatible Python environment. Tested simulator versions and SDK fingerprints are in [runtime.json](evidence/runtime.json). The simulation environment used Python 3.8, PyTorch 2.4.1, rl-games 1.6.5, NumPy 1.23.5 and SciPy 1.10.1. Support scoring also needs Eigen3 headers, FCL, libccd, a C++ compiler, and a Python interpreter with NumPy/SciPy. The native SDK, environment and training logs are external to this release.

Activate the simulator Python environment and ensure its library directory is on the dynamic linker path. If the SDK and IsaacGymEnvs are installed from source, add the SDK's `isaacgym/python` directory and the IsaacGymEnvs checkout to `PYTHONPATH`. Run from the repository root with a writable scratch directory for each axis.

```sh
scripts/replay.sh x /tmp/orca-eval-x
scripts/replay.sh y /tmp/orca-eval-y
scripts/replay.sh z /tmp/orca-eval-z
```

X evaluates three 64-start confirmation banks and saves error and drop traces. Use `ORCA_DEVICE=cpu` for CPU PhysX. Y evaluates 192 shared starts, replays for contact evidence and scores all terminal holds. Z evaluates 192 fresh starts at 20 Hz, writes the raw trajectory and scores both 10 s and 25 s. Y/Z use GPU PhysX. Set `ORCA_SCORE_PYTHON` to choose the offline scoring interpreter (default `/usr/bin/python3`). Scratch outputs can be much larger than the release. [Replay check](evidence/replay_check.md)

## Training and sources

[Training instructions](methods/README.md) describe how to initialize each task. The release distributes one selected checkpoint per axis. Per-seed evidence for the other policies is included in the tables. The simulator implementation and geometry retain their source notices under [simulator](methods/orca_reversal/simulator/). External technical references include [ORCA Hand](https://www.orcahand.com/), [PPO](https://arxiv.org/abs/1707.06347), [Aero Hand Sim and RL](https://docs.tetheria.ai/docs/hand_sim/) and [In-Hand Object Rotation via Rapid Motor Adaptation](https://proceedings.mlr.press/v205/qi23a.html).
