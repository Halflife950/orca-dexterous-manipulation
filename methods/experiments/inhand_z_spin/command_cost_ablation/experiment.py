"""Single-coefficient ablation of the seven-channel ORCA PPO experiment."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "aero_transfer"
sys.path.insert(0, str(BASE))
COEFFICIENT = 0.1


def simulate(mode, seed, raw):
    import transfer

    baseline_config = transfer.config

    def config(seed, n, training=False):
        cfg = baseline_config(seed, n, training)
        cfg["task"]["command_change_weight"] = COEFFICIENT
        return cfg

    class ReducedCommandCost(transfer.AeroORCA):
        def compute_reward(self):
            change = (self.command - self.previous).square().sum(1)
            self.rew_buf.copy_(self.control_dt * (
                self.object_angvel[:, 2] - self.cfg["command_change_weight"] * change
                - 100 * self.dropped_buf))
            self.previous.copy_(self.command)

    transfer.HERE = HERE
    transfer.config = config
    transfer.AeroORCA = ReducedCommandCost
    if mode == "train":
        transfer.train(seed)
    else:
        transfer.capture(seed, raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["train", "test", "score"])
    parser.add_argument("seed", type=int)
    parser.add_argument("raw", type=Path)
    opt = parser.parse_args()
    if opt.mode == "score":
        import metrics
        metrics.HERE = HERE
        metrics.OUT = HERE / "results"
        metrics.evaluate("aero", opt.seed, opt.raw)
    else:
        simulate(opt.mode, opt.seed, opt.raw)
