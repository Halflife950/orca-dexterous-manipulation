"""18-action variant of the native ORCA task with axial forearm rotation."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'simulator'))
sys.path.insert(0,str(ROOT/'experiments/axis_curriculum_resets'))
from common import base, parent, torch, goal

FLEXORS=['right_thumb_mcp','right_thumb_pip','right_thumb_dip']+[
    'right_'+digit+'_'+joint for digit in ('index','middle','ring','pinky') for joint in ('mcp','pip')]
class ForearmHand(parent.hold_training.ContinuingHand):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,shaping='proximity',hold=16,dwell=True,**kwargs)
        handle=self.gym.find_actor_handle(self.envs[0],'hand')
        self.joint_names=self.gym.get_actor_dof_dict(self.envs[0],handle)
        self.flexors=[self.joint_names[n] for n in FLEXORS]
        roll=self.joint_names['right_forearm_roll']
        for ptr in self.envs:
            actor=self.gym.find_actor_handle(ptr,'hand')
            props=self.gym.get_actor_dof_properties(ptr,actor)
            props['stiffness'][roll]=1500.0
            props['damping'][roll]=80.0
            props['effort'][roll]=2.0
            self.gym.set_actor_dof_properties(ptr,actor,props)
        assert len(self.joint_names)==self.num_actions==18
    def _reward_hand_posture(self):
        q=self.hand_dof_pos[:,self.flexors]
        return (torch.relu(-q-.05).square()+torch.relu(.08-q).square()).sum(-1)

def configuration(n,axis='Y',seed=975031,epochs=2):
    cfg=base.OmegaConf.load(base.CONFIG)
    cfg.num_envs=n
    cfg.seed=seed
    cfg.max_iterations=epochs
    cfg.experiment=f'forearm_{axis}_{seed}_ep{epochs}'
    cfg.task.env.numActions=18
    cfg.task.env.goal_start_quat=goal(axis,45)
    cfg.task.asset.model_file='forearm/forearm.xml'
    dims=cfg.task.observations.obs_dims
    for key in ('dof_position','dof_pos_target','dof_speed','dof_speed_numerical','dof_force','actions'):
        dims[key]=18
    dims.dof_pos_history=90
    conf=base.OmegaConf.to_container(cfg,resolve=True)
    conf['train']['params']['config']['train_dir']=str(HERE/'runs')
    return conf
