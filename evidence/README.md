# Evidence index

| File group | Unit | Interpretation |
|---|---|---|
| reversal.json and reversal_trials.csv | Paired benchmark, then trial | Announced-sequence hybrid study and the harder final reversal direction |
| x_factorial.csv | Condition, training seed, backend and start bank | Initial target entry, first streak, any streak and drop after 2 × 2 PPO training |
| x_confirmation.csv | Backend, arm, bank and slot | One selected seed and its matched controls. The final 61 and 301 samples span 1 and 5 seconds |
| axis_specialists.csv | Specialist, seed, evaluation split | Fixed-axis goals and cross-axis transfer checks with broader initial orientations |
| forearm_trials.csv | Axis, mode, seed, bank and slot | `old` and `old_end` use orientation alone. `held` and `final16` add hand support and exclude housing within the window |
| forearm_mechanism.csv and forearm_outcomes.csv | Event-conditioned average or paired endpoint | Kinematic attribution and the original fixed/free comparison including drops |
| z_command_cost.csv and z_drop_cost.csv | Seed, cost and horizon | Separate reward ablations preceding the slip experiment |
| z_late_loss_pairs.csv | Matched loss, surviving control and lead time | Palm-relative displacement before late drops. A control may be reused |
| z_trials.csv | Cohort, arm, training seed, horizon, bank and slot | Main slip ablation. `joint` requires retained, supported45 and repeated in the same trial |
| z_gate_audit.csv | Cohort/arm and training seed | Gate activation, overlap with exchanges and warning before drop |
| z_archived_contact_series.npz and z_archived_exchanges.csv | Time series and detected events | Archived fresh-bank scoring evidence for the selected Z model |
| z_replay_check.csv | Bank, slot and horizon | Per-start comparison of the archived evaluation and two replays of the selected Z model |

Each Z arm has 192 unique starting poses per cohort. The same poses are evaluated by all three trained seeds. Cohorts labeled fresh in different *experiments* have different bank seeds. Counts of trials and counts of trained policies answer different questions.
