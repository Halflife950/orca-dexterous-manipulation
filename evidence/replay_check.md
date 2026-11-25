# Frozen-checkpoint replay check

The packaged entry points were exercised on the original machine in a fresh directory. Frozen actors used deterministic actions and the archived start banks. The reported pooled comparisons use the original matched runs.

| Check on the selected checkpoint | Archived | Packaging replay 1 | Packaging replay 2 |
|---|---:|---:|---:|
| X, final 1 s / final 5 s, GPU | 189 / 187 | — | 189 / 187 |
| Y, any supported 16 / final supported 16 | 142 / 110 | — | 142 / 110 |
| Z, retained at 25 s / supported 45° | 166 / 164 | 166 / 164 | 166 / 164 |
| Z, repeated exchanges / joint outcome, 25 s | 152 / 133 | 154 / 134 | 151 / 133 |

Each row uses 192 starts. Y live orientations matched the reference trajectory exactly. For Z, saved commands, target and measured angles, rigid poses and native live masks matched element by element. Some net-body-force readings differed and the threshold-based contact classification changed. In the second replay six trials changed joint classification at 25 s, although the total matched. The cause of the force-readout variation has not been isolated. The [paired replay table](z_replay_check.csv) gives all three versions for each slot at both horizons. The [archived series](z_archived_contact_series.npz) contains angles, support bits, shell flags, live masks, initial orientations and sample period. The [archived exchange events](z_archived_exchanges.csv) list the digit identities, times and command-link test. Use these to distinguish variability in scoring from variability in motion.
