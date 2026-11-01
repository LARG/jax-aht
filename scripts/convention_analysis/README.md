# Convention clustering of Overcooked teammates

Clusters a population of Overcooked teammates (e.g. the eval teammate set of a layout)
by the convention they play, from behavioral statistics of BR-paired episodes.

## Reproduce

```bash
export PYTHONPATH=$PWD
# 1. rollouts -> features.csv + cell_support.csv per population (GPU; two lanes)
scripts/convention_analysis/extract_features.sh results/conv results/traj_cache <br_root> 0 1
# 2. derive, screen, cluster (CPU)
python scripts/convention_analysis/run_pipeline.py --pd-root results/conv --out results/conv/eval
# 3. optional figure per population
python scripts/convention_analysis/plot_convention_clusters.py \
    --features results/conv/pd_coord_ring/overcooked-coord_ring/features_derived.csv \
    --clusters results/conv/eval/coord_ring__split/clusters.csv --out coord_ring.png
```

## Pipeline

1. **Feature extraction** (`scripts/population_diversity/compute_population_diversity.py
   --full-heldout --br-paired --br-root <br_root> --batched --deterministic-reset`, driver
   `extract_features.sh`): 128 episodes per teammate with its BR, vmapped rollouts
   (~36 s per teammate), trajectory cache so feature recompute is CPU-only.
2. **Feature vocabulary** (`feature_groups.py`, `pd_events.py`, `pd_rollouts.py`): ~60
   layout-independent columns in groups role, counter_use, counter_identity, movement,
   partner_coupling, contention, throughput. Counter usage is expressed as
   activity-conditioned fractions (`derive()`), station choice as modal pot / onion-pile /
   plate-pile / goal coordinates plus a sharing rate, movement as region shares and
   circulation direction, coupling as MI between the agents' routes.
3. **Screening** (`screen_features.py`, using the `cell_support.csv` the extraction writes
   next to `features.csv`): per-task removal of structurally dead columns
   only; no single-task hacks. Four rules: exactly-constant columns, counter-cell
   columns without support, derived ratios whose numerator event averages fewer
   than `MIN_EVENT_SUPPORT` (0.01) occurrences per episode (e.g. `soup_via_counter` on
   forced_coord), and, when the two agents start in different connected components of
   the layout, the contention group plus the proximity-conditioned coupling columns
   (`PROXIMITY_CONDITIONED`). Raw counts consumed by `derive()` (`DERIVE_INPUTS`) are never
   clustered.
4. **Clustering** (`cluster_conventions.py`, driver `run_pipeline.py`): z-score ->
   `--groups split` (each column scaled by 1/sqrt(group size) so every group carries equal
   expected squared distance; a group that screening reduced to a single column on a
   task is dropped rather than given a full group's weight) -> cosine -> average
   linkage. `--exclude-groups throughput`.
   k = largest k within `--k-tolerance 0.05` of the best silhouette with
   `--min-cluster-size 2`; if every k in that band isolates a singleton, the band is
   re-taken over the cuts that satisfy the floor. `labels_by_k.csv` keeps every cut
   k=2..8. Medoids are real teammates and are the objects to watch.

## Findings

- **Delivery counts are the tracked agent's, not the team's** (r=0.10 with team score).
  The "throughput" axis of early versions was the onion-side vs plate-side role axis,
  written into the distance ~10 times through correlated counts. Deduplicating it and
  excluding throughput from the distance is what moved every task off k=2. Competence is
  checked only afterwards via eta^2 against team score.
- **Feature rollouts must start from the layout's canonical state.** With the env
  default `random_reset=True`, disconnected layouts (forced_coord, asymm_advantages)
  place the tracked agent on a random side each episode, so one feature vector averaged
  two roles. `--deterministic-reset` fixes this. Never compare features across that
  boundary.
- **Cell-identity and role features carry the convention structure**: which counter cell
  onions vs plates go through, which pot / pile is used, handoff vs dead-drop fractions,
  region shares, circulation direction (separates clockwise from counter-clockwise runners
  on coord_ring and loop carriers from shuttlers on counter_circuit).
- **What did not help**: ICC reliability weighting (no change to partitions), corridor
  contention features (too rare per episode), per-step length normalization (no-op, all
  episodes are 400 steps).
- **Frame of reference is the ceiling.** Against hand labels on coord_ring the best
  agreement is 9/10 pairs; the missed pair differs only in joint arrangement with their
  respective BRs. Fixed probe partners would remove this; BR partners were kept by choice.

## Open issues

- Positional-mode features (e.g. `dish_counter_x/y`) default to 0 when the item is never
  handled, which encodes "role absent" as a location.
- Still-frame vision reviews cannot judge temporal features (MI, contention, idle,
  circulation); use them to validate cell-identity / role features and cluster
  distinctness only.
