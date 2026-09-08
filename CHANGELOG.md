# Changelog

All notable changes to JaxAHT are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changes land under `Unreleased` as they are merged; on release, that section is
renamed to the new version and a fresh `Unreleased` section is started.

## [Unreleased]

### Added
- CI on pull requests: package build, CPU tests, and algorithm smoke tests.
- A validation teammate set (`evaluation/configs/global_validation_settings.yaml`) for tuning
  and model selection, so the heldout set is only used for final results. Downloaded from
  [jaxaht/val-teammates](https://huggingface.co/datasets/jaxaht/val-teammates).
- Configs for `liam_ego` and `meliba_ego` on all Overcooked-v1 layouts, and `trajedi` on
  `lbf/lbf_12x12`.
- `download_eval_data.py --force` to re-download data already present locally.

### Changed
- Tuned hyperparameters from the benchmark sweeps, for every algorithm and task.
- `download_eval_data.py` skips files already present locally, reports which downloads failed,
  and exits non-zero if any did.

### Performance
- MeLIBA subsamples decoder ELBO start indices, stratified to keep estimator variance low
  (~4x faster).
- TrajeDi takes a single gradient of the summed masked loss instead of a per-agent `vmap`
  (~1.5x faster).
- Heldout evaluation reuses XLA compilations across partners, and no longer OOMs on the 2D
  sweep (it vmaps over seeds and loops over iterations).
- Bootstrap confidence intervals in heldout evaluation are computed in parallel across tasks.

### Fixed
- Recurrent agent updates reset hidden states one step early, because replay passed the
  post-step `done` as the reset signal while rollouts use the pre-step `done`. Transitions now
  store `prev_done` for replay across all trainers.
- MeLIBA: KL is averaged over the batch (making `DECODER_KL_WEIGHT` minibatch-size invariant),
  ELBO sampling and encoder noise vary per seed and gradient step, and the decoder action head
  is sized from the partner action space.
- TrajeDi: corrected the second self-play entropy term, the probability-space JSD multiplier,
  and the `update_steps` increment.
- COLE: population buffer scores are stored as log-probabilities so softmax sampling recovers
  the metasolver distribution; Shapley coalition sampling, weights, and cross-play
  normalization are restricted to trained population slots; and the update budget counts
  training steps over the N-1 trained agents.
- Bootstrap confidence intervals stacked eval episodes onto rliable's runs axis alongside
  training seeds, inflating the sample size and shrinking the intervals. Episodes are now
  averaged out first, leaving seeds as the replication unit.
- Hydra run directories are timestamped to microseconds, so concurrent runs no longer share an
  output directory.

## [1.0.0] - 2025-09-15

Initial public release.

[Unreleased]: https://github.com/LARG/jax-aht/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LARG/jax-aht/releases/tag/v1.0.0
