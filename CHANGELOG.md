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
- Bootstrap confidence intervals in heldout evaluation are computed in parallel across tasks.

### Fixed
- Hydra run directories are timestamped to microseconds, so concurrent runs no longer share an
  output directory.

## [1.0.0] - 2025-09-15

Initial public release.

[Unreleased]: https://github.com/LARG/jax-aht/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LARG/jax-aht/releases/tag/v1.0.0
