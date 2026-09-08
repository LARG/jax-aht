# Changelog

All notable changes to JaxAHT are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are grouped under `Added`, `Changed`, `Fixed`, `Deprecated`, or `Removed`.
Changes land under `Unreleased` as they are merged; on release, that section is
renamed to the new version and a fresh `Unreleased` section is started.

## [Unreleased]

### Added
- Configs for `liam_ego` and `meliba_ego` on all five Overcooked-v1 layouts, and for
  `trajedi` on `lbf/lbf_12x12`. These algorithm/task combinations previously had no config.
- A validation teammate set (`evaluation/configs/global_validation_settings.yaml`), drawn from
  the public [jaxaht/val-teammates](https://huggingface.co/datasets/jaxaht/val-teammates)
  dataset, for model selection and hyperparameter tuning. Keeping tuning off the heldout set
  avoids overfitting the set used to report final results. `download_eval_data.py` fetches it
  into `val_teammates/`.
- `download_eval_data.py` takes `--force` to re-download data that is already present locally.

### Changed
- Applied tuned hyperparameters from the benchmark sweeps to every algorithm and task.
- `download_eval_data.py` skips files already present locally instead of re-downloading them,
  reports a summary of what succeeded and failed, and exits non-zero if any download fails.
- Bootstrap confidence intervals in heldout evaluation are computed in parallel across tasks.

### Fixed
- Hydra run directories are now timestamped to microsecond precision, so concurrently launched
  runs no longer share an output directory and overwrite each other's results.

## [1.0.0] - 2025-09-15

Initial public release.

[Unreleased]: https://github.com/LARG/jax-aht/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LARG/jax-aht/releases/tag/v1.0.0
