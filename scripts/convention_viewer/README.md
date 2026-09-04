# Evaluation Teammate Convention Viewer

A static site for browsing videos of this benchmark's **evaluation (held-out)
teammates** playing with their trained **best responses (BRs)**, one page per
task/layout, optionally laid out on a 2-D "convention map" derived from the
population-diversity behavioral features.

```
scripts/convention_viewer/
  tasks.py                    # task registry (slug <-> heldout/BR yaml key)
  generate_videos.py          # renders one mp4 per eval teammate + manifest.json
  export_conventions_json.py  # PD features.csv -> MDS map + clusters
  label_server.py             # serves the site, writes label edits to disk
  site/                       # the site itself (index.html, videos/, JSONs)
  site/README.md              # the data contract: every file the page reads
```

## Viewing

```bash
python scripts/convention_viewer/label_server.py --port 8000
```

then open <http://localhost:8000/?task=coord_ring>. Any static server works
(`cd scripts/convention_viewer/site && python -m http.server 8000`); the label
server just additionally persists convention descriptions into
`site/convention_labels_<task>.json` instead of browser localStorage.

Task slugs: `lbf_7x7`, `cramped_room`, `asymm_advantages`, `counter_circuit`,
`coord_ring`, `forced_coord`.

## Rendering the videos

Teammates come from `evaluation/configs/global_heldout_settings.yaml`
(`heldout_set.<task>`) and their partners from
`evaluation/configs/global_heldout_br.yaml` (`best_response_set.<task>`), matched
by name (`ippo_mlp (0)` -> `br_for_ippo_mlp_0`). A teammate whose BR checkpoint
is missing falls back to self-play, and is marked as such in the viewer.
Checkpoints must be downloaded first (`python download_eval_data.py`); orbax
deserialization needs a GPU.

```bash
conda activate bench311
PYTHONPATH=. python scripts/convention_viewer/generate_videos.py --task coord_ring
```

Renders every teammate for the task into `site/videos/<task>/<key>.mp4` (2
episodes for LBF, 1 for Overcooked) and updates `site/manifest.json`, which
drives the viewer. Useful flags: `--task all`, `--agents key1,key2`,
`--num-eps N`, `--overwrite`.

The viewer is usable with videos alone — it falls back to a teammate grid when a
task has no convention map.

## Adding the convention map

The map reuses the population-diversity features, so first run PD for the task
with the full held-out set paired against BRs:

```bash
PYTHONPATH=. python scripts/population_diversity/compute_population_diversity.py \
    --env overcooked --variant coord_ring --full-heldout --br-paired \
    --br-root eval_teammates --output-dir results/population_diversity
```

then convert its `features.csv` into the map:

```bash
python scripts/convention_viewer/export_conventions_json.py --task coord_ring \
    --features results/population_diversity/overcooked-coord_ring_full_brpaired/features.csv \
    --num-clusters 4
```

This writes `site/conventions_<task>.json` (MDS coordinates, cluster ids, medoid
flags, mean score per teammate), reusing the PD normalization: metadata and
near-constant columns dropped, z-scored features, cosine distances, classical
MDS.

The cluster ids are the one part not fixed here. By default the script falls
back to average-linkage agglomerative clustering at `--num-clusters`; pass
`--clusters <csv>` (any CSV with `agent` and `cluster` columns) to use a
clustering of your own, e.g. one built on the population-diversity code in
`scripts/population_diversity/`. The viewer does not care which produced the
ids -- see `site/README.md` for the exact shape of every file it reads.

## Convention labels

Clicking a teammate opens a "Convention description" box. With `label_server.py`
edits are written straight to `site/convention_labels_<task>.json`; otherwise
they live in localStorage and "Export labels" downloads a committable JSON.
Medoid labels are drawn next to their ringed point on the map.
