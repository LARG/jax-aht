# Site data contract

Everything the viewer (`index.html`) reads is a static file in this directory.
It fetches, relative to itself:

| Path | Required | Produced by |
|---|---|---|
| `manifest.json` | yes | `../generate_videos.py` |
| `videos/<task>/<key>.mp4` | yes | `../generate_videos.py` |
| `conventions_<task>.json` | no | `../export_conventions_json.py` |
| `convention_labels_<task>.json` | no | `../label_server.py` or the page's "Export labels" button |

`<task>` is a task slug from `../tasks.py`: `lbf_7x7`, `cramped_room`,
`asymm_advantages`, `counter_circuit`, `coord_ring`, `forced_coord`. The page
picks the task from `?task=<slug>`, defaulting to the first task in the
manifest. `videos/` is gitignored; the mp4s are regenerated, not committed.

## `manifest.json` — the teammate list (drives everything)

One block per task; each teammate needs a `key` that is stable across all files
(`sanitize()` in `../tasks.py`: the heldout label with non-alphanumerics folded
to `_`, e.g. `ippo_mlp (1)` -> `ippo_mlp_1`).

```json
{
 "tasks": {
  "counter_circuit": {
   "label": "Overcooked: Counter Circuit",
   "env": "overcooked-v1",
   "teammates": [
    {
     "key": "ippo_mlp_cc_1",
     "name": "ippo_mlp_cc (1)",
     "partner": "BR",
     "num_episodes": 1,
     "mean_return": 183.0,
     "file": "videos/counter_circuit/ippo_mlp_cc_1.mp4"
    }
   ]
  }
 }
}
```

- `name` is the display label (the held-out config's agent label).
- `partner` is `"BR"` or `"self-play"`; the viewer says which one the video shows
  and warns when a teammate had no BR checkpoint.
- `file` is a path relative to this directory.
- Tasks with an empty `teammates` list are shown in the task nav but have
  nothing to play; a manifest with no tasks makes the page report that
  `generate_videos.py` has not been run.

## `conventions_<task>.json` — the optional convention map

Without this file the page falls back to a plain grid of teammate buttons, so
videos alone are enough to use the viewer. With it, the page draws the 2-D map:
points positioned at `x`/`y`, colored by `cluster`, medoids ringed and labeled.

```json
{
 "task": "counter_circuit",
 "points": [
  {
   "name": "ippo_mlp_cc (1)",
   "key": "ippo_mlp_cc_1",
   "x": 0.41,
   "y": -0.22,
   "cluster": 2,
   "is_medoid": false,
   "score": 183.0,
   "pairing": "BR"
  }
 ]
}
```

- `key` must match the manifest key, or the point will not be clickable — the
  page treats a point with no matching video as disabled (dimmed).
- `cluster` is any integer; the palette in `index.html` covers 1–8 and falls
  back to a neutral color beyond that.
- `score` and `pairing` are optional; they only appear in the tooltip.
- Extra points with no video are fine (drawn dimmed), and extra videos with no
  point simply do not appear on the map.

`../export_conventions_json.py` writes this file from a population-diversity
`features.csv` (`compute_population_diversity.py --full-heldout --br-paired`).
It computes the MDS coordinates itself and takes cluster ids either from a
`--clusters` CSV (any file with `agent` and `cluster` columns — e.g. a
clustering built on the code in `scripts/population_diversity/`) or, without
one, from its own average-linkage fallback at a fixed `k`. Nothing about the
site depends on which clustering produced the ids.

## `convention_labels_<task>.json` — hand-written descriptions

```json
{"task": "counter_circuit", "labels": {"ippo_mlp_cc_1": "hands onions across the counter"}}
```

Keyed by teammate `key`. Shipped labels appear in the description box and next
to ringed medoids on the map. Edits in the browser go to `localStorage` unless
the site is served by `../label_server.py`, which writes them into this file.
