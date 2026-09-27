## Data and files to bring into the FYM repo

### Required (DAVIS test)
| Item | Source (ditflow machine) | Size | Notes |
|---|---|---|---|
| Cut frame folders | `E:\bench\packed\davis\<clip_id>\` | 284 MB | Each folder: 24 JPEGs + `meta.json` (the frame indices). Copy whole folders; do not re-cut or re-encode |
| Subject masks | `E:\DAVIS\Annotations\480p\<clip_id>\` | — | Only the 50 clip ids in `davis50.csv`. Needed for `mf_masked` and `lpips_bg` |
| Manifest | `benchmark/davis50.csv` | small | Use this one, not `davis.csv` (older, 80 rows, subject prompts only) |
| Subject-swap table | `benchmark/davis_subject_map.csv` | small | Record of how the Subject prompts were made |
| Scripts | `benchmark/collect.py`, `benchmark/analyse.py`, `benchmark/remap.py` | small | Standalone: no imports from the rest of ditflow |
| Saved anchor | `docs/scores_davis.csv` | small | Context only; never merged with FYM scores |

Python deps for scoring: `torch`, `numpy`, `pillow`, `imageio`, OpenAI `clip`, `lpips`.
`collect.py` also pulls CoTracker3 and DINO through `torch.hub`, so it needs network
access the first time. Pin CoTracker with `--cotracker-ref`.

### Optional (MiraData split: real camera motion, measured camera paths)
Bring this only if FYM results are meant to say anything about camera vs object motion.
| Item | Source | Size |
|---|---|---|
| Cut frame folders | `E:\bench\packed\miradata\<clip_id>\` | 191 MB |
| Camera trajectories | `E:\bench\packed\miradata_traj\<clip_id>.npy` | 120 KB |
| Manifest + swaps | `benchmark/miradata.csv`, `benchmark/subject_map.csv` | small |
| Camera analysis | `benchmark/camera_metrics.py`, `benchmark/q1_camera_realcam.py` | small |

MiraData has no subject masks: score it without `--annotations` (whole-frame metrics only),
and report it by `cam_band` (low/mid/high) in its own table. Never pool it with DAVIS.
`q1_camera_realcam.py` imports `benchmark.camera_metrics`, so keep the `benchmark/` folder name.

### Not needed
- Kubric (`kubric/`): calibrates the metrics; that work stays in ditflow.
- `grid*.yaml`, `motion_guidance*.py`, and the Wan probe/review scripts: DiTFlow-specific.

### Transfer
- Pack each split into one tar before moving it (e.g. to Drive). Thousands of loose files
  are slow and hit rate limits.
- After unpacking, remap the manifest paths and check they resolve:
      python benchmark/remap.py --manifest benchmark/davis50.csv --out davis50_local.csv \
          --from "E:\bench\packed" --to "<new packed root>" --verify
  Do the same for `miradata.csv`. `traj_path` uses the same `E:\bench\packed` prefix.
