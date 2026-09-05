# Pilot manifest v0

The first deterministic pilot manifest contains **100 IH-Challenge items**: 25 from each top-level file, selected with seed `20260905`.

## Why this is stratified

The full audit showed that uniform sampling would be dominated by system-to-user conflicts and by the input-conditioned `blue_team_auto` family. The v0 sampler therefore works hierarchically:

1. balance observed authority pairs within each source file;
2. within an authority pair, round-robin across task types before taking repeats;
3. use stable SHA-256-derived pseudorandom seeds so selection does not depend on Python hash randomization;
4. store only item identifiers and metadata in the manifest, not raw benchmark prompts.

## Resulting composition

Authority pairs across all 100 items:

- `system -> user`: **36**
- `system -> developer`: **32**
- `developer -> user`: **32**

Top-level source files are exactly balanced at 25 items each.

Task-family coverage includes:

- `blue_team_auto`: 25
- `composite`: 25
- 17 hand-constructed constraint families across the remaining 50 items

The exact composition summary is versioned in `reports/pilot_manifest_v0_summary.json`.

## Rebuild

```bash
python scripts/generate_pilot_manifest.py \
  --items-per-file 25 \
  --seed 20260905 \
  --output pilot-manifest-v0.jsonl
```

GitHub Actions also builds and uploads the complete JSONL manifest as `ih-decay-pilot-manifest-v0`.

## Next use

This manifest is a *selection layer*, not yet a model-evaluation result. The next stage materializes the selected benchmark items, constructs controlled trajectories at checkpoints 1/3/5/10/20, and evaluates a very small model pilot before scaling.
