# Unmodified Hanson baseline evidence

One user-authorized comparison after the merged candidate failed pickup. RSS d50f86f and Newton 3d3f9e4c were checked out verbatim; tracked diffs were empty before and after the run. All physical defaults are unchanged. Only the output directory and per-frame video recording were selected.

The native result and video cover all 600 frames / 20 seconds. `comparison.json` separately assesses the shared first 10 seconds and the full 20 seconds using existing recorded diagnostics. `diagnostics.json.gz` is a lossless copy of those diagnostics. The unmodified example does not save states or measured peak jaw effort.

Launchers and postprocessor are archived verbatim. Their paths refer to the original paired `hanson-baseline-20261103` worktree and its `outputs/hanson_baseline_20261103` directory, not this archive. No script modifies the simulation implementation.

See [comparison report](../BUG_FIX_hanson_baseline_20261103.md).
