# FP1-FP2 Alignment Status

This note closes the current FP1-FP2 reproducibility pass in `whpc-kit`.

## Scope

1. FP1 is aligned to the frozen M1 MM-WHPC protocol.
2. FP2 is aligned to the frozen M2 one-class/adaptive protocol.
3. FP2 keeps two different references explicit:
   1. the current research freeze reconstructed from `W-HPC/results/part3_final_thresholds/final_thresholds.json`;
   2. the published M2 table values stored in `reproducibility/reference/m2_published_reference.json`.

## FP1

Frozen FP1 runs are stored in:

1. `reproducibility/artifacts/fp1_full_summary_unsw_frozen.json`
2. `reproducibility/artifacts/fp1_full_summary_nsl_frozen.json`

Observed test results:

1. UNSW-NB15: `W-HPC uniform` F1 macro `0.7484`, `MM-WHPC uniform M=5` F1 macro `0.8128`.
2. NSL-KDD: `W-HPC uniform` F1 macro `0.7411`, `MM-WHPC uniform M=4` F1 macro `0.7602`.

These runs are treated as aligned to the frozen M1 protocol.

## FP2

Aligned FP2 runs are stored in:

1. `reproducibility/artifacts/fp2_full_summary_unsw_m2_aligned.json`
2. `reproducibility/artifacts/fp2_full_summary_nsl_m2_aligned.json`
3. `reproducibility/artifacts/fp2_full_summary_cic_frozen.json`

The JSON summaries for UNSW and NSL include `reference_alignment`, which compares the current rerun against the published M2 table values.

## Comparison Summary

1. UNSW-NB15 reproduces the published M2 values up to rounding noise in Part 3 and Part 4.
2. NSL-KDD reproduces Part 3 and the adaptive Part 4 values up to rounding noise; the fixed frozen Part 4 row shows a small drift of about `-0.0024` F1 and `-0.0015` balanced accuracy.
3. CIC-IDS2017 does not match the published M2 freeze, even though the same drift also appears when the current `W-HPC` checkout reruns the original code path.

Current CIC deltas against the published M2 table values:

1. Part 3 baseline: unseen F1 `+0.0039`.
2. Part 3 frozen: unseen F1 `-0.0101`.
3. Part 4 baseline: unseen F1 `+0.0030`, balanced accuracy `+0.0023`.
4. Part 4 frozen: unseen F1 `+0.0019`, balanced accuracy `-0.1379`.
5. Part 4 adaptive: unseen F1 `+0.1033`, balanced accuracy `-0.0717`.

## Interpretation

1. `whpc-kit` is not introducing the CIC discrepancy by itself.
2. For UNSW and NSL, the kit can be treated as manuscript-aligned.
3. For CIC, the kit should expose both:
   1. the published M2 reference values;
   2. the current recomputed values from the present research checkout.

That separation avoids silently conflating a paper freeze with a current rerun.
