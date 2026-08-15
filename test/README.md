# PixInt2D tests and golden renders

`gold/` contains compact 32x32 float arrays used by pytest. They are not
generated during a test run.

- `grid2d_eggbox/` was copied from verified Exp1 custom-render outputs:
  rigid analytic frames 0/3, affine analytic frame 3, and quadratic-saddle
  Gauss-2 frame 3.
- `speck2d_gauss/` was copied from verified Exp2 continuous-Gaussian analytic
  rigid frames 0/3.
- `speck2d_disk/` fixes the deterministic package disk Rect-4 convention for
  a rigid frame. Exact disk-overlap primitives are additionally exercised by
  the renderer path; this golden is intentionally independent of stale
  experiment output directories.

When intentionally changing a physical convention, regenerate a candidate
golden with an explicit script, compare it against the relevant study render,
and review the numerical difference before replacing the committed file.
