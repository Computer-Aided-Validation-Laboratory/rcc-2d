# PixInt2D package refactor plan

## Purpose

`pixint2d` turns the bespoke Grid2D and Speck2D study renderers into an
installable, reusable Python package for one orthographic 2D finite-element
mesh. Experiment scripts remain orchestration and I/O adapters.

## Public objects

- `Camera2D`: rectangular camera resolution, ROI and background.
- `Mesh2D`: one 2D mesh, possibly containing several elements.
- `DisplacementSeries`: nodal x/y displacements over frames.
- `RectRule`, `GaussRule` and `AnalyticRule`: pixel integration rules.
- `RenderOptions`: mapping, parallel worker count, chunk cap and optional PSF.
- `Grid2D` and `Speck2D`: renderers with `.render()` and `.render_many()`.
- `RenderResult`: normalised image, raw pre-clamp values and valid mask.
- `quantise_image`: the shared [0, 1] to b-bit camera-code operation.

Raw additive Speck2D coverage is integrated before clamping and intensity
mapping. The returned `image` is normalised [0, 1]; camera digitisation is a
separate operation.

## Mapping modes

Version 1 exposes all current mappings behind `MappingMode`: exact global
affine, robust arbitrary single-mesh VTK, exact one-Quad9 Newton and dedicated
structured Quad9 mapping. Only one `Mesh2D` is passed to a renderer, though it
may contain multiple elements.

## Package layout

```text
src/pixint2d/
  core/        camera, mesh, displacement, quadrature, PSF, digitisation
  mapping/     affine, VTK, single Quad9 Newton and structured Quad9 paths
  grid2d/      Grid2D renderer and eggbox texture
  speck2d/     Speck2D renderer and additive speckle models
  examples/    small runnable study reproductions
  data/        distributed mesh fixtures
test/gold/     compact trusted float-render regression arrays
```

## Finite-star packaged fixture

The package ships a compact multi-element Quad9 finite-star fixture:

- plate/ROI: `260 x 65` / `256 x 64` world units;
- representative final camera: `1020 x 252` pixels;
- wavelength range: 64 to 128 final pixels;
- 12 mildly graded x-elements (`ratio=1.05`) and 16 uniform y-elements;
- 192 Quad9 elements and 825 nodes;
- two frames: undeformed and +/-0.5 final-pixel peak vertical displacement.

The minimum 64-pixel wavelength is about 16.25 world units. Four Quad9
elements per wavelength require 16 elements through the 65-unit height, or
eight quadratic nodal intervals per shortest wavelength.

## Migration order

1. Add package metadata, core data classes and fixtures.
2. Extract Grid2D and Speck2D without experiment imports or output paths.
3. Preserve mapping modes and PSF halo semantics.
4. Add deterministic 32x32 regression tests from verified Exp1/Exp2 outputs.
5. Convert Exp1, Exp2 and Exp3 wrappers while retaining output naming.
6. Compare analytic, Rect, Gauss, affine, Newton, VTK and PSF outputs before
   deleting script-local renderer code.

## Test strategy

Unit tests cover quadrature, digitisation, exact eggbox integration, disk and
Gaussian coverage, and map validity. Regression tests use committed 32x32
goldens for rigid frame 0/3 Grid2D eggbox and disk/Gaussian Speck2D paths.
Serial and multi-worker results must agree. Tests never rely on `out/`.

## Examples

The distributed examples cover rigid and global-affine 32x32 Grid2D/Speck2D
renders as well as Grid2D/Speck2D renders of the compact finite-star mesh.
The finite-star examples explicitly select `MappingMode.STRUCTURED_QUAD9`, the
package's own structured Quad9 Newton inverse, rather than VTK.

## Installation and use

```bash
uv pip install -e ".[dev]"
pytest -q
python -m pixint2d.examples.grid2d_eggbox_rigid
```
