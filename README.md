# rcc-2d / PixInt2D

This repository contains the renderer-convergence study scripts and
**PixInt2D**, an installable Python package for orthographic 2D rendering of
analytic grids and additive speckle textures over one finite-element mesh.
PixInt2D supports exact analytic pixel integrals where valid, controlled Rect
and Gauss quadrature, robust VTK mapping, Quad9 Newton mapping and a Gaussian
camera PSF.

## Install

```bash
uv pip install -e ".[dev]"
```

PixInt2D deliberately depends on PyVista/VTK: robust arbitrary single-mesh FE
mapping is part of the supported public interface.

## Test

```bash
pytest -q
```

The tests use compact committed 32x32 gold renders and do not depend on the
large generated `out/` directory.

## Examples

```bash
python -m pixint2d.examples.grid2d_eggbox_rigid
python -m pixint2d.examples.speck2d_disk_rigid
python -m pixint2d.examples.speck2d_gauss_rigid
python -m pixint2d.examples.grid2d_affine
python -m pixint2d.examples.speck2d_affine
python -m pixint2d.examples.grid2d_finite_star
python -m pixint2d.examples.speck2d_finite_star
```

Each example has inputs as constants at the top of its module and writes a
float `.npy` image plus an 8-bit TIFF preview beneath `out/pixint2d_examples/`.
The finite-star examples use PixInt2D's structured Quad9 Newton inverse map,
not the VTK mapping path.

## Image convention

For additive speckles, PixInt2D integrates raw overlap coverage first, then
clamps/maps it to normalised intensity. Digitisation is explicitly separate:

```python
from pixint2d import quantise_image

codes_u12 = quantise_image(result.image, bits=12)
```

This preserves one source of truth for all camera bit depths.
