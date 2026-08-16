"""Render finite-star additive Gaussian speckles with structured Quad9 Newton."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from pixint2d import AdditiveSpeckles, Camera2D, DisplacementSeries, GaussRule, MappingMode, Mesh2D, RenderOptions, Speck2D
from ._common import save_preview

PIXELS = (128, 32)
ROI_SIZE = (256.0, 64.0)
FRAME = 1
SAMPLES_PER_AXIS = 4


def main() -> None:
    path = Path(str(files("pixint2d.data").joinpath("finite_star_coarse_q9")))
    mesh = Mesh2D.from_csv(path)
    displacement = DisplacementSeries.from_csv(path)
    pattern = AdditiveSpeckles.jittered_lattice(
        kind="gaussian", speckle_diameter=5.0, black_area_fraction=.6,
        jitter_pdf="gaussian", jitter=.12, seed=3,
        bounds=(-132.0, 132.0, -34.0, 34.0), gaussian_edge_fraction=.4,
        tail_sigmas=8.0,
    )
    renderer = Speck2D(
        mesh, Camera2D(PIXELS, ROI_SIZE), pattern,
        options=RenderOptions(mapping=MappingMode.STRUCTURED_QUAD9),
    )
    result = renderer.render(displacement, frame=FRAME, integration=GaussRule(SAMPLES_PER_AXIS))
    print(f"Wrote {save_preview(result.image, 'speck2d_finite_star')}")


if __name__ == "__main__":
    main()
