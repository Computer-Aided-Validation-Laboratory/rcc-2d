"""Render the compact finite-star fixture with structured Quad9 Newton mapping."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from pixint2d import Camera2D, DisplacementSeries, Eggbox, GaussRule, Grid2D, MappingMode, Mesh2D, RenderOptions
from ._common import save_preview

PIXELS = (128, 32)
ROI_SIZE = (256.0, 64.0)
FRAME = 1
SAMPLES_PER_AXIS = 4


def main() -> None:
    path = Path(str(files("pixint2d.data").joinpath("finite_star_coarse_q9")))
    mesh = Mesh2D.from_csv(path)
    displacement = DisplacementSeries.from_csv(path)
    renderer = Grid2D(
        mesh, Camera2D(PIXELS, ROI_SIZE), Eggbox(period=(5.0, 5.0)),
        options=RenderOptions(mapping=MappingMode.STRUCTURED_QUAD9),
    )
    result = renderer.render(displacement, frame=FRAME, integration=GaussRule(SAMPLES_PER_AXIS))
    print(f"Wrote {save_preview(result.image, 'grid2d_finite_star')}")


if __name__ == "__main__":
    main()
