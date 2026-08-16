"""Render the 32x32 global-affine continuous-Gaussian speckle case."""

from pixint2d import AdditiveSpeckles, Camera2D, GaussRule, MappingMode, RenderOptions, Speck2D
from ._common import save_preview, single_element_case

PIXELS = (32, 32)
ROI_SIZE = (32.0, 32.0)
FRAME = 3
SAMPLES_PER_AXIS = 8


def main() -> None:
    mesh, displacement = single_element_case("plate42_cam32_quad9_affine")
    pattern = AdditiveSpeckles.jittered_lattice(
        kind="gaussian", speckle_diameter=5.0, black_area_fraction=.6,
        jitter_pdf="gaussian", jitter=.12, seed=3, bounds=(-20., 20., -20., 20.),
        gaussian_edge_fraction=.4, tail_sigmas=8.0,
    )
    renderer = Speck2D(
        mesh, Camera2D(PIXELS, ROI_SIZE), pattern,
        options=RenderOptions(mapping=MappingMode.AFFINE),
    )
    result = renderer.render(displacement, frame=FRAME, integration=GaussRule(SAMPLES_PER_AXIS))
    print(f"Wrote {save_preview(result.image, 'speck2d_affine')}")


if __name__ == "__main__":
    main()
