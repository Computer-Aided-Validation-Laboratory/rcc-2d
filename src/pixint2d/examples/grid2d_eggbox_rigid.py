"""Render the 32x32 rigid eggbox study case."""
from pixint2d import AnalyticRule, Camera2D, Eggbox, Grid2D, MappingMode, RenderOptions
from ._common import save_preview, single_element_case

PIXELS = (32, 32)
ROI_SIZE = (32.0, 32.0)
FRAME = 3

def main() -> None:
    mesh, displacement = single_element_case("plate42_cam32_quad9_rigid")
    renderer = Grid2D(mesh, Camera2D(PIXELS, ROI_SIZE), Eggbox(period=(5., 5.)),
                      options=RenderOptions(mapping=MappingMode.AFFINE))
    output = save_preview(renderer.render(displacement, frame=FRAME, integration=AnalyticRule()).image, "grid2d_eggbox_rigid")
    print(f"Wrote {output}")

if __name__ == "__main__": main()
