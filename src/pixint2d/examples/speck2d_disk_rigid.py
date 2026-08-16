"""Render the 32x32 rigid additive-disk study case."""
from pixint2d import AnalyticRule, AdditiveSpeckles, Camera2D, MappingMode, RenderOptions, Speck2D
from ._common import save_preview, single_element_case

PIXELS=(32,32); ROI_SIZE=(32.,32.); FRAME=3

def main() -> None:
    mesh, displacement=single_element_case("plate42_cam32_quad9_rigid")
    pattern=AdditiveSpeckles.jittered_lattice(kind="disk",speckle_diameter=5.,black_area_fraction=.6,jitter_pdf="uniform",jitter=.25,seed=3,bounds=(-20.,20.,-20.,20.))
    image=Speck2D(mesh,Camera2D(PIXELS,ROI_SIZE),pattern,options=RenderOptions(mapping=MappingMode.AFFINE)).render(displacement,frame=FRAME,integration=AnalyticRule()).image
    print(f"Wrote {save_preview(image, 'speck2d_disk_rigid')}")

if __name__ == "__main__": main()
