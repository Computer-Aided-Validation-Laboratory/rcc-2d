"""Analyse bespoke Exp2 disk-PSF renders against their highest rectangular SSAA."""
import exp2_speckint2d_analysis as analysis
from exp2params import exp2_output_dir

analysis.OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs_psf")
analysis.RESULTS_DIR = exp2_output_dir("exp2_speckint2d_analysis_uvs_psf")
analysis.RENDER_SUFFIX = "_psf"

if __name__ == "__main__":
    analysis.main()
