"""Analyse bespoke Exp1 PSF renders against their highest rectangular SSAA."""
import exp1_gridint2d_analysis as analysis
from exp1params import exp1_output_dir

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RESULTS_DIR = exp1_output_dir("exp1_gridint2d_analysis_uvs_psf")
analysis.RENDER_SUFFIX = "_psf"

if __name__ == "__main__":
    analysis.main()
