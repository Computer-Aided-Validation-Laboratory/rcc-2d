"""Analyse Exp1 Riley floating-texture PSF renders against bespoke PSF renders."""
import exp1_riley_analysis_common as analysis
from exp1params import exp1_output_dir

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RILEY_TEX_DIR = exp1_output_dir("exp1_riley_render_texfloat_uvs_psf")
analysis.RESULTS_DIR_TEX = exp1_output_dir("exp1_riley_analysis_texfloat_psf")
analysis.CUSTOM_RENDER_SUFFIX = "_psf"
analysis.ANALYSIS_MODE = "tex"

if __name__ == "__main__":
    analysis.main()
