"""Analyse Exp1 Riley function-shader PSF renders against bespoke PSF renders."""
import exp1_riley_analysis_common as analysis
from exp1params import exp1_output_dir

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RILEY_FUNC_DIR = exp1_output_dir("exp1_riley_render_func_uvs_psf")
analysis.RESULTS_DIR_FUNC = exp1_output_dir("exp1_riley_analysis_func_uvs_psf")
analysis.CUSTOM_RENDER_SUFFIX = "_psf"
analysis.ANALYSIS_MODE = "func"

if __name__ == "__main__":
    analysis.main()
