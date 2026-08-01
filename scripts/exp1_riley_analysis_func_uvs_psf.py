"""Analyse Exp1 Riley function-shader PSF renders against bespoke PSF renders."""
from modules import exp1_riley_analysis_common as analysis
from exp1params import exp1_output_dir
from modules.render_selection import riley_enabled

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RILEY_FUNC_DIR = exp1_output_dir("exp1_riley_render_func_uvs_psf")
analysis.RESULTS_DIR_FUNC = exp1_output_dir("exp1_riley_analysis_func_uvs_psf")
analysis.CUSTOM_RENDER_SUFFIX = "_psf"
analysis.WRITE_RECTCONV = False
analysis.ANALYSIS_MODE = "func"

if __name__ == "__main__":
    if not riley_enabled("func_psf"):
        print("Experiment 1 Riley function PSF analysis disabled by RILEY_RENDER_CASES; skipping.")
    else:
        analysis.main()
