"""Analyse Exp1 Riley floating-texture PSF renders against bespoke PSF renders."""
from modules import exp1_riley_analysis_common as analysis
from exp1params import exp1_output_dir
from modules.output_naming import analysis_output_root
from modules.render_selection import riley_enabled

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RILEY_TEX_DIR = exp1_output_dir("exp1_riley_render_texfloat_psf")
analysis.RESULTS_DIR_TEX = analysis_output_root("exp1", "riley_texfloat_psf")
analysis.CUSTOM_RENDER_SUFFIX = "_psf"
analysis.WRITE_RECTCONV = False
analysis.ANALYSIS_MODE = "tex"

if __name__ == "__main__":
    if not riley_enabled("texfloat_psf"):
        print("Experiment 1 Riley texfloat PSF analysis disabled by RILEY_RENDER_CASES; skipping.")
    else:
        analysis.main()
