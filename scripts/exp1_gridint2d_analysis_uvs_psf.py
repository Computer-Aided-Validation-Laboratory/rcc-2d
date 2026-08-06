"""Analyse bespoke Exp1 PSF renders against their highest rectangular SSAA."""
import exp1_gridint2d_analysis as analysis
from exp1params import exp1_output_dir
from modules.output_naming import analysis_output_root
from modules.render_selection import custom_enabled

analysis.OUTPUT_DIR = exp1_output_dir("exp1_gridint2d_render_uvs_psf")
analysis.RESULTS_DIR = analysis_output_root("exp1", "gridint2d_uvs_psf")
analysis.RENDER_SUFFIX = "_psf"
analysis.WRITE_RECTCONV = False

if __name__ == "__main__":
    if not custom_enabled("eggbox_psf"):
        print("Experiment 1 eggbox PSF analysis disabled by CUSTOM_RENDER_CASES; skipping.")
    else:
        analysis.main()
