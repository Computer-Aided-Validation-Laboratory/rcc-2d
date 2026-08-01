"""Analyse bespoke Exp2 disk-PSF renders against their highest rectangular SSAA."""
import exp2_speckint2d_analysis as analysis
from exp2params import exp2_output_dir
from modules.render_selection import custom_enabled

analysis.OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs_psf")
analysis.RESULTS_DIR = exp2_output_dir("exp2_speckint2d_analysis_uvs_psf")
analysis.RENDER_SUFFIX = "_psf"
analysis.WRITE_RECTCONV = False

if __name__ == "__main__":
    if not custom_enabled("disk_psf"):
        print("Experiment 2 disk PSF analysis disabled by CUSTOM_RENDER_CASES; skipping.")
    else:
        analysis.main()
