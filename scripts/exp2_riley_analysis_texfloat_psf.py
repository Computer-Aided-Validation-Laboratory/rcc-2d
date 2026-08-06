"""Analyse Exp2 Riley floating-texture PSF renders against bespoke PSF renders."""
import exp2_riley_analysis_texfloat as analysis
from exp2params import exp2_output_dir
from modules.output_naming import analysis_output_root
from modules.render_selection import riley_enabled

analysis.RILEY_OUTPUT_DIR = exp2_output_dir("exp2_riley_render_texfloat_psf")
analysis.REFERENCE_OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs_psf")
analysis.RESULTS_DIR = analysis_output_root("exp2", "riley_texfloat_psf")
analysis.REFERENCE_SUFFIX = "_psf"
analysis.WRITE_RECTCONV = False
analysis.RILEY_ROWS_FLIPPED = True
# PSF renders are intentionally generated only for additive disks.
analysis.ANALYTIC_SPECKLE_TYPES = ["diskaddsat"]

if __name__ == "__main__":
    if not riley_enabled("texfloat_psf"):
        print("Experiment 2 Riley texfloat PSF analysis disabled by RILEY_RENDER_CASES; skipping.")
    else:
        analysis.main()
