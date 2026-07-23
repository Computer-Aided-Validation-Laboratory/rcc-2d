"""Analyse Exp2 Riley floating-texture PSF renders against bespoke PSF renders."""
import exp2_riley_analysis_texfloat as analysis
from exp2params import exp2_output_dir

analysis.RILEY_OUTPUT_DIR = exp2_output_dir("exp2_riley_render_texfloat_psf")
analysis.REFERENCE_OUTPUT_DIR = exp2_output_dir("exp2_speckint2d_render_uvs_psf")
analysis.RESULTS_DIR = exp2_output_dir("exp2_riley_analysis_texfloat_psf")
analysis.REFERENCE_SUFFIX = "_psf"
# PSF renders are intentionally generated only for additive disks.
analysis.ANALYTIC_SPECKLE_TYPES = ["diskaddsat"]

if __name__ == "__main__":
    analysis.main()
