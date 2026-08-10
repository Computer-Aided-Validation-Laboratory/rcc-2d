"""Reproducible physical layout and selection constants for paper figures."""
from pathlib import Path

# A4 is 21.0 x 29.7 cm.  2.5 cm margins leave a 16.0 x 24.7 cm usable area.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
PAGE_MARGIN_CM = 2.0
CONTENT_WIDTH_CM = A4_WIDTH_CM - 2.0 * PAGE_MARGIN_CM
CONTENT_HEIGHT_CM = A4_HEIGHT_CM - 2.0 * PAGE_MARGIN_CM

PAPER_OUTPUT_DIR = Path("out/paper")
# Mirror publication-ready figures and editable LaTeX input blocks to the
# working manuscript directory, while retaining the repository copy above.
PAPER_DIR = Path.home() / "paper-render-conv-uq"
PAPER_FORMATS = ("pdf", "png")
PAPER_DPI = 300

# Sized for approximately 10 pt article text: labels are only slightly smaller
# while legends remain readable in a two-column figure.
FONT_SIZE_PT = 7.0
TICK_FONT_SIZE_PT = 6.0
AXIS_LABEL_FONT_SIZE_PT = 7.0
LEGEND_FONT_SIZE_PT = 6.0
SUPTITLE_FONT_SIZE_PT = 8.0
COLORBAR_FONT_SIZE_PT = 6.0

# Matplotlib line widths and marker sizes are expressed in typographic points.
# Grid2D is deliberately heavier so it remains visible beneath Riley when
# parity makes their curves coincide.
GRID_LINE_WIDTH_PT = 1.0
GRID_MARKER_SIZE_PT = 3.2
RILEY_LINE_WIDTH_PT = 0.9
RILEY_MARKER_SIZE_PT = 2.4

# Experiment 3 specific plot styling constants
EXP3_LINE_WIDTH_PT = 1.0
EXP3_MARKER_SIZE_PT = 3.2
EXP3_ANALYTIC_LINE_WIDTH_PT = 0.8

AR = 1.2
FIG_DIM = CONTENT_WIDTH_CM/3
# (HEIGHT,WIDTH)
FIGURE_2X2_CM = (2*FIG_DIM, 2*FIG_DIM/AR)
FIGURE_2X3_CM = (3*FIG_DIM, 2*FIG_DIM/AR)
FIGURE_1X3_CM = (3*FIG_DIM, FIG_DIM/AR)
FIGURE_3X2_CM = (2*FIG_DIM, 3*FIG_DIM/AR)
FIGURE_3X3_CM = (3*FIG_DIM, 3*FIG_DIM/AR)
FIGURE_4X4_CM = (CONTENT_WIDTH_CM, CONTENT_WIDTH_CM)

# Experiment 3 Layout Sizing in centimetres
EXP3_FIGURE1_CM = (11.3, 10.4)
EXP3_FIGURE2_CM = (17.0, 5.2)
EXP3_FIGURE3_CM = (11.3, 10.4)
# The finite-star panels are wide, shallow fields.  Give the 4x2 layout
# enough height for readable profile plots while preserving useful map width.
EXP3_FIGURE4_CM = (17.0, 15.0)

# Paper selections.  Texf is evaluated at this camera digitisation depth;
# Texuint rows use the source texture depth named in their row labels.
PAPER_FRAME = 0
# Short output-directory token for Riley's cubic Catmull--Rom sampler.
PAPER_TEXTURE_INTERPOLATOR = "cubiccm"
PAPER_TEXFLOAT_BIT_DEPTH = 8
PAPER_UINT_TEXTURE_DEPTHS = (8, 12)
# Paper figures show each of these on-the-fly camera digitisations.
PAPER_EXP2_BIT_DEPTHS = (8, 12)
# Figure 3 has one error metric so its rows can compare disk and Gaussian
# textures directly.  Use the conservative maximum digitised error by default.
PAPER_EXP2_TEX_METRIC = "max_eb"
PAPER_EXP2_TEX_METRIC_LABEL = "Max. digitised err. [bits]"
# Color map for the difference images.
DIFFERENCE_CMAP = "RdBu"
# Fixed, symmetric image-difference limits make paper panels directly
# comparable even when the selected SSAA/OS matrix changes.
EXP1_FIG5_DIFF_LIMIT_BITS = 10.0
EXP2_FIG7_DIFF_LIMIT_BITS = 100.0
EXP2_FIG8_DIFF_LIMIT_BITS = 3.0

# SSAA and Oversampling constants for difference images.
EXP1_DIFF_SSAA_LEVELS = (1, 2, 4, 8, 64, 256)
EXP2_DIFF_SSAA_LEVELS = (1, 8, 32)
EXP2_DIFF_OVERSAMPLES = (1, 8, 32)

EXP2_FIGURE7_8_CM = FIGURE_3X3_CM

# Riley texture figures are each generated with one selected interpolant.  The
# token is deliberately part of every output stem, preventing accidental
# confusion when PAPER_TEXTURE_INTERPOLATOR is changed for a comparison.
EXP2_FIG3_STEM = f"exp2_fig3_riley_textures_disk_b8_{PAPER_TEXTURE_INTERPOLATOR}_rmse"
EXP2_FIG4_STEM = f"exp2_fig4_riley_textures_disk_b12_{PAPER_TEXTURE_INTERPOLATOR}_rmse"
EXP2_FIG5_STEM = f"exp2_fig5_riley_textures_gauss_b8_{PAPER_TEXTURE_INTERPOLATOR}_rmse"
EXP2_FIG6_STEM = f"exp2_fig6_riley_textures_gauss_b12_{PAPER_TEXTURE_INTERPOLATOR}_rmse"
EXP2_FIG7_STEM = f"exp2_fig7_riley_texf_disk_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"
EXP2_FIG8_STEM = f"exp2_fig8_riley_texf_gauss_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"

# Selected deformation cases and frames for difference images.
EXP1_DIFF_FUNC_CASE = "pt42_cam32_q9_aff"
EXP1_DIFF_FUNC_FRAME = 3
EXP1_DIFF_FUNC_LABEL = "0.3-pixel affine deformation"

EXP1_DIFF_TEX_CASE = "pt42_cam32_q9_rig"
EXP1_DIFF_TEX_FRAME = 3
EXP1_DIFF_TEX_LABEL = "0.3-pixel rigid-body motion"

EXP2_DIFF_TEX_CASE = "pt42_cam32_q9_rig"
EXP2_DIFF_TEX_FRAME = 3
EXP2_DIFF_TEX_LABEL = "0.3-pixel rigid-body motion"


FIGURE_CAPTIONS = {
    "exp1_fig1_eggbox_function_shaders_rmse": (
        "Digitised RMSE convergence of the eggbox function shader."
    ),
    "exp1_fig1_eggbox_function_shaders_max_eb": (
        "Max digitised error convergence of the eggbox function shader."
    ),
    "exp1_fig2_riley_textures_b8_rmse": (
        "Eight-bit digitised RMSE convergence of Riley renders for textures."
    ),
    "exp1_fig2_riley_textures_b8_max_eb": (
        "Eight-bit max digitised error convergence of Riley renders for textures."
    ),
    "exp1_fig3_riley_textures_b12_rmse": (
        "Twelve-bit digitised RMSE convergence of Riley renders for textures."
    ),
    "exp1_fig3_riley_textures_b12_max_eb": (
        "Twelve-bit max digitised error convergence of Riley renders for textures."
    ),
    "exp1_fig4_affine_eggbox_difference_maps": (
        "Signed 8-bit grey-level differences between the analytic Eggbox image "
        f"and Riley function-shader images for {EXP1_DIFF_FUNC_LABEL}."
    ),
    "exp1_fig5_riley_texf_difference_maps": (
        "Signed 8-bit grey-level differences between the analytic image "
        f"and Riley f64 texture-shader images for {EXP1_DIFF_TEX_LABEL}."
    ),
    "exp2_fig1_speck2d_disk_rmse": (
        "Digitised RMSE convergence of Speck2D for additive disk speckles."
    ),
    "exp2_fig1_speck2d_disk_max_eb": (
        "Max digitised error convergence of Speck2D for disk speckles."
    ),
    "exp2_fig2_speck2d_gauss_rmse": (
        "Digitised RMSE convergence of Speck2D for Gaussian speckles."
    ),
    "exp2_fig2_speck2d_gauss_max_eb": (
        "Max digitised error convergence of Speck2D for Gaussian speckles."
    ),
    EXP2_FIG3_STEM: (
        "Eight-bit digitised RMSE convergence of Riley renders for disk textures."
    ),
    EXP2_FIG4_STEM: (
        "Twelve-bit digitised RMSE convergence of Riley renders for disk textures."
    ),
    EXP2_FIG5_STEM: (
        "Eight-bit digitised RMSE convergence of Riley renders for Gaussian "
        "textures."
    ),
    EXP2_FIG6_STEM: (
        "Twelve-bit digitised RMSE convergence of Riley renders for Gaussian "
        "textures."
    ),
    EXP2_FIG7_STEM: (
        "Signed 8-bit grey-level differences between the analytic disk image "
        f"and Riley f64 texture-shader images for {EXP2_DIFF_TEX_LABEL}."
    ),
    EXP2_FIG8_STEM: (
        "Signed 8-bit grey-level differences between the analytic Gauss image "
        f"and Riley f64 texture-shader images for {EXP2_DIFF_TEX_LABEL}."
    ),
}
FIGURE_LABELS = {
    "exp1_fig1_eggbox_function_shaders_rmse": "fig:exp1-eggbox-rmse",
    "exp1_fig1_eggbox_function_shaders_max_eb": "fig:exp1-eggbox-max",
    "exp1_fig2_riley_textures_b8_rmse": "fig:exp1-riley-textures-b8-rmse",
    "exp1_fig2_riley_textures_b8_max_eb": "fig:exp1-riley-textures-b8-max",
    "exp1_fig3_riley_textures_b12_rmse": "fig:exp1-riley-textures-b12-rmse",
    "exp1_fig3_riley_textures_b12_max_eb": "fig:exp1-riley-textures-b12-max",
    "exp1_fig4_affine_eggbox_difference_maps": (
        "fig:exp1-affine-eggbox-difference"
    ),
    "exp1_fig5_riley_texf_difference_maps": (
        "fig:exp1-riley-texf-difference"
    ),
    "exp2_fig1_speck2d_disk_rmse": "fig:exp2-speck2d-disk-rmse",
    "exp2_fig1_speck2d_disk_max_eb": "fig:exp2-speck2d-disk-max",
    "exp2_fig2_speck2d_gauss_rmse": "fig:exp2-speck2d-gauss-rmse",
    "exp2_fig2_speck2d_gauss_max_eb": "fig:exp2-speck2d-gauss-max",
    EXP2_FIG3_STEM: "fig:exp2-riley-textures-disk-b8-rmse",
    EXP2_FIG4_STEM: (
        "fig:exp2-riley-textures-disk-b12-rmse"
    ),
    EXP2_FIG5_STEM: (
        "fig:exp2-riley-textures-gauss-b8-rmse"
    ),
    EXP2_FIG6_STEM: (
        "fig:exp2-riley-textures-gauss-b12-rmse"
    ),
    EXP2_FIG7_STEM: (
        "fig:exp2-riley-texf-disk-difference"
    ),
    EXP2_FIG8_STEM: (
        "fig:exp2-riley-texf-gauss-difference"
    ),
}

# ----------------------------------------------------
# Experiment 3 Paper Selection Constants
# ----------------------------------------------------
EXP3_RIGID_CASE = "pt516_cam512_q9_rig"
EXP3_AFFINE_CASE = "pt516_cam512_q9_aff"
EXP3_CHIRP_CASE = "pt260x65_cam256_q9_chirp"
EXP3_BIT_DEPTH = 12

FIGURE_CAPTIONS.update({
    "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12": (
        "Rigid-body translation convergence: (a) mean displacement bias "
        "and (b) displacement field RMSE relative to analytic reference."
    ),
    "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12": (
        "Displacement RMSE convergence with panel (c) inset showing "
        "SSAA/OS >= 4 zoom: (a) fixed oversampling Tex-OS=1 sweeping Px-SS, "
        "(b) fixed pixel integration Px-SS=1 sweeping Tex-OS, and "
        "(c) simultaneous Tex-OS=Px-SS refinement."
    ),
    "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid_b12": (
        "Self-convergence displacement RMSE for DIC and Grid Method "
        "under rigid translation (top row) and affine deformation (bottom)."
    ),
    "exp3_riley_gauss_fig4_chirp_combined_b12": (
        "Finite-star displacement error: spatial distribution and "
        "column-wise RMSE versus horizontal coordinate for both "
        "DIC (left column) and Grid Method (right column)."
    ),
})

FIGURE_LABELS.update({
    "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12": (
        "fig:exp3-rigid-translation-bias-rmse-refinement"
    ),
    "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12": (
        "fig:exp3-rigid-refinement-independence-os-vs-ss"
    ),
    "exp3_riley_gauss_fig3_affine_self_convergence_dic_vs_grid_b12": (
        "fig:exp3-affine-self-convergence-dic-vs-grid"
    ),
    "exp3_riley_gauss_fig4_chirp_combined_b12": (
        "fig:exp3-chirp-combined"
    ),
})
