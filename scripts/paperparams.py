"""Reproducible physical layout and selection constants for paper figures."""
from pathlib import Path

# A4 is 21.0 x 29.7 cm.  2.5 cm margins leave a 16.0 x 24.7 cm usable area.
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
PAGE_MARGIN_CM = 2.0
CONTENT_WIDTH_CM = A4_WIDTH_CM - 2.0 * PAGE_MARGIN_CM
CONTENT_HEIGHT_CM = A4_HEIGHT_CM - 2.0 * PAGE_MARGIN_CM

PAPER_OUTPUT_DIR = Path("out/paper")
PAPER_FORMATS = ("pdf", "png")
PAPER_DPI = 300

# Sized for approximately 10 pt article text: labels are only slightly smaller
# while legends remain readable in a two-column figure.
FONT_SIZE_PT = 7.0
TICK_FONT_SIZE_PT = 6.0
AXIS_LABEL_FONT_SIZE_PT = 7.0
LEGEND_FONT_SIZE_PT = 6.0
SUPTITLE_FONT_SIZE_PT = 8.0

# Matplotlib line widths and marker sizes are expressed in typographic points.
# Grid2D is deliberately heavier so it remains visible beneath Riley when
# parity makes their curves coincide.
GRID_LINE_WIDTH_PT = 1.25
GRID_MARKER_SIZE_PT = 3.0
RILEY_LINE_WIDTH_PT = 1.0
RILEY_MARKER_SIZE_PT = 2.0

AR = 1.2
FIG_DIM = CONTENT_WIDTH_CM/3
# (HEIGHT,WIDTH)
FIGURE_2X2_CM = (2*FIG_DIM, 2*FIG_DIM/AR)
FIGURE_2X3_CM = (3*FIG_DIM, 2*FIG_DIM/AR)
FIGURE_3X2_CM = (2*FIG_DIM, 3*FIG_DIM/AR)
FIGURE_4X4_CM = (CONTENT_WIDTH_CM, CONTENT_WIDTH_CM)

# Paper selections.  Texf is evaluated at this camera digitisation depth;
# Texuint rows use the source texture depth named in their row labels.
PAPER_FRAME = 0
# Short output-directory token for Riley's cubic Catmull--Rom sampler.
PAPER_TEXTURE_INTERPOLATOR = "cubiccm"
PAPER_TEXFLOAT_BIT_DEPTH = 8
PAPER_UINT_TEXTURE_DEPTHS = (8, 12)
# Exp2 Speck2D figures show each of these on-the-fly camera digitisations.
PAPER_EXP2_BIT_DEPTHS = (8, 12)
# Figure 3 has one error metric so its rows can compare disk and Gaussian
# textures directly.  Use the conservative maximum digitised error by default.
PAPER_EXP2_TEX_METRIC = "max_eb"
PAPER_EXP2_TEX_METRIC_LABEL = "Max. digitised err. [bits]"

# Captions are intentionally plain constants: edit these here without touching
# the data/plotting code.  The corresponding article labels are below.
FIGURE_CAPTIONS = {
    "exp1_fig1_eggbox_function_shaders": (
        "Digitised convergence of the eggbox function shader for the "
        "undeformed, rigid-body and affine-deformation cases."
    ),
    "exp1_fig2_riley_texf_b8": (
        "Eight-bit digitised convergence of Riley f64 texture-shader renders."
    ),
    "exp1_fig3_riley_texu8_b8": (
        "Eight-bit digitised convergence of Riley u8 texture-shader renders."
    ),
    "exp1_fig4_riley_texu12_b8": (
        "Eight-bit digitised convergence of Riley u12 texture-shader renders."
    ),
    "exp1_fig5_riley_texf_b12": (
        "Twelve-bit digitised convergence of Riley f64 texture-shader renders."
    ),
    "exp1_fig6_affine_eggbox_difference_maps": (
        "Signed 8-bit grey-level differences between the analytic Eggbox image and "
        "Riley function-shader images for 0.3-pixel affine deformation."
    ),
    "exp2_fig1_speck2d_disk": (
        "Digitised convergence of the bespoke Speck2D renderer for additive disk speckles."
    ),
    "exp2_fig2_speck2d_gauss": (
        "Digitised convergence of the bespoke Speck2D renderer for additive Gaussian speckles."
    ),
    "exp2_fig3_riley_texf": (
        "Digitised convergence of Riley f64 texture-shader renders for additive disk and Gaussian speckles."
    ),
    "exp2_fig4_riley_texf_disk_difference_maps": (
        "Signed 8-bit grey-level differences between the analytic additive-disk image and "
        "Riley f64 texture-shader images for 0.3-pixel rigid-body motion."
    ),
    "exp2_fig5_riley_texf_gauss_difference_maps": (
        "Signed 8-bit grey-level differences between the analytic additive-Gaussian image and "
        "Riley f64 texture-shader images for 0.3-pixel rigid-body motion."
    ),
}
FIGURE_LABELS = {
    "exp1_fig1_eggbox_function_shaders": "fig:exp1-eggbox-function",
    "exp1_fig2_riley_texf_b8": "fig:exp1-texf-b8",
    "exp1_fig3_riley_texu8_b8": "fig:exp1-texu8-b8",
    "exp1_fig4_riley_texu12_b8": "fig:exp1-texu12-b8",
    "exp1_fig5_riley_texf_b12": "fig:exp1-texf-b12",
    "exp1_fig6_affine_eggbox_difference_maps": "fig:exp1-affine-eggbox-difference",
    "exp2_fig1_speck2d_disk": "fig:exp2-speck2d-disk",
    "exp2_fig2_speck2d_gauss": "fig:exp2-speck2d-gauss",
    "exp2_fig3_riley_texf": "fig:exp2-riley-texf",
    "exp2_fig4_riley_texf_disk_difference_maps": "fig:exp2-riley-texf-disk-difference",
    "exp2_fig5_riley_texf_gauss_difference_maps": "fig:exp2-riley-texf-gauss-difference",
}
