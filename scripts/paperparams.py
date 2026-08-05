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

ASPECT_RATIO = 1.6
FIG_WIDTH = CONTENT_WIDTH_CM/3
FIG_HEIGHT = FIG_WIDTH/ASPECT_RATIO

FIGURE_2X2_CM = (2*FIG_WIDTH, 2*FIG_HEIGHT)
FIGURE_2X3_CM = (2*FIG_WIDTH, 3*FIG_HEIGHT)
FIGURE_3X2_CM = (3*FIG_WIDTH, 2*FIG_HEIGHT)

# Paper selections.  Texf is evaluated at this camera digitisation depth;
# Texuint rows use the source texture depth named in their row labels.
PAPER_FRAME = 0
PAPER_TEXTURE_INTERPOLATOR = "line"
PAPER_TEXFLOAT_BIT_DEPTH = 8
PAPER_UINT_TEXTURE_DEPTHS = (8, 12)

# Captions are intentionally plain constants: edit these here without touching
# the data/plotting code.  The corresponding article labels are below.
FIGURE_CAPTIONS = {
    "exp1_fig1_eggbox_function_shaders": (
        "Digitised convergence of the eggbox function shader for the "
        "undeformed, rigid-body and affine-deformation cases."
    ),
    "exp1_fig2_riley_textures_pt42_cam32_q9_rig": (
        "Digitised convergence of Riley texture-shader renders for the rigid-body case."
    ),
    "exp1_fig3_riley_textures_pt42_cam32_q9_aff": (
        "Digitised convergence of Riley texture-shader renders for the affine-deformation case."
    ),
    "exp1_fig4_riley_textures_pt42_cam32_q9_qsadd": (
        "Digitised convergence of Riley texture-shader renders for the quadratic-saddle case."
    ),
}
FIGURE_LABELS = {
    "exp1_fig1_eggbox_function_shaders": "fig:exp1-eggbox-function",
    "exp1_fig2_riley_textures_pt42_cam32_q9_rig": "fig:exp1-texture-rigid",
    "exp1_fig3_riley_textures_pt42_cam32_q9_aff": "fig:exp1-texture-affine",
    "exp1_fig4_riley_textures_pt42_cam32_q9_qsadd": "fig:exp1-texture-quadsaddle",
}
