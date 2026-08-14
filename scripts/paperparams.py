"""Reproducible physical layout and selection constants for paper figures."""
from dataclasses import dataclass
from pathlib import Path

from paperfigtex import (
    CAPTION_EXP1_FIG1, CAPTION_EXP1_FIG3, CAPTION_EXP1_FIG4,
    CAPTION_EXP2_FIG1, CAPTION_EXP2_FIG2, CAPTION_EXP3_FIG1,
    CAPTION_EXP3_FIG2, CAPTION_EXP3_FIG3, CAPTION_EXP3_FIG4,
    LABEL_EXP1_FIG1, LABEL_EXP1_FIG3, LABEL_EXP1_FIG4,
    LABEL_EXP2_FIG1, LABEL_EXP2_FIG2, LABEL_EXP3_FIG1,
    LABEL_EXP3_FIG2, LABEL_EXP3_FIG3, LABEL_EXP3_FIG4,
)

# A4 is 21.0 x 29.7 cm.  2.5 cm margins leave a 16.0 x 24.7 cm usable area.
A4_WIDTH_CM = 21.0
PAGE_MARGIN_CM = 2.5
CONTENT_WIDTH_CM = A4_WIDTH_CM - 2.0 * PAGE_MARGIN_CM

PAPER_OUTPUT_DIR = Path("out/paper")
# Supplementary/extended figures are deliberately not mirrored to the article
# repository and are never included by the journal-preview LaTeX document.
PAPER_EXT_OUTPUT_DIR = Path("out/paper_ext")
# Insets for the Exp. 3 supplementary h/2 displacement-convergence plots.
PAPER_EXT_INSET_MIN_LEVEL = 4
PAPER_EXT_INSET_BOUNDS = (0.50, 0.45, 0.45, 0.43)
# Mirror publication-ready figures and editable LaTeX input blocks to the
# working manuscript directory, while retaining the repository copy above.
PAPER_DIR = Path.home() / "paper-render-conv-uq"
PAPER_FORMATS = ("pdf", "png")
PAPER_DPI = 300

# Paper typography: match the preview article's ``lmodern`` package exactly.
# ``usetex`` delegates all labels, titles, legends, and maths to the local
# LaTeX installation, so symbols such as ``$u_y$`` match article text too.
PAPER_USE_TEX = True
PAPER_FONT_FAMILY = "serif"
PAPER_SERIF_FONT = "Latin Modern Roman"
PAPER_TEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{lmodern}"

# Sized for approximately 10 pt article text: labels are only slightly smaller
# while legends remain readable in the A4 preview article.
FONT_SIZE_PT = 9.0
TICK_FONT_SIZE_PT = 8.0
AXIS_LABEL_FONT_SIZE_PT = 9.0
LEGEND_FONT_SIZE_PT = 8.0
COLORBAR_FONT_SIZE_PT = 8.0
# Extra leading for two-line panel titles, e.g. ``(a) Case`` / ``Ref: ...``.
# With ``PAPER_USE_TEX=True`` this is passed to TeX as ``\\[...ex]``.  The
# export helper applies this consistently to all two-line panel titles.
PANEL_TITLE_LINE_GAP_EX = 0.65
PANEL_TITLE_LINE_SPACING = 1.32  # Fallback when ``PAPER_USE_TEX=False``.

# Matplotlib line widths and marker sizes are expressed in typographic points.
# Grid2D is deliberately heavier so it remains visible beneath Riley when
# parity makes their curves coincide.
GRID_LINE_WIDTH_PT = 1.0
GRID_MARKER_SIZE_PT = 4.0
RILEY_LINE_WIDTH_PT = 0.8
RILEY_MARKER_SIZE_PT = 3.2

# Experiment 3 specific plot styling constants
EXP3_LINE_WIDTH_PT = 1.0
EXP3_MARKER_SIZE_PT = 4.0
EXP3_ANALYTIC_LINE_WIDTH_PT = 0.8
# Shared muted, colourblind-friendly line palette.  Cycle this list whenever
# a paper plot has more line series than colours; line/marker styles remain a
# second independent discriminator.  Black is deliberately reserved for an
# analytic/reference trace rather than included in the cycle.
LINE_COLOURS = [
    "#332288",  # indigo
    "#44AA99",  # teal
    "#CC6677",  # muted rose
    "#999933",  # olive
    "#88CCEE",  # pale cyan
    "#882255",  # plum
    "#DDCC77",  # sand
    "#117733",  # forest green
    "#AA4499",  # purple
    "#6699CC",  # muted blue
    "#661100",  # brown
    "#DDDDDD",  # light grey (last-resort high-count series)
]
# Exp. 3 Fig. 1 panels (c) and (d) repeat every line from the top row using
# these fixed zoom windows.  Edit them directly when changing the paper view.
EXP3_FIG1_ZOOM_BIAS_YLIM = (-0.0012, 0.0012)
EXP3_FIG1_ZOOM_RMSE_YLIM = (-0.0001, 0.0030)

@dataclass(frozen=True)
class PaperLayout:
    """Physical canvas and constrained-layout settings for a figure family."""

    canvas_cm: tuple[float, float]
    legend_band: float = 0.11
    w_pad: float = 0.08
    h_pad: float = 0.08
    wspace: float = 0.08
    hspace: float = 0.12


@dataclass(frozen=True)
class PaperFigure:
    """One generated figure and the article metadata associated with it."""

    layout: PaperLayout
    caption: str
    label: str


# -------------------------------------------------------------------------
# Figure layouts — the only panel-size controls.
# -------------------------------------------------------------------------
# Matplotlib fonts are physical points, so TeX includes every PDF at this
# exact native width.  The common line layouts consequently retain the same
# printed font and panel dimensions across experiments.
LAYOUT_LINE_1X3 = PaperLayout((CONTENT_WIDTH_CM, 7.0), legend_band=0.15)
LAYOUT_LINE_2X3 = PaperLayout((CONTENT_WIDTH_CM, 14.0), legend_band=0.13)
LAYOUT_LINE_2X2_WIDE = PaperLayout((CONTENT_WIDTH_CM, 14.0))
LAYOUT_LINE_2X2_WIDE_DETACHED = PaperLayout(
    (CONTENT_WIDTH_CM, 14.0), legend_band=0.17,
)
LAYOUT_IMAGE_1X3 = PaperLayout((CONTENT_WIDTH_CM, 6.0))
LAYOUT_IMAGE_2X3 = PaperLayout((CONTENT_WIDTH_CM, 12.0))
LAYOUT_IMAGE_3X3 = PaperLayout((CONTENT_WIDTH_CM, 12.0))
LAYOUT_IMAGE_MATRIX = PaperLayout((CONTENT_WIDTH_CM, 16.0))
LAYOUT_FIELD_4X2 = PaperLayout(
    (CONTENT_WIDTH_CM, 14.0), legend_band=0.08,
    w_pad=0.025, h_pad=0.035, wspace=0.045, hspace=0.055,
)

# Paper selections.  Texf is evaluated at this camera digitisation depth;
# Texuint rows use the source texture depth named in their row labels.
PAPER_FRAME = 0
# Short output-directory token for Riley's cubic Catmull--Rom sampler.
PAPER_TEXTURE_INTERPOLATOR = "cubiccm"
# Paper figures show each of these on-the-fly camera digitisations.
PAPER_EXP2_BIT_DEPTHS = (8, 12)
# Exp. 1 Fig. 4's long third-column titles are offset slightly left in their
# own axes coordinates to keep a clear right-hand canvas margin.
EXP1_FIG4_THIRD_COLUMN_TITLE_X = 0.46
# Color map for the difference images.
DIFFERENCE_CMAP = "RdBu"
# Fixed, symmetric image-difference limits make paper panels directly
# comparable even when the selected SSAA/OS matrix changes.
EXP1_FIG2_DIFF_LIMIT_BITS = 8.0
EXP1_FIG5_DIFF_LIMIT_BITS = 10.0
EXP2_FIG7_DIFF_LIMIT_BITS = 64.0
EXP2_FIG8_DIFF_LIMIT_BITS = 2.0
# Slim, tall colourbars for multi-panel difference-map matrices.  Exp. 1
# Fig. 4 intentionally retains its original default colourbar geometry.
DIFFERENCE_MATRIX_COLORBAR_FRACTION = 0.035
DIFFERENCE_MATRIX_COLORBAR_ASPECT = 35
DIFFERENCE_MATRIX_COLORBAR_SHRINK = 0.90
DIFFERENCE_MATRIX_COLORBAR_PAD = 0.015

# SSAA and Oversampling constants for difference images.
# Function-shader difference-map panels for Exp. 1 Fig. 2.
EXP1_FIG2_DIFF_SSAA_LEVELS = (1, 2, 16)
EXP2_DIFF_SSAA_LEVELS = (1, 8, 32)
EXP2_DIFF_OVERSAMPLES = (1, 8, 32)

# Riley texture figures are each generated with one selected interpolant.  The
# token is deliberately part of every output stem, preventing accidental
# confusion when PAPER_TEXTURE_INTERPOLATOR is changed for a comparison.
# Each texture pattern has one two-row f64 convergence figure: u8 camera
# digitisation on top and u12 camera digitisation below.
EXP2_FIG2_STEM = f"exp2_fig2_texf_gauss_disk_u12_{PAPER_TEXTURE_INTERPOLATOR}_rmse"
EXP2_FIG3_STEM = f"exp2_fig3_riley_texf_disk_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"
EXP2_FIG4_STEM = f"exp2_fig4_riley_texf_gauss_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"

# Selected deformation cases and frames for difference images.
EXP1_DIFF_FUNC_CASE = "pt42_cam32_q9_rig"
EXP1_DIFF_FUNC_FRAME = 3
EXP1_DIFF_FUNC_LABEL = "0.3-pixel rigid-body motion"

EXP1_DIFF_TEX_CASE = "pt42_cam32_q9_rig"
EXP1_DIFF_TEX_FRAME = 3
EXP1_DIFF_TEX_LABEL = "0.3-pixel rigid-body motion"

EXP2_DIFF_TEX_CASE = "pt42_cam32_q9_rig"
EXP2_DIFF_TEX_FRAME = 3
EXP2_DIFF_TEX_LABEL = "0.3-pixel rigid-body motion"


# Experiment 3 paper selections.
EXP3_RIGID_CASE = "pt516_cam512_q9_rig"
EXP3_AFFINE_CASE = "pt516_cam512_q9_aff"
EXP3_CHIRP_CASE = "pt260x65_cam256_q9_chirp"
EXP3_BIT_DEPTH = 12
# Finite-star Fig. 4 compares the deliberately under-resolved render with this
# completed, highest diagonal Riley texture render for both DIC and Grid.
EXP3_FIG4_REFERENCE_SSAA = 128
EXP3_FIG4_REFERENCE_OSAMP = 128

# Generated-figure registry: layout, caption, and LaTeX label share one key.
PAPER_FIGURES = {
    "exp1_fig1_eggbox_function_shaders_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG1, LABEL_EXP1_FIG1,
    ),
    "exp1_fig3_riley_textures_b8_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG3, LABEL_EXP1_FIG3,
    ),
    "exp1_fig4_riley_textures_b12_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG4, LABEL_EXP1_FIG4,
    ),
    "exp2_fig1_speck2d_gauss_disk_rmse": PaperFigure(
        LAYOUT_LINE_2X2_WIDE, CAPTION_EXP2_FIG1, LABEL_EXP2_FIG1,
    ),
    EXP2_FIG2_STEM: PaperFigure(
        LAYOUT_LINE_2X2_WIDE, CAPTION_EXP2_FIG2, LABEL_EXP2_FIG2,
    ),
    "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12": PaperFigure(
        LAYOUT_LINE_2X2_WIDE_DETACHED, CAPTION_EXP3_FIG1, LABEL_EXP3_FIG1,
    ),
    "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12": PaperFigure(
        LAYOUT_LINE_1X3, CAPTION_EXP3_FIG2, LABEL_EXP3_FIG2,
    ),
    "exp3_riley_gauss_fig3_rigid_self_convergence_dic_vs_grid_b12": PaperFigure(
        LAYOUT_LINE_1X3, CAPTION_EXP3_FIG3, LABEL_EXP3_FIG3,
    ),
    "exp3_riley_gauss_fig4_finite_star_combined_b12": PaperFigure(
        LAYOUT_FIELD_4X2, CAPTION_EXP3_FIG4, LABEL_EXP3_FIG4,
    ),
}
