"""Reproducible physical layout and selection constants for paper figures."""
from dataclasses import dataclass
from pathlib import Path

from paperfigtex import (
    CAPTION_EXP1_FIG1, CAPTION_EXP1_FIG2, CAPTION_EXP1_FIG3,
    CAPTION_EXP2_FIG1, CAPTION_EXP2_FIG2, CAPTION_EXP2_FIG3,
    CAPTION_EXP3_FIG1,
    CAPTION_EXP3_FIG2, CAPTION_EXP3_FIG3, CAPTION_EXP3_FIG4,
    LABEL_EXP1_FIG1, LABEL_EXP1_FIG2, LABEL_EXP1_FIG3,
    LABEL_EXP2_FIG1, LABEL_EXP2_FIG2, LABEL_EXP2_FIG3,
    LABEL_EXP3_FIG1,
    LABEL_EXP3_FIG2, LABEL_EXP3_FIG3, LABEL_EXP3_FIG4,
)

# A4 is 21.0 x 29.7 cm.  2.5 cm margins leave a 16.0 x 24.7 cm usable area.
A4_WIDTH_CM = 21.0
PAGE_MARGIN_CM = 2.5
CONTENT_WIDTH_CM = A4_WIDTH_CM - 2.0 * PAGE_MARGIN_CM
#CONTENT_WIDTH = 14.0

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
# The preview article is for checking typography and float placement.  Keep
# this false to let LaTeX place figures around text just as it will in the
# manuscript; turn it on only when inspecting one figure per page.
PAPER_PREVIEW_CLEARPAGE = False

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

# Shared line styling for every journal and supplementary line plot.
# Values are Matplotlib points.  Inset traces use a fixed proportion of these
# values, so changing either constant scales the complete paper consistently.
LINE_WIDTH_PT = 0.9
MARKER_SIZE_PT = 3.2
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
    """A figure family's derived physical layout.

    The four public sizing constants below are the only controls for the
    printed panel geometry.  ``extra_height_cm`` is deliberately internal
    layout allowance for titles, tick labels, and an optional figure legend.
    """

    width_cm: float
    panel_height_cm: float
    extra_height_cm: float
    legend_band: float = 0.11
    w_pad: float = 0.08
    h_pad: float = 0.08
    wspace: float = 0.08
    hspace: float = 0.12

    def canvas_cm(self, rows: int) -> tuple[float, float]:
        """Return the native PDF dimensions for a requested subplot grid."""
        return self.width_cm, rows * self.panel_height_cm + self.extra_height_cm


@dataclass(frozen=True)
class PaperFigure:
    """One generated figure and the article metadata associated with it."""

    layout: PaperLayout
    caption: str
    label: str


# -------------------------------------------------------------------------
# Figure sizing — the only panel-size controls.
# -------------------------------------------------------------------------
# Change these four values to adjust every journal figure consistently.
# Width is used both by Matplotlib's native PDF canvas and by the generated
# ``\\includegraphics`` command, preserving the intended physical font size.
PAPER_FIGURE_WIDTH_CM = 16.0
LINE_PANEL_HEIGHT_CM = 5.25
IMAGE_PANEL_HEIGHT_CM = 5.0
FIELD_PANEL_HEIGHT_CM = 5.0

# Derived figure families.  Their short fixed allowances are not user-facing
# sizing knobs: they reserve room for labels, titles, and legends so that the
# panel-height constants above retain a direct, predictable meaning.
LAYOUT_LINE_1X3 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, LINE_PANEL_HEIGHT_CM, 1.25, legend_band=0.15,
)
LAYOUT_LINE_2X3 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, LINE_PANEL_HEIGHT_CM, 1.35, legend_band=0.13,
)
LAYOUT_LINE_2X2_WIDE = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, LINE_PANEL_HEIGHT_CM, 1.20,
)
# Two columns at two-thirds of the standard figure width give each line panel
# the same width as a panel in the full-width 2x3 figures.  Use this for the
# Exp. 1/2-style four-panel convergence figures rather than stretching them
# across the page.
LAYOUT_LINE_2X2_BALANCED = PaperLayout(
    PAPER_FIGURE_WIDTH_CM * (2.0 / 3.0), LINE_PANEL_HEIGHT_CM, 1.20,
)
LAYOUT_LINE_2X2_WIDE_DETACHED = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, LINE_PANEL_HEIGHT_CM, 2.00, legend_band=0.23,
)
# Exp. 3 Fig. 3 has a two-line title and inset in every panel.  Preserve the
# shared line-panel height by reserving their additional vertical overhead in
# the canvas rather than silently shrinking the axes.
LAYOUT_LINE_2X2_TITLED = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, LINE_PANEL_HEIGHT_CM, 4.10, legend_band=0.11,
)
LAYOUT_IMAGE_1X3 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, IMAGE_PANEL_HEIGHT_CM, 0.70,
)
LAYOUT_IMAGE_2X3 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, IMAGE_PANEL_HEIGHT_CM, 0.70,
)
LAYOUT_IMAGE_3X3 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, IMAGE_PANEL_HEIGHT_CM, 0.80,
)
LAYOUT_IMAGE_MATRIX = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, IMAGE_PANEL_HEIGHT_CM, 0.80,
)
LAYOUT_FIELD_4X2 = PaperLayout(
    PAPER_FIGURE_WIDTH_CM, FIELD_PANEL_HEIGHT_CM, 0.90, legend_band=0.08,
    w_pad=0.025, h_pad=0.035, wspace=0.045, hspace=0.055,
)

# Paper selections.  Texf is evaluated at this camera digitisation depth;
# Texuint rows use the source texture depth named in their row labels.
PAPER_FRAME = 0
# Short output-directory token for Riley's cubic Catmull--Rom sampler.
PAPER_TEXTURE_INTERPOLATOR = "cubiccm"
# Main-paper diagonal reference figures compare these three texture samplers.
PAPER_MAIN_TEXTURE_INTERPOLATORS = ("line", "cubic_bspline", "cubiccm")
# Reference type is encoded by colour and line style; marker shape identifies
# the interpolant.  These are deliberately separate from ``LINE_COLOURS``.
PAPER_DIAGONAL_ANALYTIC_COLOUR = "#332288"
PAPER_DIAGONAL_H2_COLOUR = "#44AA99"
PAPER_DIAGONAL_INTERPOLATOR_MARKERS = {
    "line": "o", "cubic_bspline": "s", "cubiccm": "^",
}
# Paper figures show each of these on-the-fly camera digitisations.
PAPER_EXP2_BIT_DEPTHS = (8, 12)
# Exp. 1 Fig. 2's long third-column titles are offset slightly left in their
# own axes coordinates to keep a clear right-hand canvas margin.
EXP1_FIG2_THIRD_COLUMN_TITLE_X = 0.46
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
EXP2_FIG3_STEM = "exp2_fig3_riley_texf_u12_diagonal_refinement_rmse"
EXP2_LEGACY_DIFF_DISK_STEM = (
    f"exp2_fig3_riley_texf_disk_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"
)
EXP2_LEGACY_DIFF_GAUSS_STEM = (
    f"exp2_fig4_riley_texf_gauss_{PAPER_TEXTURE_INTERPOLATOR}_difference_maps"
)

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
# Finite-star Figure 4 shows a resolved reference beside this diagonal
# texture/pixel-integration refinement and its difference from the reference.
EXP3_FIG4_MAP_LEVEL = 2
# Diagonal refinements included in the Figure 4 DIC and Grid Method profiles.
# Each level is plotted for both cubic B-spline and Catmull--Rom interpolation.
EXP3_FIG4_PROFILE_LEVELS = (2, 4, 8)

# Generated-figure registry: layout, caption, and LaTeX label share one key.
PAPER_FIGURES = {
    "exp1_fig1_eggbox_function_shaders_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG1, LABEL_EXP1_FIG1,
    ),
    "exp1_fig2_riley_textures_b12_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG2, LABEL_EXP1_FIG2,
    ),
    "exp1_fig3_riley_textures_u12_diagonal_refinement_rmse": PaperFigure(
        LAYOUT_LINE_2X3, CAPTION_EXP1_FIG3, LABEL_EXP1_FIG3,
    ),
    "exp2_fig1_speck2d_gauss_disk_rmse": PaperFigure(
        LAYOUT_LINE_2X2_BALANCED, CAPTION_EXP2_FIG1, LABEL_EXP2_FIG1,
    ),
    EXP2_FIG2_STEM: PaperFigure(
        LAYOUT_LINE_2X2_BALANCED, CAPTION_EXP2_FIG2, LABEL_EXP2_FIG2,
    ),
    EXP2_FIG3_STEM: PaperFigure(
        LAYOUT_LINE_2X2_BALANCED, CAPTION_EXP2_FIG3, LABEL_EXP2_FIG3,
    ),
    "exp3_riley_gauss_fig1_rigid_translation_bias_rmse_refinement_b12": PaperFigure(
        LAYOUT_LINE_2X2_WIDE_DETACHED, CAPTION_EXP3_FIG1, LABEL_EXP3_FIG1,
    ),
    "exp3_riley_gauss_fig2_rigid_refinement_independence_os_vs_ss_b12": PaperFigure(
        LAYOUT_LINE_1X3, CAPTION_EXP3_FIG2, LABEL_EXP3_FIG2,
    ),
    "exp3_riley_gauss_fig3_rigid_self_convergence_dic_vs_grid_b12": PaperFigure(
        LAYOUT_LINE_2X2_TITLED, CAPTION_EXP3_FIG3, LABEL_EXP3_FIG3,
    ),
    "exp3_riley_gauss_fig4_finite_star_combined_b12": PaperFigure(
        LAYOUT_FIELD_4X2, CAPTION_EXP3_FIG4, LABEL_EXP3_FIG4,
    ),
}
