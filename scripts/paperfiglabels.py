"""Text labels, titles, and legends for all paper figures."""

# ----------------------------------------------------
# Shared / General Labels
# ----------------------------------------------------
LABEL_AXIS_INTEGRATION = r"Axis pixel samples ($r_{px}$)"
LABEL_1_LSB = "1 LSB"
LABEL_025_LSB = "0.25 LSB"
LABEL_NO_DATA = "No completed render data"
LABEL_NO_REFERENCE = "No reference"
LABEL_COLOURBAR_PX = "px"
PANEL_PREFIX_TEMPLATE = "({letter})"
TITLE_REFERENCE_TEMPLATE = "Ref: {reference}"
TITLE_PANEL_REFERENCE_TEMPLATE = "{panel} {case}, Ref: {reference}"
TITLE_PANEL_CASE_REFERENCE_TEMPLATE = "{panel} {case}\nRef: {reference}"
TITLE_PANEL_PX_SS_TEMPLATE = r"{panel} $r_{{px}}$={ssaa}"
TITLE_PANEL_PX_SS_TEX_OS_TEMPLATE = r"{panel} $r_{{px}}$={ssaa}, $r_{{tex}}$={osamp}"
LABEL_TEX_OS_TEMPLATE = r"$r_{{tex}}$={osamp}"
LABEL_RILEY_TEMPLATE = "Riley {name}"
LABEL_REFERENCE_PX_SS_TEMPLATE = r"$r_{{px}}$ {reference}"
LABEL_ANALYTIC_REFERENCE = "Analytic Reference"
LABEL_GRID2D_METHOD_TEMPLATE = "Grid2D {method}, {bit_depth}-bit"
LABEL_RILEY_RECT_TEMPLATE = "Riley Rect, {bit_depth}-bit"
LABEL_SPECK2D_METHOD_TEMPLATE = "Speck2D {method}, {bit_depth}-bit"
LABEL_RILEY_BSPLINE = "Riley B-spline"
LABEL_RILEY_CATMULL_ROM = "Riley Catmull-Rom"
LABEL_FIG1_CATMULL_ROM_BASELINE = r"Riley Catmull-Rom ($r_{{px}}$=1, $r_{{tex}}$=1)"

# ----------------------------------------------------
# Experiment 1 & 2 Labels
# ----------------------------------------------------
LABEL_DIGITISED_RMSE = "Image RMSE [bits]"
LABEL_MAX_DIGITISED_ERR = "Max. image err. [bits]"
LABEL_DIGITISED_DIFF = "Image difference [bits]"
LABEL_MAX_DIGITISED_ERROR = "Max. digitised err. [bits]"
LABEL_MISMATCHED_PIXEL_FRACTION = "Mismatched pixel fraction"
LABEL_AXIS_REFINEMENT_LEVEL = "Axis refinement level"
TITLE_H2_PX_SS = r"2x $r_{px}$"
TITLE_H2_TEX_OS = r"2x $r_{tex}$"
TITLE_H2_DIAGONAL = r"2x ($r_{px}$, $r_{tex}$)"
TITLE_H2_DISPLACEMENT = "h/2 displacement self-conv."

TITLE_UNDEFORMED = "Undeformed"
TITLE_RIGID_03PX = "Rigid 0.3px"
TITLE_AFFINE_03PX = "Affine 0.3px"
TITLE_EXP1_TEXTURE_ROW_F64_U8 = "Riley, In: Tex f64, Out: u8"
TITLE_EXP1_TEXTURE_ROW_U8_U8 = "Riley, In: Tex u8, Out: u8"
TITLE_EXP1_TEXTURE_ROW_F64_U12 = "Riley, In: Tex f64, Out: u12"
TITLE_EXP1_TEXTURE_ROW_U12_U12 = "Riley, In: Tex u12, Out: u12"
TITLE_EXP2_SPECK2D_PANEL_TEMPLATE = (
    "{panel} {pattern} Speckle, {deformation}\nRef: {reference}"
)
TITLE_TEXTURE_CONVERGENCE_PANEL_TEMPLATE = (
    "{panel} {texture}\n{deformation}, Ref: {reference}"
)
# Supplementary diagonal self-convergence panels identify the render only;
# their two reference definitions are stated in the shared legend.
TITLE_EXT_DIAGONAL_TEXTURE_PANEL_TEMPLATE = "{panel} {texture}\n{deformation}"
LABEL_EXT_ANALYTIC_REFERENCE = "Ref: Analytic"
LABEL_EXT_H2_DIAGONAL_REFERENCE = r"Ref: 2x ($r_{px}$, $r_{tex}$)"
TITLE_EXP2_TEXF_GAUSS = "Gauss Speckle"
TITLE_EXP2_TEXF_DISK = "Disk Speckle"
TITLE_EXP1_DIAGONAL_PANEL_TEMPLATE = "{panel} {texture}\n{deformation}"
TITLE_EXP2_DIAGONAL_PANEL_TEMPLATE = "{panel} {texture}\n{deformation}"
LABEL_DIAGONAL_ANALYTIC_TEMPLATE = "{interpolator}, Ref: Analytic"
LABEL_DIAGONAL_H2_TEMPLATE = (
    r"{interpolator}, Ref: 2x ($r_{{px}}$, $r_{{tex}}$)"
)

LABEL_PIXEL_X = "X [px]"
LABEL_PIXEL_Y = "Y [px]"

# ----------------------------------------------------
# Experiment 3 Labels
# ----------------------------------------------------
LABEL_HORIZ_COORD_PX = "X [px]"
LABEL_VERT_COORD_PX = "Y [px]"
LABEL_COLUMN_RMSE_PX = "Column RMSE [px]"
LABEL_DISP_RMSE_PX = "Disp. RMSE [px]"
LABEL_DISP_BIAS_PX = "Disp. bias [px]"
LABEL_SELF_CONV_RMSE = "RMSE [px] at 0.3 px def."

TITLE_ANALYTIC_REF = "Analytic Ref."
METHOD_DIC = "DIC"
METHOD_GRID = "Grid Method"
INTERPOLATOR_LABELS = {
    "cubic_bspline": "B-spline",
    "cubiccm": "Catmull-Rom",
    "line": "Linear",
    "lanczos3": "Lanczos",
}

# Figure 1 Labels
LABEL_IMPOSED_TRANSLATION_PX = "Imposed translation [px]"
LABEL_MEAN_BIAS_PX = "Mean bias [px]"
TITLE_FIG1_A = "(a) Rigid subpixel bias (all)"
TITLE_FIG1_B = "(b) Image RMSE vs. analytic (all)"
TITLE_FIG1_C = "(c) Rigid subpixel bias (zoom)"
TITLE_FIG1_D = "(d) Image RMSE vs. analytic (zoom)"
LABEL_FIG1_RILEY_TEMPLATE = r"Riley {name} ($r_{{px}}$={ssaa}, $r_{{tex}}$={osamp})"

# Figure 2 Labels
LABEL_PX_INTEGRATION = r"Axis pixel samples ($r_{px}$)"
LABEL_TEX_OVERSAMPLING = r"Texture oversampling ($r_{tex}$)"
LABEL_REF_LEVEL_OS_SS = r"Refinement level ($r_{px}$=$r_{tex}$)"
LABEL_RMSE_AT_03PX = "Disp. RMSE [px] at 0.3 px def."
TITLE_FIG2_A = "(a) Refine $r_{px}$\nFixed $r_{tex}$=1"
TITLE_FIG2_B = "(b) Fixed $r_{px}$=1\nRefine $r_{tex}$"
TITLE_FIG2_C = "(c) Diagonal refinement\n$r_{px}$=$r_{tex}$"
LABEL_FIG2_A_TEMPLATE = r"{name} ($r_{{tex}}$=1)"
LABEL_FIG2_B_TEMPLATE = r"{name} ($r_{{px}}$=1)"
LABEL_FIG2_C_TEMPLATE = r"{name} ($r_{{px}}$=$r_{{tex}}$)"

# Figure 3 Labels
# ``reference`` is the highest completed diagonal level in the panel.  The
# second line keeps the paper layout legible at normal article width.
TITLE_FIG3_PANEL_TEMPLATE = (
    "({panel}) {method} {deformation} Def.\n"
    r"Ref. $r_{{px}}$,$r_{{tex}}$={reference}"
)
TITLE_FIG3_H2_PANEL_TEMPLATE = (
    "({panel}) {method} {deformation} Def.\n"
    r"Ref. 2x $r_{{px}}$,$r_{{tex}}$"
)
LABEL_FIG3_DIC_TEMPLATE = "Riley {name}"
LABEL_FIG3_GRID_TEMPLATE = "Grid Method ({name})"
LABEL_MISSING_METHOD_RECORDS_TEMPLATE = "Missing {method} records"
LABEL_ERROR_LOADING_METHOD_DATA_TEMPLATE = "Error loading {method} data"
LABEL_NO_METHOD_DEFORMATION_REFERENCE_TEMPLATE = "No {method} {deformation} reference"

# Figure 4 & 5 Labels
TITLE_FIG4_A_TEMPLATE = (
    "(a) Ref. disp. $u_y$\n($r_{{px}}$,$r_{{tex}}$={ref_level})"
)
TITLE_FIG4_B_TEMPLATE = (
    "(b) {name} ($r_{{px}}$,$r_{{tex}}$={ssaa}) disp. $u_y$"
)
TITLE_FIG4_C_TEMPLATE = (
    r"(c) {name} ($r_{{px}}$,$r_{{tex}}$={ssaa}) $u_y$ diff. map"
)
TITLE_FIG4_D = "(d) $u_y$ RMSE along frequency gradient"
TITLE_FIG4_E_TEMPLATE = (
    "(e) Ref. disp. $u_y$\n($r_{{px}}$,$r_{{tex}}$={ref_level})"
)
TITLE_FIG4_F_TEMPLATE = (
    "(f) {name} ($r_{{px}}$,$r_{{tex}}$={ssaa}) disp. $u_y$"
)
TITLE_FIG4_G_TEMPLATE = (
    r"(g) {name} ($r_{{px}}$,$r_{{tex}}$={ssaa}) $u_y$ diff. map"
)
TITLE_FIG4_H = "(h) $u_y$ RMSE along frequency gradient"
LABEL_FIG4_5_PROFILE_TEMPLATE = r"{name} ($r_{{px}}$,$r_{{tex}}$={ssaa})"
