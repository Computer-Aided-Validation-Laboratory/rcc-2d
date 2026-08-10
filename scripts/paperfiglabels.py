"""Text labels, titles, and legends for all paper figures."""

# ----------------------------------------------------
# Shared / General Labels
# ----------------------------------------------------
LABEL_AXIS_INTEGRATION = "Axis integration samples"
LABEL_1_LSB = "1 LSB"
LABEL_025_LSB = "0.25 LSB"
LABEL_NO_DATA = "No completed render data"

# ----------------------------------------------------
# Experiment 1 & 2 Labels
# ----------------------------------------------------
LABEL_DIGITISED_RMSE = "Image RMSE [bits]"
LABEL_MAX_DIGITISED_ERR = "Max. image err. [bits]"
LABEL_DIGITISED_DIFF = "Image difference [bits]"

TITLE_UNDEFORMED = "Undeformed"
TITLE_RIGID_03PX = "Rigid 0.3px"
TITLE_AFFINE_03PX = "Affine 0.3px"

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
LABEL_SELF_CONV_RMSE = "Self-conv. RMSE [px] at 0.3 px def."

TITLE_ANALYTIC_REF = "Analytic Ref."

# Figure 1 Labels
LABEL_IMPOSED_TRANSLATION_PX = "Imposed translation [px]"
LABEL_MEAN_BIAS_PX = "Mean bias [px]"
TITLE_FIG1_A = "(a) Rigid subpixel bias (all cases)"
TITLE_FIG1_B = "(b) Image RMSE vs. Analytic Ref. (all cases)"
TITLE_FIG1_C = "(c) Rigid subpixel bias"
TITLE_FIG1_D = "(d) Image RMSE vs. Analytic Ref."
LABEL_FIG1_RILEY_TEMPLATE = "Riley {name} (Tex-OS={osamp}, Px-SS={ssaa})"

# Figure 2 Labels
LABEL_PX_INTEGRATION = "Axis integration samples (Px-SS)"
LABEL_TEX_OVERSAMPLING = "Texture oversampling (Tex-OS)"
LABEL_REF_LEVEL_OS_SS = "Refinement level (Tex-OS=Px-SS)"
LABEL_RMSE_AT_03PX = "Disp. RMSE [px] at 0.3 px def."
TITLE_FIG2_A = "(a) Fixed Tex-OS=1, Refine Px-SS"
TITLE_FIG2_B = "(b) Fixed Px-SS=1, Refine Tex-OS"
TITLE_FIG2_C = "(c) Diagonal Refinement: Px-SS,Tex-OS"
LABEL_FIG2_A_TEMPLATE = "{name} (Tex-OS=1)"
LABEL_FIG2_B_TEMPLATE = "{name} (Px-SS=1)"
LABEL_FIG2_C_TEMPLATE = "{name} (Tex-OS=Px-SS)"

# Figure 3 Labels
TITLE_FIG3_A = "(a) DIC self-conv. (Rigid)"
TITLE_FIG3_B = "(b) Grid Method self-conv. (Rigid)"
TITLE_FIG3_C = "(c) DIC self-conv. (Affine)"
TITLE_FIG3_D = "(d) Grid Method self-conv. (Affine)"
LABEL_FIG3_DIC_TEMPLATE = "Riley {name}"
LABEL_FIG3_GRID_TEMPLATE = "Grid Method ({name})"

# Figure 4 & 5 Labels
TITLE_FIG4_A = "(a) Reference displacement $u_y$"
TITLE_FIG4_B_TEMPLATE = (
    "(b) Riley {name} (Tex-OS={osamp}, Px-SS={ssaa}) displacement $u_y$"
)
TITLE_FIG4_C_TEMPLATE = (
    "(c) Riley {name} (Tex-OS={osamp}, Px-SS={ssaa}) $u_y$ difference map"
)
TITLE_FIG4_D = "(d) $u_y$ RMSE along frequency gradient"
TITLE_FIG4_E = "(e) Reference displacement $u_y$"
TITLE_FIG4_F_TEMPLATE = (
    "(f) Riley {name} (Tex-OS={osamp}, Px-SS={ssaa}) displacement $u_y$"
)
TITLE_FIG4_G_TEMPLATE = (
    "(g) Riley {name} (Tex-OS={osamp}, Px-SS={ssaa}) $u_y$ difference map"
)
TITLE_FIG4_H = "(h) $u_y$ RMSE along frequency gradient"
LABEL_FIG4_5_PROFILE_TEMPLATE = "{name} ({osamp}, {ssaa})"
