"""LaTeX captions and labels for the journal-paper figure input blocks.
"""

# Float-placement specifier used by every generated journal-figure block.
# ``tbp`` lets LaTeX place a figure at the top or bottom of a text page, or
# on a float page when necessary.
FIGURE_PLACEMENT = "!htbp"

# Experiment 1
CAPTION_EXP1_FIG1 = (
    r"Digitised image RMSE (top) and maximum digitised error (bottom) "
    r"convergence of the eggbox function shaders for numerical integration "
    r"with \texttt{Grid2D} and \texttt{Riley}. All panels use the analytic "
    r"image reference."
)
LABEL_EXP1_FIG1 = "fig:exp1_eggbox_function_shader_rmse"

CAPTION_EXP1_FIG2 = (
    r"Digitised image RMSE convergence for \texttt{Riley} texture renders using " 
    r"a Catmull--Rom interpolant with "
    r"output at $12$~bits. \texttt{f64} represents a double precision floating point "
    r"value and \texttt{u12} is an unsigned $12$~bit integer. All panels use the "
    r"analytic image reference."
)
LABEL_EXP1_FIG2 = "fig:exp1_riley_textures_u12_rmse"

CAPTION_EXP1_FIG3 = (
    r"Diagonal texture and pixel integration refinement for $12$~bit "
    r"\texttt{Riley} texture renders. Solid curves use the analytic image "
    r"reference and dashed curves use the next diagonal $2\times$ refinement ($2 \times r_{px},r_{tex}$)."
)
LABEL_EXP1_FIG3 = "fig:exp1_riley_u12_diagonal_refinement"

# Experiment 2
CAPTION_EXP2_FIG1 = (
    r"Digitised image RMSE convergence of \texttt{Speck2D} for Gaussian "
    r"(a) undeformed, (b) rigid shift of $0.3$~px and additive disk (c) "
    r"undeformed, (d) rigid shift of $0.3$~px speckles. All panels use the "
    r"analytic image reference."
)
LABEL_EXP2_FIG1 = "fig:exp2_speck2d_gauss_disk_rmse"

CAPTION_EXP2_FIG2 = (
    r"Digitised RMSE convergence of \texttt{Riley} f64 texture renders using a "
    r"Catmull--Rom interpolant with "
    r"$12$~bit output: Gaussian speckles (a) undeformed, (b) rigid "
    r"shift of $0.3$~px and disk speckles (c) undeformed, (d) rigid shift "
    r"of $0.3$~px. All panels use the analytic image reference."
)
LABEL_EXP2_FIG2 = "fig:exp2_texf_gauss_disk_u12_rmse"

CAPTION_EXP2_FIG3 = (
    r"Diagonal texture and pixel integration refinement for $12$~bit "
    r"\texttt{Riley} \texttt{f64} texture renders. Solid curves use the analytic "
    r"image reference and dashed curves use the next diagonal $2\times$ refinement ($2 \times r_{px},r_{tex}$)."
)
LABEL_EXP2_FIG3 = "fig:exp2_texf_u12_diagonal_refinement"

# Experiment 3
CAPTION_EXP3_FIG1 = (
    r"Rigid body motion displacement convergence for DIC applied to the Gaussian "
    r"speckle model: (a), (c) mean displacement bias and (b), (d) "
    r"displacement field RMSE relative to the analytic reference."
)
LABEL_EXP3_FIG1 = "fig:exp3_rigid_translation_bias_rmse_refinement"

CAPTION_EXP3_FIG2 = (
    r"Displacement RMSE convergence for rigid body motion. The top row applies "
    r"DIC to Gaussian speckles and the bottom row applies the Grid Method "
    r"to the eggbox pattern. Columns refine (a, d) $r_{px}$ at $r_{tex}=1$, "
    r"(b, e) $r_{tex}$ at $r_{px}=1$, and (c, f) both jointly. All panels "
    r"use their corresponding analytic image reference."
)
LABEL_EXP3_FIG2 = "fig:exp3_rigid_refinement_independence_os_vs_ss"

CAPTION_EXP3_FIG3 = (
    r"Self convergence of displacement RMSE for DIC and the Grid Method applied "
    r"to $12$~bit images under rigid translation with diagonal "
    r"$r_{px}=r_{tex}$ refinement. The top row uses the highest completed "
    r"diagonal reference for each interpolant; the bottom row uses the next "
    r"available $2\times$ diagonal refinement."
)
LABEL_EXP3_FIG3 = "fig:exp3_rigid_self_convergence_dic_vs_grid"

CAPTION_EXP3_FIG4 = (
    r"Finite star displacement error: spatial displacement and error "
    r"distributions, and column wise RMSE versus horizontal coordinate, for "
    r"both DIC (left column) and the Grid Method (right column)."
)
LABEL_EXP3_FIG4 = "fig:exp3_finite_star_combined"
