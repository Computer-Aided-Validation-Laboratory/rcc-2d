"""LaTeX captions and labels for the journal-paper figure input blocks.
"""

# Experiment 1
CAPTION_EXP1_FIG1 = (
    r"Digitised image RMSE (top) and maximum digitised error (bottom) "
    r"convergence of the eggbox function shaders for numerical integration "
    r"with \texttt{Grid2D} and \texttt{Riley} against analytic renders."
)
LABEL_EXP1_FIG1 = "fig:exp1_eggbox_function_shader_rmse"

CAPTION_EXP1_FIG2 = (
    r"Digitised image RMSE convergence for \texttt{Riley} texture renders "
    r"output at 12 bits. f64 represents a double precision floating point "
    r"value and u12 is an unsigned 12 bit integer."
)
LABEL_EXP1_FIG2 = "fig:exp1_riley_textures_u12_rmse"

CAPTION_EXP1_FIG3 = (
    r"Diagonal texture and pixel-integration refinement for 12-bit "
    r"\texttt{Riley} texture renders. Solid curves use the analytic image "
    r"reference and dashed curves use the next diagonal 2x refinement."
)
LABEL_EXP1_FIG3 = "fig:exp1_riley_u12_diagonal_refinement"

# Experiment 2
CAPTION_EXP2_FIG1 = (
    r"Digitised image RMSE convergence of \texttt{Speck2D} for Gaussian "
    r"(a) undeformed, (b) rigid shift of $0.3$~px and additive-disk (c) "
    r"undeformed, (d) rigid shift of $0.3$~px speckles."
)
LABEL_EXP2_FIG1 = "fig:exp2_speck2d_gauss_disk_rmse"

CAPTION_EXP2_FIG2 = (
    r"Digitised RMSE convergence of \texttt{Riley} f64 texture renders at "
    r"12-bit camera output: Gaussian speckles (a) undeformed, (b) rigid "
    r"shift of $0.3$~px and disk speckles (c) undeformed, (d) rigid shift "
    r"of $0.3$~px."
)
LABEL_EXP2_FIG2 = "fig:exp2_texf_gauss_disk_u12_rmse"

CAPTION_EXP2_FIG3 = (
    r"Diagonal texture and pixel-integration refinement for 12-bit "
    r"\texttt{Riley} f64 texture renders. Solid curves use the analytic "
    r"image reference and dashed curves use the next diagonal 2x refinement."
)
LABEL_EXP2_FIG3 = "fig:exp2_texf_u12_diagonal_refinement"

# Experiment 3
CAPTION_EXP3_FIG1 = (
    r"Rigid-body motion displacement convergence for DIC applied to the Gaussian "
    r"blob speckle model: (a), (c) mean displacement bias and (b), (d) "
    r"displacement-field RMSE relative to the analytic reference."
)
LABEL_EXP3_FIG1 = "fig:exp3_rigid_translation_bias_rmse_refinement"

CAPTION_EXP3_FIG2 = (
    r"Displacement RMSE convergence for rigid body motion: (a) fixed texture oversampling "
    r"$r_{tex}=1$ sweeping $r_{px}$, (b) fixed pixel integration "
    r"$r_{px}=1$ sweeping $r_{tex}$, and (c) simultaneous "
    r"$r_{px}=r_{tex}$ refinement."
)
LABEL_EXP3_FIG2 = "fig:exp3_rigid_refinement_independence_os_vs_ss"

CAPTION_EXP3_FIG3 = (
    r"Self-convergence of displacement RMSE for DIC and the Grid Method applied "
    r"to 12-bit images under rigid translation with diagonal "
    r"$r_{px}=r_{tex}$ refinement. The top row uses the highest completed "
    r"diagonal reference for each interpolant; the bottom row uses the next "
    r"available 2x diagonal refinement."
)
LABEL_EXP3_FIG3 = "fig:exp3_rigid_self_convergence_dic_vs_grid"

CAPTION_EXP3_FIG4 = (
    r"Finite-star displacement error: spatial displacement and error "
    r"distributions, and column-wise RMSE versus horizontal coordinate, for "
    r"both DIC (left column) and the Grid Method (right column)."
)
LABEL_EXP3_FIG4 = "fig:exp3_finite_star_combined"
