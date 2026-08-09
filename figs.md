# Notes Figures

I want you to switch our current 1 row 3 columns figures to 1 columns 4 row figures adding the quadratic saddle case. The current width for a single panel width and height is correct keep this

Exp1 function shaders
Rows: Undeformed, Rigid 0.3px, Affine 0.3px, Saddle 0.3px
Cols: 1 column

Exp1 texture shaders
Rows: Undeformed, Rigid 0.3px, Affine 0.3px, Saddle 0.3px
Cols: Riley Texf, Riley Tex u8, Riley Tex u12


Exp 1: shows texf vs tex u8/u12 does not matter
Exp 2: shows different texture interpolation functions: linear, cubic x2, lanczos3


----------------------------------------------------------
For experiment 2 I want to reorganise the figures - we don't need to show any quantised textures these can be removed.

- exp3 analysis - need exclude 1 subset or 2x grid pitch around all edges from the difference maps if we are not already. This exclusion boundary should be a CONSTANT.
- exp3 analysis - need to use DIC on analytic reference images as the basline where possible, the reference case should be accurately stated in the figure title
- exp analysis - convergence figures should plot the following instead of individual frames
    - X Axis: Axis integration samples explicitly labelling the number i.e. 1,2,4,8,16 as ticks 
    - Y Axis: Disp. Err. [px] (clearly marking 0.01 px error)
    - Rows: Frame 1, Frame 3, Frame 5, Frame 7, Frame 10 
    - Cols: 1) RMSE error, 2) max(abs()) error


---------------------------------------------------------

A bunch of fixes for exp3 analysis scripts:

Fix this path
/home/lloydf/rcc-2d/out/exp3_analysis_dic/pt516_cam512_q9_rig/exp3_riley_render_texf/diskadd_seed3_cubic_bspline

- I have added a bunch of renders for a linear interpolant for Riley and these only seem to have SS/OS = 32,64 when many different levels exist - are these actually being processed with exp3_all_analysis or not?
- dic disp err conv figures - only goes to SS=32 but there should be renders for higher SS/OS up to 128
- disp disp err conv figures - looks like these don't refresh with exp3_all_analysis.py - make sure all exp3 analysis scripts are run by this script, might also be a problem for dic_disp_freq_err


- dic rigid interp bias - can we split the figures into sub-directories by bit depth and by speckle pattern type so sub directories here:  /home/lloydf/rcc-2d/out/exp3_analysis_dic/dic_rigid_interp_bias. Should be something like: b08_gaussadd or b012_diskadd_psf etc. 
- dic disp freq err figures - instead of having seperate figures put multiple lines on one figure for diagonal refinement of ss/os, also have figures for fixed OS=1 and refine SS and fixed SS=1 and refine OS.
- All difference map directories should have an imdiff_ prefix so this directory exp3_riley_render_texf should go to dispimdiff_riley_render_texf
- These directories should change from: riley_texfloat_cubic_bspline_disp_err_conv -> disperrconv_riley_texf_cubicbspline etc
- All sub-dirs in here: /home/lloydf/rcc-2d/out/exp3_analysis_dic/dic_disp_err_conv/pt260x65_cam256_q9_chirp and the affine folder can be removed and a concise case descriptor should be added to the file names to differentiate. Instead we should group sub-dirs here by bit depth and gauss, disk, diskpsf so b08_gausscont etc. 
- For all image difference comparisons (where we show the full difference map - not the line plots) for rigid/affine can we just plot frames 0->5. 
- Sub directories in here should be split by bit depth: /home/lloydf/rcc-2d/out/exp3_analysis_dic/pt516_cam512_q9_aff/exp3_speck2d_render_ssaa/ so we should have diskadd_seed3_b08 and gaussadd_seed3_b12 etc.
- Sub directories here should also be split by bit depth:
/home/lloydf/rcc-2d/out/exp3_analysis_dic/pt260x65_cam256_q9_chirp/exp3_riley_render_texf/ so: diskadd_seed3_cubiccm -> diskadd_seed3_cubiccm_b08 and diskadd_seed3_cubiccm_b12
- Wherever we have "render" or "exp3" in a sub directory name this can be removed for example: exp3_riley_render_texf_psf -> riley_texf_psf  
- For any directory or file name where we have cubic_bspline can we shorten this to cubicbs

Fix all of this but don't run exp3_all_analysis.py once you are done just do a smoke test on all scripts to remove syntax errors and then I will manage the massive analysis run myself.
