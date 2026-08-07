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


