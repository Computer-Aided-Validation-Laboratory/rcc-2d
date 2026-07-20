# Separating the Synthetic from the Systematic, Part 1: Renderer-Converged 2D Digital Image Correlation Uncertainty Quantification

## RUNNING ON WORKSTATION P0288
**TODO**
- grid, riley, texture shader
- grid, riley, function shader, uvs
- grid, riley, function shader, world coords
- gridint2d, function shader, uvs

**RUNNING**
- gridint2d, function shader, world coords

**COMPLETE**
- gridint2d, analytic texture generation

## Experiment 1: TODO
FIX: Riley scripts for riley's new texture shader interface
- Riley render scripts for floating point textures

## Experiment 2: TODO
- Riley render script for floating textures
- Analysis of riley convergence for floating point textures
- Riley render script for digitised textures
- Self convergence of Riley for digitised textures

## Experiment 3: TODO
When does renderer convergence matter? How many bits error can we live with?

- Grid method analysis of (displacement field diff): 
    - floating point images for ground truth
    - floating point vs converged digitised images = digitisation error
    - converged vs 1 LSB digitised images = LSB error
    - Higher error images?
- 2D DIC analysis of (displacement field diff):
    - floating point images for ground truth
    - floating point vs converged digitised images = digitisation error
    - converged vs 1 LSB digitised images = LSB error
    - Higher error images?

---------------------------------------------------------------------------
## Notes
**NOTES**
- Texture oversampling and SSAA crosstalk study!
    - Does texture oversampling need to X times SSAA?
- Need grey level bit depth parity tests:
    - Follow up - once we don't have bit depth parity how does this filter into the DIC UQ?
- Need to test under different DIC parameters: small subset, large subset, medium subset, different shape functions
- Need to test under different speckle patterns

**DATA SETS**
- TEXTURE: very, very high resolution - probably need 256x camera resolution
- IMAGES: Analytic sinusoidal grid images
- IMAGES: Sur boolean images
- MESH: single element full screen rigid translation for all element types
- MESH: Star like FE chirp field using quad9 with refined mesh from left to right

- Maybe? MESH: single element full sensor affine/polynomial deformation

For each case need:
1. Target render at final resolution "targ_"
2. Texture renders with P pixels padding around the edges "tex_usX" where X is the upsample value, X=1,2,4 etc
    - Upsampled textures at [] 
3. Need to be able to define uvs correctly so texture mapping is exact

Define TEX_PX_PAD: int = 5 in exp1common.py

I want to 

**Camera Parameters**
- Start with 256x256 pixels for exploration purposes
- Go to 1024x1024 for actual DIC analysis 

- Camera = 256, texture = camera+4 = 260x260 pixels at 1:1
- Tex x2 = 520
- Tex x4 = 1040
- Tex x8 = 2080
- Tex x16 = 4160
- Tex x32 = 8320
- Tex x64 = 16640

**Discretisation Axes**
- Texture resolution: texture oversampling ratio -> m_tex = h_tex / h_cam
    - 1,2,4,8,16,32,64,128,256,512 (approx 4x SSAA)
- Texture sampling function: nearest, linear, cubic, lanczos, quintic
    - Texture sampling strategy: Direct/Horner or LUT-Lerp
- SSAA: For a given PSF 
    - 1,2,4,8,16,32,64,128

**Deformation Cases**
- Rigid body translation
- Affine deformtion or polynomial field
- Finite star pattern
- Plate with a hole in tension

## Experiment 1: Sinusoidal Grid Deformation
1. Render the analytic sinusoidal grid images for the deformation cases at the target min/final res
2. Generate the sinusoidal grid images for the textures using the grid generator
3. Generate UVs for the different cases mapping the different resolution textures to FE meshes for the rigid body motion and the finite star case
4. Render the images with Riley in a loop using various refinement studies. 
    - Start with the highest refinement case and get the error as low as possible floating point
    - Then analyse the digitisation
    - Then do the refinement studies to see how the grey level error metrics converge

- *Error metric*: grey level difference in fp, 8bit, 12bit, 16bit

*NOTE*
- The sinusoidal grid should be larger than we need so our uvs are within the bounds of 0->1

**TODO**
- Need to move sinusoidal grid generation functions out into exp1_common.py, and add type hints!

## Experiment 2: Boolean Image Deformation
- *Error metric*: grey level difference in fp, 8bit, 12bit, 16bit 
- Follow the same steps as for experiment 1 but replace the sinusoidal grid with the boolean image generation model

## Experiment 3: 2D DIC Displacements
- Use output from experiment 2 and actually do the DIC on it
- Add in the plate with a hole experiment here

- Investigate:
    - Different speckle pattern realisations
    - Different subset sizes, different steps, different shape functions, 

## Experiment 4: 2D DIC Strains
- Use output from experiment 2 

--------------------------------------------------------------------------
**TODO**

- How do we analyse texture oversampling, texture interpolation function and texture evaluation mode? As well as pixel box convergence? What do we want to show?
    - 1) As texture oversampling -> inf, we approach the analytic shader
    - 2) As texture oversampling -> inf, all interpolants converge to LSB  
- USE ANALYTIC INTEGRAL TO GENERATE REFERENCE TEXTURES!!!!
- Figures:
    - 1) Function shader convergence (analytic reference)
    - 2) Coupled texture and pixel box convergence (analytic reference): heat map
        - X axis: sub-samples
        - Y axis: texture oversampling
        - colour: error metric: max err, rmse err, fraction diff px, 
    - 3) Interpolation kernel collapse: 
        - X axis
    - 4) 
    
## TODO:
CHECKED:

FIRST: 
    - check number of processors we can use to set the render going overnight
    - check RAM limit with Riley SSAA and number of raster threads, release tile size limits
1) Run render scripts for exp1 custom ortho generator, seems like some uv renders are missing
    - ONLY NEED: affine case rendered - rigid case is there
2) Run riley render scripts for higher SSAA up to ortho limit 


I am analysing our comparison of Riley's texture rendering to our custom 2D grid pixel integrator - Riley's function shader matches this perfectly for pixel box integration using world coords or uvs. However, once I start to analyse Riley's texture shader things are wrong and it looks like it is an error with how the texture shader and the function shader interpret the uvs leading to the texture shader deforming the texture. For the first frame there should be no deformation and the grid should appear as a constant P pixels per period but instead I see fringes which are a classic sign of deformed or incorrectly sampled grid. This indicates an issue with how the uvs are setup. Can you please analyse how we are using uvs for the eggbox function shader and the texture shader of riley in our ./scripts/exp1_*.py to see if you can diagnose the issue. The source code of Riley can be found here ~/riley-raster/ if needed. Riley's texture shader should be able to produce an initial undeformed grid image without fringing that matches our expected pixels per period with a 2D fft and is a close match to the function shader with uvs. 

Our existing exp1 script for Riley to render textures using digitised textures

-----------------------------------------------------------------------------------
Create for scripts in ./scripts/ for exp3. For exp3 we are going to analyse a tiny final image size of 32x32 pixels (with our normal 4 pixel border pad for textures). Create exp3params.py based on exp1params.py and exp2params.py. For exp3 we are going to need exp3_riley_render_eggbox_texf.py and exp3_riley_render_speckle_texf.py and variants based on digitised textures with _texu. For both cases I want to look at extremely high texture oversampling combined with extremely high SSAA in riley. We will also need a decreasing thread scaling to account for the amount of RAM we need for our sub-pixel buffers and massively oversampled textures. Come up with a scaling budget in our exp3params.py based on my currenty laptop with 8 cores and 20GB RAM available for the render. Set the combination of peak texture oversampling and peak SSAA based on these limits and let me know what the limit is - then make sure our Riley scripts dynamically reduce the number of active tiles as we get higher and higher on texture over sampling and SSAA. Any questions calrifying questions based on your analysis of our existing Riley render scripts for exp1 and exp2? 
