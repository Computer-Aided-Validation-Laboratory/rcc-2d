# Separating the Synthetic from the Systematic, Part 1: Renderer-Converged 2D Digital Image Correlation Uncertainty Quantification


## Experiment 1: TODO
- More complex deformation fields:
    - Quadratic deformation fields 

## Experiment 2: TODO
- More complex deformation fields:
    -  



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

OUT OF SCOPE?
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


## Experiments 1 & 2 Figures
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

--------------------------------------------------------------------------
**TODO**
- Experiment 3: Renders
    - Rigid body motion
    - Affine deformation
    - Finite star
    - Plate with hole   
- DIC analysis:
    - Rigid body motion
    - Affine deformation
    - Finite star

## Instructions
There is a uv venv in the .venv directory you can use.

## To Implement
