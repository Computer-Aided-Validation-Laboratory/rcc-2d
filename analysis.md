# ANALYSIS

## TODO

## Experiment 1: Eggbox Grids
- Only need to present uvs, world coords are probably not needed
- Have to show Frame 0 and one deformed frame, Frame 2 or Frame 8 seem to be worst case for rigid and affine

**KEY POINTS**
- Rigid is a best case scenario for renderer convergence, affine is tougher

**RIGID**
NOTE: all have an analytic reference for the pixel integral!

PYPXINT2D:
- SSAA: Floating point max and RMSE convergence are linear in log space
- SSAA: Converges slightly quicker than the affine case, generally equivalent or one sampling level lower
- GAUSS: always converges at 4x4 regardless of bit depth

**AFFINE**
NOTE: all have an analytic reference for the pixel integral!

PYPXINT2D:
- SSAA: Floating point max and RMSE convergence are linear in log space
- SSAA: Subtle differences between frames but:
    - Frame 0 = best case scenario
    - Frame 2 = one of worst case scenarios
- SSAA: Frame 0 is easiest and converges the quickest @:
    - 8 bit: 4x4 = 0 error
    - 12 bit: 16x16 = 1 bit err, 32x32 = 0 bits error
    - 16 bit: 64x64 = 1 bit err, 512x512 = 0 bits error
- SSAA: deformed frames are much more interesting and seem to be consistent
    - Frame 2 is worst case
    - 8 bit: 4x4 = 1 bit err, 512x512 = 0 bit err
    - 12 bit: 16x16 = 1 bit err, 512x512 = 0 bits error
    - 16 bit: 64x64 = 1 bit err, ????? = 0 bits error

*NOTE*: So convergence to 1 bit err is the same for all rigid frames but 0 bits changes and goes much higher!

- GAUSS: converges extremely quickly as expected because 
- GAUSS: all converge by 8x8 samples, 8x8 only needed at 16 bits, 8,12 bit converge at 4x4 samples
- GAUSS: Frame 2 also seems to be worst case
