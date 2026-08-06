"""Render Exp2 disk PSF textures with simulated quantised-f64 inputs."""
import os

os.environ["RCC_ENABLE_PSF"] = "1"
from exp2_riley_render_texfloat import main

if __name__ == "__main__":
    main(quantised_input=True)
