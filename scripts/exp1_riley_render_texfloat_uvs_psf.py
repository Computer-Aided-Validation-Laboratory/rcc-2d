"""Explicit Exp1 Riley floating-texture entry point with camera PSF enabled."""
import os
os.environ["RCC_ENABLE_PSF"] = "1"
from exp1_riley_render_texfloat_uvs import main
if __name__ == "__main__": main()
