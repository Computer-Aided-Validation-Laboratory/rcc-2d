"""Explicit Exp2 Riley digitised-texture entry point with camera PSF enabled."""
import os
os.environ["RCC_ENABLE_PSF"] = "1"
from exp2_riley_render_texuint import main
if __name__ == "__main__": main()
