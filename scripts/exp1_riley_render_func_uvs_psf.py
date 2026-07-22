"""Explicit Exp1 Riley function-shader entry point with camera PSF enabled."""
import os
os.environ["RCC_ENABLE_PSF"] = "1"
from exp1_riley_render_func_uvs import main
if __name__ == "__main__": main()
