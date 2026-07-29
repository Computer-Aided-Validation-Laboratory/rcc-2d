# --------------------------------------------------------------------------
# Renderer Convergence Conjecture: Data & Analysis
#
# Copyright (c) 2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
# --------------------------------------------------------------------------

"""Save a heat-map visualisation of one analytic floating-point texture."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Change this single path to select the analytic f64 texture to visualise.
TEX_PATH = Path(
    "./out/exp2_analytic_speckle_textures/"
    "tex_px256_diskaddsat_blackfrac0.6_uniform_j0.25_seed3_pad4_oversamp1_analytic.npy"
)
OUTPUT_DIR = Path("./out/vis_texture")


def main() -> None:
    if not TEX_PATH.exists():
        raise FileNotFoundError(f"Texture does not exist: {TEX_PATH}")
    texture = np.load(TEX_PATH)
    if texture.ndim != 2:
        raise ValueError(f"Texture must be 2D, got shape {texture.shape}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{TEX_PATH.stem}_heatmap.png"
    value_min = float(np.min(texture))
    value_max = float(np.max(texture))
    figure, axes = plt.subplots(figsize=(8, 7), constrained_layout=True)
    # Do not rescale the texture data: the colour bar uses the raw texture
    # values.  Keeping vmin at zero makes values above one immediately visible.
    image = axes.imshow(
        texture,
        cmap="viridis",
        origin="upper",
        vmin=0.0,
        vmax=max(1.0, value_max),
    )
    axes.set_title(f"{TEX_PATH.name}\nmin={value_min:g}, max={value_max:g}")
    axes.set_xlabel("Texture column")
    axes.set_ylabel("Texture row")
    figure.colorbar(image, ax=axes, label="Raw texture value")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
