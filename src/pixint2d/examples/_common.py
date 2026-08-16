"""Shared fixture loading and image writing for package examples."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image

from pixint2d import DisplacementSeries, Mesh2D, quantise_image


def single_element_case(name: str) -> tuple[Mesh2D, DisplacementSeries]:
    path = Path(str(files("pixint2d.data").joinpath("single_elem", name)))
    return Mesh2D.from_csv(path), DisplacementSeries.from_csv(path)


def save_preview(image: np.ndarray, name: str) -> Path:
    output = Path("out/pixint2d_examples"); output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{name}.npy", image)
    Image.fromarray(quantise_image(image, 8)).save(output / f"{name}_b8.tiff")
    return output
